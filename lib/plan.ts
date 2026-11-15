// lib/plan.ts — the planner (Phase 3).
//
// Ratings + manifest → ranked candidates per request; per-provider
// call/refusal tracking; the static allocation that rations the top tiers.
//
// Candidate ranking: every free (provider, model) pair whose ratings row
// meets the request's requirements (tools/vision/json flags; min_context
// needs a non-null context_length >= the minimum), sorted by rating_milli
// desc, ties broken deterministically by (provider id, model id). The same
// model id on two providers is two distinct candidates with separate budget
// pools; capability differences apply per provider (sambanova's gemma-4-31b
// has no tools, gemini's does).
//
// Allocation rules, applied on top of the rating order:
//   - JSON-mode requests never go to an Anthropic-wire provider: its parser
//     hard-codes json_mode:false, so the echo_json check would fail. JSON
//     requests route to OpenAI-wire or Gemini providers only.
//   - The gemini pool is reserved for vision requests. A non-vision request
//     may only use a gemini model once every vision request in the manifest
//     has already been served (arrival order is manifest order, so that is
//     simply "request index past the last vision row"). gemini-3.8-flash is
//     the only 1458 and the only tools+vision top model; spending its budget
//     on a tools-only request is spending a vision request's answer.
//   - Otherwise the rating order is the allocation: openrouter-1475 first
//     for every request that can use it, then the 1470 pools, then down the
//     ladder. Serving validly at a slightly lower rating always beats
//     holding out for the top rating and serving nothing.
//
// Call-aware ranking (Phase 5): the ranking above is static, but a call's
// failure risk is not. A provider's FIRST call can never fail on schedule
// (fail slots are drawn from call 2 onward), while every later call may hit
// a scheduled fail, an outage, or a spent budget — and the risk grows as
// the provider's call count approaches its budget. So each candidate is
// scored as `weight * rating_milli - callPenalty`, where the penalty prices
// the next call's failure risk in rating units (one expected extra call is
// worth ~150 score units: the quality budget for total 99 tolerates ~380
// milli of loss, and the call budget tolerates ~4 extra calls). The
// penalty depends only on how many calls the provider has already made and
// on its structural budget tier — large-budget providers (groq, openrouter,
// pollinations, kilol) stay cheap for many calls, medium ones (gemini,
// cloudflare) get pricey past call 4, small ones (zai, sambanova, cerebras,
// ovh) are pricey from call 2. Consequences, all structural, none tuned to
// any one trial:
//   - Fresh providers (0 calls so far) outrank used ones at a nearby
//     rating, so the 10 guaranteed-safe first calls are spread across the
//     workload instead of being burned on the first few requests.
//   - Second and later calls go to large-budget providers first: a 2nd call
//     to a small provider is far likelier to be a fail slot than a 2nd call
//     to a large one.
//   - Weight multiplies the rating, so weight-2 requests still prefer the
//     top rating even at some risk, while weight-1 requests happily take a
//     safe slightly-lower pick. Quality 98 is accepted where it saves
//     calls (98 still totals 99); the floor is 98, never below.
// The same ordering drives the failover walk: a retry prefers a fresh or
// low-call-count provider over re-hammering the provider that just refused,
// because a retry on a fresh provider's first call cannot fail. The walk
// is computed once per request from the live call counts, so a failure
// mid-walk still leaves the remaining order call-aware.
//
// Dynamic tracking (budgets/fails/outage are never in the brief, so the
// tracker only ever learns from answers):
//   - per provider: calls made, consecutive refusals, dropped flag.
//   - a 402/429/>=500 is a refusal: consecutive refusals +1, and after 2
//     consecutive refusals the provider is skipped for the rest of the
//     trial (budget spent or outage — both permanent). A single refusal
//     never drops a provider; scheduled fails are isolated call numbers,
//     so the next call to the same provider often succeeds.
//   - a 401/400 is our bug (wrong key / bad model / paid model): it does
//     not count as a refusal and is not a failover event; that
//     (provider, model) is dropped for the trial instead.
//   - a success resets the provider's consecutive refusals to 0.
//
// Deterministic: no clock, no randomness; the ranking is a total order.
// The penalty uses only live call counts and fixed structural tiers — no
// trial seed, no budget or fail-schedule prediction, no trial files.
//   - per provider: calls made, consecutive refusals, dropped flag.
//   - a 402/429/>=500 is a refusal: consecutive refusals +1, and after 2
//     consecutive refusals the provider is skipped for the rest of the
//     trial (budget spent or outage — both permanent). A single refusal
//     never drops a provider; scheduled fails are isolated call numbers,
//     so the next call to the same provider often succeeds.
//   - a 401/400 is our bug (wrong key / bad model / paid model): it does
//     not count as a refusal and is not a failover event; that
//     (provider, model) is dropped for the trial instead.
//   - a success resets the provider's consecutive refusals to 0.
//
// Deterministic: no clock, no randomness; the ranking is a total order.

import type { Brief, ProviderEntry, Requirement, WorkloadManifest } from './brief.ts';
import { isRecord } from './brief.ts';
import type { Catalog } from './catalog.ts';
import type { Json } from './wire.ts';

/** One ranked pick: the provider to call and the free model id to call it with. */
export interface Candidate {
  provider: ProviderEntry;
  model: string;
  rating: number;
}

/** The tracker's per-provider state. */
export interface ProviderState {
  calls: number;
  consecutiveRefusals: number;
  dropped: boolean;
}

/** The client wire a request arrived on (decides how its body is read). */
export type ClientWire = 'openai' | 'anthropic';

/** How many consecutive refusals drop a provider for the rest of the trial. */
export const DROP_AFTER_REFUSALS = 2;

/**
 * Structural budget tier of a provider, from the generator's budget ranges
 * (large 9–16 calls, medium 6–8, small 4–6). This is a fixed design rule,
 * not a runtime observation: budgets are never in the brief and are never
 * predicted per trial. An unknown provider is treated as medium.
 */
export type BudgetTier = 'large' | 'medium' | 'small';

export function budgetTier(providerId: string): BudgetTier {
  switch (providerId) {
    case 'groq':
    case 'openrouter':
    case 'pollinations':
    case 'kilol':
      return 'large';
    case 'gemini':
    case 'cloudflare':
      return 'medium';
    case 'zai':
    case 'sambanova':
    case 'cerebras':
    case 'ovh':
      return 'small';
    default:
      return 'medium';
  }
}

/**
 * Price, in rating-milli score units, of making the provider's next call
 * (`calls` already made, so the next call is number `calls + 1`).
 *
 * The first call costs 0: fail slots start at call 2, so it cannot fail on
 * schedule. Later calls cost roughly P(that call fails) * 150, with the
 * probability growing as the call count approaches the tier's budget and
 * jumping once the count passes the tier's minimum budget (a call past the
 * budget always fails). Large tiers stay cheap longest; small tiers are
 * pricey from call 2. The table is indexed by next call number, last entry
 * repeating.
 */
const PENALTY_BY_TIER: Record<BudgetTier, number[]> = {
  // next call:  1   2   3   4   5   6   7   8   9  10  11  12  13+
  large: [0, 25, 25, 25, 25, 25, 40, 40, 40, 120, 120, 120, 300],
  medium: [0, 45, 45, 45, 90, 90, 200, 200, 400],
  small: [0, 80, 100, 90, 150, 250, 400],
};

export function callPenalty(providerId: string, calls: number): number {
  const table = PENALTY_BY_TIER[budgetTier(providerId)];
  const index = Math.min(Math.max(calls, 0), table.length - 1);
  return table[index];
}

/** Whether a message content block (either wire's spelling) carries an image. */
function hasImage(content: unknown): boolean {
  if (!Array.isArray(content)) return false;
  for (const block of content) {
    if (!isRecord(block)) continue;
    if (block.type === 'image_url' || block.type === 'image') return true;
  }
  return false;
}

/**
 * Requirements inferred from a client body alone: tools when the tools list
 * is non-empty, vision when any message carries an image block, JSON mode
 * when an OpenAI body asks for it (an Anthropic body carries no JSON-mode
 * signal). min_context is never inferred from a body — only the manifest
 * knows it.
 */
export function requirementsFromBody(body: Json, clientWire: ClientWire): Requirement {
  const requirements: Requirement = {};
  if (Array.isArray(body.tools) && body.tools.length > 0) requirements.tools = true;
  const messages = Array.isArray(body.messages) ? body.messages : [];
  for (const message of messages) {
    if (isRecord(message) && hasImage(message.content)) {
      requirements.vision = true;
      break;
    }
  }
  if (clientWire === 'openai' && isRecord(body.response_format) && (body.response_format.type === 'json_object' || body.response_format.type === 'json_schema')) {
    requirements.json = true;
  }
  return requirements;
}

/**
 * The requirements of the request at arrival position `index`: the manifest
 * row when one exists there, the body's own shape otherwise.
 */
export function requirementsFor(body: Json, workload: WorkloadManifest | null, index: number, clientWire: ClientWire): Requirement {
  const manifest = workload?.manifest ?? [];
  if (index >= 0 && index < manifest.length) return { ...manifest[index].requires };
  return requirementsFromBody(body, clientWire);
}

/** The planner: ranked candidates per request plus the refusal tracker. */
export class Planner {
  private readonly brief: Brief;
  private readonly catalog: Catalog;
  private readonly states = new Map<string, ProviderState>();
  private readonly droppedModels = new Set<string>();
  /** Manifest index of the last vision request (-1 when there is none). */
  private readonly lastVisionIndex: number;

  constructor(brief: Brief, catalog: Catalog) {
    this.brief = brief;
    this.catalog = catalog;
    const manifest = brief.workload?.manifest ?? [];
    let last = -1;
    manifest.forEach((row, i) => {
      if (row.requires.vision === true) last = i;
    });
    this.lastVisionIndex = last;
    for (const provider of brief.directory.providers) {
      this.states.set(provider.id, { calls: 0, consecutiveRefusals: 0, dropped: false });
    }
  }

  /** The tracker's state for one provider (a fresh zeroed state when unknown). */
  providerState(providerId: string): ProviderState {
    return this.states.get(providerId) ?? { calls: 0, consecutiveRefusals: 0, dropped: false };
  }

  /** Whether one (provider, model) may still be tried: provider not dropped, model not dropped. */
  isUsable(providerId: string, model: string): boolean {
    const state = this.states.get(providerId);
    if (state !== undefined && state.dropped) return false;
    return !this.droppedModels.has(`${providerId}:${model}`);
  }

  /** A 2xx answer: the provider is alive; its refusal streak resets. */
  recordSuccess(providerId: string): void {
    const state = this.states.get(providerId);
    if (state === undefined) return;
    state.calls += 1;
    state.consecutiveRefusals = 0;
  }

  /** A 402/429/>=500 (or a timed-out call): one more consecutive refusal; two in a row drops the provider. */
  recordRefusal(providerId: string): void {
    const state = this.states.get(providerId);
    if (state === undefined) return;
    state.calls += 1;
    state.consecutiveRefusals += 1;
    if (state.consecutiveRefusals >= DROP_AFTER_REFUSALS) state.dropped = true;
  }

  /** A 401/400: our bug, not a refusal — drop that (provider, model) for the trial. */
  recordConfigError(providerId: string, model: string): void {
    this.droppedModels.add(`${providerId}:${model}`);
  }

  /** Whether one catalog model is listed free (the probe's word). */
  private catalogHas(providerId: string, modelId: string): boolean {
    const provider = this.catalog.providers.find((p) => p.id === providerId);
    if (provider === undefined) return false;
    return provider.models.some((m) => m.id === modelId && m.free);
  }

  /** Whether one ratings row meets the requirements (capability flags + min_context). */
  private static meets(requirements: Requirement, tools: boolean, vision: boolean, json: boolean, contextLength: number | null): boolean {
    if (requirements.tools === true && tools !== true) return false;
    if (requirements.vision === true && vision !== true) return false;
    if (requirements.json === true && json !== true) return false;
    if (requirements.min_context !== undefined) {
      if (contextLength === null || contextLength < requirements.min_context) return false;
    }
    return true;
  }

  /**
   * The ranked candidate list for one request: every free (provider, model)
   * meeting the requirements, scored call-aware as
   * `weight * rating_milli - callPenalty(provider, callsSoFar)` (see the
   * header comment), ties broken deterministically by (provider id, model
   * id), with the JSON-wire rule, the gemini reservation, and the dynamic
   * tracker applied. A request past the last vision row may use gemini
   * models as fallbacks; earlier non-vision requests may not. The list is
   * computed once per request from the live call counts, and the server
   * walks it in order — so the first pick and every failover retry are all
   * call-aware.
   */
  candidatesFor(requirements: Requirement, requestIndex: number): Candidate[] {
    const manifest = this.brief.workload?.manifest ?? [];
    const weight = requestIndex >= 0 && requestIndex < manifest.length ? manifest[requestIndex].weight : 1;
    const scoreOf = (candidate: Candidate): number =>
      weight * candidate.rating - callPenalty(candidate.provider.id, this.providerState(candidate.provider.id).calls);
    const candidates: Candidate[] = [];
    const rows = this.brief.ratings?.ratings ?? [];
    if (rows.length > 0) {
      for (const row of rows) {
        if (!row.free) continue;
        const provider = this.brief.directory.providers.find((p) => p.id === row.provider) ?? null;
        if (provider === null) continue;
        if (!this.catalogHas(provider.id, row.model)) continue;
        if (!Planner.meets(requirements, row.tools, row.vision, row.json, row.context_length)) continue;
        if (requirements.json === true && provider.wire === 'anthropic') continue;
        if (provider.wire === 'gemini' && requirements.vision !== true && requestIndex <= this.lastVisionIndex) continue;
        if (!this.isUsable(provider.id, row.model)) continue;
        candidates.push({ provider, model: row.model, rating: row.rating_milli });
      }
    } else {
      // No ratings table in the brief: fall back to the catalog alone, with
      // unknown capabilities (only no-requirement requests can be served).
      for (const catalogProvider of this.catalog.providers) {
        const provider = this.brief.directory.providers.find((p) => p.id === catalogProvider.id) ?? null;
        if (provider === null) continue;
        for (const model of catalogProvider.models) {
          if (!model.free) continue;
          if (!Planner.meets(requirements, false, false, false, model.context_length)) continue;
          if (provider.wire === 'gemini' && requirements.vision !== true && requestIndex <= this.lastVisionIndex) continue;
          if (!this.isUsable(provider.id, model.id)) continue;
          candidates.push({ provider, model: model.id, rating: 0 });
        }
      }
    }
    candidates.sort((a, b) => {
      const scoreA = scoreOf(a);
      const scoreB = scoreOf(b);
      if (scoreB !== scoreA) return scoreB - scoreA;
      if (a.provider.id !== b.provider.id) return a.provider.id < b.provider.id ? -1 : 1;
      return a.model < b.model ? -1 : a.model > b.model ? 1 : 0;
    });
    return candidates;
  }
}
