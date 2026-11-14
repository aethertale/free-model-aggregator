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
   * meeting the requirements, rating desc, ties by (provider id, model id),
   * with the JSON-wire rule, the gemini reservation, and the dynamic tracker
   * applied. A request past the last vision row may use gemini models as
   * fallbacks; earlier non-vision requests may not.
   */
  candidatesFor(requirements: Requirement, requestIndex: number): Candidate[] {
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
      if (b.rating !== a.rating) return b.rating - a.rating;
      if (a.provider.id !== b.provider.id) return a.provider.id < b.provider.id ? -1 : 1;
      return a.model < b.model ? -1 : a.model > b.model ? 1 : 0;
    });
    return candidates;
  }
}
