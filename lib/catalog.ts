// lib/catalog.ts — probing providers' catalog routes and building the catalog.
//
// At startup the entry asks every provider in the brief for its model list, in
// that provider's own wire:
//
//   openai     GET <base>/v1/models       Authorization: Bearer <key>
//   anthropic  GET <base>/v1/models       x-api-key: <key>, anthropic-version: 2023-06-01
//   gemini     GET <base>/v1beta/models   x-goog-api-key: <key>
//
// A probe that fails (refused connection, non-2xx, unparseable body) leaves its
// provider out of the catalog; the trial goes on without it.
//
// The catalog lists FREE models only: a paid model is never a legal pick, and
// omitting one loses nothing, while listing one as free loses a point. Where
// the quality table has a row for a model it wins for `free` and
// `context_length`; otherwise the listing's own fields decide. Models are
// id-sorted and deduplicated per provider; `base_url` is the directory's raw
// path (e.g. `/p/groq`), never joined to the env origin.

import type { AuthScheme, ProviderEntry, RatingsTable, RatingRow, Wire } from './brief.ts';
import { isRecord } from './brief.ts';

/** One model in the registered catalog. */
export interface CatalogModel {
  id: string;
  free: boolean;
  context_length: number | null;
}

/** One provider in the registered catalog. */
export interface CatalogProvider {
  id: string;
  base_url: string;
  key_env: string;
  wire: Wire;
  models: CatalogModel[];
}

/** The catalog document: what `--list` prints and what `/register` carries,
 *  byte for byte the same document. */
export interface Catalog {
  providers: CatalogProvider[];
}

/** One model as a provider's catalog route listed it. */
export interface ProbedModel {
  id: string;
  free: boolean | null;
  context_length: number | null;
}

/** The outcome of probing one provider. */
export interface Probe {
  provider: ProviderEntry;
  ok: boolean;
  status: number;
  models: ProbedModel[];
  error: string | null;
}

/** A base path joined to the env origin; an absolute base is used as it stands. */
export function joinBase(envUrl: string, base: string): string {
  const b = base.trim();
  if (/^https?:\/\//i.test(b)) return b.replace(/\/+$/, '');
  const prefix = envUrl.replace(/\/+$/, '');
  if (b === '') return prefix;
  return `${prefix}${b.startsWith('/') ? '' : '/'}${b.replace(/\/+$/, '')}`;
}

/** The headers carrying one provider's key, in its own scheme. No key, no header. */
export function authHeaders(auth: AuthScheme, key: string): Record<string, string> {
  if (key === '') return {};
  if (auth === 'bearer') return { authorization: `Bearer ${key}` };
  if (auth === 'x-api-key') return { 'x-api-key': key };
  if (auth === 'x-goog-api-key') return { 'x-goog-api-key': key };
  return {};
}

function textOf(value: unknown): string {
  return typeof value === 'string' ? value : '';
}

function numOrNull(value: unknown): number | null {
  return typeof value === 'number' && Number.isFinite(value) ? value : null;
}

function boolOrNull(value: unknown): boolean | null {
  return typeof value === 'boolean' ? value : null;
}

/** Whether a listing's price block says every price is zero (null: it says nothing). */
function zeroPriced(value: Record<string, unknown>): boolean | null {
  const pricing = isRecord(value.pricing) ? value.pricing : null;
  if (pricing === null) return null;
  const values = Object.values(pricing).flatMap((x) => {
    if (typeof x === 'number') return [x];
    if (typeof x === 'string' && x.trim() !== '') return [Number(x)];
    return [];
  });
  if (values.length === 0) return null;
  return values.every((n) => Number.isFinite(n) && n === 0);
}

/** One model as a listing wrote it, in its wire's spelling. Gemini names
 *  models `models/<id>` and reports context as `inputTokenLimit`. */
function modelIn(wire: Wire, value: unknown): ProbedModel | null {
  if (!isRecord(value)) return null;
  const id = wire === 'gemini' ? textOf(value.name).replace(/^models\//, '') : textOf(value.id);
  if (id === '') return null;
  return {
    id,
    free: boolOrNull(value.free) ?? zeroPriced(value),
    context_length: numOrNull(value.context_length) ?? numOrNull(value.context_window) ?? numOrNull(value.inputTokenLimit),
  };
}

/** The models one catalog route's answer listed, in its wire's shape. */
function modelsIn(wire: Wire, body: unknown): ProbedModel[] {
  if (!isRecord(body)) return [];
  const list = wire === 'gemini' ? body.models : body.data;
  if (!Array.isArray(list)) return [];
  return list.flatMap((v) => {
    const m = modelIn(wire, v);
    return m === null ? [] : [m];
  });
}

/** Probe one provider's catalog route, in its own wire. Never throws. */
export async function probeProvider(envUrl: string, provider: ProviderEntry, timeoutMs = 5000): Promise<Probe> {
  const base = joinBase(envUrl, provider.base);
  const url = provider.wire === 'gemini' ? `${base}/v1beta/models` : `${base}/v1/models`;
  const headers: Record<string, string> = { accept: 'application/json', ...authHeaders(provider.auth, provider.key) };
  if (provider.wire === 'anthropic') headers['anthropic-version'] = '2023-06-01';
  try {
    const res = await fetch(url, { headers, signal: AbortSignal.timeout(timeoutMs) });
    if (!res.ok) {
      return { provider, ok: false, status: res.status, models: [], error: `GET ${url} answered HTTP ${res.status}` };
    }
    const body: unknown = await res.json();
    return { provider, ok: true, status: res.status, models: modelsIn(provider.wire, body), error: null };
  } catch (err) {
    return { provider, ok: false, status: 0, models: [], error: `GET ${url} failed: ${(err as Error).message}` };
  }
}

/** The quality-table row for one provider's model, when the table has one. */
export function ratingOfModel(ratings: RatingsTable | null, providerId: string, modelId: string): RatingRow | null {
  if (ratings === null) return null;
  return ratings.ratings.find((r) => r.provider === providerId && r.model === modelId) ?? null;
}

/** Whether a model is free: the quality table's row where it has one (the
 *  sealed truth), the listing's own word where it does not, paid when neither says. */
export function modelFree(model: ProbedModel, row: RatingRow | null): boolean {
  if (row !== null) return row.free;
  return model.free ?? false;
}

/** Build the catalog from probes and the quality table: only providers that
 *  answered, only free models, id-sorted and deduplicated. The provider order
 *  follows the probe order (the brief's order). */
export function buildCatalog(probes: readonly Probe[], ratings: RatingsTable | null): Catalog {
  const providers: CatalogProvider[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    const seen = new Set<string>();
    const models: CatalogModel[] = [];
    const sorted = [...probe.models].sort((a, b) => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0));
    for (const model of sorted) {
      if (seen.has(model.id)) continue;
      seen.add(model.id);
      const row = ratingOfModel(ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      models.push({ id: model.id, free: true, context_length: row?.context_length ?? model.context_length ?? null });
    }
    providers.push({
      id: probe.provider.id,
      base_url: probe.provider.base,
      key_env: probe.provider.key_env,
      wire: probe.provider.wire,
      models,
    });
  }
  return { providers };
}

/** The flat, id-sorted, deduplicated set of free model ids in a catalog,
 *  with the provider each id was first seen on (for `owned_by`). */
export function freeModelIds(catalog: Catalog): { id: string; provider: string }[] {
  const seen = new Map<string, string>();
  for (const provider of catalog.providers) {
    for (const model of provider.models) {
      if (model.free && !seen.has(model.id)) seen.set(model.id, provider.id);
    }
  }
  return [...seen.entries()].map(([id, provider]) => ({ id, provider })).sort((a, b) => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0));
}
