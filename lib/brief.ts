// lib/brief.ts — reading the trial brief (MAKEFUN_PROMPT_FILE).
//
// The brief is prose with three JSON documents embedded in it. Each document
// carries a `schema` string; this module finds them by scanning the text for
// JSON objects that carry one, and takes the first document per schema:
//
//   free-model-aggregator-01/providers/v1  — the provider directory
//   free-model-aggregator-01/ratings/v1    — the quality table + best[]
//   free-model-aggregator-01/workload/v1   — the workload manifest
//
// Nothing here knows about the env, the network, or the trial driver; it only
// turns prompt text into typed rows the later phases can plan from.

import { readFileSync } from 'node:fs';

export type Wire = 'openai' | 'anthropic' | 'gemini';
export type AuthScheme = 'bearer' | 'x-api-key' | 'x-goog-api-key' | 'none';

export const PROVIDERS_SCHEMA = 'free-model-aggregator-01/providers/v1';
export const RATINGS_SCHEMA = 'free-model-aggregator-01/ratings/v1';
export const WORKLOAD_SCHEMA = 'free-model-aggregator-01/workload/v1';

/** Any JSON object. */
export type JsonObject = Record<string, unknown>;

export function isRecord(value: unknown): value is JsonObject {
  return value !== null && typeof value === 'object' && !Array.isArray(value);
}

/** One provider as the brief's directory lists it. `base` is the raw directory
 *  path (e.g. `/p/groq`) — never a URL joined to the env origin. */
export interface ProviderEntry {
  id: string;
  name: string;
  wire: Wire;
  auth: AuthScheme;
  base: string;
  key_env: string;
  key: string;
}

export interface ProviderDirectory {
  schema: string;
  providers: ProviderEntry[];
}

/** One row of the quality table: a model's rating and capabilities on one provider. */
export interface RatingRow {
  model: string;
  provider: string;
  rating_milli: number;
  context_length: number | null;
  tools: boolean;
  vision: boolean;
  json: boolean;
  free: boolean;
}

export interface RatingsTable {
  schema: string;
  ratings: RatingRow[];
  best: { id: string; rating_milli: number }[];
}

/** What one request needs from the model that serves it. */
export interface Requirement {
  tools?: boolean;
  vision?: boolean;
  json?: boolean;
  min_context?: number;
}

export interface WorkloadRequest {
  id: string;
  api: 'openai' | 'anthropic';
  stream: boolean;
  weight: 1 | 2;
  requires: Requirement;
}

export interface WorkloadManifest {
  schema: string;
  manifest: WorkloadRequest[];
}

/** The parsed brief. `error` is non-null when the file could not be read or
 *  holds no provider directory; ratings/workload may still be null on their own. */
export interface Brief {
  directory: ProviderDirectory;
  ratings: RatingsTable | null;
  workload: WorkloadManifest | null;
  error: string | null;
}

function textOf(value: unknown): string {
  return typeof value === 'string' ? value : '';
}

function numOrNull(value: unknown): number | null {
  return typeof value === 'number' && Number.isFinite(value) ? value : null;
}

function wireOf(value: unknown): Wire | null {
  return value === 'openai' || value === 'anthropic' || value === 'gemini' ? value : null;
}

function authOf(value: unknown): AuthScheme {
  if (value === 'bearer' || value === 'x-api-key' || value === 'x-goog-api-key' || value === 'none') return value;
  return 'bearer';
}

/** The closing brace of the object that opens at `start`, or -1. String
 *  contents (which may hold braces) are skipped while scanning. */
function objectEnd(text: string, start: number): number {
  let depth = 0;
  let inString = false;
  for (let i = start; i < text.length; i += 1) {
    const ch = text[i];
    if (inString) {
      if (ch === '\\') i += 1;
      else if (ch === '"') inString = false;
    } else if (ch === '"') {
      inString = true;
    } else if (ch === '{') {
      depth += 1;
    } else if (ch === '}') {
      depth -= 1;
      if (depth === 0) return i;
    }
  }
  return -1;
}

/** Every JSON object in `text` that carries a string `schema`, in file order. */
function schemaDocuments(text: string): JsonObject[] {
  const docs: JsonObject[] = [];
  for (let start = text.indexOf('{'); start !== -1; start = text.indexOf('{', start + 1)) {
    const end = objectEnd(text, start);
    if (end === -1) continue;
    try {
      const value: unknown = JSON.parse(text.slice(start, end + 1));
      if (isRecord(value) && typeof value.schema === 'string') docs.push(value);
    } catch {
      // Not a JSON object on its own; keep scanning.
    }
  }
  return docs;
}

function providerOf(value: unknown): ProviderEntry | null {
  if (!isRecord(value)) return null;
  const id = textOf(value.id);
  const wire = wireOf(value.wire);
  if (id === '' || wire === null) return null;
  // The directory writes `path` (/p/<id>); accept base_path/base_url/base too,
  // `path` first, so an absolute base_url also survives.
  const base =
    textOf(value.path) !== ''
      ? textOf(value.path)
      : textOf(value.base_path) !== ''
        ? textOf(value.base_path)
        : textOf(value.base_url) !== ''
          ? textOf(value.base_url)
          : textOf(value.base);
  return {
    id,
    name: textOf(value.name) !== '' ? textOf(value.name) : id,
    wire,
    auth: authOf(value.auth),
    base,
    key_env: textOf(value.key_env),
    key: textOf(value.key),
  };
}

function ratingOf(value: unknown): RatingRow | null {
  if (!isRecord(value)) return null;
  const model = textOf(value.model);
  const provider = textOf(value.provider);
  if (model === '' || provider === '') return null;
  return {
    model,
    provider,
    rating_milli: numOrNull(value.rating_milli) ?? 0,
    context_length: numOrNull(value.context_length),
    tools: value.tools === true,
    vision: value.vision === true,
    json: value.json === true,
    free: value.free === true,
  };
}

function workloadRequestOf(value: unknown): WorkloadRequest | null {
  if (!isRecord(value)) return null;
  const id = textOf(value.id);
  if (id === '') return null;
  const requires = isRecord(value.requires) ? value.requires : {};
  const minContext = numOrNull(requires.min_context);
  return {
    id,
    api: value.api === 'anthropic' ? 'anthropic' : 'openai',
    stream: value.stream === true,
    weight: value.weight === 2 ? 2 : 1,
    requires: {
      ...(requires.tools === true ? { tools: true } : {}),
      ...(requires.vision === true ? { vision: true } : {}),
      ...(requires.json === true ? { json: true } : {}),
      ...(minContext !== null ? { min_context: minContext } : {}),
    },
  };
}

/** Parse the brief at `path`. An empty path means "no brief was given". */
export function readBrief(path: string): Brief {
  const empty: ProviderDirectory = { schema: PROVIDERS_SCHEMA, providers: [] };
  if (path === '') {
    return { directory: empty, ratings: null, workload: null, error: 'MAKEFUN_PROMPT_FILE is not set' };
  }
  let text: string;
  try {
    text = readFileSync(path, 'utf8');
  } catch (err) {
    return { directory: empty, ratings: null, workload: null, error: `cannot read ${path}: ${(err as Error).message}` };
  }
  const docs = schemaDocuments(text);
  const directoryDoc = docs.find((d) => d.schema === PROVIDERS_SCHEMA && Array.isArray(d.providers)) ?? null;
  if (directoryDoc === null || !Array.isArray(directoryDoc.providers)) {
    return { directory: empty, ratings: null, workload: null, error: `the brief at ${path} holds no ${PROVIDERS_SCHEMA} document` };
  }
  const providers = directoryDoc.providers.flatMap((v) => {
    const p = providerOf(v);
    return p === null ? [] : [p];
  });
  const ratingsDoc = docs.find((d) => d.schema === RATINGS_SCHEMA && Array.isArray(d.ratings)) ?? null;
  const ratings: RatingsTable | null =
    ratingsDoc === null || !Array.isArray(ratingsDoc.ratings)
      ? null
      : {
          schema: RATINGS_SCHEMA,
          ratings: ratingsDoc.ratings.flatMap((v) => {
            const r = ratingOf(v);
            return r === null ? [] : [r];
          }),
          best: Array.isArray(ratingsDoc.best)
            ? ratingsDoc.best.flatMap((v) => {
                if (!isRecord(v) || textOf(v.id) === '') return [];
                return [{ id: textOf(v.id), rating_milli: numOrNull(v.rating_milli) ?? 0 }];
              })
            : [],
        };
  const workloadDoc = docs.find((d) => d.schema === WORKLOAD_SCHEMA && Array.isArray(d.manifest)) ?? null;
  const workload: WorkloadManifest | null =
    workloadDoc === null || !Array.isArray(workloadDoc.manifest)
      ? null
      : {
          schema: WORKLOAD_SCHEMA,
          manifest: workloadDoc.manifest.flatMap((v) => {
            const w = workloadRequestOf(v);
            return w === null ? [] : [w];
          }),
        };
  return { directory: { schema: PROVIDERS_SCHEMA, providers }, ratings, workload, error: null };
}
