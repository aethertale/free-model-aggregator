// lib/providers.ts — the provider clients, one per wire (Phase 3).
//
// One call, no retry: the caller walks the planner's ranked candidates and
// owns failover. Every call carries a 5 s timeout; an abort is a failed call
// the caller maps to a refusal and fails over from.
//
//   openai     POST <base>/v1/chat/completions, body verbatim except `model`
//   anthropic  POST <base>/v1/messages, `anthropic-version: 2023-06-01`
//   gemini     POST <base>/v1beta/models/<model>:generateContent or
//              :streamGenerateContent — the model lives in the URL path,
//              URL-decoded by the provider; no `model` in the body.
//
// Auth follows each provider's own scheme (see catalog.authHeaders): zai is
// an Anthropic-wire provider that still takes a bearer token; gemini takes
// `x-goog-api-key` (the env also accepts `?key=`, but the header is the
// brief's scheme).

import type { ProviderEntry } from './brief.ts';
import { isRecord } from './brief.ts';
import { authHeaders, joinBase } from './catalog.ts';
import type { Json } from './wire.ts';

/** Timeout for one provider call, in milliseconds. */
export const PROVIDER_TIMEOUT_MS = 5000;

/** The outcome of one provider call: a stream when the request streamed and
 *  the provider answered 2xx, otherwise the raw response text. */
export interface ProviderResult {
  ok: boolean;
  status: number;
  headers: Headers;
  bodyText: string | null;
  stream: ReadableStream<Uint8Array> | null;
}

/**
 * POST <base>/v1/chat/completions. `body` is sent verbatim except `model`,
 * which is replaced with the routed model id. `stream` asks for
 * `Accept: text/event-stream`; a 2xx streaming answer comes back as `stream`,
 * everything else as `bodyText`.
 */
export async function callOpenAI(
  envUrl: string,
  provider: ProviderEntry,
  body: Json,
  stream: boolean,
  model: string,
): Promise<ProviderResult> {
  const url = `${joinBase(envUrl, provider.base)}/v1/chat/completions`;
  const headers: Record<string, string> = {
    'content-type': 'application/json',
    accept: stream ? 'text/event-stream' : 'application/json',
    ...authHeaders(provider.auth, provider.key),
  };
  const res = await fetch(url, {
    method: 'POST',
    headers,
    body: JSON.stringify({ ...body, model }),
    signal: AbortSignal.timeout(PROVIDER_TIMEOUT_MS),
  });
  if (stream && res.ok && res.body !== null) {
    return { ok: true, status: res.status, headers: res.headers, bodyText: null, stream: res.body };
  }
  return { ok: res.ok, status: res.status, headers: res.headers, bodyText: await res.text(), stream: null };
}

/**
 * POST <base>/v1/messages. `body` is the Anthropic-shaped request naming
 * `model`; `stream` asks for `Accept: text/event-stream`. A 2xx streaming
 * answer comes back as `stream`, everything else as `bodyText`.
 */
export async function callAnthropic(
  envUrl: string,
  provider: ProviderEntry,
  body: Json,
  stream: boolean,
  model: string,
): Promise<ProviderResult> {
  const url = `${joinBase(envUrl, provider.base)}/v1/messages`;
  const headers: Record<string, string> = {
    'content-type': 'application/json',
    accept: stream ? 'text/event-stream' : 'application/json',
    ...authHeaders(provider.auth, provider.key),
    'anthropic-version': '2023-06-01',
  };
  const res = await fetch(url, {
    method: 'POST',
    headers,
    body: JSON.stringify({ ...body, model }),
    signal: AbortSignal.timeout(PROVIDER_TIMEOUT_MS),
  });
  if (stream && res.ok && res.body !== null) {
    return { ok: true, status: res.status, headers: res.headers, bodyText: null, stream: res.body };
  }
  return { ok: res.ok, status: res.status, headers: res.headers, bodyText: await res.text(), stream: null };
}

/**
 * POST <base>/v1beta/models/<model>:generateContent (or
 * :streamGenerateContent when streaming). The model id is in the URL path,
 * percent-encoded; the body carries no `model`. A 2xx streaming answer
 * comes back as `stream`, everything else as `bodyText`.
 */
export async function callGemini(
  envUrl: string,
  provider: ProviderEntry,
  model: string,
  body: Json,
  stream: boolean,
): Promise<ProviderResult> {
  const action = stream ? 'streamGenerateContent' : 'generateContent';
  const url = `${joinBase(envUrl, provider.base)}/v1beta/models/${encodeURIComponent(model)}:${action}`;
  const headers: Record<string, string> = {
    'content-type': 'application/json',
    accept: stream ? 'text/event-stream' : 'application/json',
    ...authHeaders(provider.auth, provider.key),
  };
  const res = await fetch(url, {
    method: 'POST',
    headers,
    body: JSON.stringify(body),
    signal: AbortSignal.timeout(PROVIDER_TIMEOUT_MS),
  });
  if (stream && res.ok && res.body !== null) {
    return { ok: true, status: res.status, headers: res.headers, bodyText: null, stream: res.body };
  }
  return { ok: res.ok, status: res.status, headers: res.headers, bodyText: await res.text(), stream: null };
}

/**
 * One provider error body, parsed in its wire's shape, for logging:
 *   OpenAI    {"error": {"message", "type", "code"}}
 *   Anthropic {"type": "error", "error": {"type", "message"}} (no `code`)
 *   Gemini    {"error": {"code": <int>, "message", "status": <string>}}
 * The raw text (first 500 chars) when the body is not one of those.
 */
export function providerErrorInfo(bodyText: string): { message: string; code: string | null } {
  try {
    const parsed: unknown = JSON.parse(bodyText);
    if (isRecord(parsed) && isRecord(parsed.error)) {
      const error = parsed.error;
      const message = typeof error.message === 'string' ? error.message : bodyText.slice(0, 500);
      const code =
        typeof error.code === 'string' ? error.code : typeof error.code === 'number' ? String(error.code) : typeof error.status === 'string' ? error.status : typeof error.type === 'string' ? error.type : null;
      return { message, code };
    }
  } catch {
    // Not JSON: fall through to the raw text.
  }
  return { message: bodyText.slice(0, 500), code: null };
}

/** The message of a provider's error body, for logging; the raw text when it is not one. */
export function providerErrorText(bodyText: string): string {
  return providerErrorInfo(bodyText).message;
}
