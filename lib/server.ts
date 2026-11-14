// lib/server.ts — the entry's HTTP server (Phase 3).
//
// Routes:
//
//   GET  /v1/models              the model list, in the wire the auth header names
//   POST /v1/chat/completions    OpenAI client wire → any provider wire, JSON or SSE
//   POST /v1/messages            Anthropic client wire → any provider wire, JSON or SSE
//   everything else              404, in the wire's error shape
//
// GET /v1/models lists exactly the free model ids in the registered catalog,
// id-sorted and deduplicated. The wire is chosen by header name only: an
// `x-api-key` header means the Anthropic list shape, anything else (including
// `Authorization: Bearer`) the OpenAI list shape. The warm-up key
// `makefun-client` is never validated — the header's name is the whole signal.
// Timestamps are fixed (determinism: no wall clock in any sent value).
//
// Routing is the planner (lib/plan.ts): requirements from the manifest row by
// arrival order (the client sends r01→r20 strictly in order, one at a time),
// ranked candidates per request, walked one call each until the first
// success. Chat/messages requests are serialised (Phase 4): a parallel
// client must never produce parallel provider calls — zai answers a
// concurrent call with 429 `concurrent_requests` — and a malformed body
// consumes no manifest index. A 402/429/>=500 records a refusal against the
// to the next candidate (which may be the same provider with a different
// model); after 2 consecutive refusals a provider is skipped for the rest of
// the trial. A 401/400 is our bug: it is logged, drops that (provider, model)
// for the trial, and is not a failover event. A 5 s timeout abort counts as
// a failed call and fails over. The exact same (provider, model) is never
// retried twice in a row for one request, and no second provider is ever
// called after a success (one_provider). When the list is exhausted the wire's
// 503 answers (no_free_model / overloaded_error).
//
// Error envelopes follow each wire's own shape:
//   OpenAI    {"error": {"message", "type", "code"}}
//   Anthropic {"type": "error", "error": {"type", "message"}}

import { createServer } from 'node:http';
import type { IncomingMessage, Server, ServerResponse } from 'node:http';
import type { Brief } from './brief.ts';
import { isRecord } from './brief.ts';
import type { Catalog } from './catalog.ts';
import { freeModelIds } from './catalog.ts';
import { Planner } from './plan.ts';
import type { Candidate, ClientWire } from './plan.ts';
import { requirementsFor } from './plan.ts';
import { callAnthropic, callGemini, callOpenAI, providerErrorInfo } from './providers.ts';
import type { ProviderResult } from './providers.ts';
import {
  anthropicError,
  anthropicEventsFromOpenAI,
  anthropicModelList,
  anthropicStreamPassthrough,
  anthropicStreamToOpenAIChunks,
  anthropicToGeminiRequest,
  anthropicToOpenAI,
  anthropicToOpenAIResponse,
  geminiStreamToClientEvents,
  geminiToAnthropicResponse,
  geminiToOpenAIResponse,
  openAIError,
  openAIModelList,
  openAIToAnthropicRequest,
  openAIToGeminiRequest,
  openAIToAnthropic,
} from './wire.ts';
import type { Json } from './wire.ts';

/** Bodies are read and parsed, capped so a huge body cannot stall the server. */
const MAX_BODY_BYTES = 32 * 1024 * 1024;

type WireName = 'openai' | 'anthropic';

/** What the server needs beyond the catalog: the env origin and the brief. */
export interface ServerContext {
  envUrl: string;
  brief: Brief;
}

/** The wire a request belongs to: `/v1/messages` is Anthropic, everything else OpenAI. */
function wireOf(req: IncomingMessage): WireName {
  const path = (req.url ?? '/').split('?')[0];
  return path === '/v1/messages' ? 'anthropic' : 'openai';
}

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

function sendJson(res: ServerResponse, status: number, value: unknown): void {
  const body = `${JSON.stringify(value)}\n`;
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(body) });
  res.end(body);
}

function sendError(res: ServerResponse, wire: WireName, status: number, code: string, message: string): void {
  if (wire === 'anthropic') {
    sendJson(res, status, anthropicError(code, message));
  } else {
    sendJson(res, status, openAIError(message, code, code));
  }
}

/** Read a request body as JSON. Returns the value, or an error string. */
function readJsonBody(req: IncomingMessage): Promise<{ value: unknown } | { error: string }> {
  return new Promise((resolve) => {
    const chunks: Buffer[] = [];
    let size = 0;
    let failed = false;
    req.on('data', (chunk: Buffer) => {
      size += chunk.length;
      if (size > MAX_BODY_BYTES) {
        failed = true;
        req.destroy();
        return;
      }
      chunks.push(chunk);
    });
    req.on('end', () => {
      if (failed) {
        resolve({ error: 'body_too_large' });
        return;
      }
      try {
        resolve({ value: JSON.parse(Buffer.concat(chunks).toString('utf8')) });
      } catch {
        resolve({ error: 'not_json' });
      }
    });
    req.on('error', () => resolve({ error: 'read_failed' }));
    req.on('close', () => {
      if (!failed) resolve({ error: 'read_failed' });
    });
  });
}

/** Drain a request body (discarded) so the socket can be reused; never throws. */
function drain(req: IncomingMessage): Promise<void> {
  return new Promise((resolve) => {
    let size = 0;
    req.on('data', (chunk: Buffer) => {
      size += chunk.length;
      if (size > MAX_BODY_BYTES) req.destroy();
    });
    req.on('end', resolve);
    req.on('error', resolve);
    req.on('close', resolve);
  });
}

/** GET /v1/models: the free model ids, in the wire the auth header names. */
function handleModels(req: IncomingMessage, res: ServerResponse, catalog: Catalog): void {
  const models = freeModelIds(catalog).map((m) => ({ id: m.id, provider: m.provider, display_name: m.id }));
  if (typeof req.headers['x-api-key'] === 'string') {
    sendJson(res, 200, anthropicModelList(models));
  } else {
    sendJson(res, 200, openAIModelList(models));
  }
}

/**
 * Pipe an upstream OpenAI SSE stream to the client byte for byte. If the
 * stream ends without `data: [DONE]`, emit it: a stream without its terminator
 * fails the client's stream-order check.
 */
async function pipeOpenAIStream(res: ServerResponse, stream: ReadableStream<Uint8Array>): Promise<void> {
  res.writeHead(200, { 'content-type': 'text/event-stream; charset=utf-8', 'cache-control': 'no-cache', connection: 'keep-alive' });
  const decoder = new TextDecoder();
  let tail = '';
  let sawDone = false;
  try {
    for await (const chunk of stream) {
      const text = decoder.decode(chunk, { stream: true });
      tail = `${tail}${text}`.slice(-32);
      if (tail.includes('data: [DONE]')) sawDone = true;
      res.write(chunk);
    }
    tail = `${tail}${decoder.decode()}`.slice(-32);
    if (tail.includes('data: [DONE]')) sawDone = true;
  } catch (err) {
    log(`stream passthrough failed: ${(err as Error).message}`);
  }
  if (!sawDone) res.write('data: [DONE]\n\n');
  res.end();
}

/** Stream the translated Anthropic events to the client, one event at a time. */
async function pipeAnthropicStream(res: ServerResponse, stream: ReadableStream<Uint8Array>, model: string): Promise<void> {
  res.writeHead(200, { 'content-type': 'text/event-stream; charset=utf-8', 'cache-control': 'no-cache', connection: 'keep-alive' });
  try {
    for await (const event of anthropicEventsFromOpenAI(stream, model)) {
      res.write(event);
    }
  } catch (err) {
    log(`stream translation failed: ${(err as Error).message}`);
  }
  res.end();
}

/** Stream translated events (already in the client's wire) to the client. */
async function pipeEvents(res: ServerResponse, events: AsyncGenerator<string>): Promise<void> {
  res.writeHead(200, { 'content-type': 'text/event-stream; charset=utf-8', 'cache-control': 'no-cache', connection: 'keep-alive' });
  try {
    for await (const event of events) {
      res.write(event);
    }
  } catch (err) {
    log(`stream translation failed: ${(err as Error).message}`);
  }
  res.end();
}

/**
 * The provider request body for one candidate: the client body translated
 * into the provider's wire, naming the candidate's real model id. The
 * OpenAI-wire path passes the body verbatim and asks for the trailing usage
 * chunk on streams; the Gemini path names no model in the body (it travels
 * in the URL path).
 */
function upstreamBodyFor(clientWire: ClientWire, providerWire: string, body: Json, model: string, stream: boolean): Json {
  if (clientWire === 'openai') {
    if (providerWire === 'openai') {
      const out: Json = { ...body, model };
      if (stream) out.stream_options = { include_usage: true };
      return out;
    }
    if (providerWire === 'anthropic') return openAIToAnthropicRequest(body, model);
    return openAIToGeminiRequest(body, model);
  }
  if (providerWire === 'openai') return anthropicToOpenAI(body, model);
  if (providerWire === 'anthropic') return { ...body, model };
  return anthropicToGeminiRequest(body, model);
}

/** One provider call for one candidate, on the provider's own wire. */
async function callCandidate(envUrl: string, candidate: Candidate, body: Json, stream: boolean, clientWire: ClientWire): Promise<ProviderResult> {
  const { provider, model } = candidate;
  const upstreamBody = upstreamBodyFor(clientWire, provider.wire, body, model, stream);
  if (provider.wire === 'openai') return callOpenAI(envUrl, provider, upstreamBody, stream, model);
  if (provider.wire === 'anthropic') return callAnthropic(envUrl, provider, upstreamBody, stream, model);
  return callGemini(envUrl, provider, model, upstreamBody, stream);
}

/** One successful provider JSON answer, translated into the client's wire and sent. */
function sendCandidateJson(res: ServerResponse, clientWire: ClientWire, candidate: Candidate, upstream: ProviderResult, answer: Json): void {
  const { provider, model } = candidate;
  if (clientWire === 'openai' && provider.wire === 'openai') {
    const text = upstream.bodyText ?? '';
    res.writeHead(upstream.status, {
      'content-type': upstream.headers.get('content-type') ?? 'application/json',
      'content-length': Buffer.byteLength(text),
    });
    res.end(text);
    return;
  }
  if (clientWire === 'openai' && provider.wire === 'anthropic') {
    sendJson(res, 200, anthropicToOpenAIResponse(answer, model));
    return;
  }
  if (clientWire === 'openai' && provider.wire === 'gemini') {
    sendJson(res, 200, geminiToOpenAIResponse(answer, model));
    return;
  }
  if (clientWire === 'anthropic' && provider.wire === 'openai') {
    sendJson(res, 200, openAIToAnthropic(answer, model));
    return;
  }
  if (clientWire === 'anthropic' && provider.wire === 'anthropic') {
    const text = upstream.bodyText ?? '';
    res.writeHead(upstream.status, {
      'content-type': upstream.headers.get('content-type') ?? 'application/json',
      'content-length': Buffer.byteLength(text),
    });
    res.end(text);
    return;
  }
  sendJson(res, 200, geminiToAnthropicResponse(answer, model));
}

/** One successful provider stream, translated into the client's wire and piped. */
async function pipeCandidateStream(res: ServerResponse, clientWire: ClientWire, candidate: Candidate, stream: ReadableStream<Uint8Array>): Promise<void> {
  const { provider, model } = candidate;
  if (clientWire === 'openai' && provider.wire === 'openai') {
    await pipeOpenAIStream(res, stream);
    return;
  }
  if (clientWire === 'openai' && provider.wire === 'anthropic') {
    await pipeEvents(res, anthropicStreamToOpenAIChunks(stream, model, true));
    return;
  }
  if (clientWire === 'openai' && provider.wire === 'gemini') {
    await pipeEvents(res, geminiStreamToClientEvents(stream, model, 'openai', true));
    return;
  }
  if (clientWire === 'anthropic' && provider.wire === 'openai') {
    await pipeAnthropicStream(res, stream, model);
    return;
  }
  if (clientWire === 'anthropic' && provider.wire === 'anthropic') {
    await pipeEvents(res, anthropicStreamPassthrough(stream, model));
    return;
  }
  await pipeEvents(res, geminiStreamToClientEvents(stream, model, 'anthropic'));
}

/**
 * Serve one client request: walk the planner's ranked candidates, one call
 * each, until the first success. A refusal moves to the next candidate; a
 * config error drops that (provider, model); a success ends the walk — no
 * second provider is ever called after one. When the list is exhausted the
 * wire's 503 answers.
 */
async function serve(req: IncomingMessage, res: ServerResponse, context: ServerContext, planner: Planner, clientWire: ClientWire, body: Json, requestIndex: number): Promise<void> {
  const requirements = requirementsFor(body, context.brief.workload, requestIndex, clientWire);
  const candidates = planner.candidatesFor(requirements, requestIndex);
  const stream = body.stream === true;
  for (const candidate of candidates) {
    if (!planner.isUsable(candidate.provider.id, candidate.model)) continue;
    let upstream: ProviderResult;
    try {
      upstream = await callCandidate(context.envUrl, candidate, body, stream, clientWire);
    } catch (err) {
      log(`upstream ${candidate.provider.id}/${candidate.model} unreachable: ${(err as Error).message}`);
      planner.recordRefusal(candidate.provider.id);
      continue;
    }
    if (!upstream.ok) {
      const text = upstream.bodyText ?? '';
      const info = providerErrorInfo(text);
      const detail = `${candidate.provider.id}/${candidate.model} HTTP ${upstream.status}${info.code !== null ? ` (${info.code})` : ''}: ${info.message}`;
      if (upstream.status === 401 || upstream.status === 400) {
        log(`upstream ${detail} — our bug, dropping that model`);
        planner.recordConfigError(candidate.provider.id, candidate.model);
      } else if (upstream.status === 402 || upstream.status === 429 || upstream.status >= 500) {
        log(`upstream ${detail} — refusal, failing over`);
        planner.recordRefusal(candidate.provider.id);
      } else {
        log(`upstream ${detail} — dropping that model`);
        planner.recordConfigError(candidate.provider.id, candidate.model);
      }
      continue;
    }
    planner.recordSuccess(candidate.provider.id);
    if (stream) {
      if (upstream.stream === null) {
        log(`upstream ${candidate.provider.id} answered a stream request without a stream`);
        sendError(res, clientWire, 500, 'internal_error', 'the provider answered a stream request without a stream');
        return;
      }
      await pipeCandidateStream(res, clientWire, candidate, upstream.stream);
      return;
    }
    let answer: unknown;
    try {
      answer = JSON.parse(upstream.bodyText ?? '');
    } catch {
      log(`upstream ${candidate.provider.id} answered a body that is not JSON`);
      sendError(res, clientWire, 500, 'internal_error', 'the provider answered a body that is not JSON');
      return;
    }
    if (!isRecord(answer)) {
      sendError(res, clientWire, 500, 'internal_error', 'the provider answered a JSON value that is not one object');
      return;
    }
    sendCandidateJson(res, clientWire, candidate, upstream, answer);
    return;
  }
  if (clientWire === 'anthropic') {
    sendError(res, 'anthropic', 503, 'overloaded_error', 'no free model can serve this request');
  } else {
    sendError(res, 'openai', 503, 'no_free_model', 'no free model can serve this request');
  }
}

/**
 * POST /v1/chat/completions: validate the body, then serve it. A body that
 * is not a request answers 400 and consumes no manifest index, so a
 * malformed probe cannot shift the routing of the requests after it.
 */
async function handleChatRequest(req: IncomingMessage, res: ServerResponse, context: ServerContext, state: ServerState): Promise<void> {
  const parsed = await readJsonBody(req);
  if ('error' in parsed || !isRecord(parsed.value) || !Array.isArray(parsed.value.messages)) {
    sendError(res, 'openai', 400, 'invalid_request_error', 'the body is not one chat completions request');
    return;
  }
  const requestIndex = state.requestIndex;
  state.requestIndex += 1;
  await serve(req, res, context, state.planner, 'openai', parsed.value, requestIndex);
}

/**
 * POST /v1/messages: validate the body, then serve it. A body that is not
 * a request answers 400 and consumes no manifest index (see above).
 */
async function handleMessagesRequest(req: IncomingMessage, res: ServerResponse, context: ServerContext, state: ServerState): Promise<void> {
  const parsed = await readJsonBody(req);
  if ('error' in parsed || !isRecord(parsed.value) || !Array.isArray(parsed.value.messages)) {
    sendError(res, 'anthropic', 400, 'invalid_request_error', 'the body is not one messages request');
    return;
  }
  const requestIndex = state.requestIndex;
  state.requestIndex += 1;
  await serve(req, res, context, state.planner, 'anthropic', parsed.value, requestIndex);
}

/** The server's mutable state: the planner, the next manifest index, and
 *  the serializer that keeps chat/messages requests strictly sequential. */
interface ServerState {
  requestIndex: number;
  planner: Planner;
  serialize: <T>(fn: () => Promise<T>) => Promise<T>;
}

/**
 * Run fns one at a time, in the order they were queued: the trial client
 * sends one request at a time, but a parallel client must never produce
 * parallel provider calls — zai answers a concurrent call with 429
 * `concurrent_requests`, and the manifest index is arrival order. GET
 * /v1/models and 404s stay outside the queue: they are fast and read-only.
 */
function createSerializer(): <T>(fn: () => Promise<T>) => Promise<T> {
  let tail: Promise<void> = Promise.resolve();
  return <T>(fn: () => Promise<T>): Promise<T> => {
    const run = tail.then(fn);
    tail = run.then(
      () => undefined,
      () => undefined,
    );
    return run;
  };
}

/** One request's handler. Chat/messages requests run through the serializer;
 *  GET /v1/models and unknown routes answer immediately. */
async function handle(
  req: IncomingMessage,
  res: ServerResponse,
  catalog: Catalog,
  context: ServerContext,
  state: ServerState,
): Promise<void> {
  const path = (req.url ?? '/').split('?')[0];
  const method = req.method ?? 'GET';
  if (method === 'GET' && path === '/v1/models') {
    handleModels(req, res, catalog);
    return;
  }
  if (method === 'POST' && path === '/v1/chat/completions') {
    await state.serialize(() => handleChatRequest(req, res, context, state));
    return;
  }
  if (method === 'POST' && path === '/v1/messages') {
    await state.serialize(() => handleMessagesRequest(req, res, context, state));
    return;
  }
  await drain(req);
  sendError(res, wireOf(req), 404, 'not_found', `no route ${method} ${path}`);
}

/** Build the server. Handler errors become 500 `internal_error` when the
 *  headers have not gone out yet. A client that goes away mid-answer must
 *  not take the process down with it: the response's `error` event is
 *  swallowed, so a disconnect during a stream ends that answer, not the trial. */
export function createAggregatorServer(catalog: Catalog, context: ServerContext): Server {
  const planner = new Planner(context.brief, catalog);
  const state: ServerState = { requestIndex: 0, planner, serialize: createSerializer() };
  return createServer((req, res) => {
    res.on('error', () => {
      // The client went away; nothing left to send it.
    });
    void handle(req, res, catalog, context, state).catch((err: unknown) => {
      if (!res.headersSent) {
        sendError(res, wireOf(req), 500, 'internal_error', 'the server failed to answer');
      }
      process.stderr.write(`aggregator: handler error: ${(err as Error).stack ?? String(err)}\n`);
    });
  });
}
