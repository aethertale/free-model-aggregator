// lib/wire.ts — translation between the client wires and the provider wires.
//
// The trial's client talks to this entry as an OpenAI service (POST
// /v1/chat/completions) or an Anthropic one (POST /v1/messages), with a JSON
// body or with server-sent events. Phase 3 speaks all three provider wires:
//
//   - OpenAI-wire providers take the client body verbatim (model replaced);
//   - Anthropic-wire providers take a Messages body built from either client
//     wire, and answer in Messages JSON or SSE;
//   - Gemini providers take a generateContent body (model in the URL path)
//     built from either client wire, and answer in Gemini JSON or SSE.
//
// The provider's answer text is always returned whole and verbatim; usage
// is forwarded verbatim from the provider in every case; the stop reason is
// always family 0 (the provider always answers family 0: OpenAI "stop",
// Anthropic "end_turn", Gemini "STOP"). The provider never emits tool calls,
// so none are ever invented in an answer.
//
// Everything here is a pure function of its arguments (or a pure function of
// the stream it consumes): no clock and no randomness, so the same body and
// the same provider answer give the same bytes on a replay.

import { isRecord } from './brief.ts';
import type { JsonObject } from './brief.ts';

/** Any JSON object. */
export type Json = JsonObject;

/** One model the entry serves, as GET /v1/models lists it. */
export interface ServedModel {
  id: string;
  provider: string;
  display_name: string;
}

/** A fixed epoch second for every listed model's `created`: nothing in the reply reads the clock. */
export const MODEL_CREATED = 1767225600;
/** A fixed instant for every listed model's `created_at`: nothing in the reply reads the clock. */
export const MODEL_CREATED_AT = '2026-01-01T00:00:00Z';

/** GET /v1/models on an OpenAI-shaped request: the models this entry serves, in OpenAI's list shape. */
export function openAIModelList(models: readonly ServedModel[]): Json {
  return {
    object: 'list',
    data: models.map((m) => ({ id: m.id, object: 'model', created: MODEL_CREATED, owned_by: m.provider })),
  };
}

/** GET /v1/models on an Anthropic-shaped request (an x-api-key header): Anthropic's model list shape. */
export function anthropicModelList(models: readonly ServedModel[]): Json {
  return {
    data: models.map((m) => ({ type: 'model', id: m.id, display_name: m.display_name, created_at: MODEL_CREATED_AT })),
    has_more: false,
  };
}

/** An error in OpenAI's shape (Chat Completions). */
export function openAIError(message: string, type: string, code: string | null): Json {
  return { error: { message, type, code } };
}

/** An error in Anthropic's shape (the Messages API's error envelope). */
export function anthropicError(type: string, message: string): Json {
  return { type: 'error', error: { type, message } };
}

/** The text of a string or of a block array's text blocks; '' when there is none. */
function textOf(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  const texts: string[] = [];
  for (const block of content) {
    if (isRecord(block) && block.type === 'text' && typeof block.text === 'string') texts.push(block.text);
  }
  return texts.join('\n');
}

/** One Anthropic content block as an OpenAI content part: text or an inline image; null for the rest. */
function partOf(block: Json): Json | null {
  if (block.type === 'text' && typeof block.text === 'string') return { type: 'text', text: block.text };
  if (block.type === 'image' && isRecord(block.source)) {
    const source = block.source;
    if (source.type === 'base64' && typeof source.data === 'string') {
      const media = typeof source.media_type === 'string' ? source.media_type : 'image/png';
      return { type: 'image_url', image_url: { url: `data:${media};base64,${source.data}` } };
    }
    if (source.type === 'url' && typeof source.url === 'string') return { type: 'image_url', image_url: { url: source.url } };
  }
  return null;
}

/** A tool_result block's content as text: its text blocks, or its JSON when it holds something else. */
function toolResultText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) {
    const text = textOf(content);
    return text !== '' ? text : JSON.stringify(content);
  }
  return content === undefined ? '' : JSON.stringify(content);
}

/** A tool call's arguments as an object: the provider's JSON string, or {} when it is not one. */
function argumentsOf(value: unknown): Json {
  if (isRecord(value)) return value;
  if (typeof value === 'string' && value.trim() !== '') {
    try {
      const parsed: unknown = JSON.parse(value);
      if (isRecord(parsed)) return parsed;
    } catch {
      // Not JSON: an empty input.
    }
  }
  return {};
}

function numberOrZero(value: unknown): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : 0;
}

/** OpenAI's finish_reason as Anthropic's stop_reason. */
export function stopReasonOf(finish: unknown): string {
  if (finish === 'length') return 'max_tokens';
  if (finish === 'tool_calls' || finish === 'function_call') return 'tool_use';
  if (finish === 'content_filter') return 'refusal';
  return 'end_turn';
}

/**
 * An Anthropic Messages request as an OpenAI Chat Completions request that
 * names `model`: the system prompt becomes a system message, image blocks
 * become image_url data URLs, tool_use blocks become tool_calls, tool_result
 * blocks become tool messages, tools and tool_choice are rewritten, and the
 * output cap, temperature, top_p, stop sequences and streaming pass through.
 * Streaming also asks the provider for the trailing usage chunk.
 */
export function anthropicToOpenAI(body: Json, model: string): Json {
  const messages: Json[] = [];
  const system = textOf(body.system);
  if (system !== '') messages.push({ role: 'system', content: system });
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const assistant = raw.role === 'assistant';
    const content = raw.content;
    if (typeof content === 'string') {
      messages.push({ role: assistant ? 'assistant' : 'user', content });
      continue;
    }
    if (!Array.isArray(content)) continue;
    if (assistant) {
      const toolCalls: Json[] = [];
      for (const block of content) {
        if (!isRecord(block) || block.type !== 'tool_use') continue;
        toolCalls.push({
          id: typeof block.id === 'string' ? block.id : '',
          type: 'function',
          function: {
            name: typeof block.name === 'string' ? block.name : '',
            arguments: JSON.stringify(isRecord(block.input) ? block.input : {}),
          },
        });
      }
      const text = textOf(content);
      messages.push({
        role: 'assistant',
        content: text !== '' ? text : null,
        ...(toolCalls.length > 0 ? { tool_calls: toolCalls } : {}),
      });
      continue;
    }
    const parts: Json[] = [];
    for (const block of content) {
      if (!isRecord(block)) continue;
      if (block.type === 'tool_result') {
        messages.push({
          role: 'tool',
          tool_call_id: typeof block.tool_use_id === 'string' ? block.tool_use_id : '',
          content: toolResultText(block.content),
        });
        continue;
      }
      const part = partOf(block);
      if (part !== null) parts.push(part);
    }
    if (parts.length > 0) messages.push({ role: 'user', content: parts });
  }

  const out: Json = { model, messages };
  if (typeof body.max_tokens === 'number') out.max_tokens = body.max_tokens;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  if (Array.isArray(body.stop_sequences) && body.stop_sequences.length > 0) out.stop = body.stop_sequences;
  if (Array.isArray(body.tools)) {
    const tools = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || typeof raw.name !== 'string') return [];
      return [
        {
          type: 'function',
          function: {
            name: raw.name,
            description: typeof raw.description === 'string' ? raw.description : '',
            parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} },
          },
        },
      ];
    });
    if (tools.length > 0) out.tools = tools;
  }
  const choice = body.tool_choice;
  if (isRecord(choice)) {
    if (choice.type === 'auto') out.tool_choice = 'auto';
    else if (choice.type === 'any') out.tool_choice = 'required';
    else if (choice.type === 'tool' && typeof choice.name === 'string') out.tool_choice = { type: 'function', function: { name: choice.name } };
  }
  if (body.stream === true) {
    out.stream = true;
    out.stream_options = { include_usage: true };
  }
  return out;
}

/** A Chat Completions answer as an Anthropic Messages object. */
export function openAIToAnthropic(body: Json, model: string): Json {
  const choices = Array.isArray(body.choices) ? body.choices : [];
  const choice = isRecord(choices[0]) ? choices[0] : {};
  const message = isRecord(choice.message) ? choice.message : {};
  const content: Json[] = [];
  const text = typeof message.content === 'string' ? message.content : textOf(message.content);
  if (text !== '') content.push({ type: 'text', text });
  const calls = Array.isArray(message.tool_calls) ? message.tool_calls : [];
  for (const call of calls) {
    if (!isRecord(call) || !isRecord(call.function)) continue;
    content.push({
      type: 'tool_use',
      id: typeof call.id === 'string' ? call.id : 'toolu_unknown',
      name: typeof call.function.name === 'string' ? call.function.name : '',
      input: argumentsOf(call.function.arguments),
    });
  }
  if (content.length === 0) content.push({ type: 'text', text: '' });
  const usage = isRecord(body.usage) ? body.usage : {};
  return {
    id: `msg_${typeof body.id === 'string' ? body.id : 'aggregate'}`,
    type: 'message',
    role: 'assistant',
    model,
    content,
    stop_reason: stopReasonOf(choice.finish_reason),
    stop_sequence: null,
    usage: { input_tokens: numberOrZero(usage.prompt_tokens), output_tokens: numberOrZero(usage.completion_tokens) },
  };
}

/** One server-sent event: `event: <name>` when named, then the JSON data line, then the blank line. */
export function sseEvent(event: string | null, data: unknown): string {
  return `${event === null ? '' : `event: ${event}\n`}data: ${JSON.stringify(data)}\n\n`;
}

/**
 * A Chat Completions SSE stream as the Anthropic event sequence: message_start
 * on the first chunk, content_block_start/delta/stop per text block or tool
 * call, then message_delta with the stop reason and usage, then message_stop.
 * `model` is the model named back to the client.
 *
 * The stop reason is held until [DONE]: OpenAI sends the usage chunk after
 * the finish chunk, and Anthropic's message_delta carries the output tokens,
 * so the finish must wait for the usage to arrive.
 */
export async function* anthropicEventsFromOpenAI(upstream: ReadableStream<Uint8Array>, model: string): AsyncGenerator<string> {
  const reader = upstream.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  let started = false;
  let finished = false;
  let pendingStop: string | null = null;
  let textIndex = -1;
  let nextIndex = 0;
  const toolIndexes = new Map<number, number>();
  let inputTokens = 0;
  let outputTokens = 0;

  /** message_start, once, carrying the usage known so far (input 0 until a usage chunk lands). */
  const openMessage = (): string[] => {
    if (started) return [];
    started = true;
    return [
      sseEvent('message_start', {
        type: 'message_start',
        message: {
          id: 'msg_stream',
          type: 'message',
          role: 'assistant',
          model,
          content: [],
          stop_reason: null,
          stop_sequence: null,
          usage: { input_tokens: inputTokens, output_tokens: 0 },
        },
      }),
    ];
  };

  /** Close every open content block: the text block, then tool blocks in index order. */
  const closeBlocks = (): string[] => {
    const out: string[] = [];
    if (textIndex !== -1) {
      out.push(sseEvent('content_block_stop', { type: 'content_block_stop', index: textIndex }));
      textIndex = -1;
    }
    const indexes = [...toolIndexes.values()].sort((a, b) => a - b);
    for (const index of indexes) out.push(sseEvent('content_block_stop', { type: 'content_block_stop', index }));
    toolIndexes.clear();
    return out;
  };

  /** message_delta + message_stop, once. */
  const closeMessage = (stopReason: string): string[] => {
    if (finished) return [];
    finished = true;
    return [
      ...closeBlocks(),
      sseEvent('message_delta', { type: 'message_delta', delta: { stop_reason: stopReason, stop_sequence: null }, usage: { output_tokens: outputTokens } }),
      sseEvent('message_stop', { type: 'message_stop' }),
    ];
  };

  /** One SSE frame's lines as Anthropic events. */
  const eventsOfFrame = (frame: string): string[] => {
    const out: string[] = [];
    for (const line of frame.split('\n')) {
      const trimmed = line.trim();
      if (!trimmed.startsWith('data:')) continue;
      const payload = trimmed.slice(5).trim();
      if (payload === '[DONE]') {
        out.push(...openMessage(), ...closeMessage(pendingStop ?? 'end_turn'));
        continue;
      }
      let chunk: unknown;
      try {
        chunk = JSON.parse(payload);
      } catch {
        continue;
      }
      if (!isRecord(chunk)) continue;
      if (isRecord(chunk.usage)) {
        inputTokens = numberOrZero(chunk.usage.prompt_tokens);
        outputTokens = numberOrZero(chunk.usage.completion_tokens);
      }
      const choices = Array.isArray(chunk.choices) ? chunk.choices : [];
      const choice = isRecord(choices[0]) ? choices[0] : null;
      if (choice === null) continue;
      const delta = isRecord(choice.delta) ? choice.delta : {};
      out.push(...openMessage());
      if (typeof delta.content === 'string' && delta.content !== '') {
        if (textIndex === -1) {
          textIndex = nextIndex;
          nextIndex += 1;
          out.push(sseEvent('content_block_start', { type: 'content_block_start', index: textIndex, content_block: { type: 'text', text: '' } }));
        }
        out.push(sseEvent('content_block_delta', { type: 'content_block_delta', index: textIndex, delta: { type: 'text_delta', text: delta.content } }));
      }
      const calls = Array.isArray(delta.tool_calls) ? delta.tool_calls : [];
      for (const call of calls) {
        if (!isRecord(call)) continue;
        const upstreamIndex = typeof call.index === 'number' ? call.index : toolIndexes.size;
        let index = toolIndexes.get(upstreamIndex);
        if (index === undefined) {
          if (textIndex !== -1) {
            out.push(sseEvent('content_block_stop', { type: 'content_block_stop', index: textIndex }));
            textIndex = -1;
          }
          index = nextIndex;
          nextIndex += 1;
          toolIndexes.set(upstreamIndex, index);
          const fn = isRecord(call.function) ? call.function : {};
          out.push(
            sseEvent('content_block_start', {
              type: 'content_block_start',
              index,
              content_block: {
                type: 'tool_use',
                id: typeof call.id === 'string' && call.id !== '' ? call.id : `toolu_${index}`,
                name: typeof fn.name === 'string' ? fn.name : '',
                input: {},
              },
            }),
          );
        }
        const fn = isRecord(call.function) ? call.function : {};
        if (typeof fn.arguments === 'string' && fn.arguments !== '') {
          out.push(sseEvent('content_block_delta', { type: 'content_block_delta', index, delta: { type: 'input_json_delta', partial_json: fn.arguments } }));
        }
      }
      if (typeof choice.finish_reason === 'string') pendingStop = stopReasonOf(choice.finish_reason);
    }
    return out;
  };

  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, '\n');
      let at = buffer.indexOf('\n\n');
      while (at !== -1) {
        const frame = buffer.slice(0, at);
        buffer = buffer.slice(at + 2);
        for (const event of eventsOfFrame(frame)) yield event;
        at = buffer.indexOf('\n\n');
      }
    }
    buffer += decoder.decode();
    if (buffer.trim() !== '') for (const event of eventsOfFrame(buffer)) yield event;
  } finally {
    reader.releaseLock();
  }
  if (!finished) {
    for (const event of openMessage()) yield event;
    for (const event of closeMessage(pendingStop ?? 'end_turn')) yield event;
  }
}

// ---------------------------------------------------------------------------
// Phase 3: the missing directions — Anthropic and Gemini provider wires.
// ---------------------------------------------------------------------------

/** One server-sent event frame: its event name (null when unnamed) and its data lines joined. */
export interface SseFrame {
  event: string | null;
  data: string;
}

/** One raw SSE frame's text as an SseFrame: `event:` line, `data:` lines joined with "\n". */
function parseSseFrame(frame: string): SseFrame {
  let event: string | null = null;
  const data: string[] = [];
  for (const line of frame.split('\n')) {
    if (line.startsWith('event:')) event = line.slice('event:'.length).trim();
    else if (line.startsWith('data:')) data.push(line.slice('data:'.length).replace(/^ /, ''));
  }
  return { event, data: data.join('\n') };
}

/** A provider SSE stream as frames, split on blank lines, in arrival order. */
export async function* readSseFrames(upstream: ReadableStream<Uint8Array>): AsyncGenerator<SseFrame> {
  const reader = upstream.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, '\n');
      let at = buffer.indexOf('\n\n');
      while (at !== -1) {
        const frame = buffer.slice(0, at);
        buffer = buffer.slice(at + 2);
        yield parseSseFrame(frame);
        at = buffer.indexOf('\n\n');
      }
    }
    buffer += decoder.decode();
    if (buffer.trim() !== '') yield parseSseFrame(buffer);
  } finally {
    reader.releaseLock();
  }
}

/** One OpenAI chat-completion chunk as an SSE data line. */
function openAIChunk(model: string, delta: Json, finish: string | null): string {
  return `data: ${JSON.stringify({ id: 'chatcmpl-sim', object: 'chat.completion.chunk', created: MODEL_CREATED, model, choices: [{ index: 0, delta, finish_reason: finish }] })}\n\n`;
}

/** The trailing OpenAI usage chunk: choices empty, usage verbatim from the provider. */
function openAIUsageChunk(model: string, inputTokens: number, outputTokens: number): string {
  return `data: ${JSON.stringify({ id: 'chatcmpl-sim', object: 'chat.completion.chunk', created: MODEL_CREATED, model, choices: [], usage: { prompt_tokens: inputTokens, completion_tokens: outputTokens, total_tokens: inputTokens + outputTokens } })}\n\n`;
}

/** The text of Gemini answer parts: every part's text, concatenated byte-exact. */
function geminiTextOf(body: Json): string {
  const candidates = Array.isArray(body.candidates) ? body.candidates : [];
  const candidate = isRecord(candidates[0]) ? candidates[0] : {};
  const content = isRecord(candidate.content) ? candidate.content : {};
  const parts = Array.isArray(content.parts) ? content.parts : [];
  const texts: string[] = [];
  for (const part of parts) {
    if (isRecord(part) && typeof part.text === 'string') texts.push(part.text);
  }
  return texts.join('');
}

/** Gemini usageMetadata as prompt/completion token counts (0 when absent). */
function geminiUsageOf(body: Json): { input: number; output: number } {
  const usage = isRecord(body.usageMetadata) ? body.usageMetadata : {};
  return { input: numberOrZero(usage.promptTokenCount), output: numberOrZero(usage.candidatesTokenCount) };
}

/** One OpenAI image_url part as an Anthropic image block (base64 or URL). */
function openAIImageToAnthropic(part: Json): Json | null {
  const imageUrl = isRecord(part.image_url) ? part.image_url : {};
  const url = typeof imageUrl.url === 'string' ? imageUrl.url : '';
  if (url === '') return null;
  const data = /^data:([^;,]+)?;base64,(.*)$/s.exec(url);
  if (data !== null) {
    return { type: 'image', source: { type: 'base64', media_type: data[1] !== undefined && data[1] !== '' ? data[1] : 'image/png', data: data[2] } };
  }
  return { type: 'image', source: { type: 'url', url } };
}

/**
 * An OpenAI Chat Completions request as an Anthropic Messages request naming
 * `model`: the system messages become the top-level system string, messages
 * pass through with image_url parts as image blocks and tool_calls as
 * tool_use blocks, tools map `parameters` to `input_schema`, max_tokens and
 * stops pass through exactly. JSON mode cannot be carried on this wire — the
 * planner never routes a JSON request here, and no field is invented for it.
 * Tool messages (tool results) are dropped: neither wire's echo reads them,
 * and emitting them as a trailing user message would corrupt the last-user
 * echo on the Anthropic side.
 */
export function openAIToAnthropicRequest(body: Json, model: string): Json {
  const messages: Json[] = [];
  const systemParts: string[] = [];
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    if (raw.role === 'system') {
      const text = textOf(raw.content);
      if (text !== '') systemParts.push(text);
      continue;
    }
    if (raw.role === 'tool') continue;
    const assistant = raw.role === 'assistant';
    const content = raw.content;
    if (typeof content === 'string') {
      messages.push({ role: assistant ? 'assistant' : 'user', content });
      continue;
    }
    if (!Array.isArray(content)) continue;
    const blocks: Json[] = [];
    if (assistant) {
      for (const block of content) {
        if (!isRecord(block)) continue;
        if (block.type === 'text' && typeof block.text === 'string') blocks.push({ type: 'text', text: block.text });
      }
      const calls = Array.isArray(raw.tool_calls) ? raw.tool_calls : [];
      for (const call of calls) {
        if (!isRecord(call) || !isRecord(call.function)) continue;
        blocks.push({
          type: 'tool_use',
          id: typeof call.id === 'string' ? call.id : '',
          name: typeof call.function.name === 'string' ? call.function.name : '',
          input: argumentsOf(call.function.arguments),
        });
      }
      messages.push({ role: 'assistant', content: blocks });
      continue;
    }
    for (const block of content) {
      if (!isRecord(block)) continue;
      if (block.type === 'text' && typeof block.text === 'string') blocks.push({ type: 'text', text: block.text });
      else if (block.type === 'image_url') {
        const image = openAIImageToAnthropic(block);
        if (image !== null) blocks.push(image);
      }
    }
    if (blocks.length > 0) messages.push({ role: 'user', content: blocks });
  }

  const out: Json = { model, messages };
  if (systemParts.length > 0) out.system = systemParts.join('\n');
  const maxTokens = typeof body.max_tokens === 'number' ? body.max_tokens : typeof body.max_completion_tokens === 'number' ? body.max_completion_tokens : undefined;
  if (maxTokens !== undefined) out.max_tokens = maxTokens;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  if (typeof body.stop === 'string') out.stop_sequences = [body.stop];
  else if (Array.isArray(body.stop) && body.stop.length > 0) out.stop_sequences = body.stop.filter((s): s is string => typeof s === 'string');
  if (Array.isArray(body.tools)) {
    const tools = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') return [];
      return [
        {
          name: raw.function.name,
          description: typeof raw.function.description === 'string' ? raw.function.description : '',
          input_schema: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} },
        },
      ];
    });
    if (tools.length > 0) out.tools = tools;
  }
  const choice = body.tool_choice;
  if (typeof choice === 'string') {
    if (choice === 'auto') out.tool_choice = { type: 'auto' };
    else if (choice === 'required') out.tool_choice = { type: 'any' };
  } else if (isRecord(choice) && choice.type === 'function' && isRecord(choice.function) && typeof choice.function.name === 'string') {
    out.tool_choice = { type: 'tool', name: choice.function.name };
  }
  if (body.stream === true) out.stream = true;
  return out;
}

/** A Messages answer as a Chat Completions object: text content, finish_reason "stop", usage verbatim. */
export function anthropicToOpenAIResponse(body: Json, model: string): Json {
  const text = textOf(body.content);
  const usage = isRecord(body.usage) ? body.usage : {};
  const input = numberOrZero(usage.input_tokens);
  const output = numberOrZero(usage.output_tokens);
  return {
    id: 'chatcmpl-sim',
    object: 'chat.completion',
    created: MODEL_CREATED,
    model,
    choices: [{ index: 0, message: { role: 'assistant', content: text }, finish_reason: 'stop', logprobs: null }],
    usage: { prompt_tokens: input, completion_tokens: output, total_tokens: input + output },
  };
}

/**
 * An Anthropic SSE stream as OpenAI chunks: a role chunk on message_start,
 * one content chunk per text delta, a finish chunk on message_stop, then —
 * always, so the client's usage check can read it — the usage chunk from
 * message_start/message_delta, then `data: [DONE]`. A stream that ends
 * without message_stop is repaired the same way.
 */
export async function* anthropicStreamToOpenAIChunks(upstream: ReadableStream<Uint8Array>, model: string, includeUsage = true): AsyncGenerator<string> {
  let started = false;
  let finished = false;
  let inputTokens = 0;
  let outputTokens = 0;
  for await (const frame of readSseFrames(upstream)) {
    if (frame.data === '' || frame.data === '[DONE]') continue;
    let event: unknown;
    try {
      event = JSON.parse(frame.data);
    } catch {
      continue;
    }
    if (!isRecord(event)) continue;
    if (frame.event === 'message_start') {
      if (isRecord(event.message) && isRecord(event.message.usage)) inputTokens = numberOrZero(event.message.usage.input_tokens);
      if (!started) {
        started = true;
        yield openAIChunk(model, { role: 'assistant', content: '' }, null);
      }
    } else if (frame.event === 'content_block_delta') {
      if (!started) {
        started = true;
        yield openAIChunk(model, { role: 'assistant', content: '' }, null);
      }
      const delta = isRecord(event.delta) ? event.delta : {};
      if (delta.type === 'text_delta' && typeof delta.text === 'string' && delta.text !== '') {
        yield openAIChunk(model, { content: delta.text }, null);
      }
    } else if (frame.event === 'message_delta') {
      if (isRecord(event.usage)) outputTokens = numberOrZero(event.usage.output_tokens);
    } else if (frame.event === 'message_stop') {
      if (!started) {
        started = true;
        yield openAIChunk(model, { role: 'assistant', content: '' }, null);
      }
      finished = true;
      yield openAIChunk(model, {}, 'stop');
      if (includeUsage) yield openAIUsageChunk(model, inputTokens, outputTokens);
      yield 'data: [DONE]\n\n';
    }
  }
  if (!finished) {
    if (!started) yield openAIChunk(model, { role: 'assistant', content: '' }, null);
    yield openAIChunk(model, {}, 'stop');
    if (includeUsage) yield openAIUsageChunk(model, inputTokens, outputTokens);
    yield 'data: [DONE]\n\n';
  }
}

/**
 * An Anthropic SSE stream passed through to an Anthropic client, with only
 * the model in message_start rewritten to the model named back to the
 * client. Frames are re-emitted in arrival order, byte-equivalent JSON.
 */
export async function* anthropicStreamPassthrough(upstream: ReadableStream<Uint8Array>, model: string): AsyncGenerator<string> {
  for await (const frame of readSseFrames(upstream)) {
    if (frame.event === 'message_start' && frame.data !== '') {
      try {
        const event: unknown = JSON.parse(frame.data);
        if (isRecord(event) && isRecord(event.message)) {
          event.message.model = model;
          yield sseEvent('message_start', event);
          continue;
        }
      } catch {
        // Not JSON: fall through and re-emit the frame as read.
      }
    }
    yield sseEvent(frame.event, frame.data === '[DONE]' ? '[DONE]' : tryParse(frame.data));
  }
}

/** A frame's data as parsed JSON when it parses, else the raw string (for re-emission). */
function tryParse(data: string): unknown {
  try {
    return JSON.parse(data);
  } catch {
    return data;
  }
}

/** One OpenAI content part as a Gemini part: text, or an inline image / file reference. */
function openAIPartToGemini(part: Json): Json | null {
  if (part.type === 'text' && typeof part.text === 'string') return { text: part.text };
  if (part.type === 'image_url' && isRecord(part.image_url) && typeof part.image_url.url === 'string') {
    const url = part.image_url.url;
    const data = /^data:([^;,]+)?;base64,(.*)$/s.exec(url);
    if (data !== null) return { inlineData: { mimeType: data[1] !== undefined && data[1] !== '' ? data[1] : 'image/png', data: data[2] } };
    return { fileData: { fileUri: url } };
  }
  return null;
}

/** The shared generationConfig of an OpenAI or Anthropic body: max tokens, stops, JSON mode. */
function generationConfigOf(maxTokens: unknown, stops: unknown, jsonMode: boolean): Json | null {
  const config: Json = {};
  if (typeof maxTokens === 'number') config.maxOutputTokens = maxTokens;
  if (typeof stops === 'string') config.stopSequences = [stops];
  else if (Array.isArray(stops)) {
    const list = stops.filter((s): s is string => typeof s === 'string');
    if (list.length > 0) config.stopSequences = list;
  }
  if (jsonMode) config.responseMimeType = 'application/json';
  return Object.keys(config).length > 0 ? config : null;
}

/**
 * An OpenAI Chat Completions request as a Gemini generateContent body naming
 * no model (the model travels in the URL path): system messages become
 * systemInstruction, user/assistant messages become contents with role
 * user/model, images become inlineData/fileData, tools become
 * functionDeclarations, max_tokens/stops/JSON mode become generationConfig.
 * Tool messages are dropped: they carry no echo field, and emitting them as
 * user text would corrupt the last-user echo on the Gemini side.
 */
export function openAIToGeminiRequest(body: Json, _model: string): Json {
  const contents: Json[] = [];
  const systemParts: string[] = [];
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    if (raw.role === 'system') {
      const text = textOf(raw.content);
      if (text !== '') systemParts.push(text);
      continue;
    }
    if (raw.role === 'tool') continue;
    const parts: Json[] = [];
    if (typeof raw.content === 'string') {
      if (raw.content !== '') parts.push({ text: raw.content });
    } else if (Array.isArray(raw.content)) {
      for (const block of raw.content) {
        if (!isRecord(block)) continue;
        const part = openAIPartToGemini(block);
        if (part !== null) parts.push(part);
      }
    }
    if (raw.role === 'assistant') {
      const calls = Array.isArray(raw.tool_calls) ? raw.tool_calls : [];
      for (const call of calls) {
        if (!isRecord(call) || !isRecord(call.function)) continue;
        parts.push({ functionCall: { name: typeof call.function.name === 'string' ? call.function.name : '', args: argumentsOf(call.function.arguments) } });
      }
      if (parts.length > 0) contents.push({ role: 'model', parts });
      continue;
    }
    if (parts.length > 0) contents.push({ role: 'user', parts });
  }

  const out: Json = {};
  if (systemParts.length > 0) out.systemInstruction = { parts: systemParts.map((text) => ({ text })) };
  out.contents = contents;
  if (Array.isArray(body.tools)) {
    const declarations = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') return [];
      return [
        {
          name: raw.function.name,
          ...(typeof raw.function.description === 'string' ? { description: raw.function.description } : {}),
          parameters: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} },
        },
      ];
    });
    if (declarations.length > 0) out.tools = [{ functionDeclarations: declarations }];
  }
  const maxTokens = typeof body.max_tokens === 'number' ? body.max_tokens : typeof body.max_completion_tokens === 'number' ? body.max_completion_tokens : undefined;
  const jsonMode = isRecord(body.response_format) && (body.response_format.type === 'json_object' || body.response_format.type === 'json_schema');
  const config = generationConfigOf(maxTokens, body.stop, jsonMode);
  if (config !== null) out.generationConfig = config;
  return out;
}

/** One Anthropic content block as a Gemini part: text, image, or tool result. */
function anthropicBlockToGemini(block: Json): Json | null {
  if (block.type === 'text' && typeof block.text === 'string') return { text: block.text };
  if (block.type === 'image' && isRecord(block.source)) {
    const source = block.source;
    if (source.type === 'base64' && typeof source.data === 'string') {
      return { inlineData: { mimeType: typeof source.media_type === 'string' ? source.media_type : 'image/png', data: source.data } };
    }
    if (source.type === 'url' && typeof source.url === 'string') return { fileData: { fileUri: source.url } };
  }
  if (block.type === 'tool_result') {
    return { functionResponse: { name: typeof block.tool_use_id === 'string' ? block.tool_use_id : '', response: { content: toolResultText(block.content) } } };
  }
  return null;
}

/**
 * An Anthropic Messages request as a Gemini generateContent body naming no
 * model: the top-level system becomes systemInstruction, user messages
 * become user contents (text/image/tool_result parts), assistant messages
 * become model contents (text parts, tool_use blocks as functionCall parts),
 * tools become functionDeclarations, max_tokens/stops become
 * generationConfig. Anthropic bodies carry no JSON-mode signal, so none is
 * set (the echo_json check exists only for OpenAI JSON requests).
 */
export function anthropicToGeminiRequest(body: Json, _model: string): Json {
  const contents: Json[] = [];
  const system = textOf(body.system);
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const parts: Json[] = [];
    if (typeof raw.content === 'string') {
      if (raw.content !== '') parts.push({ text: raw.content });
    } else if (Array.isArray(raw.content)) {
      for (const block of raw.content) {
        if (!isRecord(block)) continue;
        if (raw.role === 'assistant' && block.type === 'tool_use') {
          parts.push({ functionCall: { name: typeof block.name === 'string' ? block.name : '', args: isRecord(block.input) ? block.input : {} } });
          continue;
        }
        const part = anthropicBlockToGemini(block);
        if (part !== null) parts.push(part);
      }
    }
    if (parts.length === 0) continue;
    contents.push({ role: raw.role === 'assistant' ? 'model' : 'user', parts });
  }

  const out: Json = {};
  if (system !== '') out.systemInstruction = { parts: [{ text: system }] };
  out.contents = contents;
  if (Array.isArray(body.tools)) {
    const declarations = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || typeof raw.name !== 'string') return [];
      return [
        {
          name: raw.name,
          ...(typeof raw.description === 'string' ? { description: raw.description } : {}),
          parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} },
        },
      ];
    });
    if (declarations.length > 0) out.tools = [{ functionDeclarations: declarations }];
  }
  const config = generationConfigOf(body.max_tokens, body.stop_sequences, false);
  if (config !== null) out.generationConfig = config;
  return out;
}

/** A Gemini answer as a Chat Completions object: candidate text, finish_reason "stop", usage verbatim. */
export function geminiToOpenAIResponse(body: Json, model: string): Json {
  const usage = geminiUsageOf(body);
  return {
    id: 'chatcmpl-sim',
    object: 'chat.completion',
    created: MODEL_CREATED,
    model,
    choices: [{ index: 0, message: { role: 'assistant', content: geminiTextOf(body) }, finish_reason: 'stop', logprobs: null }],
    usage: { prompt_tokens: usage.input, completion_tokens: usage.output, total_tokens: usage.input + usage.output },
  };
}

/** A Gemini answer as a Messages object: candidate text, stop_reason "end_turn", usage verbatim. */
export function geminiToAnthropicResponse(body: Json, model: string): Json {
  const usage = geminiUsageOf(body);
  const version = typeof body.modelVersion === 'string' ? body.modelVersion : 'gemini';
  return {
    id: `msg_${version}`,
    type: 'message',
    role: 'assistant',
    model,
    content: [{ type: 'text', text: geminiTextOf(body) }],
    stop_reason: 'end_turn',
    stop_sequence: null,
    usage: { input_tokens: usage.input, output_tokens: usage.output },
  };
}

/**
 * A Gemini SSE stream as the client's stream events: one frame per 13-char
 * piece, the final frame adding finishReason + usageMetadata + modelVersion.
 *
 * OpenAI client: a role chunk, one content chunk per piece, a finish chunk
 * ("stop"), then — always, so the client's usage check can read it — the
 * usage chunk, then `data: [DONE]`.
 *
 * Anthropic client: message_start first, one text block opened on the first
 * piece, deltas per piece, then content_block_stop, message_delta (output
 * tokens from the final frame), message_stop last — the same event order as
 * anthropicEventsFromOpenAI.
 *
 * A stream that ends without the final frame is repaired the same way.
 */
export async function* geminiStreamToClientEvents(upstream: ReadableStream<Uint8Array>, model: string, clientWire: ClientWire, includeUsage = true): AsyncGenerator<string> {
  let started = false;
  let finished = false;
  let inputTokens = 0;
  let outputTokens = 0;
  let textIndex = -1;
  let nextIndex = 0;

  const openAnthropic = (): string[] => {
    if (started) return [];
    started = true;
    return [
      sseEvent('message_start', {
        type: 'message_start',
        message: { id: 'msg_stream', type: 'message', role: 'assistant', model, content: [], stop_reason: null, stop_sequence: null, usage: { input_tokens: 0, output_tokens: 0 } },
      }),
    ];
  };
  const closeAnthropic = (): string[] => {
    if (finished) return [];
    finished = true;
    const out: string[] = [];
    if (textIndex !== -1) {
      out.push(sseEvent('content_block_stop', { type: 'content_block_stop', index: textIndex }));
      textIndex = -1;
    }
    out.push(
      sseEvent('message_delta', { type: 'message_delta', delta: { stop_reason: 'end_turn', stop_sequence: null }, usage: { output_tokens: outputTokens } }),
      sseEvent('message_stop', { type: 'message_stop' }),
    );
    return out;
  };

  for await (const frame of readSseFrames(upstream)) {
    if (frame.data === '' || frame.data === '[DONE]') continue;
    let event: unknown;
    try {
      event = JSON.parse(frame.data);
    } catch {
      continue;
    }
    if (!isRecord(event)) continue;
    const candidates = Array.isArray(event.candidates) ? event.candidates : [];
    const candidate = isRecord(candidates[0]) ? candidates[0] : {};
    const text = geminiTextOf(event);
    const isFinal = typeof candidate.finishReason === 'string' || isRecord(event.usageMetadata);
    if (isRecord(event.usageMetadata)) {
      const usage = geminiUsageOf(event);
      inputTokens = usage.input;
      outputTokens = usage.output;
    }
    if (clientWire === 'openai') {
      if (!started) {
        started = true;
        yield openAIChunk(model, { role: 'assistant', content: '' }, null);
      }
      if (text !== '') yield openAIChunk(model, { content: text }, null);
      if (isFinal) {
        finished = true;
        yield openAIChunk(model, {}, 'stop');
        if (includeUsage) yield openAIUsageChunk(model, inputTokens, outputTokens);
        yield 'data: [DONE]\n\n';
      }
    } else {
      for (const e of openAnthropic()) yield e;
      if (text !== '') {
        if (textIndex === -1) {
          textIndex = nextIndex;
          nextIndex += 1;
          yield sseEvent('content_block_start', { type: 'content_block_start', index: textIndex, content_block: { type: 'text', text: '' } });
        }
        yield sseEvent('content_block_delta', { type: 'content_block_delta', index: textIndex, delta: { type: 'text_delta', text } });
      }
      if (isFinal) {
        for (const e of closeAnthropic()) yield e;
      }
    }
  }
  if (!finished) {
    if (clientWire === 'openai') {
      if (!started) yield openAIChunk(model, { role: 'assistant', content: '' }, null);
      yield openAIChunk(model, {}, 'stop');
      if (includeUsage) yield openAIUsageChunk(model, inputTokens, outputTokens);
      yield 'data: [DONE]\n\n';
    } else {
      for (const e of openAnthropic()) yield e;
      for (const e of closeAnthropic()) yield e;
    }
  }
}
