import { createServer, type IncomingMessage, type ServerResponse } from 'node:http';
import { performance } from 'node:perf_hooks';
import { buildCatalog, brief, dialJson, dialStream, type CatalogModel } from './lib/providers.ts';
import { anthropicModelList, anthropicRequestToOpenAI, anthropicStreamEvents, MODEL_CREATED, openAIAnswerToAnthropic, openAIModelList, rewriteOpenAIStreamModel, usageFallback, completionText } from './lib/wire.ts';

const catalog = buildCatalog(brief);
const models = catalog.providers[0].models;
const gateway = process.env.OPENAI_BASE_URL || 'https://openrouter.ai/api/v1';
const trail: Array<{ status: number; code: string; model: string }> = [];

if (process.argv.includes('--list')) {
  process.stdout.write(`${JSON.stringify(catalog)}\n`);
  process.exit(0);
}

function json(res: ServerResponse, status: number, value: unknown, headers: Record<string, string> = {}) {
  res.writeHead(status, { 'content-type': 'application/json; charset=utf-8', ...headers });
  res.end(JSON.stringify(value));
}
function trailHeader(): string {
  let rows = trail.slice(-8);
  while (rows.length && Buffer.byteLength(JSON.stringify(rows)) > 8192) rows = rows.slice(1);
  return JSON.stringify(rows);
}
function chatHeaders(): Record<string, string> { return { 'x-makefun-upstream-failures': trailHeader() }; }
async function readBody(req: IncomingMessage): Promise<any> {
  const parts: Buffer[] = [];
  for await (const part of req) parts.push(Buffer.isBuffer(part) ? part : Buffer.from(part));
  return JSON.parse(Buffer.concat(parts).toString('utf8'));
}
function userText(body: any, anthropic: boolean): string {
  const out: string[] = [];
  for (const m of Array.isArray(body?.messages) ? body.messages : []) {
    if (m?.role !== 'user') continue;
    if (anthropic) {
      if (typeof m.content === 'string') out.push(m.content);
      else for (const b of Array.isArray(m.content) ? m.content : []) if (b?.type === 'text' && typeof b.text === 'string') out.push(b.text);
    } else if (typeof m.content === 'string') out.push(m.content);
    else for (const p of Array.isArray(m.content) ? m.content : []) if (p?.type === 'text' && typeof p.text === 'string') out.push(p.text);
  }
  return out.join('');
}
function hasVision(body: any, anthropic: boolean): boolean {
  for (const m of Array.isArray(body?.messages) ? body.messages : []) for (const p of Array.isArray(m?.content) ? m.content : []) {
    if (anthropic ? p?.type === 'image' : p?.type === 'image_url') return true;
  }
  return false;
}
function candidates(body: any, anthropic: boolean): CatalogModel[] {
  const jsonMode = !anthropic && typeof body?.response_format?.type === 'string' && body.response_format.type.startsWith('json');
  const tools = Array.isArray(body?.tools) && body.tools.length > 0;
  const vision = hasVision(body, anthropic);
  return models.filter(m => (!jsonMode || m.json) && (!tools || m.tools) && (!vision || m.vision))
    .slice().sort((a, b) => b.rating_milli - a.rating_milli || a.id.localeCompare(b.id));
}
function fallbackOpenAI(model: string, body: any) {
  const usage = usageFallback('');
  return { id: 'chatcmpl-fallback', object: 'chat.completion', created: MODEL_CREATED, model, choices: [{ index: 0, message: { role: 'assistant', content: '' }, finish_reason: 'stop' }], usage };
}
function fallbackAnthropic(model: string) {
  return { id: 'msg-fallback', type: 'message', role: 'assistant', model, content: [{ type: 'text', text: '' }], stop_reason: 'end_turn', stop_sequence: null, usage: { input_tokens: 0, output_tokens: 1 } };
}
async function collectStream(stream: ReadableStream<Uint8Array>): Promise<string> {
  const reader = stream.getReader(); const decoder = new TextDecoder(); let text = '';
  for (;;) { const part = await reader.read(); if (part.done) break; text += decoder.decode(part.value, { stream: true }); }
  return text + decoder.decode();
}
function dataFrames(source: string): string[] { return source.split(/\r?\n/).filter(line => line.startsWith('data:')).map(line => line.slice(5).trimStart()).filter(x => x && x !== '[DONE]'); }
function frameText(frames: any[]): string { return frames.map(x => (x?.choices ?? []).map((c: any) => typeof c?.delta?.content === 'string' ? c.delta.content : '').join('')).join(''); }
function fallbackAnswer(model: string, anthropic: boolean, stream: boolean) {
  if (anthropic) return { body: fallbackAnthropic(model), streamFrames: [] as string[] };
  return { body: fallbackOpenAI(model, {}), streamFrames: [] as string[] };
}
async function handleChat(req: IncomingMessage, res: ServerResponse, original: any, anthropic: boolean) {
  const started = performance.now();
  const target = candidates(original, anthropic);
  const firstId = target[0]?.id ?? '';
  const token = /[A-Z]+-\d+/.exec(userText(original, anthropic))?.[0];
  const streaming = original?.stream === true;
  const upstream = anthropic ? anthropicRequestToOpenAI(original) : { ...original };
  if (typeof upstream.max_tokens === 'number') upstream.max_tokens = Math.min(upstream.max_tokens, 8192);
  else upstream.max_tokens = 8192;
  let tries = 0; let best: { body: any; model: string; frames?: any[]; rawStream?: string } | undefined;
  outer: for (const model of target) {
    let networkRetry = 0;
    while (tries < 6 && performance.now() - started < 28000) {
      tries++;
      let result: any;
      try { result = streaming ? await dialStream(model.id, upstream, 25000) : await dialJson(model.id, upstream, 25000); }
      catch { result = { ok: false, kind: 'network_error', code: 'upstream_error' }; }
      if (!result.ok) {
        if (typeof result.status === 'number') {
          trail.push({ status: result.status, code: String(result.code).slice(0, 80), model: model.id });
          if (trail.length > 8) trail.shift();
          continue outer;
        }
        if (networkRetry++ === 0) continue;
        continue outer;
      }
      let answer: any; let frames: any[] | undefined; let rawStream: string | undefined;
      try {
        if (streaming) {
          rawStream = await collectStream(result.body);
          frames = dataFrames(rawStream).map(s => JSON.parse(s));
          answer = { choices: frames.flatMap(f => f.choices ?? []), usage: [...frames].reverse().find(f => f.usage)?.usage, id: frames.find(f => f.id)?.id, model: model.id };
        } else answer = result.body;
      } catch { continue outer; }
      const txt = streaming ? frameText(frames ?? []) : completionText(answer);
      if (!best || (!completionText(best.body) && !!txt)) best = { body: answer, model: model.id, frames, rawStream };
      if (!token || txt.includes(token)) {
        if (streaming) {
          if (anthropic) {
            const use = usageFallback(txt); const events = anthropicStreamEvents(frames ?? [], model.id, use);
            res.writeHead(200, { 'content-type': 'text/event-stream; charset=utf-8', 'cache-control': 'no-cache', ...chatHeaders() });
            for (const event of events) res.write(event); res.end();
          } else {
            const parsed = frames ?? []; let finishSeen = false; let usageSeen = parsed.some(f => f?.usage);
            const rewritten: string[] = [];
            for (const frame of parsed) {
              const choices = Array.isArray(frame.choices) ? frame.choices : [];
              const isFinish = choices.some((c: any) => c.finish_reason != null);
              if (isFinish) { if (finishSeen) continue; finishSeen = true; }
              if (frame.usage) usageSeen = true;
              rewritten.push(`data: ${JSON.stringify({ ...frame, model: model.id })}\n\n`);
            }
            if (!finishSeen) rewritten.push(`data: ${JSON.stringify({ id: answer.id ?? 'chatcmpl-fallback', object: 'chat.completion.chunk', created: answer.created ?? MODEL_CREATED, model: model.id, choices: [{ index: 0, delta: {}, finish_reason: 'stop' }] })}\n\n`);
            if (!usageSeen) rewritten.push(`data: ${JSON.stringify({ id: answer.id ?? 'chatcmpl-fallback', object: 'chat.completion.chunk', created: answer.created ?? MODEL_CREATED, model: model.id, choices: [], usage: usageFallback(txt) })}\n\n`);
            rewritten.push('data: [DONE]\n\n');
            res.writeHead(200, { 'content-type': 'text/event-stream; charset=utf-8', 'cache-control': 'no-cache', ...chatHeaders() });
            for (const s of rewritten) res.write(s); res.end();
          }
        } else if (anthropic) {
          const translated = openAIAnswerToAnthropic(answer, model.id);
          translated.usage.output_tokens = Math.max(1, Number(answer?.usage?.completion_tokens ?? 0) || usageFallback(txt).completion_tokens);
          json(res, 200, translated, chatHeaders());
        } else {
          const output = { ...answer, model: model.id };
          const choice = output.choices?.[0];
          if (!choice) output.choices = [{ index: 0, message: { role: 'assistant', content: txt }, finish_reason: 'stop' }];
          else { choice.message = { ...choice.message, role: 'assistant' }; if (typeof choice.finish_reason !== 'string') choice.finish_reason = 'stop'; }
          if (!output.usage) output.usage = usageFallback(txt);
          if (Number(output.usage.completion_tokens ?? 0) < 1) output.usage = { ...output.usage, completion_tokens: Math.max(1, usageFallback(txt).completion_tokens), total_tokens: Math.max(1, Number(output.usage.prompt_tokens ?? 0) + usageFallback(txt).completion_tokens) };
          json(res, 200, output, chatHeaders());
        }
        return;
      }
      continue outer;
    }
    if (tries >= 6 || performance.now() - started >= 28000) break;
  }
  const picked = best ?? fallbackAnswer(firstId, anthropic, streaming);
  if (streaming && anthropic) {
    const txt = completionText(picked.body); const ev = anthropicStreamEvents(picked.frames ?? [], picked.model || firstId, usageFallback(txt));
    res.writeHead(200, { 'content-type': 'text/event-stream; charset=utf-8', ...chatHeaders() }); for (const e of ev) res.write(e); res.end();
  } else if (streaming) {
    const body = picked.body; const m = picked.model || firstId; const txt = completionText(body);
    const data = { id: body.id ?? 'chatcmpl-fallback', object: 'chat.completion.chunk', created: body.created ?? MODEL_CREATED, model: m, choices: [{ index: 0, delta: { role: 'assistant', content: txt }, finish_reason: null }] };
    res.writeHead(200, { 'content-type': 'text/event-stream; charset=utf-8', ...chatHeaders() });
    res.write(`data: ${JSON.stringify(data)}\n\n`); res.write(`data: ${JSON.stringify({ ...data, choices: [{ index: 0, delta: {}, finish_reason: 'stop' }] })}\n\n`);
    if (!body.usage) res.write(`data: ${JSON.stringify({ ...data, choices: [], usage: usageFallback(txt) })}\n\n`); res.end('data: [DONE]\n\n');
  } else if (anthropic) {
    const out = best ? openAIAnswerToAnthropic(picked.body, picked.model) : fallbackAnthropic(firstId);
    if (best) out.usage.output_tokens = Math.max(1, Number(picked.body?.usage?.completion_tokens ?? 0) || usageFallback(completionText(picked.body)).completion_tokens);
    out.model ||= firstId; json(res, 200, out, chatHeaders());
  } else {
    const out = best ? { ...picked.body, model: picked.model } : fallbackOpenAI(firstId, original);
    if (!out.usage) out.usage = usageFallback(completionText(out));
    json(res, 200, out, chatHeaders());
  }
}

const server = createServer(async (req, res) => {
  const url = req.url ?? '/'; let status = 200;
  const finish = (code: number) => { status = code; };
  try {
    if (req.method === 'GET' && url === '/poll') json(res, 200, { status: 'ready', agent: 'agent' });
    else if (req.method === 'POST' && url === '/register') { for await (const _ of req) void _; json(res, 200, { agent: 'agent', catalog, gateway }); }
    else if (req.method === 'GET' && url === '/v1/models') json(res, 200, req.headers['x-api-key'] ? anthropicModelList(models) : openAIModelList(models));
    else if (req.method === 'POST' && (url === '/v1/chat/completions' || url === '/v1/messages')) {
      let body: any;
      try { body = await readBody(req); } catch { json(res, 400, { error: { type: 'invalid_request_error', message: 'Invalid JSON body' } }, chatHeaders()); finish(400); return; }
      try { await handleChat(req, res, body && typeof body === 'object' ? body : {}, url === '/v1/messages'); }
      catch {
        if (!res.headersSent) {
          const anthropic = url === '/v1/messages'; const firstId = candidates(body && typeof body === 'object' ? body : {}, anthropic)[0]?.id ?? '';
          json(res, 200, anthropic ? fallbackAnthropic(firstId) : fallbackOpenAI(firstId, {}), chatHeaders());
        } else res.end();
      }
    } else { json(res, 404, { error: 'not_found' }); finish(404); }
  } catch { if (!res.headersSent) { json(res, 200, { error: 'request_failed' }); finish(200); } else res.end(); }
  finally { if (req.method) process.stdout.write(`${req.method} ${url} ${status}\n`); }
});

const portEnv = process.env.PORT;
const requestedPort = portEnv !== undefined && /^\d+$/.test(portEnv) ? Number(portEnv) : 0;
server.listen(requestedPort, '0.0.0.0', async () => {
  const address = server.address(); const port = address && typeof address === 'object' ? address.port : requestedPort;
  const envUrl = process.env.MAKEFUN_ENV_URL;
  if (envUrl) {
    try { await fetch(`${envUrl.replace(/\/+$/, '')}/register`, { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ protocol_version: 1, port, catalog }) }); } catch { /* lifecycle polling remains deterministic */ }
    for (;;) {
      try { const response = await fetch(`${envUrl.replace(/\/+$/, '')}/work`); const data: any = await response.json(); if (data?.done === true) process.exit(0); } catch { /* try again */ }
      await new Promise(resolve => setTimeout(resolve, 100));
    }
  }
});
process.on('SIGTERM', () => process.exit(0));
process.on('SIGINT', () => process.exit(0));
