import { readFileSync } from 'node:fs';

export type BriefModel = {
  id: string;
  rating_milli: number;
  context_length: number;
  tools: boolean;
  vision: boolean;
  json: boolean;
  free: boolean;
};

export type PromptManifestEntry = Record<string, unknown>;
export type Brief = {
  models: BriefModel[];
  prompts: PromptManifestEntry[];
};

export type CatalogModel = BriefModel;
export type Catalog = {
  providers: [{
    id: 'openrouter';
    base_url: 'https://openrouter.ai/api/v1';
    key_env: 'OPENROUTER_API_KEY';
    wire: 'openai';
    models: CatalogModel[];
  }];
};

type GatewayError = { error?: { code?: unknown; message?: unknown } };
export type DialResult =
  | { ok: true; status: number; body: any }
  | { ok: false; status: number; code: string }
  | { ok: false; kind: 'timeout' | 'network_error'; code: string };
export type StreamDialResult =
  | { ok: true; status: number; body: ReadableStream<Uint8Array> }
  | { ok: false; status: number; code: string }
  | { ok: false; kind: 'timeout' | 'network_error'; code: string };

function documentAfterSchema(text: string, schema: string): unknown {
  const marker = `\"schema\"\\s*:\\s*\"${schema.replace(/[.*+?^${}()|[\\]\\]/g, '\\$&')}\"`;
  const found = new RegExp(marker).exec(text);
  if (!found) throw new Error('schema missing');
  const start = text.lastIndexOf('{', found.index);
  if (start < 0) throw new Error('document missing');
  let depth = 0;
  let quoted = false;
  let escaped = false;
  for (let i = start; i < text.length; i++) {
    const c = text[i];
    if (quoted) {
      if (escaped) escaped = false;
      else if (c === '\\') escaped = true;
      else if (c === '"') quoted = false;
      continue;
    }
    if (c === '"') quoted = true;
    else if (c === '{') depth++;
    else if (c === '}' && --depth === 0) return JSON.parse(text.slice(start, i + 1));
  }
  throw new Error('unterminated document');
}

function readBrief(): Brief {
  const path = process.env.MAKEFUN_PROMPT_FILE;
  if (!path) return { models: [], prompts: [] };
  try {
    const source = readFileSync(path, 'utf8');
    const modelDoc = documentAfterSchema(source, 'free-model-aggregator-02/models/v1') as { models?: BriefModel[] };
    const promptDoc = documentAfterSchema(source, 'free-model-aggregator-02/prompts/v1') as { prompts?: PromptManifestEntry[] };
    if (!Array.isArray(modelDoc.models) || !Array.isArray(promptDoc.prompts)) throw new Error('invalid brief');
    return { models: modelDoc.models, prompts: promptDoc.prompts };
  } catch {
    return { models: [], prompts: [] };
  }
}

export const brief: Brief = readBrief();
export const models: BriefModel[] = brief.models;

export function buildCatalog(source: Brief = brief): Catalog {
  return { providers: [{
    id: 'openrouter',
    base_url: 'https://openrouter.ai/api/v1',
    key_env: 'OPENROUTER_API_KEY',
    wire: 'openai',
    models: source.models.map(m => ({ id: m.id, free: true, context_length: m.context_length,
      rating_milli: m.rating_milli, tools: m.tools, vision: m.vision, json: m.json })),
  }] };
}

export function errorCodeForStatus(status: number): string {
  const known: Record<number, string> = { 400: 'bad_request', 401: 'unauthorized', 403: 'forbidden', 404: 'not_found', 408: 'request_timeout', 429: 'rate_limit', 500: 'server_error', 502: 'bad_gateway', 503: 'service_unavailable', 504: 'gateway_timeout' };
  return known[status] ?? `http_${status}`;
}

function gatewayConfig() {
  const base = process.env.OPENAI_BASE_URL || process.env.ANTHROPIC_BASE_URL || 'https://openrouter.ai/api/v1';
  const key = process.env.OPENAI_API_KEY || process.env.ANTHROPIC_API_KEY || process.env.OPENROUTER_API_KEY;
  return { url: `${base.replace(/\/+$/, '')}/chat/completions`, key };
}

function requestBody(model: string, request: Record<string, any>, stream: boolean): Record<string, unknown> {
  const max = typeof request.max_tokens === 'number' ? Math.min(request.max_tokens, 8192) : 1024;
  const body: Record<string, unknown> = { model, messages: request.messages ?? [], max_tokens: max, stream };
  if ('temperature' in request) body.temperature = request.temperature;
  if ('response_format' in request) body.response_format = request.response_format;
  if ('tools' in request) body.tools = request.tools;
  if (stream) body.stream_options = { include_usage: true };
  return body;
}

async function dial(model: string, request: Record<string, any>, timeoutMs: number, stream: boolean): Promise<DialResult | StreamDialResult> {
  if (!brief.models.some(item => item.id === model)) return { ok: false, kind: 'network_error', code: 'model_unavailable' };
  const { url, key } = gatewayConfig();
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  let streaming = false;
  try {
    const headers: Record<string, string> = { 'Content-Type': 'application/json' };
    if (key) headers.Authorization = `Bearer ${key}`;
    const response = await fetch(url, { method: 'POST', headers, body: JSON.stringify(requestBody(model, request, stream)), signal: controller.signal });
    if (!response.ok) {
      let data: GatewayError | undefined;
      try { data = await response.json() as GatewayError; } catch { /* status fallback */ }
      const raw = data?.error?.code ?? data?.error?.message;
      const code = typeof raw === 'string' && raw.trim() ? raw.trim().slice(0, 80) : errorCodeForStatus(response.status);
      return { ok: false, status: response.status, code };
    }
    if (stream) {
      streaming = true;
      const reader = response.body!.getReader();
      const body = new ReadableStream<Uint8Array>({
        async pull(target) {
          try {
            const part = await reader.read();
            if (part.done) { clearTimeout(timer); target.close(); }
            else target.enqueue(part.value);
          } catch (error) { clearTimeout(timer); target.error(error); }
        },
        async cancel(reason) { clearTimeout(timer); await reader.cancel(reason); },
      });
      return { ok: true, status: response.status, body };
    }
    return { ok: true, status: response.status, body: await response.json() };
  } catch (error) {
    const timeout = controller.signal.aborted || (error instanceof Error && error.name === 'AbortError');
    return { ok: false, kind: timeout ? 'timeout' : 'network_error', code: timeout ? 'upstream_timeout' : 'upstream_error' };
  } finally {
    if (!streaming) clearTimeout(timer);
  }
}

export function dialJson(model: string, request: Record<string, any>, timeoutMs = 25000): Promise<DialResult> {
  return dial(model, request, timeoutMs, false) as Promise<DialResult>;
}

export function dialStream(model: string, request: Record<string, any>, timeoutMs = 25000): Promise<StreamDialResult> {
  return dial(model, request, timeoutMs, true) as Promise<StreamDialResult>;
}
