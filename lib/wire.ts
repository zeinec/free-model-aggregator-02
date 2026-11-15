import type { CatalogModel } from './providers.ts';

export const MODEL_CREATED = 1763610000;
type AnyObj = Record<string, any>;

export function anthropicRequestToOpenAI(body: AnyObj): AnyObj {
  const messages: AnyObj[] = [];
  if (typeof body.system === 'string') messages.push({ role: 'system', content: body.system });
  else if (Array.isArray(body.system)) {
    for (const block of body.system) if (block?.type === 'text' && typeof block.text === 'string') messages.push({ role: 'system', content: block.text });
  }
  for (const message of body.messages ?? []) {
    const role = message.role;
    const blocks = Array.isArray(message.content) ? message.content : [{ type: 'text', text: message.content ?? '' }];
    if (role === 'assistant') {
      const text: AnyObj[] = [];
      const tool_calls: AnyObj[] = [];
      for (const block of blocks) {
        if (block?.type === 'text') text.push({ type: 'text', text: block.text ?? '' });
        else if (block?.type === 'tool_use') tool_calls.push({ id: block.id, type: 'function', function: { name: block.name, arguments: JSON.stringify(block.input ?? {}) } });
      }
      const content = text.length === 0 ? '' : text.map(x => x.text).join('');
      const out: AnyObj = { role: 'assistant', content };
      if (tool_calls.length) out.tool_calls = tool_calls;
      messages.push(out);
    } else {
      let text = '';
      const parts: AnyObj[] = [];
      for (const block of blocks) {
        if (block?.type === 'text') {
          text += block.text ?? '';
          parts.push({ type: 'text', text: block.text ?? '' });
        } else if (block?.type === 'image' && block.source?.type === 'base64') {
          parts.push({ type: 'image_url', image_url: { url: `data:${block.source.media_type};base64,${block.source.data}` } });
        } else if (block?.type === 'tool_result') {
          const result = typeof block.content === 'string' ? block.content : Array.isArray(block.content) ? block.content.filter((v: AnyObj) => v?.type === 'text').map((v: AnyObj) => v.text).join('') : JSON.stringify(block.content ?? '');
          messages.push({ role: 'tool', tool_call_id: block.tool_use_id, content: result });
        }
      }
      if (parts.some(p => p.type === 'image_url')) messages.push({ role, content: parts });
      else if (text || !blocks.length) messages.push({ role, content: text });
    }
  }
  const result: AnyObj = { messages };
  for (const key of ['max_tokens', 'temperature', 'stream', 'stream_options'] as const) if (key in body) result[key] = body[key];
  if (typeof body.max_tokens === 'number') result.max_tokens = Math.min(body.max_tokens, 8192);
  if (Array.isArray(body.tools)) result.tools = body.tools.map((tool: AnyObj) => ({ type: 'function', function: { name: tool.name, description: tool.description, parameters: tool.input_schema } }));
  return result;
}

export function completionText(body: AnyObj): string {
  return (body.choices ?? []).map((choice: AnyObj) => typeof choice?.message?.content === 'string' ? choice.message.content : '').join('');
}

export function extractToolCalls(body: AnyObj): AnyObj[] {
  return (body.choices ?? []).flatMap((choice: AnyObj) => choice?.message?.tool_calls ?? []);
}

export function usageFromBody(body: AnyObj): AnyObj | undefined {
  const usage = body?.usage;
  if (!usage || typeof usage !== 'object') return undefined;
  return { prompt_tokens: Number(usage.prompt_tokens ?? 0), completion_tokens: Number(usage.completion_tokens ?? 0), total_tokens: Number(usage.total_tokens ?? 0) };
}

export function usageFallback(text: string): { prompt_tokens: number; completion_tokens: number; total_tokens: number } {
  const completion_tokens = Math.max(1, Math.ceil(text.length / 4));
  return { prompt_tokens: 0, completion_tokens, total_tokens: completion_tokens };
}

export function finishReasonToStopReason(reason: unknown): 'end_turn' | 'max_tokens' | 'tool_use' {
  if (reason === 'length') return 'max_tokens';
  if (reason === 'tool_calls') return 'tool_use';
  return 'end_turn';
}

export function openAIAnswerToAnthropic(body: AnyObj, model: string): AnyObj {
  const choice = body?.choices?.[0] ?? {};
  const content: AnyObj[] = [{ type: 'text', text: typeof choice.message?.content === 'string' ? choice.message.content : '' }];
  for (const call of choice.message?.tool_calls ?? []) {
    let input: unknown = {};
    try { input = JSON.parse(call.function?.arguments ?? '{}'); } catch { input = {}; }
    content.push({ type: 'tool_use', id: call.id, name: call.function?.name ?? '', input });
  }
  return {
    id: `msg-${body?.id ?? ''}`, type: 'message', role: 'assistant', model, content,
    stop_reason: finishReasonToStopReason(choice.finish_reason), stop_sequence: null,
    usage: { input_tokens: Number(body?.usage?.prompt_tokens ?? 0), output_tokens: Number(body?.usage?.completion_tokens ?? 0) },
  };
}

export function openAIModelList(catalogModels: CatalogModel[]): AnyObj {
  return { data: catalogModels.map(model => ({ id: model.id, object: 'model', created: MODEL_CREATED, owned_by: 'openrouter' })) };
}

export function anthropicModelList(catalogModels: CatalogModel[]): AnyObj {
  return { data: catalogModels.map(model => ({ type: 'model', id: model.id, display_name: model.id, created_at: MODEL_CREATED })) };
}

export function rewriteOpenAIStreamModel(line: string, dialledId: string): string {
  const match = /^(\s*data:\s*)(.*?)(\s*)$/.exec(line);
  if (!match || match[2] === '[DONE]') return line;
  try {
    const value = JSON.parse(match[2]);
    if (!value || typeof value !== 'object' || !('model' in value)) return line;
    value.model = dialledId;
    return `${match[1]}${JSON.stringify(value)}${match[3]}`;
  } catch { return line; }
}

export function extractCompletionText(body: AnyObj): string { return completionText(body); }
export function extractUsage(body: AnyObj): AnyObj | undefined { return usageFromBody(body); }

function event(name: string, data: AnyObj): string { return `event: ${name}\ndata: ${JSON.stringify(data)}\n\n`; }

export function anthropicStreamEvents(chunks: AnyObj[], model: string, fallback: { prompt_tokens: number; completion_tokens: number; total_tokens: number }): string[] {
  const providerId = chunks.find(chunk => chunk?.id)?.id ?? 'response';
  const usageChunk = [...chunks].reverse().find(chunk => chunk?.usage);
  const inputTokens = Number(usageChunk?.usage?.prompt_tokens ?? 0);
  const outputTokens = Math.max(1, Number(usageChunk?.usage?.completion_tokens ?? fallback.completion_tokens));
  const textParts: string[] = [];
  const tools = new Map<number, { id: string; name: string; args: string[] }>();
  let reason: unknown;
  for (const chunk of chunks) {
    for (const choice of chunk?.choices ?? []) {
      const delta = choice?.delta ?? {};
      if (typeof delta.content === 'string') textParts.push(delta.content);
      for (const call of delta.tool_calls ?? []) {
        const idx = Number(call.index ?? 0);
        const item = tools.get(idx) ?? { id: '', name: '', args: [] };
        if (call.id) item.id = call.id;
        if (call.function?.name) item.name += call.function.name;
        if (typeof call.function?.arguments === 'string') item.args.push(call.function.arguments);
        tools.set(idx, item);
      }
      if (choice.finish_reason != null) reason = choice.finish_reason;
    }
  }
  const output: string[] = [event('message_start', { message: { id: `msg-${providerId}`, type: 'message', role: 'assistant', model, content: [], stop_reason: null, stop_sequence: null, usage: { input_tokens: inputTokens, output_tokens: 0 } } })];
  if (textParts.length) {
    output.push(event('content_block_start', { index: 0, content_block: { type: 'text', text: '' } }));
    for (const text of textParts) output.push(event('content_block_delta', { index: 0, delta: { type: 'text_delta', text } }));
    output.push(event('content_block_stop', { index: 0 }));
  }
  const toolList = [...tools.entries()].sort((a, b) => a[0] - b[0]);
  toolList.forEach(([, tool], pos) => {
    const index = pos + (textParts.length ? 1 : 0);
    output.push(event('content_block_start', { index, content_block: { type: 'tool_use', id: tool.id, name: tool.name, input: {} } }));
    for (const args of tool.args) output.push(event('content_block_delta', { index, delta: { type: 'input_json_delta', partial_json: args } }));
    output.push(event('content_block_stop', { index }));
  });
  output.push(event('message_delta', { delta: { stop_reason: finishReasonToStopReason(reason), stop_sequence: null }, usage: { output_tokens: outputTokens } }));
  output.push(event('message_stop', {}));
  return output;
}
