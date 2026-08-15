import { readSSEStream, apiGet, apiDelete } from './client';
import type { MultimodalChatRequest, ChatSessionData } from '../types';

/** Union type: either a display token or a stage progress indicator */
export type StreamEvent =
  | { type: 'token'; value: string }
  | { type: 'stage'; stage: string; text: string };

const STAGE_LABELS: Record<string, string> = {
  routing: '正在分析问题类型...',
  small_model: '正在理解问题内容...',
  rag: '正在检索相关知识库...',
  llm_prompt: '正在整理上下文...',
  llm_stream: '正在生成回复...',
  finalize: '正在整理结果...',
};

/**
 * Send a chat message via SSE streaming.
 * Yields both token strings and stage progress objects.
 */
export async function* sendChatMessage(
  sessionId: string,
  userInput: string,
  llmModel?: string
): AsyncGenerator<StreamEvent> {
  const body: MultimodalChatRequest = {
    session_id: sessionId,
    user_input: userInput,
  };
  if (llmModel) body.llm_model = llmModel;

  for await (const event of readSSEStream('/v1/multimodal/chat/stream', body)) {
    if (event.event === 'error') {
      let message = '对话服务异常';
      try {
        const err = JSON.parse(event.data);
        if (err.message) message = err.message;
      } catch { /* use default */ }
      throw new Error(message);
    }

    // Try to parse event data
    let parsed: Record<string, unknown> | null = null;
    try {
      parsed = JSON.parse(event.data);
    } catch {
      continue;
    }

    // Stage progress events (non-token)
    if (event.event !== 'token' && parsed?.stage) {
      const stage = parsed.stage as string;
      const label = STAGE_LABELS[stage];
      if (label) {
        yield { type: 'stage', stage, text: label };
      }
      continue;
    }

    // Token events — extract content token
    if (parsed?.token && typeof parsed.token === 'string') {
      yield { type: 'token', value: parsed.token };
    }
  }
}

export async function getSession(sessionId: string): Promise<ChatSessionData> {
  const res = await apiGet(`/v1/multimodal/chat/session/${sessionId}`);
  if (!res.ok) {
    return { session_id: sessionId, history: [] };
  }
  return res.json();
}

export async function deleteSession(sessionId: string): Promise<void> {
  await apiDelete(`/v1/multimodal/chat/session/${sessionId}`);
}
