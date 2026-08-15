const TOKEN_KEY = 'access_token';
const FETCH_TIMEOUT_MS = 15_000;
const STREAM_READ_TIMEOUT_MS = 30_000;

export function getToken(): string | null {
  return localStorage.getItem(TOKEN_KEY);
}

export function setToken(token: string): void {
  localStorage.setItem(TOKEN_KEY, token);
}

export function clearToken(): void {
  localStorage.removeItem(TOKEN_KEY);
}

type LogoutHandler = () => void;
let onLogout: LogoutHandler | null = null;

export function setLogoutHandler(handler: LogoutHandler): void {
  onLogout = handler;
}

function abortableFetch(url: string, options: RequestInit = {}): Promise<Response> {
  const controller = new AbortController();
  const signal = controller.signal;
  const timeout = setTimeout(() => controller.abort(), FETCH_TIMEOUT_MS);

  return fetch(url, { ...options, signal }).finally(() => clearTimeout(timeout));
}

export async function apiFetch(
  url: string,
  options: RequestInit = {}
): Promise<Response> {
  const token = getToken();
  const headers: Record<string, string> = {
    ...(options.headers as Record<string, string>),
  };

  if (token) {
    headers['Authorization'] = `Bearer ${token}`;
  }

  if (!(options.body instanceof FormData)) {
    headers['Content-Type'] = 'application/json';
  }

  const response = await abortableFetch(url, {
    ...options,
    headers,
  });

  if (response.status === 401 && onLogout) {
    clearToken();
    onLogout();
  }

  return response;
}

export async function apiGet(url: string): Promise<Response> {
  return apiFetch(url, { method: 'GET' });
}

export async function apiPost(
  url: string,
  body: unknown
): Promise<Response> {
  return apiFetch(url, {
    method: 'POST',
    body: body instanceof FormData ? body : JSON.stringify(body),
    headers: body instanceof FormData ? {} : undefined,
  });
}

export async function apiDelete(url: string): Promise<Response> {
  return apiFetch(url, { method: 'DELETE' });
}

/**
 * Read a single chunk from the stream reader with a timeout.
 * Times out after STREAM_READ_TIMEOUT_MS if no data arrives.
 */
async function readWithTimeout(
  reader: ReadableStreamDefaultReader<Uint8Array>
): Promise<ReadableStreamReadResult<Uint8Array>> {
  const result = await Promise.race([
    reader.read(),
    new Promise<never>((_, reject) =>
      setTimeout(
        () => reject(new DOMException('流式读取超时（30秒无数据）', 'TimeoutError')),
        STREAM_READ_TIMEOUT_MS
      )
    ),
  ]);
  return result;
}

/**
 * SSE stream reader for POST endpoints (can't use EventSource).
 * Yields parsed { event, data } objects.
 */
export async function* readSSEStream(
  url: string,
  body: unknown
): AsyncGenerator<{ event: string; data: string }> {
  const token = getToken();
  const headers: Record<string, string> = {
    'Content-Type': 'application/json',
  };
  if (token) {
    headers['Authorization'] = `Bearer ${token}`;
  }

  let response: Response;
  try {
    response = await abortableFetch(url, {
      method: 'POST',
      headers,
      body: JSON.stringify(body),
    });
  } catch (e: unknown) {
    if (e instanceof DOMException && e.name === 'AbortError') {
      throw new Error('请求超时（15秒），请确认后端服务已启动');
    }
    if (e instanceof TypeError && e.message.includes('fetch')) {
      throw new Error('无法连接到服务器，请确认后端已启动 (http://localhost:8000)');
    }
    throw e;
  }

  if (response.status === 401 && onLogout) {
    clearToken();
    onLogout();
    throw new Error('认证已过期，请重新登录');
  }

  if (!response.ok) {
    const text = await response.text().catch(() => '');
    throw new Error(text || `服务器错误 (HTTP ${response.status})`);
  }

  const contentType = response.headers.get('content-type') || '';
  if (!contentType.includes('text/event-stream')) {
    const text = await response.text().catch(() => '');
    throw new Error(text || `接口返回非流式数据 (${contentType})`);
  }

  const reader = response.body?.getReader();
  if (!reader) throw new Error('无法读取响应数据');

  const decoder = new TextDecoder();
  let buffer = '';

  while (true) {
    let done: boolean;
    let value: Uint8Array | undefined;

    try {
      const result = await readWithTimeout(reader);
      done = result.done;
      value = result.value;
    } catch (e: unknown) {
      reader.cancel();
      if (e instanceof DOMException && e.name === 'TimeoutError') {
        throw new Error('AI 服务响应超时（30秒无数据），请检查 API Key 配置或网络连接');
      }
      throw e;
    }

    if (done || !value) break;

    // Normalize: replace \r\n with \n (HTTP may use CRLF)
    const chunk = decoder.decode(value, { stream: true }).replace(/\r\n/g, '\n');
    buffer += chunk;

    // Split on double-newline (SSE event boundary)
    const parts = buffer.split('\n\n');
    buffer = parts.pop() || '';

    for (const part of parts) {
      const lines = part.split('\n');
      let eventType = 'message';
      let data = '';

      for (const line of lines) {
        if (line.startsWith('event: ')) {
          eventType = line.slice(7).trim();
        } else if (line.startsWith('data: ')) {
          // Accumulate multi-line data fields
          data += line.slice(6);
        }
      }

      if (data || eventType !== 'message') {
        yield { event: eventType, data };
      }
    }
  }
}
