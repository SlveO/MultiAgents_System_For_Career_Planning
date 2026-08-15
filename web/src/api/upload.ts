import { getToken } from './client';
import type { UploadResponse } from '../types';

const BACKEND = 'http://localhost:8000';
const UPLOAD_TIMEOUT_MS = 60_000;

/**
 * Upload a file directly to the backend (bypasses Vite proxy —
 * multipart/form-data can get mangled by the proxy layer).
 */
export async function uploadFile(file: File): Promise<UploadResponse> {
  const formData = new FormData();
  formData.append('file', file);

  const token = getToken();
  const headers: Record<string, string> = {};
  if (token) {
    headers['Authorization'] = `Bearer ${token}`;
  }
  // Do NOT set Content-Type — browser sets it with boundary for FormData

  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), UPLOAD_TIMEOUT_MS);

  let res: Response;
  try {
    res = await fetch(`${BACKEND}/v1/upload`, {
      method: 'POST',
      headers,
      body: formData,
      signal: controller.signal,
    });
  } catch (e: unknown) {
    clearTimeout(timeout);
    if (e instanceof DOMException && e.name === 'AbortError') {
      throw new Error(`上传超时（${UPLOAD_TIMEOUT_MS / 1000}秒），文件可能过大或网络不通`);
    }
    if (e instanceof TypeError && e.message.includes('fetch')) {
      throw new Error('无法连接到后端服务器 (localhost:8000)，请确认后端已启动');
    }
    throw e;
  }
  clearTimeout(timeout);

  if (!res.ok) {
    let detail = `HTTP ${res.status}`;
    try {
      const body = await res.json();
      if (body.detail) detail = body.detail;
      if (body.message) detail = body.message;
    } catch {
      const text = await res.text().catch(() => '');
      if (text) detail = `${detail} — ${text.slice(0, 200)}`;
    }
    throw new Error(detail);
  }

  return res.json();
}
