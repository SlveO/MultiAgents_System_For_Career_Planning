export type FrozenRequest = { target: string | null; path: string; key: string; method: 'POST' | 'DELETE'; json?: string; upload?: { file: File; subject: string } };

export function submissionRequest(item: FrozenRequest): RequestInit {
  let body: BodyInit | undefined = item.json;
  const headers: Record<string, string> = { 'Idempotency-Key': item.key };
  if (item.upload) {
    const form = new FormData();
    form.set('file', item.upload.file);
    form.set('subject', item.upload.subject);
    body = form;
  } else if (item.json !== undefined) headers['Content-Type'] = 'application/json';
  return { method: item.method, headers, body };
}

export function validReceipt(item: FrozenRequest, value: unknown): boolean {
  if (!value || typeof value !== 'object') return false;
  const data = value as Record<string, unknown>;
  if (item.method === 'DELETE') return data.deleted === true;
  if (typeof data.id !== 'string' || !data.id) return false;
  const kind = item.path.match(/\/(perceive|answers|plans|feedback)$/)?.[1];
  if (kind) return data.session_id === item.target && data.kind === kind && ['queued', 'running', 'succeeded', 'failed', 'interrupted', 'cancelled'].includes(String(data.status)) && typeof data.created_at === 'string' && typeof data.updated_at === 'string';
  if (item.upload) return typeof data.name === 'string' && typeof data.size === 'number' && data.subject === item.upload.subject;
  if (item.path.endsWith('/evaluations')) {
    const original = JSON.parse(item.json || '{}');
    return data.plan_id === original.plan_id && data.output_version === original.output_version;
  }
  return typeof data.state_version === 'number' && Array.isArray(data.uploads) && Array.isArray(data.versions);
}
