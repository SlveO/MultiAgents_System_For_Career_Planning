import type { Task } from './career-types';

export function elapsedLabel(task: Task, now = Date.now()): string {
  const start = Date.parse(task.created_at);
  const end = ['queued', 'running'].includes(task.status) ? now : Date.parse(task.updated_at);
  if (!Number.isFinite(start) || !Number.isFinite(end)) return '耗时暂不可用';
  const seconds = Math.max(0, Math.floor((end - start) / 1000));
  const hours = Math.floor(seconds / 3600), minutes = Math.floor(seconds % 3600 / 60);
  return `提交后已过去 ${hours ? `${hours}小时` : ''}${hours || minutes ? `${minutes}分` : ''}${seconds % 60}秒`;
}

export type RecoveryAction = 'confirm' | 'query' | 'review' | 'edit' | 'new_session' | 'history' | 'sign_in' | 'service' | 'none';
export type ErrorPolicy = { message: string; action: RecoveryAction };

/** Only known codes determine displayed text; server-supplied messages are never displayed. */
export function errorPolicy(code?: string, status?: number): ErrorPolicy {
  switch (code) {
    case 'stale_state': case 'stale_question': case 'stale_version':
      return { message: '会话状态、问题或规划版本已更新。请恢复当前会话，重新检查答案或版本后再提交。', action: 'review' };
    case 'queue_full':
      return { message: '当前队列已满，本次未提交。请等待并查询状态，稍后手动提交。', action: 'query' };
    case 'user_task_active':
      return { message: '已有任务正在处理，请查询任务状态并等待完成。', action: 'query' };
    case 'file_size_limit':
      return { message: '文件必须非空且不超过 25 MiB，上传请求不得超过 26 MiB。请调整后重新选择。', action: 'edit' };
    case 'file_count_limit':
      return { message: '每个会话最多上传 3 个文件。请调整文件后再提交。', action: 'edit' };
    case 'unsupported_file':
      return { message: '文件格式暂不支持，当前不接受音频。请更换文件后再提交。', action: 'edit' };
    case 'invalid_request':
      return { message: '提交内容缺失、格式错误或超出限制。请检查表单字段后再提交。', action: 'edit' };
    case 'materials_expired': case 'materials_locked':
      return { message: '原始材料已过保留期或已完成解析。需要更换材料时，请新建会话并重新上传材料。', action: 'new_session' };
    case 'not_found':
      return { message: '记录不存在或已删除。请返回历史记录，或新建会话。', action: 'history' };
    case 'server_restarted':
      return { message: '服务重启导致任务中断；已保存内容仍在，系统不会自动重新提交。', action: 'review' };
    case 'identity_required': case 'unauthorized':
      return { message: '身份验证未通过。请登录后恢复当前会话。', action: 'sign_in' };
    case 'storage_unavailable': case 'schema_incompatible': case 'invalid_storage_path':
      return { message: '存储服务暂不可用。请检查服务状态后恢复会话，无需反复修改答案。', action: 'service' };
    case 'cleanup_failed':
      return { message: '记录访问已撤销，但文件清理失败。请检查服务并处理剩余文件。', action: 'service' };
    case 'idempotency_conflict': case 'idempotency_key_required':
      return { message: '提交标识无效或已用于不同请求。请恢复会话并检查当前状态。', action: 'review' };
    case 'not_ready': case 'plan_unavailable':
      return { message: '当前会话尚未满足此操作条件。请恢复会话并检查当前步骤。', action: 'review' };
    case 'service_timeout': case 'service_unavailable': case 'connection_failed':
      return { message: '连接中断或超时，提交结果尚未确认。请先查询当前状态；如仍未确认，可使用原标识确认。', action: 'confirm' };
  }
  if (status === 401 || status === 403) return { message: '身份验证未通过。请登录后恢复当前会话。', action: 'sign_in' };
  if (status === 429) return { message: '当前服务繁忙。请等待并查询状态，稍后手动提交。', action: 'query' };
  if (status === 502 || status === 504 || status === 0) return { message: '连接中断或网关超时，提交结果尚未确认。请查询当前状态或使用原标识确认。', action: 'confirm' };
  if (status === 409) return { message: '当前状态已变化。请恢复会话并检查当前步骤。', action: 'review' };
  return { message: '操作未完成。请恢复会话查看已保存的状态，再决定下一步。', action: 'review' };
}

export function errorMessage(code?: string, status?: number): string {
  return errorPolicy(code, status).message;
}

const TASK_FAILURE: Record<string, string> = {
  perceive: '材料解析未完成。此前保存的目标和材料仍在，请恢复会话检查材料后再决定下一步。',
  answers: '引导任务未完成。此前保存的会话内容仍在，请恢复会话并检查当前问题后再继续。',
  plans: '规划生成未完成。此前保存的材料和回答仍在，请恢复会话检查当前状态后再继续。',
  feedback: '规划详略调整未完成。此前保存的规划仍在，请恢复会话检查当前版本后再继续。',
};

export function taskError(task: Task): string {
  const code = typeof task.error === 'object' && task.error ? task.error.code : undefined;
  // A terminal task was already accepted; even a timeout here is not an unknown submission.
  if (code && ['server_restarted', 'materials_expired', 'stale_state', 'stale_question', 'stale_version', 'identity_required', 'unauthorized', 'storage_unavailable', 'schema_incompatible'].includes(code)) return errorMessage(code);
  return TASK_FAILURE[task.kind] || '任务未完成。此前保存的内容仍在，请恢复会话检查当前状态后再继续。';
}
