import test from 'node:test';
import assert from 'node:assert/strict';
import { elapsedLabel, errorMessage, taskError } from '../lib/task-status.mts';

const task = { id: 'abc', session_id: 's', kind: 'plans', status: 'running', created_at: '2026-01-01T00:00:00Z', updated_at: '2026-01-01T00:00:05Z' };
test('elapsed uses server creation and freezes at terminal update', () => {
  assert.equal(elapsedLabel(task, Date.parse('2026-01-01T01:02:03Z')), '提交后已过去 1小时2分3秒');
  for (const status of ['failed', 'interrupted', 'succeeded', 'cancelled']) assert.equal(elapsedLabel({ ...task, status }, Date.parse('2027-01-01')), '提交后已过去 5秒');
});
test('recovery guidance distinguishes errors without exposing server messages', () => {
  for (const [code, phrase] of Object.entries({ stale_state: '检查答案', stale_question: '检查答案', queue_full: '手动提交', materials_expired: '重新上传', server_restarted: '不会自动', connection_failed: '尚未确认', service_timeout: '尚未确认', service_unavailable: '尚未确认' })) assert.match(errorMessage(code), new RegExp(phrase));
  assert.doesNotMatch(taskError({ ...task, error: { code: 'unknown', message: '/private/path secret' } }), /private|secret/);
});

test('error policy classifies safe recovery actions and known limits', async () => {
  const { errorPolicy } = await import('../lib/task-status.mts');
  for (const [code, action] of Object.entries({ stale_version: 'review', queue_full: 'query', user_task_active: 'query', file_size_limit: 'edit', file_count_limit: 'edit', unsupported_file: 'edit', invalid_request: 'edit', materials_expired: 'new_session', not_found: 'history', unauthorized: 'sign_in', storage_unavailable: 'service', connection_failed: 'confirm', cleanup_failed: 'service' })) {
    assert.equal(errorPolicy(code).action, action, code);
  }
  assert.match(errorPolicy('file_size_limit').message, /25 MiB/);
  assert.match(errorPolicy('file_count_limit').message, /3 个/);
  assert.match(errorPolicy('cleanup_failed').message, /访问已撤销.*清理失败/);
  assert.equal(errorPolicy(undefined, 502).action, 'confirm');
  assert.equal(errorPolicy(undefined, 504).action, 'confirm');
  assert.equal(errorPolicy(undefined, 429).action, 'query');
  assert.equal(errorPolicy(undefined, 409).action, 'review');
  assert.doesNotMatch(errorPolicy('/private/secret').message, /private|secret/);
});

test('accepted task failures describe their stage without retrying an unknown submission', () => {
  for (const [kind, stage] of Object.entries({ perceive: '材料解析', answers: '引导', plans: '规划生成', feedback: '详略调整' })) {
    const message = taskError({ ...task, kind, status: 'failed', error: { code: 'service_timeout', message: '/private/secret' } });
    assert.match(message, new RegExp(stage));
    assert.match(message, /保存/);
    assert.doesNotMatch(message, /尚未确认|原标识|private|secret/);
  }
});
