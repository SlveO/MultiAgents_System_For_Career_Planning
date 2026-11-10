import test from 'node:test';
import assert from 'node:assert/strict';
import { submissionRequest, validReceipt } from '../lib/pending-submission.mts';

const task = { id: 't', session_id: 's', kind: 'answers', status: 'queued', created_at: '2026-01-01', updated_at: '2026-01-01' };
test('replay retains method, key, JSON body and version across all JSON writes', () => {
  for (const path of ['sessions', 'sessions/s/answers', 'sessions/s/plans', 'sessions/s/feedback', 'sessions/s/evaluations']) {
    const item = { target: 's', path, method: 'POST', key: 'original', json: JSON.stringify({ state_version: 4, answer: '原答案', output_version: 2 }) };
    const first = submissionRequest(item), second = submissionRequest(item);
    assert.deepEqual(first, second);
    assert.equal(second.headers['Idempotency-Key'], 'original');
    assert.equal(second.body, item.json);
  }
  const deletion = { target: 's', path: 'sessions/s', method: 'DELETE', key: 'same' };
  assert.deepEqual(submissionRequest(deletion), submissionRequest(deletion));
});
test('upload replay rebuilds form with the original file and subject', async () => {
  const file = new File(['original bytes'], 'material.txt', { type: 'text/plain' });
  const item = { target: 's', path: 'sessions/s/uploads', method: 'POST', key: 'same', upload: { file, subject: 'student' } };
  const first = submissionRequest(item), second = submissionRequest(item);
  assert.notEqual(first.body, second.body);
  for (const req of [first, second]) {
    assert.equal(req.headers['Idempotency-Key'], 'same');
    assert.equal(req.body.get('subject'), 'student');
    assert.equal(req.body.get('file').name, file.name);
    assert.equal(await req.body.get('file').text(), 'original bytes');
  }
});
test('receipt must identify the submitted operation and target', () => {
  const item = { target: 's', path: 'sessions/s/answers', method: 'POST', key: 'k' };
  assert.equal(validReceipt(item, task), true);
  assert.equal(validReceipt(item, { ...task, session_id: 'other' }), false);
  assert.equal(validReceipt(item, { ...task, kind: 'plans' }), false);
  assert.equal(validReceipt(item, { ...task, status: 'unknown' }), false);
  assert.equal(validReceipt(item, null), false);
  assert.equal(validReceipt({ ...item, method: 'DELETE' }, { deleted: true }), true);
  const evaluation = { ...item, path: 'sessions/s/evaluations', json: '{"plan_id":"p","output_version":2}' };
  assert.equal(validReceipt(evaluation, { id: 'e', plan_id: 'p', output_version: 1 }), false);
});
