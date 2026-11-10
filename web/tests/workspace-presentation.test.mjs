import test from "node:test";
import assert from "node:assert/strict";
import { feedbackNotice, taskResultNotice, workspaceStage } from "../lib/workspace-presentation.mts";

test("successful task with unchanged compression does not claim that output became shorter", () => {
  const session = { versions: [{ version: 1, display_text: "长报告" }], feedback: [{ status: "unchanged", reason: "length_constraint", source_version: 1, output_version: { version: 1, display_text: "长报告" } }] };
  const message = taskResultNotice({ kind: "feedback", status: "succeeded" }, session);
  assert.match(message, /未改变内容/);
  assert.doesNotMatch(message, /已更新|已变短|已完成/);
});

test("feedback notice uses actual source and result lengths rather than promising a fixed brief mode", () => {
  const session = { versions: [{ version: 2, display_text: "甲".repeat(2408) }], feedback: [{ status: "adapted", source_version: 2, output_version: { version: 3, display_text: "乙".repeat(1712) } }] };
  assert.equal(feedbackNotice(session), "已更新为第 3 版，2408 → 1712 字。");
  assert.doesNotMatch(feedbackNotice(session), /简略版|500/);
});

test("suitable and missing feedback records are described without creating a new display version", () => {
  assert.match(feedbackNotice({ feedback: [{ status: "unchanged", reason: "suitable" }] }), /保留这一版/);
  assert.match(feedbackNotice({}), /读取当前规划/);
});

test("only saved business states determine the three presentation stages", () => {
  assert.equal(workspaceStage(null), 0);
  assert.equal(workspaceStage({ status: "collecting" }), 0);
  assert.equal(workspaceStage({ status: "awaiting_answer", question: { prompt: "当前问题" } }), 1);
  assert.equal(workspaceStage({ status: "ready" }), 1);
  assert.equal(workspaceStage({ status: "planned", plan: { plan_id: "saved" } }), 2);
});


test("reading view transitions describe presentation rather than compression", () => {
  const feedback = { status: "adapted", reason: "brief_view", output_version: { version: 2, display_text: "原文" } };
  assert.match(feedbackNotice({ feedback: [feedback] }), /已切换简略阅读/);
  feedback.reason = "full_view";
  assert.match(feedbackNotice({ feedback: [feedback] }), /已恢复完整阅读/);
  feedback.reason = "already_brief"; feedback.status = "unchanged";
  assert.match(feedbackNotice({ feedback: [feedback] }), /当前已是简略阅读/);
});
