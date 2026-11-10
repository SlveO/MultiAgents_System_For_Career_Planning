import test from "node:test";
import assert from "node:assert/strict";
import { coreGraph, displayCharacterCount, edgeEndpoints, planConstraints, splitReportSections } from "../components/career/report-utils.mts";

test("report headings support saved plain labels without discarding unknown text", () => {
  const sections = splitReportSections("目标岗位\n- 数据分析师\n\n## 下一阶段\n包含0元预算和不能全职的原文。\n其他未识别说明");
  assert.equal(sections[0].title, "目标岗位");
  assert.equal(sections[1].title, "下一阶段");
  assert.match(sections[1].lines.join("\n"), /0元预算和不能全职/);
  assert.match(sections[1].lines.join("\n"), /其他未识别说明/);
  assert.equal(splitReportSections("仅有原始建议，不补造章节。")[0].title, "规划正文");
});

test("core graph keeps only recorded edges and accepts stored from/to contracts", () => {
  const graph = { nodes: [{ id: "plan", kind: "plan", label: "规划" }, { id: "role", kind: "role_suggestion", label: "方向" }, { id: "gap", kind: "gap_suggestion", label: "差距" }, { id: "knowledge", kind: "knowledge_skill", label: "岗位要求" }], edges: [{ from: "plan", to: "role", label: "建议目标" }, { from: "plan", to: "knowledge" }] };
  const original = JSON.stringify(graph);
  const projection = coreGraph(graph);
  assert.equal(projection.nodes.length, 3);
  assert.deepEqual(projection.edges, [graph.edges[0]]);
  assert.deepEqual(edgeEndpoints(graph.edges[0]), ["plan", "role"]);
  assert.equal(projection.edges.some(edge => edge.from === "role" && edge.to === "gap"), false);
  assert.equal(JSON.stringify(graph), original);
  assert.equal(coreGraph({ nodes: [], edges: [] }).nodes.length, 0);
});

test("constraint projection keeps zero budgets and refusal distinct from missing", () => {
  const rows = planConstraints({ profile: { constraints: { time_budget_hours_per_week: 0, financial_budget_cny: 0 }, main_constraints: ["不能全职"], guidance: { fields: { education: { status: "refused" }, interests: { status: "unknown" } } } } });
  assert.deepEqual(rows.slice(0, 2), [{ label: "每周时间", value: "0小时" }, { label: "资金预算", value: "0元" }]);
  assert.ok(rows.some(row => row.value === "不能全职"));
  assert.ok(rows.some(row => row.label === "学历" && row.value === "暂不回答"));
  assert.ok(rows.some(row => row.label === "兴趣" && row.value === "暂不确定"));
  assert.equal(displayCharacterCount("规划🙂"), 3);
});
