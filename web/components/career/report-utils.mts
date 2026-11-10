import type { Graph, Plan } from "../../lib/career-types";

const SECTION_LABELS = new Set(["规划建议", "目标岗位", "能力差距", "行动路线", "下一行动", "学习资源", "当前约束", "注意事项", "待确认信息"]);

export type ReportSection = { id: string; title: string; lines: string[] };

/** Read existing headings only; unknown prose stays available as original text. */
export function splitReportSections(text: string): ReportSection[] {
  const result: ReportSection[] = [];
  let current: ReportSection = { id: "report-text-1", title: "规划正文", lines: [] };
  for (const line of text.split("\n")) {
    const heading = /^#{1,3}\s+(.+?)\s*$/.exec(line)?.[1] || (SECTION_LABELS.has(line.trim()) ? line.trim() : undefined);
    if (heading) {
      if (current.lines.some(value => value.trim())) result.push(current);
      current = { id: `report-text-${result.length + 1}`, title: heading, lines: [] };
    } else current.lines.push(line);
  }
  if (current.lines.some(value => value.trim())) result.push(current);
  return result;
}

export function displayCharacterCount(text: string): number {
  return Array.from(text).length;
}

export function edgeEndpoints(edge: Graph["edges"][number]): [string, string] {
  return [String(edge.source || edge.from || ""), String(edge.target || edge.to || "")];
}

/** A projection only: every displayed node and edge remains an original object. */
export function coreGraph(graph: Graph, limit = 10): Graph {
  const preferred: [string[], number][] = [
    [["plan"], 1], [["role_suggestion"], 2], [["gap_suggestion"], 2],
    [["milestone"], 3], [["action_suggestion"], 2],
  ];
  const nodes: Graph["nodes"] = [];
  for (const [kinds, count] of preferred) {
    nodes.push(...graph.nodes.filter(node => kinds.includes(node.kind || node.type || "")).slice(0, count));
  }
  if (!nodes.length) nodes.push(...graph.nodes.slice(0, limit));
  const retained = nodes.slice(0, limit);
  const ids = new Set(retained.map(node => node.id));
  return { nodes: retained, edges: graph.edges.filter(edge => edgeEndpoints(edge).every(id => ids.has(id))) };
}

function record(value: unknown): Record<string, unknown> {
  return value && typeof value === "object" && !Array.isArray(value) ? value as Record<string, unknown> : {};
}

export function planConstraints(plan: Plan): { label: string; value: string }[] {
  const rows: { label: string; value: string }[] = [];
  const constraints = plan.profile?.constraints;
  const number = new Intl.NumberFormat("zh-CN");
  if (typeof constraints?.time_budget_hours_per_week === "number") rows.push({ label: "每周时间", value: `${number.format(constraints.time_budget_hours_per_week)}小时` });
  if (typeof constraints?.financial_budget_cny === "number") rows.push({ label: "资金预算", value: `${number.format(constraints.financial_budget_cny)}元` });
  if (constraints?.city) rows.push({ label: "城市", value: constraints.city });
  if (constraints?.preferred_industries?.length) rows.push({ label: "行业偏好", value: constraints.preferred_industries.join("、") });
  if (plan.profile?.preference) rows.push({ label: "偏好", value: plan.profile.preference });
  for (const value of plan.profile?.main_constraints || []) rows.push({ label: "限制", value });
  const fields = record(record(plan.profile?.guidance).fields);
  const labels: Record<string, string> = { education: "学历", major: "专业", skills: "技能", interests: "兴趣", target_role: "目标岗位", time_budget: "时间", preference: "偏好", constraints: "限制" };
  const statuses: Record<string, string> = { missing: "未提供", unknown: "暂不确定", refused: "暂不回答", conflict: "有矛盾，待确认" };
  for (const [field, state] of Object.entries(fields)) {
    const status = String(record(state).status || "");
    if (labels[field] && statuses[status]) rows.push({ label: labels[field], value: statuses[status] });
  }
  return rows;
}
