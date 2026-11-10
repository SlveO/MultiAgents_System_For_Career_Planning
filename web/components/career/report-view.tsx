"use client";

import { useId, useRef, useState } from "react";
import { DownloadIcon, FileTextIcon, InfoIcon, ListIcon, MessageSquareTextIcon, NetworkIcon } from "lucide-react";
import type { FeedbackChoice, Graph, OutputVersion, Session } from "../../lib/career-types";
import { Button } from "../ui/button";
import { ButtonGroup } from "../ui/button-group";
import { Badge } from "../ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "../ui/tabs";
import { Select, SelectContent, SelectGroup, SelectItem, SelectTrigger, SelectValue } from "../ui/select";
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from "../ui/accordion";
import { Sheet, SheetContent, SheetDescription, SheetHeader, SheetTitle, SheetTrigger } from "../ui/sheet";
import { Alert, AlertDescription, AlertTitle } from "../ui/alert";
import { Empty, EmptyDescription, EmptyHeader, EmptyTitle } from "../ui/empty";
import { coreGraph, displayCharacterCount, edgeEndpoints, planConstraints, splitReportSections } from "./report-utils.mts";

type ReportViewProps = {
  session: Session;
  version: OutputVersion | undefined;
  latestVersion: OutputVersion | undefined;
  pending: boolean;
  busy: boolean;
  onVersionChange: (version: number) => void;
  onFeedback: (feedback: FeedbackChoice) => void;
  onEvaluate: () => void;
  reportTab: string;
  onReportTabChange: (tab: string) => void;
};

const CHAPTERS = [
  ["report-overview", "规划概览"], ["report-actions", "近期行动"],
  ["report-roadmap", "阶段路线"], ["report-risks", "风险与约束"],
  ["report-original", "完整正文"],
];
const PERIODS: Record<string, string> = { "30d": "30天", "90d": "90天", "180d": "180天" };
const SUBJECTS: Record<string, string> = { student: "我的经历 / 能力", job: "岗位要求", reference: "参考资料", unknown: "主体尚未确定" };
const FIELD_LABELS: Record<string, string> = { education: "学历", major: "专业", skills: "技能", interests: "兴趣", target_role: "目标岗位", time_budget: "每周时间", preference: "偏好", constraints: "限制" };

function displayDate(value: string): string {
  const date = new Date(value);
  return Number.isNaN(date.valueOf()) ? "时间未记录" : new Intl.DateTimeFormat("zh-CN", { timeZone: "Asia/Shanghai", year: "numeric", month: "long", day: "numeric", hour: "2-digit", minute: "2-digit" }).format(date);
}

export function ReportView({ session, version, latestVersion, pending, busy, onVersionChange, onFeedback, onEvaluate, reportTab, onReportTabChange }: ReportViewProps) {
  const [tocOpen, setTocOpen] = useState(false);
  const nextAnchor = useRef<string | null>(null);
  const plan = session.plan;
  if (!plan) return null;
  const currentVersion = !version || version.version === latestVersion?.version;
  const text = version?.display_text || plan.user_facing_advice || "";
  const count = displayCharacterCount(text);
  const brief = version?.reading_view?.mode === "brief" ? version.reading_view : null;
  const actions = plan.next_actions || [];
  const roadmap = [...(plan.roadmap_30_90_180 || [])].sort((left, right) => ["30d", "90d", "180d"].indexOf(left.period) - ["30d", "90d", "180d"].indexOf(right.period));
  const constraints = planConstraints(plan);
  const chapterList = brief ? [["report-overview", "简要正文"], ["report-actions", "全部行动"], ["report-roadmap", "完整路线"], ["report-risks", "风险与约束"], ["report-questions", "待确认信息"], ["report-original", "完整正文"]] : CHAPTERS;
  const chapters = <nav className="report-chapter-nav" aria-label="报告章节">{chapterList.map(([id, label]) => <a href={`#${id}`} key={id} onClick={() => { if (tocOpen) { nextAnchor.current = id; setTocOpen(false); } }}>{label}</a>)}</nav>;

  return <section className="report-view" aria-labelledby="report-title">
    <header className="report-heading">
      <div><h1 id="report-title" tabIndex={-1}>{plan.target_roles?.[0] || "下一步行动规划"}</h1><p className="report-heading-meta"><span>第 {version?.version || 1} 版</span><span>{brief ? `简要正文 ${displayCharacterCount(brief.excerpt).toLocaleString("zh-CN")} 字符` : `正文 ${count.toLocaleString("zh-CN")} 字符`}</span>{brief && <span>完整正文 {count.toLocaleString("zh-CN")} 字符</span>}<time dateTime={version?.created_at || session.updated_at} title={displayDate(version?.created_at || session.updated_at)} aria-label={displayDate(version?.created_at || session.updated_at)}>{new Intl.DateTimeFormat("zh-CN", { timeZone: "Asia/Shanghai", month: "long", day: "numeric" }).format(new Date(version?.created_at || session.updated_at))}</time></p></div>
    </header>

    <div className="report-actions">
      <ButtonGroup aria-label="调整当前规划的阅读详略" className="report-feedback">
        <Button variant="outline" disabled={pending || !currentVersion || !version} onClick={() => onFeedback("过于详细")}>更简要</Button>
        <Button variant="outline" disabled={pending || !currentVersion || !version} onClick={() => onFeedback("合适")}>这样刚好</Button>
        <Button variant="outline" disabled={pending || !currentVersion || !version} onClick={() => onFeedback("过短")}>更详细</Button>
      </ButtonGroup>
      <div className="report-version-tools">
        <Select value={String(version?.version || "")} onValueChange={value => onVersionChange(Number(value))}>
          <SelectTrigger aria-label="选择规划展示版本"><SelectValue placeholder="选择版本" /></SelectTrigger>
          <SelectContent><SelectGroup>{session.versions.map(item => <SelectItem key={item.version} value={String(item.version)}>第 {item.version} 版{item.version === latestVersion?.version ? "（最新）" : ""}</SelectItem>)}</SelectGroup></SelectContent>
        </Select>
        <Button variant="outline" asChild><a href={`/api/sessions/${session.id}/export?format=markdown`} download><DownloadIcon data-icon="inline-start" aria-hidden="true" />完整导出最新版</a></Button>
      </div>
    </div>
    {!currentVersion && <Alert className="report-version-note"><InfoIcon aria-hidden="true" /><AlertTitle>正在阅读历史版本</AlertTitle><AlertDescription>本版可以阅读和评价。详略调整需切换至最新版；导出报告始终使用最新版。</AlertDescription></Alert>}
    {session.warning && <Alert className="report-version-note"><InfoIcon aria-hidden="true" /><AlertTitle>本次规划的信息提示</AlertTitle><AlertDescription>{session.warning}</AlertDescription></Alert>}

    <Tabs value={reportTab} onValueChange={onReportTabChange} className="report-tabs">
      <TabsList variant="line" aria-label="规划视图">
        <TabsTrigger value="plan"><FileTextIcon aria-hidden="true" />行动规划</TabsTrigger>
        <TabsTrigger value="graph"><NetworkIcon aria-hidden="true" />职业图谱</TabsTrigger>
        <TabsTrigger value="sources"><InfoIcon aria-hidden="true" />材料与来源</TabsTrigger>
      </TabsList>
      <TabsContent value="plan">
        <div className="report-mobile-toc"><Sheet open={tocOpen} onOpenChange={setTocOpen}>
          <SheetTrigger asChild><Button variant="outline"><ListIcon data-icon="inline-start" aria-hidden="true" />报告章节</Button></SheetTrigger>
          <SheetContent side="left" onCloseAutoFocus={event => { if (nextAnchor.current) { event.preventDefault(); document.getElementById(nextAnchor.current)?.focus(); nextAnchor.current = null; } }}><SheetHeader><SheetTitle>报告章节</SheetTitle><SheetDescription>跳转到你现在想读的部分。</SheetDescription></SheetHeader>{chapters}</SheetContent>
        </Sheet></div>
        <div className="report-layout">
          <aside className="report-toc"><p>本页目录</p>{chapters}</aside>
          <article className="report-article" aria-label="规划正文">
            {brief ? <BriefReport brief={brief} text={text} actions={actions} roadmap={roadmap} constraints={constraints} plan={plan} /> : <>
            <section className="report-section report-overview" aria-labelledby="report-overview"><h2 id="report-overview" tabIndex={-1}>规划概览</h2><p className="report-section-description">本次目标：{session.user_goal}</p>
              {(plan.target_roles?.length || 0) > 1 && <p className="report-alternatives">其他建议方向：{plan.target_roles!.slice(1).join("、")}</p>}{!plan.target_roles?.length && <p>本次规划未列出具体岗位，请结合正文中的建议阅读。</p>}

            </section>

            <section className="report-section" aria-labelledby="report-actions"><h2 id="report-actions" tabIndex={-1}>近期先做什么</h2>
              {actions.length ? <><ol className="report-action-list">{actions.slice(0, 3).map((action, index) => <li key={index}><p>{action}</p></li>)}</ol>{actions.length > 3 && <Accordion type="single" collapsible><AccordionItem value="more-actions"><AccordionTrigger>还有{actions.length - 3}项行动</AccordionTrigger><AccordionContent><ol start={4} className="report-list">{actions.slice(3).map((action, index) => <li key={index}>{action}</li>)}</ol></AccordionContent></AccordionItem></Accordion>}</> : <p className="report-section-description">本次未单独列出下一行动，完整正文仍可展开阅读。</p>}
              {!!plan.gap_analysis?.length && <Accordion type="single" collapsible><AccordionItem value="gaps"><AccordionTrigger>需要补足的能力（{plan.gap_analysis.length}项）</AccordionTrigger><AccordionContent><ul className="report-list">{plan.gap_analysis.map((item, index) => <li key={index}>{item}</li>)}</ul></AccordionContent></AccordionItem></Accordion>}
            </section>

            <section className="report-section" aria-labelledby="report-roadmap"><h2 id="report-roadmap" tabIndex={-1}>把方向拆成阶段</h2>
              {roadmap.length ? <div className="report-timeline">{roadmap.map((milestone, index) => <section className="report-period" key={`${milestone.period}-${index}`} aria-labelledby={`report-period-${index}`}><p className="report-period-label">{PERIODS[milestone.period] || milestone.period}</p><h3 id={`report-period-${index}`}>{milestone.objective}</h3>{!!(milestone.deliverables?.length || milestone.metrics?.length) && <Accordion type="single" collapsible><AccordionItem value="details"><AccordionTrigger>交付物与检查指标</AccordionTrigger><AccordionContent>{!!milestone.deliverables?.length && <><h4>交付物</h4><ul className="report-list">{milestone.deliverables.map((item, itemIndex) => <li key={itemIndex}>{item}</li>)}</ul></>}{!!milestone.metrics?.length && <><h4>检查指标</h4><ul className="report-list">{milestone.metrics.map((item, itemIndex) => <li key={itemIndex}>{item}</li>)}</ul></>}</AccordionContent></AccordionItem></Accordion>}</section>)}</div> : <p className="report-section-description">本次没有单独的阶段路线，请阅读完整规划。</p>}
            </section>

            <section className="report-section" aria-labelledby="report-risks"><h2 id="report-risks" tabIndex={-1}>风险与现实约束</h2>
              {!!constraints.length && <dl className="report-constraints">{constraints.map((row, index) => <div key={index}><dt>{row.label}</dt><dd>{row.value}</dd></div>)}</dl>}
              {plan.risk_flags?.length ? <ul className="report-list">{plan.risk_flags.map((risk, index) => <li key={index}>{risk}</li>)}</ul> : <p className="report-section-description">本次未单独记录风险提示，不代表实际选择没有风险。</p>}
              {!!plan.follow_up_questions?.length && <Accordion type="single" collapsible><AccordionItem value="open-questions"><AccordionTrigger>还需要确认的信息（{plan.follow_up_questions.length}项）</AccordionTrigger><AccordionContent><ul className="report-list">{plan.follow_up_questions.map((item, index) => <li key={index}>{item}</li>)}</ul></AccordionContent></AccordionItem></Accordion>}
            </section>

            <section className="report-section" aria-labelledby="report-original"><h2 id="report-original" tabIndex={-1}>完整正文</h2><p className="report-section-description">这里保留第{version?.version || 1}版的全部内容。章节展开只影响阅读，顶部详略反馈才会提交内容调整。</p>
              <OriginalText text={text} />
              {!!plan.learning_resources?.length && <Accordion type="single" collapsible><AccordionItem value="resources"><AccordionTrigger>学习资源（{plan.learning_resources.length}项）</AccordionTrigger><AccordionContent><ul className="report-list">{plan.learning_resources.map((item, index) => <li key={index}>{item}</li>)}</ul></AccordionContent></AccordionItem></Accordion>}
            </section>
            </>}
          </article>
        </div>
      </TabsContent>
      <TabsContent value="graph"><CareerGraph graph={plan.graph || plan.career_graph} /></TabsContent>
      <TabsContent value="sources"><Sources session={session} /></TabsContent>
    </Tabs>

    <footer className="report-footer"><Button variant="outline" disabled={busy} onClick={onEvaluate}><MessageSquareTextIcon data-icon="inline-start" aria-hidden="true" />评价第{version?.version || 1}版</Button><Button variant="ghost" asChild><a href={`/api/sessions/${session.id}/export?format=json`} download><DownloadIcon data-icon="inline-start" aria-hidden="true" />导出完整记录</a></Button></footer>
  </section>;
}

function BriefReport({ brief, text, actions, roadmap, constraints, plan }: { brief: NonNullable<OutputVersion["reading_view"]>; text: string; actions: string[]; roadmap: NonNullable<Session["plan"]>["roadmap_30_90_180"]; constraints: { label: string; value: string }[]; plan: NonNullable<Session["plan"]> }) {
  const milestones = roadmap || [];
  const risks = plan.risk_flags || [];
  const questions = plan.follow_up_questions || [];
  const visibleZeroBudgets = constraints.filter(row => ["每周时间", "资金预算"].includes(row.label) && /^0(?:小时|元)$/.test(row.value) && !brief.excerpt.includes(row.value));
  return <>
    <section className="report-section" aria-labelledby="report-overview"><h2 id="report-overview" tabIndex={-1}>简要正文</h2>
      <pre data-testid="brief-excerpt" className="report-raw-text">{brief.excerpt}</pre>
      {brief.notice && <p className="report-section-description" role="note">{brief.notice}</p>}
      {!!visibleZeroBudgets.length && <dl className="report-constraints">{visibleZeroBudgets.map(row => <div key={row.label}><dt>{row.label}</dt><dd>{row.value}</dd></div>)}</dl>}
    </section>
    <Accordion type="multiple" className="report-full-text">
      <AccordionItem value="actions"><AccordionTrigger id="report-actions">全部行动（{actions.length}项）</AccordionTrigger><AccordionContent>{actions.length ? <ol className="report-list">{actions.map ( (action, index) => <li key={index}>{action}</li>)}</ol> : <p>暂无单独列出的行动。</p>}{!!plan.gap_analysis?.length && <><h3>需要补足的能力</h3><ul className="report-list">{plan.gap_analysis.map((item, index) => <li key={index}>{item}</li>)}</ul></>}</AccordionContent></AccordionItem>
      <AccordionItem value="roadmap"><AccordionTrigger id="report-roadmap">完整路线（{milestones.length}个阶段）</AccordionTrigger><AccordionContent>{milestones.length ? milestones.map((milestone, index) => <section key={index}><h3>{PERIODS[milestone.period] || milestone.period}：{milestone.objective}</h3>{!!milestone.deliverables?.length && <><h4>交付物</h4><ul className="report-list">{milestone.deliverables.map((item, i) => <li key={i}>{item}</li>)}</ul></>}{!!milestone.metrics?.length && <><h4>检查指标</h4><ul className="report-list">{milestone.metrics.map((item, i) => <li key={i}>{item}</li>)}</ul></>}</section>) : <p>暂无单独列出的阶段路线。</p>}</AccordionContent></AccordionItem>
      <AccordionItem value="risks"><AccordionTrigger id="report-risks">风险与约束（{risks.length + constraints.length}项）</AccordionTrigger><AccordionContent>{!!constraints.length && <dl className="report-constraints">{constraints.map((row, index) => <div key={index}><dt>{row.label}</dt><dd>{row.value}</dd></div>)}</dl>}{risks.length ? <ul className="report-list">{risks.map((risk, index) => <li key={index}>{risk}</li>)}</ul> : <p>本次未单独记录风险提示，不代表实际选择没有风险。</p>}</AccordionContent></AccordionItem>
      <AccordionItem value="questions"><AccordionTrigger id="report-questions">待确认信息（{questions.length}项）</AccordionTrigger><AccordionContent>{questions.length ? <ul className="report-list">{questions.map((item, index) => <li key={index}>{item}</li>)}</ul> : <p>暂无单独列出的待确认信息。</p>}</AccordionContent></AccordionItem>
      <AccordionItem value="original"><AccordionTrigger id="report-original">完整正文（{displayCharacterCount(text).toLocaleString("zh-CN")}字符）</AccordionTrigger><AccordionContent><pre className="report-raw-text">{text}</pre>{!!plan.learning_resources?.length && <><h3>学习资源</h3><ul className="report-list">{plan.learning_resources.map((item, index) => <li key={index}>{item}</li>)}</ul></>}</AccordionContent></AccordionItem>
    </Accordion>
  </>;
}

function OriginalText({ text }: { text: string }) {
  const sections = splitReportSections(text);
  if (!text) return <p className="report-section-description">本版暂无可显示正文，可导出完整记录查看已保存内容。</p>;
  return <Accordion type="multiple" className="report-full-text">{sections.map(section => <AccordionItem value={section.id} key={section.id}><AccordionTrigger>{section.title}</AccordionTrigger><AccordionContent><div className="report-prose">{section.lines.map((line, index) => !line.trim() ? <div className="report-paragraph-gap" key={index} /> : <p key={index}>{line}</p>)}</div></AccordionContent></AccordionItem>)}<AccordionItem value="original"><AccordionTrigger>查看本版完整原文（{displayCharacterCount(text).toLocaleString("zh-CN")}字符）</AccordionTrigger><AccordionContent><pre className="report-raw-text">{text}</pre></AccordionContent></AccordionItem></Accordion>;
}

const KIND_LABELS: Record<string, string> = { plan: "规划", profile: "个人信息", profile_constraint: "个人约束", role_suggestion: "建议方向", gap_suggestion: "能力差距", resource_suggestion: "学习资源", action_suggestion: "下一行动", milestone: "阶段目标", knowledge_role: "岗位知识", knowledge_skill: "岗位要求", risk: "风险", open_question: "待确认", deliverable: "交付物", metric: "检查指标" };

function CareerGraph({ graph }: { graph?: Graph }) {
  const [selected, setSelected] = useState<string | null>(null);
  const [filter, setFilter] = useState("all");
  const [visibleCount, setVisibleCount] = useState(20);
  const [connectionCount, setConnectionCount] = useState(20);
  const arrowId = `graph-arrow-${useId().replace(/:/g, "")}`;
  if (!graph?.nodes?.length) return <Empty><EmptyHeader><EmptyTitle>本次规划暂无图谱</EmptyTitle><EmptyDescription>规划正文和已保存的来源仍可以阅读与导出。</EmptyDescription></EmptyHeader></Empty>;
  const core = coreGraph(graph);
  const columns = [core.nodes.filter(node => (node.kind || node.type) === "plan"), core.nodes.filter(node => ["role_suggestion", "gap_suggestion"].includes(node.kind || node.type || "")), core.nodes.filter(node => (node.kind || node.type) === "milestone"), core.nodes.filter(node => (node.kind || node.type) === "action_suggestion")];
  const positioned = new Map<string, { x: number; y: number }>();
  columns.forEach((column, columnIndex) => column.forEach((node, index) => positioned.set(node.id, { x: 16 + columnIndex * 248, y: ((600 - column.length * 104) / (column.length + 1)) * (index + 1) + index * 104 })));
  core.nodes.filter(node => !positioned.has(node.id)).forEach((node, index) => positioned.set(node.id, { x: 16 + (index % 4) * 248, y: 72 + Math.floor(index / 4) * 160 }));
  const visible = graph.nodes.filter(node => filter === "all" || (node.kind || node.type || "other") === filter);
  const kinds = [...new Set(graph.nodes.map(node => node.kind || node.type || "other"))];
  const focus = graph.nodes.find(node => node.id === selected);
  function selectNode(id: string) { setSelected(id); setConnectionCount(20); }
  const connections = focus ? graph.edges.flatMap(edge => { const [from, to] = edgeEndpoints(edge); if (from !== focus.id && to !== focus.id) return []; const other = graph.nodes.find(node => node.id === (from === focus.id ? to : from)); return other ? [{ edge, other }] : []; }) : [];
  return <div className="career-graph">
    <div className="report-section-heading"><h2>你的职业路线图</h2><p className="report-section-description">先看{core.nodes.length}个核心节点；点击节点查看已有关系。岗位要求不代表你已经掌握的能力。</p></div>
    <p className="career-graph-key">连线来自已保存的关系记录。图中没有连线的内容仍可在全部节点中查看。</p>
    <div className="career-graph-canvas" tabIndex={0} role="region" aria-label="核心关系图，可横向滚动；下方也提供节点列表">
      <svg viewBox="0 0 1000 600" role="group" aria-label="规划核心关系。每个节点可以用Tab键聚焦并按回车查看详情。"><defs><marker id={arrowId} viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0 0L10 5L0 10Z" fill="currentColor" /></marker></defs>
        {core.edges.map((edge, index) => { const [from, to] = edgeEndpoints(edge); const start = positioned.get(from); const end = positioned.get(to); if (!start || !end) return null; return <path key={index} className="career-graph-edge" d={`M${start.x + 216} ${start.y + 52} C${start.x + 244} ${start.y + 52},${end.x - 28} ${end.y + 52},${end.x} ${end.y + 52}`} fill="none" markerEnd={`url(#${arrowId})`} />; })}
        {core.nodes.map(node => { const position = positioned.get(node.id)!; const chars = Array.from(node.label); const lines = Array.from({ length: Math.min(3, Math.ceil(chars.length / 12)) }, (_, index) => chars.slice(index * 12, (index + 1) * 12).join("")); if (chars.length > 36) lines[2] = lines[2].slice(0, -1) + "…"; return <g className="career-graph-node" data-selected={selected === node.id} key={node.id} transform={`translate(${position.x} ${position.y})`} role="button" tabIndex={0} aria-label={`${KIND_LABELS[node.kind || node.type || ""] || "规划节点"}：${node.label}`} aria-pressed={selected === node.id} onClick={() => selectNode(node.id)} onKeyDown={event => { if (event.key === "Enter" || event.key === " ") { event.preventDefault(); selectNode(node.id); } }}><title>{node.label}</title><rect width="216" height="104" rx="12" /><text x="12" y="24" aria-hidden="true">{KIND_LABELS[node.kind || node.type || ""] || "规划节点"}</text><text x="12" y="49" aria-hidden="true">{lines.map((line, index) => <tspan key={index} x="12" dy={index ? 20 : 0}>{line}</tspan>)}</text></g>; })}
      </svg>
    </div>
    <Accordion type="single" collapsible><AccordionItem value="all-nodes"><AccordionTrigger>查看全部{graph.nodes.length}个节点</AccordionTrigger><AccordionContent><div className="career-node-filter"><label htmlFor="graph-kind">节点类别</label><Select value={filter} onValueChange={value => { setFilter(value); setVisibleCount(20); }}><SelectTrigger id="graph-kind"><SelectValue /></SelectTrigger><SelectContent><SelectGroup><SelectItem value="all">全部类别</SelectItem>{kinds.map(kind => <SelectItem value={kind} key={kind}>{KIND_LABELS[kind] || kind}</SelectItem>)}</SelectGroup></SelectContent></Select></div><ul className="career-node-list">{visible.slice(0, visibleCount).map(node => <li key={node.id}><Button variant={selected === node.id ? "secondary" : "outline"} aria-pressed={selected === node.id} onClick={() => selectNode(node.id)}><span>{KIND_LABELS[node.kind || node.type || ""] || "规划节点"}</span>{node.label}</Button></li>)}</ul>{visible.length > visibleCount && <Button variant="outline" onClick={() => setVisibleCount(value => value + 20)}>再显示{Math.min(20, visible.length - visibleCount)}个节点</Button>}</AccordionContent></AccordionItem></Accordion>
    {focus && <section className="career-node-detail" aria-live="polite"><h3>{focus.label}</h3><p className="report-section-description">{KIND_LABELS[focus.kind || focus.type || ""] || "规划节点"} · {connections.length}条已记录关系</p>{connections.length ? <ul className="career-node-connections">{connections.slice(0, connectionCount).map(({ edge, other }, index) => <li key={index}><span>{edge.label || edge.relation || "关联"}</span><Button variant="link" onClick={() => selectNode(other.id)}>{other.label}</Button></li>)}</ul> : <p>本节点未记录关联边。</p>}{connections.length > connectionCount && <Button variant="outline" onClick={() => setConnectionCount(value => value + 20)}>再显示{Math.min(20, connections.length - connectionCount)}条关系</Button>}<Accordion type="single" collapsible><AccordionItem value="provenance"><AccordionTrigger>查看来源引用{Array.isArray(focus.provenance_ids) ? `（${focus.provenance_ids.length}条）` : ""}</AccordionTrigger><AccordionContent><p className="report-source-meta">{Array.isArray(focus.provenance_ids) ? focus.provenance_ids.join("、") : "见完整导出记录"}</p></AccordionContent></AccordionItem></Accordion></section>}
  </div>;
}

function Sources({ session }: { session: Session }) {
  const snapshot = session.knowledge_snapshot || session.plan?.knowledge_snapshot;
  const claims = session.source_claims || [];
  return <div className="report-sources"><div className="report-section-heading"><h2>这份规划的依据</h2><p className="report-section-description">使用本次规划保存时的知识快照；后续知识库更新不会改变这份报告。</p></div><dl className="report-facts"><div><dt>知识版本</dt><dd>{snapshot?.version_id || "本次未使用知识库"}</dd></div></dl>
    <section className="report-section"><h3>材料记录</h3>{session.uploads.length ? <ul className="report-source-list">{session.uploads.map(upload => <li className="report-source-item" key={upload.id}><strong>{upload.name}</strong><p>{SUBJECTS[upload.subject] || SUBJECTS.unknown}</p><p className="report-source-meta">{upload.deleted_at ? "已提取并清理原文件" : "已上传"}；原始来源未经核验</p></li>)}</ul> : <p>本次使用你提交的文字和回答。</p>}</section>
    <section className="report-section"><h3>字段来源与确认</h3>{claims.length ? <ul className="report-source-list">{claims.map((claim, index) => <li className="report-source-item" key={String(claim.claim_id || index)}><strong>{FIELD_LABELS[String(claim.field)] || String(claim.field || "来源记录")}</strong>{claim.value !== undefined && <p>{typeof claim.value === "string" ? claim.value : JSON.stringify(claim.value)}</p>}<div className="report-source-meta"><span>{SUBJECTS[String(claim.subject)] || SUBJECTS.unknown}</span><Badge variant="outline">{({ user_confirmed: "用户确认", source_verified: "外部来源已核验", unverified: "未经核验" } as Record<string, string>)[String(claim.verification)] || "未经核验"}</Badge></div></li>)}</ul> : <p>本次暂无字段级来源记录，完整信息可通过导出查看。</p>}</section>
    <Alert><InfoIcon aria-hidden="true" /><AlertTitle>确认与核验分别记录</AlertTitle><AlertDescription>用户确认表示你认可这一描述，不等同于对外部来源的独立核验。规划中的模型建议也不属于已经核实的事实。</AlertDescription></Alert>
  </div>;
}
