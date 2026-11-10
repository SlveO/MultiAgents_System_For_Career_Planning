"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { ArrowUpRightIcon, CheckIcon, FileTextIcon, HistoryIcon, PlusIcon, Trash2Icon, UploadIcon, XIcon, ChevronRightIcon } from "lucide-react";
import { Button } from "../components/ui/button";
import { Badge } from "../components/ui/badge";
import { Alert, AlertDescription } from "../components/ui/alert";
import { Field, FieldDescription, FieldError, FieldGroup, FieldLabel, FieldSet, FieldLegend } from "../components/ui/field";
import { Textarea } from "../components/ui/textarea";
import { ToggleGroup, ToggleGroupItem } from "../components/ui/toggle-group";
import { NativeSelect, NativeSelectOption } from "../components/ui/native-select";
import { InputGroup, InputGroupTextarea } from "../components/ui/input-group";
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from "../components/ui/accordion";
import { Sheet, SheetClose, SheetContent, SheetDescription, SheetHeader, SheetTitle, SheetTrigger } from "../components/ui/sheet";
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle } from "../components/ui/dialog";
import { AlertDialog, AlertDialogAction, AlertDialogCancel, AlertDialogContent, AlertDialogDescription, AlertDialogFooter, AlertDialogHeader, AlertDialogTitle } from "../components/ui/alert-dialog";
import { Attachment, AttachmentContent, AttachmentDescription, AttachmentMedia, AttachmentTitle } from "../components/ui/attachment";
import { Empty, EmptyDescription, EmptyHeader, EmptyTitle } from "../components/ui/empty";
import { Skeleton } from "../components/ui/skeleton";
import { Spinner } from "../components/ui/spinner";
import { ReportView } from "../components/career/report-view";
import { elapsedLabel, errorMessage, taskError } from "../lib/task-status.mts";
import { submissionRequest, validReceipt } from "../lib/pending-submission.mts";
import type { FrozenRequest } from "../lib/pending-submission.mts";
import { taskResultNotice, workspaceStage } from "../lib/workspace-presentation.mts";
import type { OutputVersion, Session, Task } from "../lib/career-types";

const SUPPORTED = ".txt,.md,.csv,.tsv,.pdf,.docx,.xlsx,.png,.jpg,.jpeg,.webp,.mp4,.avi,.mov,.mkv";
const SUBJECTS: Record<string, string> = { unknown: "尚不确定", student: "我的经历 / 能力", job: "岗位要求", reference: "参考资料" };
const STATUS: Record<string, string> = { collecting: "准备材料", awaiting_answer: "等待你的回答", ready: "可以生成规划", planned: "已生成规划", queued: "正在排队", running: "正在处理", succeeded: "已完成", failed: "处理失败", interrupted: "任务已中断", cancelled: "任务已取消" };
const KINDS: Record<string, string> = { perceive: "理解材料", answers: "梳理下一步问题", plans: "生成职业规划", feedback: "调整规划详略" };

class ApiError extends Error {
  constructor(message: string, readonly status: number, readonly code?: string) { super(message); }
}
async function api<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try { response = await fetch(`/api/${path}`, { ...init, signal: init?.signal || AbortSignal.timeout(30000), credentials: "same-origin", cache: "no-store" }); }
  catch { throw new ApiError(errorMessage("connection_failed"), 0, "connection_failed"); }
  const data: unknown = await response.json().catch(() => null);
  const failure = data as { error?: { message?: string; code?: string } } | null;
  if (!response.ok) {
    if (!data || typeof data !== "object") throw new ApiError("未能读取有效回执，提交结果尚未确认。", 0, "invalid_receipt");
    throw new ApiError(errorMessage(failure?.error?.code, response.status), response.status, failure?.error?.code);
  }
  return data as T;
}
function jsonRequest(body: unknown, key: string): RequestInit {
  return { method: "POST", headers: { "Content-Type": "application/json", "Idempotency-Key": key }, body: JSON.stringify(body) };
}
type PendingSubmission = FrozenRequest & { label: string; accept: (value: Session & Task) => void; sync?: (value: Session & Task) => Promise<void> };
function dateLabel(value: string) { return new Date(value).toLocaleString("zh-CN", { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" }); }
function liveTask(task: Task | null) { return !!task && ["queued", "running"].includes(task.status); }

export function CareerWorkspace({ displayName, signOutPath, localDemo = false }: { displayName: string; signOutPath: string; localDemo?: boolean }) {
  const [sessions, setSessions] = useState<Session[]>([]);
  const [session, setSession] = useState<Session | null>(null);
  const [task, setTask] = useState<Task | null>(null);
  const [now, setNow] = useState(() => Date.now());
  const [pollEpoch, setPollEpoch] = useState(0);
  const [loading, setLoading] = useState(true);
  const [busy, setBusy] = useState(false);
  const [busyLabel, setBusyLabel] = useState("");
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const [submission, setSubmission] = useState<PendingSubmission | null>(null);
  const submissionRef = useRef<PendingSubmission | null>(null);
  const [retry, setRetry] = useState<null | (() => void)>(null);
  const [history, setHistory] = useState(false);
  const [deleteDialog, setDeleteDialog] = useState(false);
  const [materialsOpen, setMaterialsOpen] = useState(false);
  const [goalError, setGoalError] = useState("");
  const [answerError, setAnswerError] = useState("");
  const [pendingNavigation, setPendingNavigation] = useState<null | { kind: "new" } | { kind: "open"; id: string }>(null);
  const [reportTab, setReportTab] = useState("plan");
  const [selectedVersion, setSelectedVersion] = useState<number | null>(null);
  const [goal, setGoal] = useState("");
  const [background, setBackground] = useState("");
  const [textSubject, setTextSubject] = useState("unknown");
  const [detail, setDetail] = useState("standard");
  const [answer, setAnswer] = useState("");
  const [subject, setSubject] = useState("unknown");
  const [file, setFile] = useState<File | null>(null);
  const [evaluation, setEvaluation] = useState(false);
  const [evaluationVersion, setEvaluationVersion] = useState<number | null>(null);
  const [scores, setScores] = useState<Record<string, string>>({});
  const [comment, setComment] = useState("");
  const currentId = useRef<string | null>(null);
  const navigationEpoch = useRef(0);
  const revokedIds = useRef(new Set<string>());
  const [confirmedTarget, setConfirmedTarget] = useState<string | null>(null);
  const actionLock = useRef(false);
  const fileInput = useRef<HTMLInputElement>(null);
  const stopped = useRef(false);
  const headingRef = useRef<HTMLHeadingElement>(null);
  const goalRef = useRef<HTMLTextAreaElement>(null);
  const answerRef = useRef<HTMLTextAreaElement>(null);
  const evaluationReturnRef = useRef<HTMLElement | null>(null);
  const deleteReturnRef = useRef<HTMLButtonElement>(null);

  const loadList = useCallback(async () => {
    const result = await api<{ sessions: Session[] }>("sessions");
    const visible = result.sessions.filter(item => !revokedIds.current.has(item.id));
    if (!stopped.current) setSessions(visible);
    return visible;
  }, []);
  const loadSession = useCallback(async (id: string) => {
    const value = await api<Session>(`sessions/${id}`);
    if (!stopped.current && currentId.current === id) {
      setSession(value);
      setTask(value.active_task || value.last_task || null);
    }
    return value;
  }, []);
  const openSession = useCallback(async (id: string) => {
    if (revokedIds.current.has(id)) return;
    navigationEpoch.current += 1;
    currentId.current = id;
    sessionStorage.setItem("career-session", id);
    setSession(null); setTask(null); setHistory(false); setError(""); setNotice(""); setRetry(null); if (!submissionRef.current) { setAnswer(""); setFile(null); setScores({}); setComment(""); } setSelectedVersion(null); setEvaluation(false); setLoading(true);
    try {
      const restored = await loadSession(id);
      if (currentId.current === id) {
        const latest = restored.versions?.at(-1);
        const event = { event: "restore", ...(restored.plan && latest ? { plan_id: restored.plan.plan_id, output_version: latest.version } : {}) };
        void api(`sessions/${id}/events`, jsonRequest(event, crypto.randomUUID())).catch(() => {
          if (currentId.current === id) setNotice("会话已恢复；恢复事件暂未记录。");
        });
      }
    }
    catch (err) { if (currentId.current === id) setError(err instanceof Error ? err.message : "无法恢复会话。"); }
    finally { if (currentId.current === id) setLoading(false); }
  }, [loadSession]);
  useEffect(() => {
    stopped.current = false;
    void (async () => {
      try {
        const list = await loadList();
        const last = sessionStorage.getItem("career-session");
        if (last && list.some(item => item.id === last)) await openSession(last);
      } catch (err) { setError(err instanceof Error ? err.message : "无法读取历史记录。"); }
      finally { if (!stopped.current) setLoading(false); }
    })();
    return () => { stopped.current = true; };
  }, [loadList, openSession]);

  useEffect(() => {
    if (!task || !liveTask(task)) return;
    setNow(Date.now());
    const timer = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(timer);
  }, [task?.id, task?.status]);

  useEffect(() => {
    if (!task || !liveTask(task)) return;
    const watchedTask = task;
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout>;
    async function poll() {
      try {
        const next = await api<Task>(`tasks/${watchedTask.id}`);
        if (cancelled) return;
        setTask(next);
        if (liveTask(next)) timer = setTimeout(poll, 2000);
        else {
          const [, restored] = await Promise.all([loadList(), currentId.current === next.session_id ? loadSession(next.session_id) : Promise.resolve(null)]);
          if (!cancelled) {
            if (next.status === "succeeded" && !submissionRef.current) {
              setNotice(taskResultNotice(next, restored));
              setAnswer("");
              if (["plans", "feedback"].includes(next.kind)) {
                setSelectedVersion(null); setReportTab("plan");
                const url = new URL(window.location.href); url.searchParams.delete("version"); url.searchParams.set("tab", "plan"); url.hash = ""; window.history.replaceState(null, "", url);
                window.scrollTo({ top: 0, behavior: "instant" });
              }
            } else if (next.status !== "succeeded") setError(taskError(next));
          }
        }
      } catch (err) {
        if (!cancelled) setError(`${err instanceof Error ? err.message : "无法查询任务。"} 页面已停止轮询，点击刷新任务状态重新查询。`);
      }
    }
    timer = setTimeout(poll, 1200);
    return () => { cancelled = true; clearTimeout(timer); };
  // Polling is keyed by task identity; status updates do not restart a poll.
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [task?.id, pollEpoch, loadList, loadSession]);

  async function run(label: string, action: () => Promise<void>) {
    if (actionLock.current) return;
    actionLock.current = true; setBusy(true); setBusyLabel(label); setError(""); setNotice("");
    try { await action(); } catch (err) { setError(err instanceof Error ? err.message : "操作未完成。"); }
    finally { actionLock.current = false; setBusy(false); setBusyLabel(""); }
  }
  function revokeLocal(id: string) {
    revokedIds.current.add(id);
    setSessions(items => items.filter(item => item.id !== id));
    if (currentId.current === id) { currentId.current = null; sessionStorage.removeItem("career-session"); setSession(null); setTask(null); setFile(null); setGoal(""); setBackground(""); }
  }
  async function confirmSubmission(item: PendingSubmission) {
    if (actionLock.current || submissionRef.current !== item) return;
    actionLock.current = true; setBusy(true); setBusyLabel(item.label); setError(""); setNotice("");
    try {
      const value = await api<unknown>(item.path, submissionRequest(item));
      if (!validReceipt(item, value)) throw new ApiError("未能读取有效回执，提交结果尚未确认。", 0, "invalid_receipt");
      submissionRef.current = null; setSubmission(null); item.accept(value as Session & Task);
      if (item.method !== "DELETE") { const target = item.target || (value as { id: string }).id; if (currentId.current !== target) setConfirmedTarget(target); }
      try { await item.sync?.(value as Session & Task); } catch { setError("操作已确认，但页面同步失败。请查询当前状态或刷新记录。"); }
    } catch (err) {
      if (err instanceof ApiError && item.method === "DELETE" && err.code === "cleanup_failed" && item.target) {
        submissionRef.current = null; setSubmission(null); revokeLocal(item.target); setError("记录访问已撤销，但文件清理失败；请检查服务端存储，不能确认原文件已清理。");
      } else if (err instanceof ApiError && err.status > 0 && err.status < 500 && err.status !== 408) {
        submissionRef.current = null; setSubmission(null); setError(err.message);
      } else setError(err instanceof Error ? err.message : "提交结果尚未确认。");
    } finally { actionLock.current = false; setBusy(false); setBusyLabel(""); }
  }
  function submit(item: PendingSubmission) {
    if (submissionRef.current || actionLock.current) return;
    submissionRef.current = item; setSubmission(item); void confirmSubmission(item);
  }
  async function querySubmission() {
    const item = submissionRef.current;
    try {
      if (item?.target) { const value = await api<Session>(`sessions/${item.target}`); if (currentId.current === item.target) { setSession(value); setTask(value.active_task || value.last_task || null); } }
      else await loadList();
      setNotice("已查询当前状态；查询结果不能确认这次提交，请使用原标识确认。");
    } catch { setError("无法查询当前状态；原提交仍可用原标识确认。"); }
  }
  async function refreshTaskStatus() {
    if (!currentId.current) return;
    try {
      const restored = await loadSession(currentId.current);
      if (!submissionRef.current && session?.question?.id !== restored.question?.id) setAnswer("");
      const next = restored.active_task || restored.last_task;
      setError(""); setNotice(next ? "任务状态已刷新。" : "当前会话没有已记录的任务；提交结果仍未确认，可使用原标识重试。");
      if (next && liveTask(next)) setPollEpoch(value => value + 1);
    } catch { setError("无法查询任务状态，请检查连接后再试。"); }
  }
  const pending = busy || Boolean(submission) || liveTask(task);
  const latestVersion = session?.versions?.at(-1);
  const version: OutputVersion | undefined = session?.versions?.find(item => item.version === selectedVersion) || latestVersion;

  function newSession() {
    if (busy) return;
    navigationEpoch.current += 1;
    currentId.current = null; sessionStorage.removeItem("career-session");
    setSession(null); setTask(null); if (!submissionRef.current) { setGoal(""); setBackground(""); setTextSubject("unknown"); setFile(null); setAnswer(""); } setError(""); setNotice(""); setRetry(null); setHistory(false); setLoading(false); setSelectedVersion(null); setEvaluation(false);
  }
  function create() {
    if (!goal.trim()) { setGoalError("先写下你希望解决的职业问题。"); goalRef.current?.focus(); return; }
    setGoalError("");
    const body = { user_goal: goal.trim(), text_input: background.trim(), text_subject: textSubject, output_preferences: { detail, knowledge_level: "unspecified", presentation: "bullets", user_reported: true, source: "user_input" } };
    const origin = currentId.current; const epoch = navigationEpoch.current;
    submit({ label: "保存目标", target: null, path: "sessions", key: crypto.randomUUID(), method: "POST", json: JSON.stringify(body),
      accept: (value: Session) => { if (currentId.current === origin && navigationEpoch.current === epoch) { currentId.current = value.id; sessionStorage.setItem("career-session", value.id); setSession(value); setTask(value.active_task || value.last_task || null); setNotice("目标已保存。可以添加材料，或直接开始梳理。"); } else setNotice("目标已保存，可从记录中查看原记录。"); }, sync: async () => { await loadList(); } });
  }
  function chooseFile(value?: File) {
    if (!value) return;
    if (!SUPPORTED.split(",").includes(`.${value.name.split(".").at(-1)?.toLowerCase()}`)) { setError("暂不支持此格式，请选择文档、图片或视频。音频暂未开放。"); return; }
    if (value.size > 25 * 1024 * 1024) { setError("单个文件不能超过 25 MiB。"); return; }
    setError(""); setFile(value);
  }
  function upload() {
    if (!session || !file) return;
    const id = session.id;
    submit({ label: "上传材料", target: id, path: `sessions/${id}/uploads`, key: crypto.randomUUID(), method: "POST", upload: { file, subject },
      accept: () => { if (currentId.current === id) { setFile(null); if (fileInput.current) fileInput.current.value = ""; setNotice("材料已保存，等待开始理解。"); } else setNotice("材料已保存，可从记录中查看原记录。"); }, sync: async () => { if (currentId.current === id) await loadSession(id); } });
  }
  function submitTask(kind: string, extra: Record<string, unknown> = {}) {
    if (!session) return;
    if (kind === "answers" && !String(extra.answer || "").trim()) { setAnswerError("写下你的回答，或选择暂不确定。"); answerRef.current?.focus(); return; }
    setAnswerError(""); const id = session.id;
    submit({ label: KINDS[kind], target: id, path: `sessions/${id}/${kind}`, key: crypto.randomUUID(), method: "POST", json: JSON.stringify({ state_version: session.state_version, ...extra }),
      accept: (next: Task) => { if (currentId.current === id) { setTask(next); if (next.status !== "succeeded" && !liveTask(next)) setError(taskError(next)); } else setNotice("任务已受理，可从记录中查看原记录。"); },
      sync: async (next: Task) => { if (!liveTask(next) && currentId.current === id) { const restored = await loadSession(id); if (next.status === "succeeded") { setNotice(taskResultNotice(next, restored)); setAnswer(""); } } } });
  }
  function remove() {
    if (!session) return;
    const id = session.id; setDeleteDialog(false);
    submit({ label: "删除会话", target: id, path: `sessions/${id}`, key: crypto.randomUUID(), method: "DELETE", accept: () => { revokeLocal(id); setNotice("会话已删除，无法恢复。"); }, sync: async () => { await loadList(); } });
  }
  function saveEvaluation() {
    const target = session?.versions.find(item => item.version === evaluationVersion);
    if (!session?.plan || !target) return;
    const id = session.id;
    const body = { plan_id: session.plan.plan_id, output_version: target.version, scores: Object.fromEntries(Object.entries(scores).filter(([, value]) => value).map(([key, value]) => [key, Number(value)])), comment };
    if (!Object.keys(body.scores).length && !comment.trim()) { setError("请选择至少一项评分，或写下你的意见。"); document.getElementById("evaluation-comment")?.focus(); return; }
    submit({ label: "保存评价", target: id, path: `sessions/${id}/evaluations`, key: crypto.randomUUID(), method: "POST", json: JSON.stringify(body), accept: () => { if (currentId.current === id) { setEvaluation(false); setScores({}); setComment(""); } setNotice(`已保存对第 ${target.version} 版的评价，可从记录中查看原记录。`); } });
  }
  function switchReport(value: string) {
    const url = new URL(window.location.href); url.searchParams.set("tab", value); window.history.replaceState(null, "", url);
    setReportTab(value);
    if (value === "graph" && session?.plan && version) {
      const key = crypto.randomUUID();
      void api(`sessions/${session.id}/events`, jsonRequest({ event: "graph_view", plan_id: session.plan.plan_id, output_version: version.version }, key)).catch(() => setNotice("图谱已显示；查看事件暂未记录。"));
    }
  }

  const stage = workspaceStage(session);
  const draft = (!session && Boolean(goal.trim() || background.trim())) || Boolean(file) || Boolean(session?.question && answer.trim());

  useEffect(() => {
    if (!draft && !submission) return;
    const warn = (event: BeforeUnloadEvent) => { event.preventDefault(); event.returnValue = ""; };
    window.addEventListener("beforeunload", warn);
    return () => window.removeEventListener("beforeunload", warn);
  }, [draft, submission]);

  useEffect(() => {
    if (!loading && session && stage > 0) {
      const target = stage === 2 ? document.getElementById("report-title") : headingRef.current;
      target?.focus({ preventScroll: true });
      if (stage === 1) target?.scrollIntoView({ block: "start", behavior: "instant" });
    }
  }, [session?.question?.id, session?.status, loading, stage]);

  useEffect(() => {
    if (!session?.plan) return;
    const restoreReadingState = () => {
      const url = new URL(window.location.href);
      const tab = url.searchParams.get("tab");
      setReportTab(tab && ["plan", "graph", "sources"].includes(tab) ? tab : "plan");
      const requestedVersion = Number(url.searchParams.get("version"));
      setSelectedVersion(session.versions.some(item => item.version === requestedVersion) ? requestedVersion : null);
    };
    restoreReadingState();
    window.addEventListener("popstate", restoreReadingState);
    return () => window.removeEventListener("popstate", restoreReadingState);
  }, [session?.id, Boolean(session?.plan)]);

  function navigate(destination: { kind: "new" } | { kind: "open"; id: string }) {
    if (draft) { setPendingNavigation(destination); return; }
    performNavigation(destination);
  }
  function performNavigation(destination: { kind: "new" } | { kind: "open"; id: string }) {
    setPendingNavigation(null); setGoalError(""); setAnswerError("");
    const url = new URL(window.location.href); url.searchParams.delete("tab"); url.searchParams.delete("version"); url.hash = ""; window.history.replaceState(null, "", url);
    setReportTab("plan"); setSelectedVersion(null);
    if (destination.kind === "new") newSession();
    else void openSession(destination.id);
    window.scrollTo({ top: 0, behavior: "instant" });
  }
  function changeVersion(value: number) {
    setSelectedVersion(value); setEvaluation(false); setScores({}); setComment("");
    const url = new URL(window.location.href); url.searchParams.set("version", String(value)); window.history.replaceState(null, "", url);
  }
  function evaluate() {
    if (!version) return;
    evaluationReturnRef.current = document.activeElement as HTMLElement | null;
    setEvaluationVersion(version.version);
    setError(""); setRetry(null);
    setScores({}); setComment(""); setEvaluation(true);
  }

  return <div className="app-shell">
    <a className="skip-link" href="#workspace-main">跳到主要内容</a>
    <header className="topbar">
      <a href="/" className="brand" aria-label="行向首页"><span className="brand-mark" aria-hidden="true"><ArrowUpRightIcon /></span><span>行向</span><span className="brand-sub">职业规划</span></a>
      <div className="topbar-actions">
        <Badge variant="secondary" title={displayName}>{localDemo ? "本地 Demo · 单用户" : "私人空间"}</Badge>
        <Sheet open={history} onOpenChange={setHistory}>
          <SheetTrigger asChild><Button variant="ghost"><HistoryIcon aria-hidden="true" data-icon="inline-start" />我的记录</Button></SheetTrigger>
          <SheetContent className="history-sheet">
            <SheetHeader><SheetTitle>我的规划记录</SheetTitle><SheetDescription>从上次保存的状态继续。记录保留30天，删除后无法恢复。</SheetDescription></SheetHeader>
            <div className="history-list">
              <Button onClick={() => navigate({ kind: "new" })} disabled={busy}><PlusIcon aria-hidden="true" data-icon="inline-start" />新建规划</Button>
              {!sessions.length ? <Empty><EmptyHeader><EmptyTitle>还没有规划记录</EmptyTitle><EmptyDescription>先从一个想解决的职业问题开始。</EmptyDescription></EmptyHeader></Empty> : sessions.map(item => <Button className="history-record" variant="ghost" key={item.id} disabled={busy} aria-current={item.id === session?.id ? "true" : undefined} onClick={() => navigate({ kind: "open", id: item.id })}><span className="history-record-content"><strong>{item.title || item.user_goal}</strong><span>{dateLabel(item.updated_at)} · {STATUS[item.status] || "进行中"}</span></span><ChevronRightIcon aria-hidden="true" data-icon="inline-end" /></Button>)}
              <Button variant="outline" disabled={busy} onClick={() => void run("刷新记录", async () => { await loadList(); })}>刷新记录</Button>
            </div>
          </SheetContent>
        </Sheet>
        {session && <Button variant="outline" disabled={busy} onClick={() => navigate({ kind: "new" })}><PlusIcon aria-hidden="true" data-icon="inline-start" />新建<span className="desktop-label">规划</span></Button>}
        {!localDemo && <Button variant="ghost" asChild><a href={signOutPath} target="_top">退出</a></Button>}
      </div>
    </header>

    <main id="workspace-main" className="workspace-main" data-stage={session?.plan ? "report" : "flow"}>
      <nav aria-label="规划步骤" className="stage-nav"><ol>{["准备材料", "补充信息", "查看规划"].map((label, index) => <li key={label} className="stage-step" data-state={stage === index ? "current" : stage > index ? "complete" : "upcoming"} aria-current={stage === index ? "step" : undefined}><span className="stage-index" aria-hidden="true">{stage > index ? <CheckIcon /> : index + 1}</span><span>{label}</span></li>)}</ol></nav>

      {!evaluation && !error && session?.plan && notice === "规划已生成，可以查看结论与行动路线。" && <p className="sr-only" role="status" aria-live="polite">{notice}</p>}
      {!evaluation && (error || (notice && (!session?.plan || notice !== "规划已生成，可以查看结论与行动路线。"))) && <Alert className="workspace-notice" variant={error ? "destructive" : "default"} role={error ? "alert" : "status"} aria-live={error ? "assertive" : "polite"}>
        <AlertDescription><p>{error || notice}</p><div className="notice-actions" hidden={!error || (!retry && !currentId.current)}>{error && retry && <Button size="sm" variant="outline" onClick={retry} disabled={busy}>用原标识重试</Button>}{error && currentId.current && <><Button size="sm" variant="outline" disabled={busy} onClick={() => void refreshTaskStatus()}>刷新任务状态</Button><Button size="sm" variant="outline" disabled={busy} onClick={() => void openSession(currentId.current!)}>恢复会话</Button></>}</div></AlertDescription>
        <Button className="notice-close" variant="ghost" size="icon" aria-label="关闭提示" onClick={() => { setError(""); setNotice(""); }}><XIcon aria-hidden="true" /></Button>
      </Alert>}

      {confirmedTarget && !submission && <Button variant="outline" disabled={busy} onClick={() => { const id = confirmedTarget; setConfirmedTarget(null); void openSession(id); }}>查看原记录</Button>}
      {submission && !evaluation && <Alert className="workspace-notice" role="status"><AlertDescription><p>{submission.label}的提交结果尚未确认。查询不会清除此入口。</p><div className="notice-actions"><Button size="sm" variant="outline" disabled={busy} onClick={() => void querySubmission()}>查询当前状态</Button><Button size="sm" variant="outline" disabled={busy} onClick={() => void confirmSubmission(submission)}>用原标识确认</Button>{submission.target && currentId.current !== submission.target && <Button size="sm" variant="outline" disabled={busy} onClick={() => void openSession(submission.target!)}>查看原记录</Button>}</div></AlertDescription></Alert>}
      {(pending || (task && task.status !== "succeeded")) && <div className="task-status" role="status" aria-live="polite">{pending && <Spinner aria-label="处理中" />}<div><strong>{busy ? `${busyLabel}…` : task ? `${KINDS[task.kind] || "任务"} · ${STATUS[task.status] || task.status}` : "正在处理…"}</strong><p>{task ? elapsedLabel(task, now) : "正在确认提交结果…"}</p><p>{task && liveTask(task) ? "已提交的任务会继续处理，请勿重复提交。" : task && task.status !== "succeeded" ? taskError(task) : ""}</p>{task && <><Button size="sm" variant="outline" disabled={busy} onClick={() => void refreshTaskStatus()}>刷新任务状态</Button>{["failed", "interrupted", "cancelled"].includes(task.status) && <p>任务编号：<code>{task.id}</code> <Button size="sm" variant="ghost" onClick={() => void navigator.clipboard.writeText(task.id).then(() => setNotice("任务编号已复制。")).catch(() => setNotice("无法自动复制，请手动选择任务编号。"))}>复制编号</Button></p>}</>}</div></div>}

      {loading ? <section className="flow-view" aria-label="正在恢复工作台" aria-busy="true"><Skeleton className="h-8 w-2/3" /><Skeleton className="mt-8 h-32 w-full" /><Skeleton className="mt-6 h-11 w-40" /><p className="field-note">正在读取已保存的状态…</p></section> : !session ? <section className="flow-view intake-view">
        <div className="flow-heading"><h1 ref={headingRef} tabIndex={-1}>这次希望解决什么职业问题？</h1><p>从你的目标开始。背景与材料可以逐步补充。</p></div>
        <form noValidate onSubmit={event => { event.preventDefault(); create(); }}>
          <FieldGroup>
            <Field data-invalid={Boolean(goalError)}><FieldLabel htmlFor="goal">你的目标 <span className="field-note">必填</span></FieldLabel><Textarea ref={goalRef} id="goal" name="career-goal" autoComplete="off" value={goal} onChange={event => { setGoal(event.target.value); if (goalError) setGoalError(""); }} placeholder="例如：明确数据分析和后端开发的选择，准备一份实习申请计划…" rows={3} maxLength={6000} aria-invalid={Boolean(goalError)} aria-describedby={goalError ? "goal-error" : "goal-help"} disabled={pending} /><FieldDescription id="goal-help">可以写下目标岗位，也可以先说你目前最困惑的事。</FieldDescription>{goalError && <FieldError id="goal-error">{goalError}</FieldError>}</Field>
            <Accordion type="single" collapsible><AccordionItem value="optional-background"><AccordionTrigger><span>补充背景与阅读偏好 <span className="optional-label">选填</span></span></AccordionTrigger><AccordionContent><FieldGroup>
            <Field><FieldLabel htmlFor="background">补充背景 <span className="field-note">选填</span></FieldLabel><Textarea id="background" name="career-background" autoComplete="off" value={background} onChange={event => setBackground(event.target.value)} placeholder="专业、已有经历、每周可投入时间，或你在意的现实约束…" rows={4} maxLength={20000} disabled={pending} /></Field>
            <Field><FieldLabel htmlFor="text-subject">这段背景描述的是</FieldLabel><NativeSelect id="text-subject" name="background-subject" value={textSubject} onChange={event => setTextSubject(event.target.value)} disabled={pending}>{Object.entries(SUBJECTS).map(([value, label]) => <NativeSelectOption key={value} value={value}>{label}</NativeSelectOption>)}</NativeSelect><FieldDescription>自我介绍选“我的经历 / 能力”，招聘信息选“岗位要求”。</FieldDescription></Field>
            <FieldSet><FieldLegend variant="label">初次规划的阅读偏好</FieldLegend><ToggleGroup type="single" variant="outline" value={detail} onValueChange={value => { if (value) setDetail(value); }} disabled={pending} aria-label="初次规划的阅读偏好"><ToggleGroupItem value="brief">简要</ToggleGroupItem><ToggleGroupItem value="standard">标准</ToggleGroupItem><ToggleGroupItem value="detailed">详细</ToggleGroupItem></ToggleGroup></FieldSet>
</FieldGroup></AccordionContent></AccordionItem></Accordion>
            <div className="flow-actions"><Button type="submit" disabled={pending}>{busy ? <Spinner aria-label="保存中" data-icon="inline-start" /> : null}保存目标，添加材料</Button><p className="field-note">此时只保存目标，开始理解材料后才运行模型。</p></div>
          </FieldGroup>
        </form>
      </section> : session.plan ? <ReportView session={session} version={version} latestVersion={latestVersion} pending={pending} busy={busy} onVersionChange={changeVersion} onFeedback={feedback => submitTask("feedback", { feedback, plan_id: session.plan!.plan_id, source_version: version!.version })} onEvaluate={evaluate} reportTab={reportTab} onReportTabChange={switchReport} /> : session.status === "collecting" ? <section className="flow-view intake-view">
        <div className="flow-heading"><h1 ref={headingRef} tabIndex={-1}>添加与你有关的材料</h1><p>可以补充文档、图片或视频，也可以直接用已保存的背景开始。</p></div>
        <section className="saved-goal" aria-label="已保存的目标"><h2>你的目标</h2><p>{session.user_goal}</p>{session.text_input && <Accordion type="single" collapsible><AccordionItem value="background"><AccordionTrigger>查看已保存背景</AccordionTrigger><AccordionContent><p className="user-text">{session.text_input}</p></AccordionContent></AccordionItem></Accordion>}</section>
        <div className="material-heading"><h2>材料</h2><span className="field-note">{session.uploads.length} / 3</span></div>
        <MaterialList session={session} />
        <input ref={fileInput} className="sr-only" type="file" name="career-material" accept={SUPPORTED} tabIndex={-1} aria-label="选择材料文件" onChange={event => chooseFile(event.target.files?.[0])} disabled={pending || session.uploads.length >= 3} />
        <Button className="upload-zone" variant="upload" type="button" onClick={() => fileInput.current?.click()} disabled={pending || session.uploads.length >= 3} onDragOver={event => event.preventDefault()} onDrop={event => { event.preventDefault(); if (!pending && session.uploads.length < 3) { if (event.dataTransfer.files.length !== 1) setError("请每次添加一个文件。"); else chooseFile(event.dataTransfer.files[0]); } }}><UploadIcon aria-hidden="true" data-icon="inline-start" /><span>{file ? file.name : "选择或拖入一份材料"}<small>文档、图片、扫描 PDF 或视频 · 单文件不超过25 MiB</small></span></Button>
        {file && <div className="upload-confirm"><FieldGroup><Field><FieldLabel htmlFor="subject">这份材料描述的是</FieldLabel><NativeSelect id="subject" name="file-subject" value={subject} onChange={event => setSubject(event.target.value)} disabled={pending}>{Object.entries(SUBJECTS).map(([value, label]) => <NativeSelectOption key={value} value={value}>{label}</NativeSelectOption>)}</NativeSelect><FieldDescription>岗位要求会作为参考，不会当作你已经具备的能力。</FieldDescription></Field><div className="flow-actions"><Button variant="outline" disabled={pending} onClick={upload}>上传这份材料</Button><Button variant="ghost" disabled={pending} onClick={() => { setFile(null); if (fileInput.current) fileInput.current.value = ""; }}>取消选择</Button></div></FieldGroup></div>}
        <div className="flow-actions"><Button disabled={pending || Boolean(file)} onClick={() => submitTask("perceive")}>开始理解材料</Button><p className="field-note">{file ? "先上传选中的文件，再开始理解材料。" : "材料提取完成后会清理原文件，音频暂未开放。"}</p></div>
      </section> : <section className="flow-view question-view">
        {session.warning && <Alert><AlertDescription>{session.warning}</AlertDescription></Alert>}
        {session.question ? <>
          <div className="flow-heading"><p className="question-kicker">补充一个关键信息</p><h1 className="question-prompt" ref={headingRef} tabIndex={-1}>{session.question.prompt}</h1><p>{session.question.reason === "conflict" ? "已有描述存在不同说法，请说明你现在的实际情况。" : session.question.reason === "clarify" ? "进一步明确你的想法，有助于形成适合当前目标的规划。" : "按目前的实际情况回答即可，不确定的内容可以暂时留空。"}</p></div>
          <form noValidate onSubmit={event => { event.preventDefault(); submitTask("answers", { question_id: session.question!.id, answer }); }}><FieldGroup><Field data-invalid={Boolean(answerError)}><FieldLabel htmlFor="answer">你的回答</FieldLabel><InputGroup><InputGroupTextarea ref={answerRef} id="answer" name="guidance-answer" autoComplete="off" rows={5} value={answer} onChange={event => { setAnswer(event.target.value); if (answerError) setAnswerError(""); }} placeholder="写下你的想法，也可以回答“不知道”或“不想回答”…" maxLength={2000} aria-invalid={Boolean(answerError)} aria-describedby={answerError ? "answer-error" : undefined} disabled={pending} /></InputGroup>{answerError && <FieldError id="answer-error">{answerError}</FieldError>}</Field><div className="flow-actions"><Button type="submit" disabled={pending}>提交回答</Button><Button type="button" variant="outline" disabled={pending} onClick={() => submitTask("answers", { question_id: session.question!.id, answer: "不知道" })}>暂不确定</Button></div></FieldGroup></form>
          <Button variant="link" className="end-guidance" disabled={pending} onClick={() => submitTask("answers", { question_id: session.question!.id, answer: "结束引导" })}>先用已有信息生成规划</Button><p className="field-note">结束本轮引导后，还需由你点击生成规划。</p>
        </> : <div className="flow-heading"><h1 ref={headingRef} tabIndex={-1}>可以形成你的规划了</h1><p>已提交的信息会作为本次规划的依据，未确定的部分仍会保留说明。</p><div className="saved-goal"><h2>你的目标</h2><p>{session.user_goal}</p></div><Button disabled={pending} onClick={() => submitTask("plans")}>生成我的职业规划</Button></div>}
        <Button variant="ghost" onClick={() => setMaterialsOpen(true)}>查看已保存的目标与材料</Button>
      </section>}

      {session && !loading && <div className="workspace-session-actions"><Button variant="ghost" onClick={() => setMaterialsOpen(true)}><FileTextIcon aria-hidden="true" data-icon="inline-start" />目标与材料</Button><Button ref={deleteReturnRef} variant="ghost" disabled={pending} onClick={() => setDeleteDialog(true)}><Trash2Icon aria-hidden="true" data-icon="inline-start" />删除本次规划</Button></div>}
    </main>
    <footer className="page-footer"><span>规划是决策参考，重要选择由你判断。</span><span>记录保留30天，可随时主动删除。</span></footer>

    <Sheet open={materialsOpen} onOpenChange={setMaterialsOpen}><SheetContent className="materials-sheet"><SheetHeader><SheetTitle>目标与材料</SheetTitle><SheetDescription>这里展示已保存的信息，不会重新解析材料。</SheetDescription></SheetHeader>{session && <div className="material-summary"><h3>你的目标</h3><p className="user-text">{session.user_goal}</p>{session.text_input && <><h3>背景</h3><p className="user-text">{session.text_input}</p><p className="field-note">主体：{SUBJECTS[session.text_subject || "unknown"]}</p></>}<h3>材料</h3><MaterialList session={session} />{!session.uploads.length && <p className="field-note">本次使用文字与回答，没有上传文件。</p>}<SheetClose asChild><Button variant="outline">返回当前步骤</Button></SheetClose></div>}</SheetContent></Sheet>

    <Dialog open={evaluation} onOpenChange={open => { setEvaluation(open); if (!open) { setError(""); setRetry(null); } }}><DialogContent className="evaluation-dialog" onCloseAutoFocus={event => { event.preventDefault(); evaluationReturnRef.current?.focus(); }}><DialogHeader><DialogTitle>评价第 {evaluationVersion || 1} 版规划</DialogTitle><DialogDescription>评分和意见均为选填，请根据实际体验填写，至少填写一项。</DialogDescription></DialogHeader>{error && <Alert variant="destructive"><AlertDescription><p>{error}</p>{retry && <Button variant="outline" disabled={busy} onClick={retry}>用原标识重试</Button>}</AlertDescription></Alert>}{submission && <Alert className="workspace-notice" role="status"><AlertDescription><p>{submission.label}的提交结果尚未确认。查询不会清除此入口。</p><div className="notice-actions"><Button size="sm" variant="outline" disabled={busy} onClick={() => void querySubmission()}>查询当前状态</Button><Button size="sm" variant="outline" disabled={busy} onClick={() => void confirmSubmission(submission)}>用原标识确认</Button>{submission.target && currentId.current !== submission.target && <Button size="sm" variant="outline" disabled={busy} onClick={() => void openSession(submission.target!)}>查看原记录</Button>}</div></AlertDescription></Alert>}<form onSubmit={event => { event.preventDefault(); saveEvaluation(); }}><FieldGroup><FieldSet><FieldLegend variant="label">你的体验</FieldLegend><div className="evaluation-fields">{[["usefulness", "对我有帮助"], ["clarity", "清晰易懂"], ["feasibility", "可以执行"]].map(([key, label]) => <Field key={key}><FieldLabel htmlFor={`rating-${key}`}>{label}</FieldLabel><NativeSelect disabled={pending} id={`rating-${key}`} name={`rating-${key}`} value={scores[key] || ""} onChange={event => setScores(previous => ({ ...previous, [key]: event.target.value }))}><NativeSelectOption value="">暂不评分</NativeSelectOption>{[1, 2, 3, 4, 5].map(number => <NativeSelectOption value={number} key={number}>{number} 分</NativeSelectOption>)}</NativeSelect></Field>)}</div></FieldSet><Field><FieldLabel htmlFor="evaluation-comment">你的意见 <span className="field-note">选填</span></FieldLabel><Textarea disabled={pending} id="evaluation-comment" name="evaluation-comment" autoComplete="off" value={comment} onChange={event => setComment(event.target.value)} placeholder="哪些内容有帮助？还有哪里不清楚…" maxLength={4000} rows={4} /></Field><div className="flow-actions"><Button type="submit" disabled={pending}>保存评价</Button><Button type="button" variant="ghost" disabled={busy} onClick={() => { setEvaluation(false); setError(""); setRetry(null); }}>取消</Button></div></FieldGroup></form></DialogContent></Dialog>

    <AlertDialog open={deleteDialog} onOpenChange={setDeleteDialog}><AlertDialogContent onCloseAutoFocus={event => { event.preventDefault(); (deleteReturnRef.current || headingRef.current)?.focus(); }}><AlertDialogHeader><AlertDialogTitle>删除本次规划？</AlertDialogTitle><AlertDialogDescription>材料、规划、评价和交互记录将被删除，无法恢复。排队任务会取消，运行中的结果将被丢弃。</AlertDialogDescription></AlertDialogHeader><AlertDialogFooter><AlertDialogCancel>保留规划</AlertDialogCancel><AlertDialogAction variant="destructive" onClick={remove}>确认删除</AlertDialogAction></AlertDialogFooter></AlertDialogContent></AlertDialog>
    <AlertDialog open={Boolean(pendingNavigation)} onOpenChange={open => { if (!open) setPendingNavigation(null); }}><AlertDialogContent><AlertDialogHeader><AlertDialogTitle>离开尚未提交的内容？</AlertDialogTitle><AlertDialogDescription>本页未提交的文字或选中的文件不会保存，已保存的会话记录不受影响。</AlertDialogDescription></AlertDialogHeader><AlertDialogFooter><AlertDialogCancel>继续填写</AlertDialogCancel><AlertDialogAction onClick={() => { if (pendingNavigation) performNavigation(pendingNavigation); }}>放弃未提交内容</AlertDialogAction></AlertDialogFooter></AlertDialogContent></AlertDialog>
  </div>;
}

function MaterialList({ session }: { session: Session }) {
  return <div className="material-list">{session.uploads.map(upload => <Attachment className="material-row" key={upload.id}><AttachmentMedia><FileTextIcon aria-hidden="true" /></AttachmentMedia><AttachmentContent><AttachmentTitle className="line-clamp-none break-words">{upload.name}</AttachmentTitle><AttachmentDescription className="line-clamp-none break-words">{SUBJECTS[upload.subject] || "尚不确定"} · {(upload.size / 1024 / 1024).toFixed(1)} MiB · {upload.deleted_at ? "已提取并清理原件" : "已上传"}</AttachmentDescription></AttachmentContent></Attachment>)}</div>;
}
