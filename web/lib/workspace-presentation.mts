import type { FeedbackRecord, Session, Task } from "./career-types";

export function workspaceStage(session: Session | null): number {
  if (session?.plan) return 2;
  if (session && session.status !== "collecting") return 1;
  return 0;
}

export function feedbackNotice(session: Session): string {
  const feedback: FeedbackRecord | undefined = session.feedback?.at(-1);
  if (!feedback) return "已读取当前规划，请查看版本与内容。";
  if (feedback.reason === "brief_view") return `已切换简略阅读，保存为第 ${feedback.output_version.version} 版。`;
  if (feedback.reason === "full_view") return `已恢复完整阅读，保存为第 ${feedback.output_version.version} 版。`;
  if (feedback.reason === "already_brief") return "当前已是简略阅读，版本未改变。";
  if (feedback.status === "unchanged") {
    if (feedback.reason === "suitable") return "已记录：当前详略合适，保留这一版。";
    return "本次未改变内容，保留原版本。现有内容未能按本次要求进一步调整。";
  }
  if (feedback.status === "failed") return "本次调整未通过校验，原版本已保留。";
  const next = feedback.output_version;
  const previous = session.versions.find(item => item.version === feedback.source_version);
  const count = Array.from(next.display_text).length;
  const from = previous ? `${Array.from(previous.display_text).length} → ` : "";
  return `已更新为第 ${next.version} 版，${from}${count} 字。`;
}

export function taskResultNotice(task: Task, session: Session | null): string {
  if (task.kind === "feedback" && session) return feedbackNotice(session);
  const labels: Record<string, string> = { perceive: "材料已处理，可以继续补充信息。", answers: "回答已保存。", plans: "规划已生成，可以查看结论与行动路线。" };
  return labels[task.kind] || "操作已完成。";
}
