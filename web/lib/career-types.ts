export type Task = { id: string; session_id: string; kind: string; status: "queued" | "running" | "succeeded" | "failed" | "interrupted" | "cancelled"; error?: string | { message?: string; code?: string } | null; created_at: string; updated_at: string };
export type Upload = { id: string; name: string; size: number; subject: string; created_at: string; deleted_at?: string | null };
export type OutputVersion = { plan_id: string; version: number; parent_version?: number | null; display_text: string; created_at: string; reading_view?: { mode: "brief"; strategy_version: string; item_ids: string[]; excerpt: string; notice: string } | null };
export type Graph = { nodes: { id: string; label: string; kind?: string; type?: string; source?: string; verification?: string; [key: string]: unknown }[]; edges: { source: string; target: string; relation?: string; label?: string; [key: string]: unknown }[] };
export type FeedbackChoice = "过短" | "合适" | "过于详细";
export type FeedbackRecord = { request_id: string; feedback: FeedbackChoice; status: "adapted" | "unchanged" | "failed"; reason?: string; source_version: number; output_version: OutputVersion };
export type Milestone = { period: "30d" | "90d" | "180d"; objective: string; deliverables: string[]; metrics: string[] };
export type Plan = {
  plan_id: string; user_facing_advice?: string; target_roles?: string[]; gap_analysis?: string[];
  risk_flags?: string[]; next_actions?: string[]; learning_resources?: string[]; follow_up_questions?: string[];
  roadmap_30_90_180?: Milestone[];
  profile?: { preference?: string; main_constraints?: string[]; constraints?: { time_budget_hours_per_week?: number | null; financial_budget_cny?: number | null; city?: string | null; preferred_industries?: string[] }; [key: string]: unknown };
  graph?: Graph; career_graph?: Graph; knowledge_snapshot?: { version_id?: string }; [key: string]: unknown;
};
export type Session = { id: string; title: string; user_goal: string; text_input: string; text_subject?: string; status: string; state_version: number; question: { id: string; field: string; prompt: string; reason: string } | null; uploads: Upload[]; active_task: Task | null; last_task?: Task | null; plan: Plan | null; versions: OutputVersion[]; created_at: string; updated_at: string; source_claims?: Record<string, unknown>[]; knowledge_snapshot?: { version_id?: string }; feedback?: FeedbackRecord[]; evaluations?: unknown[]; warning?: string };
