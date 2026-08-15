// ===== API Request Types =====

export interface UserConstraints {
  time_budget_hours_per_week?: number;
  financial_budget_cny?: number;
  city?: string;
  education_level?: string;
  preferred_industries: string[];
}

export interface TaskRequest {
  session_id: string;
  user_goal: string;
  text_input?: string;
  image_paths?: string[];
  document_paths?: string[];
  audio_paths?: string[];
  video_paths?: string[];
  brain_model?: string;
  stream?: boolean;
  debug_trace?: boolean;
  constraints?: UserConstraints;
  metadata?: Record<string, string>;
}

export interface MultimodalChatRequest {
  session_id: string;
  user_input: string;
  llm_model?: string;
}

export interface RegisterRequest {
  username: string;
  password: string;
}

export interface LoginRequest {
  username: string;
  password: string;
}

export interface FeedbackRequest {
  session_id: string;
  feedback: string;
  rating?: number;
}

// ===== API Response Types =====

export interface UserRow {
  id: string;
  username: string;
  api_key: string;
  created_at: string;
}

export interface LoginResponse {
  access_token: string;
  token_type: string;
  user_id: string;
  username: string;
}

export interface UploadResponse {
  file_path: string;
  file_name: string;
  file_type: string;
  size: number;
}

export interface UserProfile {
  strengths: string[];
  weaknesses: string[];
  interests: string[];
  current_stage: string;
  constraints: UserConstraints;
}

export interface Milestone {
  period: '30d' | '90d' | '180d';
  objective: string;
  deliverables: string[];
  metrics: string[];
}

export interface EvidenceItem {
  source: string;
  quote: string;
}

export interface PerceptionResult {
  modality: 'text' | 'image' | 'document' | 'audio' | 'video';
  summary: string;
  facts: string[];
  evidence: EvidenceItem[];
  confidence: number;
  missing_info: string[];
  raw_output: string;
}

export interface CareerPlanResponse {
  session_id: string;
  intent: string;
  profile: UserProfile;
  target_roles: string[];
  gap_analysis: string[];
  roadmap_30_90_180: Milestone[];
  learning_resources: string[];
  next_actions: string[];
  risk_flags: string[];
  follow_up_questions: string[];
  confidence: number;
  user_facing_advice: string;
  perception_results: PerceptionResult[];
  knowledge_hits: string[];
  model_trace: string[];
  served_by: string;
  retry_count: number;
  latency_ms: number;
}

export interface SessionResponse {
  session_id: string;
  profile?: UserProfile;
  history?: ChatMessage[];
}

/** Backend returns history as [{user, assistant}] — one turn per entry */
export interface ChatSessionData {
  session_id: string;
  history: Array<{ user: string; assistant: string }>;
}

// ===== SSE Event Types =====

export interface SSEEvent {
  event: string;
  data: string;
}

export interface TokenEvent {
  token?: string;
}

export interface ErrorEvent {
  code?: string;
  message?: string;
  session_id?: string;
}

// ===== App State Types =====

export interface ChatMessage {
  role: 'user' | 'ai';
  content: string;
}

export interface ChatSession {
  id: string;
  title: string;
  messages: ChatMessage[];
  timestamp: string;
}

export interface TodoItem {
  id: string;
  title: string;
  description: string;
  completed: boolean;
  createdAt: string;
}

export type TimerPhase = 'work' | 'shortBreak' | 'longBreak';
export type TimerMode = 'countdown' | 'countup';

export interface PomodoroState {
  isRunning: boolean;
  isPaused: boolean;
  currentPhase: TimerPhase;
  timeLeft: number;
  elapsed: number;
  workTime: number;
  shortBreakTime: number;
  longBreakTime: number;
  longBreakInterval: number;
  completedPomodoros: number;
  timerMode: TimerMode;
}

export interface PomodoroSettings {
  workTime: number;
  shortBreakTime: number;
  longBreakTime: number;
  longBreakInterval: number;
  timerMode: TimerMode;
}

export type InfoModalType = 'major-gpa' | 'major-only' | 'exam-work';

export type PageTab = 'career' | 'todo';
