import React, { createContext, useContext, useState, useCallback, useRef } from 'react';
import { sendChatMessage, getSession, deleteSession as apiDeleteSession } from '../api/chat';
import type { StreamEvent } from '../api/chat';
import type { ChatSession, ChatMessage } from '../types';

interface ChatState {
  sessions: ChatSession[];
  currentSession: ChatSession | null;
  isLoading: boolean;
  isStreaming: boolean;
  historyVisible: boolean;
  createSession: () => void;
  switchSession: (id: string) => void;
  deleteSession: (id: string) => void;
  sendMessage: (text: string) => Promise<void>;
  toggleHistory: () => void;
}

const ChatContext = createContext<ChatState | null>(null);

function newSession(): ChatSession {
  return {
    id: Date.now().toString(),
    title: '新对话',
    messages: [
      { role: 'ai', content: '你好！我是大学生职业规划助手，有什么可以帮助你的吗？' },
    ],
    timestamp: new Date().toISOString(),
  };
}

export function ChatProvider({ children }: { children: React.ReactNode }) {
  const [sessions, setSessions] = useState<ChatSession[]>([newSession()]);
  const [currentId, setCurrentId] = useState<string>(sessions[0]?.id || '');
  const [isLoading, setIsLoading] = useState(false);
  const [isStreaming, setIsStreaming] = useState(false);
  const [historyVisible, setHistoryVisible] = useState(true);
  const abortRef = useRef<AbortController | null>(null);

  const currentSession = sessions.find((s) => s.id === currentId) || null;

  const updateSession = useCallback((id: string, updater: (s: ChatSession) => ChatSession) => {
    setSessions((prev) => prev.map((s) => (s.id === id ? updater(s) : s)));
  }, []);

  const createSession = useCallback(() => {
    const session = newSession();
    setSessions((prev) => [session, ...prev]);
    setCurrentId(session.id);
  }, []);

  const switchSession = useCallback(
    (id: string) => {
      setCurrentId(id);
      // Load history from backend (format: [{user, assistant}])
      // Only seed from backend if local session has no user conversations yet.
      // Backend only stores successful LLM turns; failed exchanges exist only locally.
      getSession(id).then((data) => {
        if (data.history.length > 0) {
          const remoteMessages: ChatMessage[] = [];
          for (const turn of data.history) {
            if (turn.user) remoteMessages.push({ role: 'user', content: turn.user });
            if (turn.assistant) remoteMessages.push({ role: 'ai', content: turn.assistant });
          }
          if (remoteMessages.length > 0) {
            updateSession(id, (s) => {
              const hasLocalConversations = s.messages.some((m) => m.role === 'user');
              if (hasLocalConversations) {
                // Keep local data — it may include failed exchanges backend doesn't know about
                return s;
              }
              return {
                ...s,
                messages: remoteMessages,
                title: remoteMessages.find((m) => m.role === 'user')?.content.slice(0, 20) || s.title,
              };
            });
          }
        }
      }).catch(() => { /* session may not exist yet */ });
    },
    [updateSession]
  );

  const deleteSession = useCallback(
    (id: string) => {
      apiDeleteSession(id).catch(() => {});
      setSessions((prev) => {
        const filtered = prev.filter((s) => s.id !== id);
        if (filtered.length === 0) {
          const s = newSession();
          setCurrentId(s.id);
          return [s];
        }
        if (currentId === id) {
          setCurrentId(filtered[0].id);
        }
        return filtered;
      });
    },
    [currentId]
  );

  const setAI = useCallback(
    (content: string) => {
      if (!currentSession) return;
      updateSession(currentSession.id, (s) => {
        const msgs = [...s.messages];
        const last = msgs[msgs.length - 1];
        if (last && last.role === 'ai') {
          msgs[msgs.length - 1] = { ...last, content };
        }
        return { ...s, messages: msgs };
      });
    },
    [currentSession, updateSession]
  );

  const sendMessage = useCallback(
    async (text: string) => {
      if (!currentSession || isStreaming) return;

      const userMsg: ChatMessage = { role: 'user', content: text };
      updateSession(currentSession.id, (s) => ({
        ...s,
        messages: [...s.messages, userMsg],
        title: s.messages.length === 1 ? text.slice(0, 20) : s.title,
        timestamp: new Date().toISOString(),
      }));

      setIsLoading(true);
      setIsStreaming(true);

      // Placeholder shows initial progress
      const aiPlaceholder: ChatMessage = { role: 'ai', content: '▊ 正在连接AI服务...' };
      updateSession(currentSession.id, (s) => ({
        ...s,
        messages: [...s.messages, aiPlaceholder],
      }));

      try {
        let fullContent = '';
        let tokenStarted = false;
        let lastStage = '';

        for await (const event of sendChatMessage(currentSession.id, text)) {
          if (event.type === 'stage') {
            lastStage = event.text;
            if (!tokenStarted) {
              setAI(`▊ ${event.text}`);
            }
          } else if (event.type === 'token') {
            if (!tokenStarted) {
              fullContent = '';
              tokenStarted = true;
            }
            fullContent += event.value;
            setAI(fullContent);
          }
        }

        // Stream ended with no tokens — diagnostic message
        if (!tokenStarted) {
          const stageInfo = lastStage ? `\n\n流程已到达：${lastStage}` : '\n\n后端未返回任何事件，请检查后端日志。';
          setAI(`AI 未返回回复内容。${stageInfo}\n\n常见原因：\n1. 未配置 DEEPSEEK_API_KEY 环境变量\n2. DeepSeek API 密钥无效或额度不足\n3. 网络无法访问 api.deepseek.com`);
        }
      } catch (e) {
        const errMsg = e instanceof Error ? e.message : '发送失败，请重试';
        setAI(`抱歉，AI服务暂时不可用。\n\n错误详情：${errMsg}`);
      } finally {
        setIsLoading(false);
        setIsStreaming(false);
      }
    },
    [currentSession, isStreaming, updateSession, setAI]
  );

  const toggleHistory = useCallback(() => {
    setHistoryVisible((v) => !v);
  }, []);

  return (
    <ChatContext.Provider
      value={{
        sessions,
        currentSession,
        isLoading,
        isStreaming,
        historyVisible,
        createSession,
        switchSession,
        deleteSession,
        sendMessage,
        toggleHistory,
      }}
    >
      {children}
    </ChatContext.Provider>
  );
}

export function useChat(): ChatState {
  const ctx = useContext(ChatContext);
  if (!ctx) throw new Error('useChat must be used within ChatProvider');
  return ctx;
}
