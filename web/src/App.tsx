import { useState } from 'react';
import { GraduationCap } from 'lucide-react';
import { useAuth } from './store/AuthContext';
import { useChat } from './store/ChatContext';
import { usePomodoro } from './store/PomodoroContext';

import Header from './components/layout/Header';
import Navbar from './components/layout/Navbar';
import LoginForm from './components/auth/LoginForm';
import RegisterForm from './components/auth/RegisterForm';
import HistorySidebar from './components/history/HistorySidebar';
import ChatContainer from './components/chat/ChatContainer';
import ChatInput from './components/chat/ChatInput';
import QuickQuestions from './components/chat/QuickQuestions';
import TodoForm from './components/todo/TodoForm';
import TodoList from './components/todo/TodoList';
import InfoModal from './components/modals/InfoModal';
import TodoDetailModal from './components/modals/TodoDetailModal';
import PomodoroModal from './components/modals/PomodoroModal';
import FlipClockModal from './components/modals/FlipClockModal';

import type { PageTab, InfoModalType, TodoItem } from './types';

export default function App() {
  const { user } = useAuth();
  const { historyVisible } = useChat();
  const { setCurrentTask } = usePomodoro();

  const [authMode, setAuthMode] = useState<'login' | 'register'>('login');
  const [activeTab, setActiveTab] = useState<PageTab>('career');

  // Modal states
  const [infoModalType, setInfoModalType] = useState<InfoModalType | null>(null);
  const [selectedTodo, setSelectedTodo] = useState<TodoItem | null>(null);
  const [pomodoroOpen, setPomodoroOpen] = useState(false);
  const [flipClockOpen, setFlipClockOpen] = useState(false);

  // ---- Auth page ----
  if (!user) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-slate-50 to-slate-200 p-4">
        <div className="bg-white rounded-2xl shadow-xl max-w-md w-full p-8">
          <div className="flex items-center justify-center gap-2 mb-8">
            <GraduationCap className="w-8 h-8 text-primary-500" />
            <h1 className="text-xl font-bold text-gray-800">大学生职业规划助手</h1>
          </div>
          {authMode === 'login' ? (
            <LoginForm onSwitchToRegister={() => setAuthMode('register')} />
          ) : (
            <RegisterForm onSwitchToLogin={() => setAuthMode('login')} />
          )}
        </div>
      </div>
    );
  }

  // ---- Main app ----
  const handleOpenPomodoro = (todo: TodoItem) => {
    setSelectedTodo(null);
    setCurrentTask({ title: todo.title, description: todo.description });
    setPomodoroOpen(true);
  };

  const handleOpenFlipClock = () => {
    setFlipClockOpen(true);
  };

  return (
    <div className="min-h-screen flex flex-col">
      <Header />
      <Navbar activeTab={activeTab} onNavigate={setActiveTab} />

      <main className="flex-1 flex max-w-7xl mx-auto w-full">
        {activeTab === 'career' ? (
          <>
            {/* History sidebar — always mounted so collapse/expand button works */}
            <HistorySidebar />

            {/* Chat area */}
            <div className="flex-1 flex flex-col min-w-0">
              <ChatContainer />
              <ChatInput />
            </div>

            {/* Quick questions — right sidebar */}
            {historyVisible && (
              <div className="w-52 bg-white border-l border-gray-100 shrink-0">
                <QuickQuestions onOpenInfoModal={setInfoModalType} />
              </div>
            )}
          </>
        ) : (
          /* Todo page */
          <div className="flex-1 max-w-2xl mx-auto px-4 py-6 space-y-6">
            <TodoForm />
            <TodoList onTodoClick={setSelectedTodo} />
          </div>
        )}
      </main>

      {/* Modals */}
      <InfoModal
        isOpen={infoModalType !== null}
        type={infoModalType || 'major-gpa'}
        onClose={() => setInfoModalType(null)}
      />

      <TodoDetailModal
        isOpen={selectedTodo !== null}
        todo={selectedTodo}
        onClose={() => setSelectedTodo(null)}
        onOpenPomodoro={handleOpenPomodoro}
        onOpenFlipClock={handleOpenFlipClock}
      />

      <PomodoroModal
        isOpen={pomodoroOpen}
        onClose={() => setPomodoroOpen(false)}
      />

      <FlipClockModal
        isOpen={flipClockOpen}
        onClose={() => setFlipClockOpen(false)}
      />
    </div>
  );
}
