import { useChat } from '../../store/ChatContext';
import type { InfoModalType } from '../../types';

interface QuickQuestionsProps {
  onOpenInfoModal: (type: InfoModalType) => void;
}

const QUESTIONS = [
  { text: '我适合什么职业？', triggersModal: 'major-gpa' as const },
  { text: '我的专业就业方向', triggersModal: 'major-only' as const },
  { text: '如何做大学四年规划？', triggersModal: null },
  { text: '考研还是就业？', triggersModal: 'exam-work' as const },
  { text: '如何提升简历竞争力？', triggersModal: null },
];

export default function QuickQuestions({ onOpenInfoModal }: QuickQuestionsProps) {
  const { sendMessage, isLoading } = useChat();

  const handleClick = (q: typeof QUESTIONS[0]) => {
    if (isLoading) return;
    if (q.triggersModal) {
      onOpenInfoModal(q.triggersModal);
    } else {
      sendMessage(q.text);
    }
  };

  return (
    <div className="p-3 space-y-2">
      <p className="text-xs font-semibold text-primary-500 uppercase tracking-wider px-1 mb-1">职业规划提问</p>
      {QUESTIONS.map((q) => (
        <button
          key={q.text}
          onClick={() => handleClick(q)}
          disabled={isLoading}
          className="quick-question w-full text-left px-3 py-2.5 bg-white rounded-xl text-sm text-gray-600 hover:bg-primary-50 hover:text-primary-600 hover:-translate-y-0.5 transition-all shadow-sm border border-gray-100 disabled:opacity-50"
        >
          {q.text}
        </button>
      ))}
    </div>
  );
}
