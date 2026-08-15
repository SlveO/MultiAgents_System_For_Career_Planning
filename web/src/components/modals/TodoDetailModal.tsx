import Modal from './Modal';
import { Timer, Clock } from 'lucide-react';
import type { TodoItem } from '../../types';

interface TodoDetailModalProps {
  isOpen: boolean;
  todo: TodoItem | null;
  onClose: () => void;
  onOpenPomodoro: (todo: TodoItem) => void;
  onOpenFlipClock: () => void;
}

export default function TodoDetailModal({
  isOpen,
  todo,
  onClose,
  onOpenPomodoro,
  onOpenFlipClock,
}: TodoDetailModalProps) {
  if (!todo) return null;

  return (
    <Modal isOpen={isOpen} onClose={onClose} title={todo.title}>
      <div className="space-y-4 mt-2">
        {todo.description ? (
          <div>
            <p className="text-xs font-medium text-gray-400 uppercase mb-1">描述</p>
            <p className="text-sm text-gray-600 bg-gray-50 rounded-xl p-3">{todo.description}</p>
          </div>
        ) : (
          <p className="text-sm text-gray-400 italic">无描述</p>
        )}
        <div className="text-xs text-gray-400">
          创建于 {new Date(todo.createdAt).toLocaleDateString('zh-CN')}
        </div>
        <div className="flex gap-3 pt-2">
          <button
            onClick={() => onOpenPomodoro(todo)}
            className="flex-1 flex items-center justify-center gap-2 py-2.5 bg-red-50 text-red-600 rounded-xl text-sm font-medium hover:bg-red-100 transition-colors"
          >
            <Timer className="w-4 h-4" />
            番茄钟
          </button>
          <button
            onClick={() => {
              onClose();
              onOpenFlipClock();
            }}
            className="flex-1 flex items-center justify-center gap-2 py-2.5 bg-gray-100 text-gray-700 rounded-xl text-sm font-medium hover:bg-gray-200 transition-colors"
          >
            <Clock className="w-4 h-4" />
            翻页时钟
          </button>
        </div>
      </div>
    </Modal>
  );
}
