import { GripVertical, Circle, CheckCircle2, Trash2 } from 'lucide-react';
import type { TodoItem as TodoItemType } from '../../types';

interface TodoItemProps {
  todo: TodoItemType;
  index: number;
  onToggle: (id: string) => void;
  onDelete: (id: string) => void;
  onClick: (todo: TodoItemType) => void;
  onDragStart: (e: React.DragEvent, index: number) => void;
  onDragOver: (e: React.DragEvent, index: number) => void;
  onDrop: (e: React.DragEvent, index: number) => void;
}

export default function TodoItem({
  todo,
  index,
  onToggle,
  onDelete,
  onClick,
  onDragStart,
  onDragOver,
  onDrop,
}: TodoItemProps) {
  return (
    <div
      draggable
      onDragStart={(e) => onDragStart(e, index)}
      onDragOver={(e) => onDragOver(e, index)}
      onDrop={(e) => onDrop(e, index)}
      className={`flex items-center gap-3 px-4 py-3 rounded-xl transition-all cursor-pointer group ${
        todo.completed ? 'bg-gray-50 opacity-60' : 'bg-white hover:bg-gray-50 hover:shadow-sm'
      } border border-gray-100`}
    >
      {/* Drag handle */}
      <div className="text-gray-300 hover:text-gray-500 cursor-grab transition-colors shrink-0">
        <GripVertical className="w-4 h-4" />
      </div>

      {/* Checkbox */}
      <button
        onClick={(e) => {
          e.stopPropagation();
          onToggle(todo.id);
        }}
        className="shrink-0 transition-colors"
      >
        {todo.completed ? (
          <CheckCircle2 className="w-5 h-5 text-green-500" />
        ) : (
          <Circle className="w-5 h-5 text-gray-300 hover:text-primary-500" />
        )}
      </button>

      {/* Content */}
      <div className="flex-1 min-w-0" onClick={() => onClick(todo)}>
        <p className={`text-sm font-medium truncate ${todo.completed ? 'line-through text-gray-400' : 'text-gray-700'}`}>
          {todo.title}
        </p>
        {todo.description && (
          <p className="text-xs text-gray-400 truncate mt-0.5">{todo.description}</p>
        )}
      </div>

      {/* Delete */}
      <button
        onClick={(e) => {
          e.stopPropagation();
          onDelete(todo.id);
        }}
        className="opacity-0 group-hover:opacity-100 p-1.5 rounded-lg hover:bg-red-50 text-gray-400 hover:text-red-500 transition-all shrink-0"
      >
        <Trash2 className="w-4 h-4" />
      </button>
    </div>
  );
}
