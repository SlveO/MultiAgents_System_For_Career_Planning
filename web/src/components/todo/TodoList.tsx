import { useState, useCallback } from 'react';
import { useTodos } from '../../store/TodoContext';
import TodoItemComponent from './TodoItem';
import type { TodoItem } from '../../types';

interface TodoListProps {
  onTodoClick: (todo: TodoItem) => void;
}

export default function TodoList({ onTodoClick }: TodoListProps) {
  const { todos, toggleTodo, deleteTodo, reorderTodos } = useTodos();
  const [dragIndex, setDragIndex] = useState<number | null>(null);

  const handleDragStart = useCallback((_e: React.DragEvent, index: number) => {
    setDragIndex(index);
  }, []);

  const handleDragOver = useCallback((e: React.DragEvent, _index: number) => {
    e.preventDefault();
    e.dataTransfer.dropEffect = 'move';
  }, []);

  const handleDrop = useCallback(
    (e: React.DragEvent, toIndex: number) => {
    e.preventDefault();
    if (dragIndex !== null && dragIndex !== toIndex) {
      reorderTodos(dragIndex, toIndex);
    }
    setDragIndex(null);
  },
  [dragIndex, reorderTodos]
  );

  if (todos.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center py-16 text-gray-400">
        <p className="text-sm">还没有待办事项</p>
        <p className="text-xs mt-1">添加一个开始管理你的任务吧</p>
      </div>
    );
  }

  return (
    <div className="space-y-2">
      {todos.map((todo, index) => (
        <TodoItemComponent
          key={todo.id}
          todo={todo}
          index={index}
          onToggle={toggleTodo}
          onDelete={deleteTodo}
          onClick={onTodoClick}
          onDragStart={handleDragStart}
          onDragOver={handleDragOver}
          onDrop={handleDrop}
        />
      ))}
    </div>
  );
}
