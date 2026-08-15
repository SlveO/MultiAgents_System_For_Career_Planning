import React, { useState, useRef } from 'react';
import { Plus } from 'lucide-react';
import { useTodos } from '../../store/TodoContext';

export default function TodoForm() {
  const { addTodo } = useTodos();
  const [title, setTitle] = useState('');
  const [description, setDescription] = useState('');
  const titleRef = useRef<HTMLInputElement>(null);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!title.trim()) return;
    addTodo(title.trim(), description.trim());
    setTitle('');
    setDescription('');
    titleRef.current?.focus();
  };

  return (
    <form onSubmit={handleSubmit} className="bg-white rounded-2xl shadow-sm border border-gray-100 p-4 space-y-3">
      <div className="flex gap-3">
        <input
          ref={titleRef}
          type="text"
          value={title}
          onChange={(e) => setTitle(e.target.value)}
          placeholder="待办标题..."
          className="flex-1 px-4 py-2.5 border border-gray-200 rounded-xl focus:border-primary-500 focus:ring-2 focus:ring-primary-100 outline-none transition-all text-sm"
        />
        <button
          type="submit"
          disabled={!title.trim()}
          className="flex items-center gap-1.5 px-5 py-2.5 bg-primary-500 text-white rounded-xl hover:bg-primary-600 disabled:opacity-50 transition-all text-sm font-medium"
        >
          <Plus className="w-4 h-4" />
          添加
        </button>
      </div>
      <textarea
        value={description}
        onChange={(e) => setDescription(e.target.value)}
        placeholder="描述（选填）"
        className="w-full px-4 py-2.5 border border-gray-200 rounded-xl focus:border-primary-500 focus:ring-2 focus:ring-primary-100 outline-none transition-all text-sm resize-none"
        rows={2}
      />
    </form>
  );
}
