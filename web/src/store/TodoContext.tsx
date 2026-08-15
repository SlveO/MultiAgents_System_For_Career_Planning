import React, { createContext, useContext, useState, useCallback } from 'react';
import { getItem, setItem } from '../utils/storage';
import type { TodoItem } from '../types';

const STORAGE_KEY = 'career_todos';

interface TodoState {
  todos: TodoItem[];
  addTodo: (title: string, description: string) => void;
  toggleTodo: (id: string) => void;
  deleteTodo: (id: string) => void;
  reorderTodos: (fromIndex: number, toIndex: number) => void;
}

const TodoContext = createContext<TodoState | null>(null);

export function TodoProvider({ children }: { children: React.ReactNode }) {
  const [todos, setTodos] = useState<TodoItem[]>(() => getItem<TodoItem[]>(STORAGE_KEY, []));

  const persist = useCallback((updated: TodoItem[]) => {
    setTodos(updated);
    setItem(STORAGE_KEY, updated);
  }, []);

  const addTodo = useCallback(
    (title: string, description: string) => {
      const todo: TodoItem = {
        id: Date.now().toString(),
        title,
        description,
        completed: false,
        createdAt: new Date().toISOString(),
      };
      persist([...todos, todo]);
    },
    [todos, persist]
  );

  const toggleTodo = useCallback(
    (id: string) => {
      persist(todos.map((t) => (t.id === id ? { ...t, completed: !t.completed } : t)));
    },
    [todos, persist]
  );

  const deleteTodo = useCallback(
    (id: string) => {
      persist(todos.filter((t) => t.id !== id));
    },
    [todos, persist]
  );

  const reorderTodos = useCallback(
    (fromIndex: number, toIndex: number) => {
      const copy = [...todos];
      const [item] = copy.splice(fromIndex, 1);
      copy.splice(toIndex, 0, item);
      persist(copy);
    },
    [todos, persist]
  );

  return (
    <TodoContext.Provider value={{ todos, addTodo, toggleTodo, deleteTodo, reorderTodos }}>
      {children}
    </TodoContext.Provider>
  );
}

export function useTodos(): TodoState {
  const ctx = useContext(TodoContext);
  if (!ctx) throw new Error('useTodos must be used within TodoProvider');
  return ctx;
}
