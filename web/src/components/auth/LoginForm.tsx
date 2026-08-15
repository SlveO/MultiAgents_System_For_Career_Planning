import React, { useState } from 'react';
import { LogIn } from 'lucide-react';
import { useAuth } from '../../store/AuthContext';

interface LoginFormProps {
  onSwitchToRegister: () => void;
}

export default function LoginForm({ onSwitchToRegister }: LoginFormProps) {
  const { login, isLoading, error } = useAuth();
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!username.trim() || !password.trim()) return;
    try {
      await login(username.trim(), password);
    } catch { /* error handled by context */ }
  };

  return (
    <form onSubmit={handleSubmit} className="space-y-4">
      <h2 className="text-2xl font-bold text-gray-800 text-center mb-6">登录</h2>
      {error && (
        <div className="bg-red-50 text-red-600 text-sm px-4 py-2 rounded-lg">{error}</div>
      )}
      <div>
        <label htmlFor="login-username" className="block text-sm font-medium text-gray-600 mb-1">
          用户名
        </label>
        <input
          id="login-username"
          type="text"
          value={username}
          onChange={(e) => setUsername(e.target.value)}
          className="w-full px-4 py-2.5 border border-gray-200 rounded-xl focus:border-primary-500 focus:ring-2 focus:ring-primary-100 outline-none transition-all"
          placeholder="请输入用户名"
          autoComplete="username"
        />
      </div>
      <div>
        <label htmlFor="login-password" className="block text-sm font-medium text-gray-600 mb-1">
          密码
        </label>
        <input
          id="login-password"
          type="password"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          className="w-full px-4 py-2.5 border border-gray-200 rounded-xl focus:border-primary-500 focus:ring-2 focus:ring-primary-100 outline-none transition-all"
          placeholder="请输入密码"
          autoComplete="current-password"
        />
      </div>
      <button
        type="submit"
        disabled={isLoading || !username.trim() || !password.trim()}
        className="w-full flex items-center justify-center gap-2 py-2.5 bg-primary-500 text-white rounded-xl hover:bg-primary-600 disabled:opacity-50 disabled:cursor-not-allowed transition-all font-medium"
      >
        <LogIn className="w-4 h-4" />
        {isLoading ? '登录中...' : '登录'}
      </button>
      <p className="text-center text-sm text-gray-500">
        还没有账号？{' '}
        <button
          type="button"
          onClick={onSwitchToRegister}
          className="text-primary-500 hover:text-primary-600 font-medium"
        >
          立即注册
        </button>
      </p>
    </form>
  );
}
