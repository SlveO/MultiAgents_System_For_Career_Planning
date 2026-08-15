import React, { useState } from 'react';
import { UserPlus } from 'lucide-react';
import { useAuth } from '../../store/AuthContext';

interface RegisterFormProps {
  onSwitchToLogin: () => void;
}

export default function RegisterForm({ onSwitchToLogin }: RegisterFormProps) {
  const { register, isLoading, error } = useAuth();
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [localError, setLocalError] = useState<string | null>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLocalError(null);
    if (!username.trim() || !password.trim()) return;
    if (password !== confirmPassword) {
      setLocalError('两次输入的密码不一致');
      return;
    }
    if (password.length < 4) {
      setLocalError('密码长度至少4位');
      return;
    }
    try {
      await register(username.trim(), password);
    } catch { /* error handled by context */ }
  };

  const displayError = localError || error;

  return (
    <form onSubmit={handleSubmit} className="space-y-4">
      <h2 className="text-2xl font-bold text-gray-800 text-center mb-6">注册</h2>
      {displayError && (
        <div className="bg-red-50 text-red-600 text-sm px-4 py-2 rounded-lg">{displayError}</div>
      )}
      <div>
        <label htmlFor="reg-username" className="block text-sm font-medium text-gray-600 mb-1">
          用户名
        </label>
        <input
          id="reg-username"
          type="text"
          value={username}
          onChange={(e) => setUsername(e.target.value)}
          className="w-full px-4 py-2.5 border border-gray-200 rounded-xl focus:border-primary-500 focus:ring-2 focus:ring-primary-100 outline-none transition-all"
          placeholder="请输入用户名"
          autoComplete="username"
        />
      </div>
      <div>
        <label htmlFor="reg-password" className="block text-sm font-medium text-gray-600 mb-1">
          密码
        </label>
        <input
          id="reg-password"
          type="password"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          className="w-full px-4 py-2.5 border border-gray-200 rounded-xl focus:border-primary-500 focus:ring-2 focus:ring-primary-100 outline-none transition-all"
          placeholder="至少4位"
          autoComplete="new-password"
        />
      </div>
      <div>
        <label htmlFor="reg-confirm" className="block text-sm font-medium text-gray-600 mb-1">
          确认密码
        </label>
        <input
          id="reg-confirm"
          type="password"
          value={confirmPassword}
          onChange={(e) => setConfirmPassword(e.target.value)}
          className="w-full px-4 py-2.5 border border-gray-200 rounded-xl focus:border-primary-500 focus:ring-2 focus:ring-primary-100 outline-none transition-all"
          placeholder="再次输入密码"
          autoComplete="new-password"
        />
      </div>
      <button
        type="submit"
        disabled={isLoading || !username.trim() || !password.trim() || !confirmPassword.trim()}
        className="w-full flex items-center justify-center gap-2 py-2.5 bg-primary-500 text-white rounded-xl hover:bg-primary-600 disabled:opacity-50 disabled:cursor-not-allowed transition-all font-medium"
      >
        <UserPlus className="w-4 h-4" />
        {isLoading ? '注册中...' : '注册'}
      </button>
      <p className="text-center text-sm text-gray-500">
        已有账号？{' '}
        <button
          type="button"
          onClick={onSwitchToLogin}
          className="text-primary-500 hover:text-primary-600 font-medium"
        >
          立即登录
        </button>
      </p>
    </form>
  );
}
