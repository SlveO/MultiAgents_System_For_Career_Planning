import { GraduationCap, LogOut } from 'lucide-react';
import { useAuth } from '../../store/AuthContext';

export default function Header() {
  const { user, logout } = useAuth();

  return (
    <header className="bg-white shadow-sm border-b border-gray-100">
      <div className="max-w-7xl mx-auto px-4 py-3 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <GraduationCap className="w-6 h-6 text-primary-500" />
          <h1 className="text-lg font-semibold text-gray-800">大学生职业规划助手</h1>
        </div>
        {user && (
          <div className="flex items-center gap-3">
            <span className="text-sm text-gray-500">{user.username}</span>
            <button
              onClick={logout}
              className="flex items-center gap-1 text-sm text-gray-400 hover:text-red-500 transition-colors"
              title="退出登录"
            >
              <LogOut className="w-4 h-4" />
            </button>
          </div>
        )}
      </div>
    </header>
  );
}
