import { MessageCircle, CheckSquare } from 'lucide-react';
import type { PageTab } from '../../types';

interface NavbarProps {
  activeTab: PageTab;
  onNavigate: (tab: PageTab) => void;
}

export default function Navbar({ activeTab, onNavigate }: NavbarProps) {
  return (
    <nav className="bg-white border-b border-gray-100">
      <div className="max-w-7xl mx-auto px-4 flex gap-1">
        <button
          onClick={() => onNavigate('career')}
          className={`nav-button flex items-center gap-2 px-5 py-3 text-sm font-medium rounded-t-lg transition-all ${
            activeTab === 'career'
              ? 'text-primary-600 bg-primary-50/50 border-b-2 border-primary-500'
              : 'text-gray-500 hover:text-gray-700 hover:bg-gray-50'
          }`}
        >
          <MessageCircle className="w-4 h-4" />
          对话助手
        </button>
        <button
          onClick={() => onNavigate('todo')}
          className={`nav-button flex items-center gap-2 px-5 py-3 text-sm font-medium rounded-t-lg transition-all ${
            activeTab === 'todo'
              ? 'text-primary-600 bg-primary-50/50 border-b-2 border-primary-500'
              : 'text-gray-500 hover:text-gray-700 hover:bg-gray-50'
          }`}
        >
          <CheckSquare className="w-4 h-4" />
          今日待办
        </button>
      </div>
    </nav>
  );
}
