import { useChat } from '../../store/ChatContext';
import { Plus, PanelLeftClose, PanelLeft, Trash2 } from 'lucide-react';
import { truncateText } from '../../utils/format';

export default function HistorySidebar() {
  const {
    sessions,
    currentSession,
    historyVisible,
    createSession,
    switchSession,
    deleteSession,
    toggleHistory,
  } = useChat();

  if (!historyVisible) {
    return (
      <button
        onClick={toggleHistory}
        className="p-2 m-2 rounded-lg hover:bg-gray-100 transition-colors text-gray-400"
        title="展开历史对话"
      >
        <PanelLeft className="w-5 h-5" />
      </button>
    );
  }

  return (
    <div className="w-60 bg-white border-r border-gray-100 flex flex-col h-full shrink-0">
      <div className="flex items-center justify-between px-3 py-3 border-b border-gray-50">
        <button
          onClick={createSession}
          className="flex items-center gap-1.5 text-xs font-medium text-primary-600 hover:bg-primary-50 px-2 py-1.5 rounded-lg transition-colors"
        >
          <Plus className="w-3.5 h-3.5" />
          新对话
        </button>
        <button
          onClick={toggleHistory}
          className="p-1.5 rounded-lg hover:bg-gray-100 transition-colors text-gray-400"
          title="收起侧边栏"
        >
          <PanelLeftClose className="w-4 h-4" />
        </button>
      </div>
      <div className="flex-1 overflow-y-auto px-2 py-2 space-y-1">
        {sessions.map((session) => (
          <div
            key={session.id}
            onClick={() => switchSession(session.id)}
            className={`history-item group flex items-center justify-between px-3 py-2.5 rounded-xl cursor-pointer transition-all ${
              session.id === currentSession?.id
                ? 'bg-primary-50 border-l-3 border-l-primary-500'
                : 'hover:bg-gray-50'
            }`}
          >
            <div className="flex-1 min-w-0">
              <p className="text-sm text-gray-700 truncate">
                {truncateText(session.title, 12)}
              </p>
              <p className="text-xs text-gray-400 mt-0.5">
                {session.messages.length} 条消息
              </p>
            </div>
            <button
              onClick={(e) => {
                e.stopPropagation();
                deleteSession(session.id);
              }}
              className="opacity-0 group-hover:opacity-100 p-1 rounded hover:bg-red-50 text-gray-400 hover:text-red-500 transition-all"
              title="删除对话"
            >
              <Trash2 className="w-3.5 h-3.5" />
            </button>
          </div>
        ))}
      </div>
    </div>
  );
}
