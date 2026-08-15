import type { ChatMessage } from '../../types';

interface MessageBubbleProps {
  message: ChatMessage;
}

export default function MessageBubble({ message }: MessageBubbleProps) {
  const isUser = message.role === 'user';

  return (
    <div className={`flex ${isUser ? 'justify-end' : 'justify-start'} mb-3`}>
      <div
        className={`message-bubble ${
          isUser
            ? 'bg-gradient-to-br from-primary-500 to-primary-600 text-white rounded-[18px] rounded-br-[4px] ml-auto'
            : 'bg-white text-gray-800 rounded-[18px] rounded-bl-[4px] mr-auto'
        } px-4 py-3 max-w-[75%] shadow-sm hover:-translate-y-0.5 hover:shadow-md transition-all leading-relaxed text-sm whitespace-pre-wrap`}
      >
        {message.content || (
          isUser ? null : <span className="text-gray-400 italic">思考中...</span>
        )}
      </div>
    </div>
  );
}
