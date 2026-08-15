import { useEffect, useRef } from 'react';
import { useChat } from '../../store/ChatContext';
import MessageBubble from './MessageBubble';
import TypingIndicator from './TypingIndicator';

export default function ChatContainer() {
  const { currentSession, isStreaming } = useChat();
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [currentSession?.messages, isStreaming]);

  const messages = currentSession?.messages || [];

  return (
    <div className="flex-1 overflow-y-auto px-4 py-4 space-y-1" style={{ scrollBehavior: 'smooth' }}>
      {messages.length === 0 && (
        <div className="flex items-center justify-center h-full">
          <p className="text-gray-400 text-sm">开始你的职业规划之旅吧</p>
        </div>
      )}
      {messages.map((msg, i) => (
        <MessageBubble key={i} message={msg} />
      ))}
      {isStreaming && messages[messages.length - 1]?.role === 'ai' && messages[messages.length - 1]?.content.startsWith('▊') && (
        <TypingIndicator />
      )}
      <div ref={bottomRef} />
    </div>
  );
}
