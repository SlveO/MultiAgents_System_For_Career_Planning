export default function TypingIndicator() {
  return (
    <div className="flex justify-start mb-3">
      <div className="bg-white rounded-[18px] rounded-bl-[4px] px-5 py-3 shadow-sm">
        <div className="typing-indicator flex items-center gap-1">
          <span className="inline-block w-2 h-2 bg-primary-400 rounded-full animate-bounce" style={{ animationDelay: '-0.32s' }} />
          <span className="inline-block w-2 h-2 bg-primary-400 rounded-full animate-bounce" style={{ animationDelay: '-0.16s' }} />
          <span className="inline-block w-2 h-2 bg-primary-400 rounded-full animate-bounce" style={{ animationDelay: '0s' }} />
        </div>
      </div>
    </div>
  );
}
