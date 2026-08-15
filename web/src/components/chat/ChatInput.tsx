import React, { useState, useRef, useEffect } from 'react';
import { Send } from 'lucide-react';
import { useChat } from '../../store/ChatContext';
import { uploadFile } from '../../api/upload';
import FileAttachment from './FileAttachment';

export default function ChatInput() {
  const { sendMessage, isLoading } = useChat();
  const [input, setInput] = useState('');
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (!isLoading) {
      inputRef.current?.focus();
    }
  }, [isLoading]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    const text = input.trim();
    if (!text || isLoading) return;

    let message = text;

    // If file attached, upload it first and append path
    if (selectedFile) {
      try {
        const result = await uploadFile(selectedFile);
        message = `${text}\n[附件: ${result.file_name}]\n${result.file_path}`;
        setUploadError(null);
      } catch (err) {
        const detail = err instanceof Error ? err.message : '未知错误';
        setUploadError(`文件上传失败: ${detail}`);
        setSelectedFile(null);
        return; // Stop — don't send message without file
      }
      setSelectedFile(null);
    }

    setInput('');
    setUploadError(null);
    await sendMessage(message);
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  };

  return (
    <form onSubmit={handleSubmit} className="border-t border-gray-100 bg-white px-4 py-3">
      {uploadError && (
        <div className="max-w-3xl mx-auto mb-2 px-3 py-2 bg-red-50 border border-red-200 rounded-lg text-sm text-red-600">
          {uploadError}
        </div>
      )}
      <div className="flex items-end gap-2 max-w-3xl mx-auto">
        <FileAttachment selectedFile={selectedFile} onFileChange={setSelectedFile} />
        <div className="flex-1 relative">
          <input
            ref={inputRef}
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="输入你的职业规划问题..."
            className="w-full px-4 py-2.5 border border-gray-200 rounded-xl focus:border-primary-500 focus:ring-2 focus:ring-primary-100 outline-none transition-all text-sm"
            disabled={isLoading}
            autoComplete="off"
          />
        </div>
        <button
          type="submit"
          disabled={isLoading || !input.trim()}
          className="p-2.5 bg-primary-500 text-white rounded-xl hover:bg-primary-600 hover:-translate-y-0.5 hover:shadow-md disabled:opacity-50 disabled:hover:translate-y-0 disabled:hover:shadow-none transition-all"
          title="发送"
        >
          <Send className="w-4 h-4" />
        </button>
      </div>
    </form>
  );
}
