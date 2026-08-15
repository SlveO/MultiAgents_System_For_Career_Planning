import React, { useState, useEffect, useRef } from 'react';
import Modal from './Modal';
import { useChat } from '../../store/ChatContext';
import type { InfoModalType } from '../../types';

interface InfoModalProps {
  isOpen: boolean;
  type: InfoModalType;
  onClose: () => void;
}

const CONFIG: Record<InfoModalType, {
  title: string;
  fields: ('major' | 'gpa' | 'additional')[];
  buildQuestion: (values: Record<string, string>) => string;
}> = {
  'major-gpa': {
    title: '专业与成绩信息',
    fields: ['major', 'gpa', 'additional'],
    buildQuestion: (v) =>
      `我适合什么职业？我是${v.major}专业的学生，绩点是${v.gpa}。${v.additional ? `补充信息：${v.additional}` : ''}`,
  },
  'major-only': {
    title: '专业信息',
    fields: ['major'],
    buildQuestion: (v) => `我的专业就业方向？我是${v.major}专业的学生。`,
  },
  'exam-work': {
    title: '考研与就业决策信息',
    fields: ['major', 'gpa', 'additional'],
    buildQuestion: (v) =>
      `考研还是就业？我是${v.major}专业的学生，绩点是${v.gpa}。${v.additional ? `补充信息：${v.additional}` : ''}`,
  },
};

export default function InfoModal({ isOpen, type, onClose }: InfoModalProps) {
  const { sendMessage } = useChat();
  const [major, setMajor] = useState('');
  const [gpa, setGpa] = useState('');
  const [additional, setAdditional] = useState('');
  const majorRef = useRef<HTMLInputElement>(null);

  const config = CONFIG[type];

  useEffect(() => {
    if (isOpen) {
      setMajor('');
      setGpa('');
      setAdditional('');
    }
  }, [isOpen, type]);

  useEffect(() => {
    if (isOpen && config.fields.includes('major')) {
      setTimeout(() => majorRef.current?.focus(), 100);
    }
  }, [isOpen, config.fields]);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (config.fields.includes('major') && !major.trim()) return;
    const question = config.buildQuestion({ major: major.trim(), gpa: gpa.trim(), additional: additional.trim() });
    onClose();
    sendMessage(question);
  };

  return (
    <Modal isOpen={isOpen} onClose={onClose} title={config.title}>
      <form onSubmit={handleSubmit} className="space-y-4 mt-2">
        {config.fields.includes('major') && (
          <div>
            <label htmlFor="info-major" className="block text-sm font-medium text-gray-600 mb-1">
              专业
            </label>
            <input
              ref={majorRef}
              id="info-major"
              type="text"
              value={major}
              onChange={(e) => setMajor(e.target.value)}
              className="w-full px-4 py-2.5 border border-gray-200 rounded-xl focus:border-primary-500 focus:ring-2 focus:ring-primary-100 outline-none transition-all text-sm"
              placeholder="例如：计算机科学与技术"
              required
            />
          </div>
        )}
        {config.fields.includes('gpa') && (
          <div>
            <label htmlFor="info-gpa" className="block text-sm font-medium text-gray-600 mb-1">
              绩点
            </label>
            <input
              id="info-gpa"
              type="number"
              min="0"
              max="4"
              step="0.1"
              value={gpa}
              onChange={(e) => setGpa(e.target.value)}
              className="w-full px-4 py-2.5 border border-gray-200 rounded-xl focus:border-primary-500 focus:ring-2 focus:ring-primary-100 outline-none transition-all text-sm"
              placeholder="例如：3.5"
            />
          </div>
        )}
        {config.fields.includes('additional') && (
          <div>
            <label htmlFor="info-additional" className="block text-sm font-medium text-gray-600 mb-1">
              补充信息
            </label>
            <textarea
              id="info-additional"
              value={additional}
              onChange={(e) => setAdditional(e.target.value)}
              className="w-full px-4 py-2.5 border border-gray-200 rounded-xl focus:border-primary-500 focus:ring-2 focus:ring-primary-100 outline-none transition-all text-sm resize-none"
              rows={3}
              placeholder="其他你想补充的信息（选填）"
            />
          </div>
        )}
        <div className="flex gap-3 pt-2">
          <button
            type="button"
            onClick={onClose}
            className="flex-1 py-2.5 border border-gray-200 rounded-xl text-sm text-gray-600 hover:bg-gray-50 transition-colors"
          >
            取消
          </button>
          <button
            type="submit"
            disabled={config.fields.includes('major') && !major.trim()}
            className="flex-1 py-2.5 bg-primary-500 text-white rounded-xl text-sm font-medium hover:bg-primary-600 disabled:opacity-50 transition-all"
          >
            发送问题
          </button>
        </div>
      </form>
    </Modal>
  );
}
