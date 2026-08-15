import { useState } from 'react';
import Modal from './Modal';
import { Play, Pause, RotateCcw, SkipForward, Settings2 } from 'lucide-react';
import { usePomodoro } from '../../store/PomodoroContext';

interface PomodoroModalProps {
  isOpen: boolean;
  onClose: () => void;
}

export default function PomodoroModal({ isOpen, onClose }: PomodoroModalProps) {
  const [showSettings, setShowSettings] = useState(false);
  const pom = usePomodoro();

  const phaseColor =
    pom.currentPhase === 'work' ? 'text-red-500' :
    pom.currentPhase === 'shortBreak' ? 'text-green-500' : 'text-blue-500';

  const phaseBg =
    pom.currentPhase === 'work' ? 'bg-red-50' :
    pom.currentPhase === 'shortBreak' ? 'bg-green-50' : 'bg-blue-50';

  return (
    <Modal isOpen={isOpen} onClose={onClose} title="番茄钟">
      <div className="flex flex-col items-center space-y-6 mt-4">
        {pom.currentTask && (
          <div className="w-full bg-gray-50 rounded-xl p-3 text-center">
            <p className="text-xs text-gray-400">当前任务</p>
            <p className="text-sm font-medium text-gray-700">{pom.currentTask.title}</p>
          </div>
        )}

        <div className={`${phaseBg} rounded-3xl px-10 py-8`}>
          <p className={`text-xs font-medium uppercase tracking-wider text-center mb-2 ${phaseColor}`}>
            {pom.phaseLabel}
          </p>
          <p className="text-6xl font-mono font-bold text-gray-800 tracking-wider tabular-nums">
            {pom.display}
          </p>
          <p className="text-xs text-gray-400 text-center mt-2">
            完成 {pom.completedPomodoros} 个番茄
          </p>
        </div>

        <div className="flex items-center gap-3">
          <button
            onClick={pom.reset}
            className="p-2.5 rounded-full hover:bg-gray-100 transition-colors text-gray-400"
            title="重置"
          >
            <RotateCcw className="w-5 h-5" />
          </button>
          <button
            onClick={pom.isRunning && !pom.isPaused ? pom.pause : pom.start}
            className="p-4 bg-primary-500 text-white rounded-full hover:bg-primary-600 hover:shadow-lg transition-all"
            title={pom.isRunning && !pom.isPaused ? '暂停' : '开始'}
          >
            {pom.isRunning && !pom.isPaused ? (
              <Pause className="w-6 h-6" />
            ) : (
              <Play className="w-6 h-6 ml-0.5" />
            )}
          </button>
          <button
            onClick={pom.skip}
            className="p-2.5 rounded-full hover:bg-gray-100 transition-colors text-gray-400"
            title="跳过"
          >
            <SkipForward className="w-5 h-5" />
          </button>
        </div>

        <button
          onClick={() => setShowSettings(!showSettings)}
          className="flex items-center gap-1.5 text-xs text-gray-400 hover:text-gray-600 transition-colors"
        >
          <Settings2 className="w-3.5 h-3.5" />
          设置
        </button>

        {showSettings && (
          <div className="w-full bg-gray-50 rounded-xl p-4 space-y-3">
            <div className="grid grid-cols-2 gap-3">
              <div>
                <label className="block text-xs text-gray-500 mb-1">工作时长 (分钟)</label>
                <input
                  type="number"
                  value={pom.workTime / 60}
                  onChange={(e) => pom.updateSettings({ workTime: Math.max(1, +e.target.value) * 60 })}
                  className="w-full px-3 py-1.5 border border-gray-200 rounded-lg text-sm focus:border-primary-500 outline-none"
                  min={1}
                />
              </div>
              <div>
                <label className="block text-xs text-gray-500 mb-1">短休息 (分钟)</label>
                <input
                  type="number"
                  value={pom.shortBreakTime / 60}
                  onChange={(e) => pom.updateSettings({ shortBreakTime: Math.max(1, +e.target.value) * 60 })}
                  className="w-full px-3 py-1.5 border border-gray-200 rounded-lg text-sm focus:border-primary-500 outline-none"
                  min={1}
                />
              </div>
              <div>
                <label className="block text-xs text-gray-500 mb-1">长休息 (分钟)</label>
                <input
                  type="number"
                  value={pom.longBreakTime / 60}
                  onChange={(e) => pom.updateSettings({ longBreakTime: Math.max(1, +e.target.value) * 60 })}
                  className="w-full px-3 py-1.5 border border-gray-200 rounded-lg text-sm focus:border-primary-500 outline-none"
                  min={1}
                />
              </div>
              <div>
                <label className="block text-xs text-gray-500 mb-1">长休息间隔</label>
                <input
                  type="number"
                  value={pom.longBreakInterval}
                  onChange={(e) => pom.updateSettings({ longBreakInterval: Math.max(1, +e.target.value) })}
                  className="w-full px-3 py-1.5 border border-gray-200 rounded-lg text-sm focus:border-primary-500 outline-none"
                  min={1}
                />
              </div>
            </div>
            <div>
              <label className="block text-xs text-gray-500 mb-1">计时模式</label>
              <select
                value={pom.timerMode}
                onChange={(e) => pom.updateSettings({ timerMode: e.target.value as 'countdown' | 'countup' })}
                className="w-full px-3 py-1.5 border border-gray-200 rounded-lg text-sm focus:border-primary-500 outline-none"
              >
                <option value="countdown">倒计时</option>
                <option value="countup">正计时</option>
              </select>
            </div>
          </div>
        )}
      </div>
    </Modal>
  );
}
