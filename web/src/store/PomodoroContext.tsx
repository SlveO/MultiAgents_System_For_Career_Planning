import React, { createContext, useContext, useState, useCallback, useRef, useEffect } from 'react';
import { getItem, setItem } from '../utils/storage';
import { formatTime } from '../utils/format';
import type { TimerPhase, TimerMode } from '../types';

const SETTINGS_KEY = 'pomodoro_settings';

const DEFAULT_WORK = 25 * 60;
const DEFAULT_SHORT_BREAK = 5 * 60;
const DEFAULT_LONG_BREAK = 15 * 60;
const DEFAULT_INTERVAL = 4;

interface PomodoroState {
  isRunning: boolean;
  isPaused: boolean;
  currentPhase: TimerPhase;
  timeLeft: number;
  elapsed: number;
  workTime: number;
  shortBreakTime: number;
  longBreakTime: number;
  longBreakInterval: number;
  completedPomodoros: number;
  timerMode: TimerMode;
  display: string;
  phaseLabel: string;
  currentTask: { title: string; description: string } | null;
  // Actions
  setCurrentTask: (task: { title: string; description: string } | null) => void;
  start: () => void;
  pause: () => void;
  reset: () => void;
  skip: () => void;
  updateSettings: (s: Partial<PomodoroSettings>) => void;
}

interface PomodoroSettings {
  workTime: number;
  shortBreakTime: number;
  longBreakTime: number;
  longBreakInterval: number;
  timerMode: TimerMode;
}

const PomodoroContext = createContext<PomodoroState | null>(null);

function loadSettings(): PomodoroSettings {
  return getItem<PomodoroSettings>(SETTINGS_KEY, {
    workTime: DEFAULT_WORK,
    shortBreakTime: DEFAULT_SHORT_BREAK,
    longBreakTime: DEFAULT_LONG_BREAK,
    longBreakInterval: DEFAULT_INTERVAL,
    timerMode: 'countdown',
  });
}

function notify(title: string, body: string) {
  if (Notification.permission === 'granted') {
    new Notification(title, { body });
  }
}

export function PomodoroProvider({ children }: { children: React.ReactNode }) {
  const settings = useRef<PomodoroSettings>(loadSettings());
  const [isRunning, setIsRunning] = useState(false);
  const [isPaused, setIsPaused] = useState(false);
  const [currentPhase, setCurrentPhase] = useState<TimerPhase>('work');
  const [timeLeft, setTimeLeft] = useState(settings.current.workTime);
  const [elapsed, setElapsed] = useState(0);
  const [completedPomodoros, setCompletedPomodoros] = useState(0);
  const [currentTask, setCurrentTask] = useState<{ title: string; description: string } | null>(null);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const clearTimer = useCallback(() => {
    if (timerRef.current) {
      clearInterval(timerRef.current);
      timerRef.current = null;
    }
  }, []);

  const getPhaseTime = useCallback((phase: TimerPhase): number => {
    switch (phase) {
      case 'work': return settings.current.workTime;
      case 'shortBreak': return settings.current.shortBreakTime;
      case 'longBreak': return settings.current.longBreakTime;
    }
  }, []);

  const switchPhase = useCallback(() => {
    setCurrentPhase((prev) => {
      if (prev === 'work') {
        const newCount = completedPomodoros + 1;
        setCompletedPomodoros(newCount);
        const next = newCount % settings.current.longBreakInterval === 0 ? 'longBreak' : 'shortBreak';
        notify('番茄钟', newCount % settings.current.longBreakInterval === 0 ? '完成一组！该长休息了' : '专注完成，休息一下');
        setTimeLeft(getPhaseTime(next));
        setElapsed(0);
        return next;
      }
      notify('番茄钟', '休息结束，开始新的专注吧');
      setTimeLeft(getPhaseTime('work'));
      setElapsed(0);
      return 'work';
    });
  }, [completedPomodoros, getPhaseTime]);

  const tick = useCallback(() => {
    if (settings.current.timerMode === 'countdown') {
      setTimeLeft((prev) => {
        if (prev <= 1) {
          switchPhase();
          return 0;
        }
        return prev - 1;
      });
    } else {
      setElapsed((prev) => prev + 1);
    }
  }, [switchPhase]);

  const start = useCallback(() => {
    if (isRunning && !isPaused) return;
    if (isPaused) {
      setIsPaused(false);
      timerRef.current = setInterval(tick, 1000);
      return;
    }
    if (Notification.permission === 'default') {
      Notification.requestPermission();
    }
    setIsRunning(true);
    setIsPaused(false);
    timerRef.current = setInterval(tick, 1000);
  }, [isRunning, isPaused, tick]);

  const pause = useCallback(() => {
    clearTimer();
    setIsPaused(true);
  }, [clearTimer]);

  const reset = useCallback(() => {
    clearTimer();
    setIsRunning(false);
    setIsPaused(false);
    setCurrentPhase('work');
    setTimeLeft(settings.current.workTime);
    setElapsed(0);
    setCompletedPomodoros(0);
  }, [clearTimer]);

  const skip = useCallback(() => {
    clearTimer();
    switchPhase();
    if (isRunning) {
      setIsRunning(false);
      setIsPaused(false);
    }
  }, [clearTimer, switchPhase, isRunning]);

  const updateSettings = useCallback((s: Partial<PomodoroSettings>) => {
    const updated = { ...settings.current, ...s };
    settings.current = updated;
    setItem(SETTINGS_KEY, updated);
    if (!isRunning) {
      setTimeLeft(updated.workTime);
      setElapsed(0);
    }
  }, [isRunning]);

  // Cleanup on unmount
  useEffect(() => clearTimer, [clearTimer]);

  const phaseLabel =
    currentPhase === 'work' ? '专注' :
    currentPhase === 'shortBreak' ? '短休息' : '长休息';

  const display = settings.current.timerMode === 'countdown'
    ? formatTime(timeLeft)
    : formatTime(elapsed);

  return (
    <PomodoroContext.Provider
      value={{
        isRunning,
        isPaused,
        currentPhase,
        timeLeft,
        elapsed,
        workTime: settings.current.workTime,
        shortBreakTime: settings.current.shortBreakTime,
        longBreakTime: settings.current.longBreakTime,
        longBreakInterval: settings.current.longBreakInterval,
        completedPomodoros,
        timerMode: settings.current.timerMode,
        display,
        phaseLabel,
        currentTask,
        setCurrentTask,
        start,
        pause,
        reset,
        skip,
        updateSettings,
      }}
    >
      {children}
    </PomodoroContext.Provider>
  );
}

export function usePomodoro(): PomodoroState {
  const ctx = useContext(PomodoroContext);
  if (!ctx) throw new Error('usePomodoro must be used within PomodoroProvider');
  return ctx;
}
