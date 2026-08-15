import { useState, useEffect, useRef } from 'react';

interface FlipClockState {
  hours: [string, string];
  minutes: [string, string];
  seconds: [string, string];
  date: string;
  weekday: string;
}

function padTwo(n: number): [string, string] {
  const s = String(n).padStart(2, '0');
  return [s[0], s[1]];
}

function formatDate(d: Date): string {
  return `${d.getFullYear()}年${d.getMonth() + 1}月${d.getDate()}日`;
}

function formatWeekday(d: Date): string {
  const days = ['星期日', '星期一', '星期二', '星期三', '星期四', '星期五', '星期六'];
  return days[d.getDay()];
}

function getCurrentState(): FlipClockState {
  const now = new Date();
  return {
    hours: padTwo(now.getHours()),
    minutes: padTwo(now.getMinutes()),
    seconds: padTwo(now.getSeconds()),
    date: formatDate(now),
    weekday: formatWeekday(now),
  };
}

export function useFlipClock(active: boolean): FlipClockState {
  const [state, setState] = useState<FlipClockState>(getCurrentState);
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  useEffect(() => {
    if (active) {
      setState(getCurrentState());
      intervalRef.current = setInterval(() => {
        setState(getCurrentState());
      }, 1000);
    } else {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
        intervalRef.current = null;
      }
    }
    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
        intervalRef.current = null;
      }
    };
  }, [active]);

  return state;
}
