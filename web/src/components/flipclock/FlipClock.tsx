import { useRef } from 'react';
import { useFlipClock } from '../../hooks/useFlipClock';
import FlipCard from './FlipCard';

export default function FlipClock() {
  const clock = useFlipClock(true);
  const prevRef = useRef(clock);

  // Determine which digits changed
  const flipped = {
    hours0: clock.hours[0] !== prevRef.current.hours[0],
    hours1: clock.hours[1] !== prevRef.current.hours[1],
    minutes0: clock.minutes[0] !== prevRef.current.minutes[0],
    minutes1: clock.minutes[1] !== prevRef.current.minutes[1],
    seconds0: clock.seconds[0] !== prevRef.current.seconds[0],
    seconds1: clock.seconds[1] !== prevRef.current.seconds[1],
  };

  prevRef.current = clock;

  return (
    <div className="flip-clock-wrapper h-full flex flex-col items-center justify-center bg-[#0a0a0a]">
      {/* Date */}
      <div className="text-white text-right mb-4">
        <p className="text-2xl font-light tracking-wide">{clock.date}</p>
        <p className="text-xl font-light tracking-wide">{clock.weekday}</p>
      </div>

      {/* Clock digits */}
      <div className="flex items-center gap-1.5 mt-8">
        <div className="flex gap-1">
          <FlipCard digit={clock.hours[0]} prevDigit={prevRef.current.hours[0]} flipping={flipped.hours0} />
          <FlipCard digit={clock.hours[1]} prevDigit={prevRef.current.hours[1]} flipping={flipped.hours1} />
        </div>

        <div className="flex flex-col gap-8 px-2">
          <div className="w-4 h-4 rounded-full bg-white" />
          <div className="w-4 h-4 rounded-full bg-white" />
        </div>

        <div className="flex gap-1">
          <FlipCard digit={clock.minutes[0]} prevDigit={prevRef.current.minutes[0]} flipping={flipped.minutes0} />
          <FlipCard digit={clock.minutes[1]} prevDigit={prevRef.current.minutes[1]} flipping={flipped.minutes1} />
        </div>

        <div className="flex flex-col gap-8 px-2">
          <div className="w-4 h-4 rounded-full bg-white" />
          <div className="w-4 h-4 rounded-full bg-white" />
        </div>

        <div className="flex gap-1">
          <FlipCard digit={clock.seconds[0]} prevDigit={prevRef.current.seconds[0]} flipping={flipped.seconds0} />
          <FlipCard digit={clock.seconds[1]} prevDigit={prevRef.current.seconds[1]} flipping={flipped.seconds1} />
        </div>
      </div>
    </div>
  );
}
