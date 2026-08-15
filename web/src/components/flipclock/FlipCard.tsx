interface FlipCardProps {
  digit: string;
  prevDigit: string;
  flipping: boolean;
}

export default function FlipCard({ digit, prevDigit, flipping }: FlipCardProps) {
  return (
    <div className={`flip-card ${flipping ? 'flipping' : ''}`}>
      <div className="flip-card-inner">
        {/* Static top half — shows current */}
        <div className="flip-card-top">
          <span>{digit}</span>
        </div>
        {/* Static bottom half — shows current */}
        <div className="flip-card-bottom">
          <span>{digit}</span>
        </div>
        {/* Animated top half — flips away with old digit */}
        {flipping && (
          <div className="flip-top-active">
            <span>{prevDigit}</span>
          </div>
        )}
        {/* Animated bottom half — flips in with new digit */}
        {flipping && (
          <div className="flip-bottom-active">
            <span>{digit}</span>
          </div>
        )}
      </div>
    </div>
  );
}
