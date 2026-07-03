import './Card.css';

// Renders a single card from its 2-3 char string (e.g. 'H5', 'ST', 'SB',
// 'HR'). Interaction (selection) is handled by the parent wrapper, so this
// component is purely presentational.
const SUIT_MAP = {
  H: { symbol: '♥', color: 'red' },
  S: { symbol: '♠', color: 'black' },
  C: { symbol: '♣', color: 'black' },
  D: { symbol: '♦', color: 'red' },
};

// Jester-hat icon for the jokers. `fill="currentColor"` makes it follow the
// card's text colour, so the small joker (card-black) is black and the big
// joker (card-red) is coloured — no Chinese label needed.
const JokerIcon = ({ className }) => (
  <svg className={className} viewBox="0 0 64 64" fill="currentColor" aria-hidden="true">
    <path d="M32 13 L43 47 H21 Z" />
    <path d="M14 19 L27 47 H4 Z" />
    <path d="M50 19 L60 47 H37 Z" />
    <rect x="9" y="45" width="46" height="10" rx="5" />
    <circle cx="32" cy="9" r="5.5" />
    <circle cx="11" cy="15" r="4.5" />
    <circle cx="53" cy="15" r="4.5" />
  </svg>
);

const Card = ({ cardString, size = 'normal' }) => {
  if (!cardString || cardString.length < 2) {
    return (
      <div className={`card card-${size} card-empty`}>
        <div className="card-content">?</div>
      </div>
    );
  }

  const suit = cardString[0];
  const rank = cardString.slice(1);
  const isJoker = rank === 'B' || rank === 'R';
  const suitInfo = isJoker ? null : SUIT_MAP[suit];
  const displayRank = rank === 'T' ? '10' : rank;

  const cardClasses = [
    'card',
    `card-${size}`,
    suitInfo ? `card-${suitInfo.color}` : 'card-joker',
  ].join(' ');

  if (isJoker) {
    // 小王 (SB) -> black icon, 大王 (HR) -> coloured (red) icon.
    const isSmall = rank === 'B';
    return (
      <div className={`${cardClasses} card-joker ${isSmall ? 'card-black' : 'card-red'}`}>
        <div className="card-content">
          <JokerIcon className="joker-corner-icon" />
        </div>
        <JokerIcon className="joker-center-icon" />
      </div>
    );
  }

  return (
    <div className={cardClasses}>
      <div className="card-content">
        <div className="card-rank">{displayRank}</div>
        <div className="card-suit">{suitInfo.symbol}</div>
      </div>
      <div className="card-body-suit">{suitInfo.symbol}</div>
    </div>
  );
};

export default Card;
