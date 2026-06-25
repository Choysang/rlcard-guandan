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
    const isSmall = rank === 'B';
    return (
      <div className={`${cardClasses} ${isSmall ? 'card-black' : 'card-red'}`}>
        <div className="card-content">
          <div className="card-rank">{isSmall ? 'S' : 'B'}</div>
          <div className="card-suit">JOKER</div>
          <div className="card-body-suit">{isSmall ? '小王' : '大王'}</div>
        </div>
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
