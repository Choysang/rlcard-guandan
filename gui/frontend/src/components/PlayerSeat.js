import Card from './Card';
import './PlayerSeat.css';
import { actionCards } from '../utils/actionCards';

const actionLabel = (action) => {
  if (!action) return '';
  if (action[0] === 'PASS') return '不出';
  if (action[0] === 'Bomb') return `${actionCards(action).length}张炸弹`;
  const labels = {
    Single: '单牌',
    Pair: '对子',
    Trips: '三张',
    ThreeWithTwo: '三带二',
    ThreePair: '三连对',
    TwoTrips: '钢板',
    Straight: '顺子',
    StraightFlush: '同花顺',
  };
  return labels[action[0]] || action[0];
};

const renderActionCards = (action) => {
  if (!action) return null;
  if (action[0] === 'PASS') {
    return <span className="seat-pass-chip">不出</span>;
  }
  const cards = actionCards(action);
  return (
    <div className="seat-play-cards">
      {cards.map((card, index) => (
        <Card key={`${card}-${index}`} cardString={card} size="small" />
      ))}
    </div>
  );
};

const PlayerSeat = ({
  position,
  playerId,
  playerName,
  cardCount,
  isActive,
  isHuman,
  latestAction,
  debugHand,
  showMeta = true,
  showCardBacks = true,
}) => {
  const backCount = Math.min(cardCount || 0, position === 'top' ? 13 : 9);

  return (
    <section className={`player-seat player-seat-${position} ${isActive ? 'is-active' : ''}`}>
      {showMeta && (
        <div className="seat-meta">
          <div className="seat-avatar">{isHuman ? '人' : 'AI'}</div>
          <div>
            <div className="seat-name">{playerName}</div>
            <div className="seat-count">{cardCount || 0} 张</div>
          </div>
        </div>
      )}

      {showCardBacks && !debugHand && (
        <div className={`seat-card-backs ${position === 'top' ? 'horizontal' : 'vertical'}`}>
          {Array.from({ length: backCount }).map((_, index) => (
            <div key={`${playerId}-back-${index}`} className="seat-card-back" />
          ))}
        </div>
      )}

      <div className="seat-latest-play">
        {latestAction && latestAction[0] !== 'PASS' && (
          <span className="seat-action-label">{actionLabel(latestAction)}</span>
        )}
        {renderActionCards(latestAction)}
      </div>

      {debugHand && (
        <div className="seat-debug-hand">
          {debugHand.map((card, index) => (
            <Card key={`debug-${playerId}-${card}-${index}`} cardString={card} size="small" />
          ))}
        </div>
      )}
    </section>
  );
};

export default PlayerSeat;
