import Card from './Card';
import './HandCards.css';

// Card width and the widest the fan is allowed to get. The horizontal step
// between cards shrinks as the hand grows so a full 27-card opening hand
// still fits without overflowing the table.
const CARD_WIDTH = 72;
const MAX_FAN_WIDTH = 820;
const MAX_STEP = 44; // most spacing (fewest cards)
const MIN_STEP = 20; // tightest spacing (many cards)

const HandCards = ({ cards, selectedCards, onCardSelect, isInteractive }) => {
  if (!cards || cards.length === 0) {
    return (
      <div className="hand-cards-container">
        <div className="hand-cards-header">
          <h3>我的手牌</h3>
          <span className="card-count">0 张</span>
        </div>
        <div className="hand-cards empty">
          <p>没有手牌</p>
        </div>
      </div>
    );
  }

  const count = cards.length;
  const fittedStep = count > 1 ? (MAX_FAN_WIDTH - CARD_WIDTH) / (count - 1) : MAX_STEP;
  const step = Math.max(MIN_STEP, Math.min(MAX_STEP, fittedStep));
  const totalWidth = (count - 1) * step + CARD_WIDTH;

  return (
    <div className="hand-cards-container">
      <div className="hand-cards-header">
        <h3>我的手牌</h3>
        <span className="card-count">{count} 张</span>
        {selectedCards.length > 0 && (
          <span className="selected-count">已选择 {selectedCards.length} 张</span>
        )}
      </div>

      <div
        className="hand-cards"
        data-card-count={count}
        style={{ width: `${totalWidth}px` }}
      >
        {cards.map((cardString, index) => {
          const isSelected = selectedCards.includes(index);
          const leftPosition = index * step;

          const handleSelect = () => isInteractive && onCardSelect(index);
          return (
            <div
              key={`${cardString}-${index}`}
              className={`card-container ${isSelected ? 'selected' : ''} ${isInteractive ? 'interactive' : ''}`}
              style={{
                left: `${leftPosition}px`,
                top: isSelected ? '-20px' : '0px',
                zIndex: isSelected ? 1000 + index : index,
              }}
              role="button"
              tabIndex={isInteractive ? 0 : -1}
              aria-pressed={isSelected}
              aria-label={`${cardString}${isSelected ? '（已选）' : ''}`}
              onClick={handleSelect}
              onKeyDown={(e) => {
                if (e.key === 'Enter' || e.key === ' ') {
                  e.preventDefault();
                  handleSelect();
                }
              }}
            >
              <Card cardString={cardString} isSelected={isSelected} />
            </div>
          );
        })}
      </div>

      <div className="hand-cards-tip">
        <p>💡 点击卡牌选择 / 取消，凑齐有效牌型后点“出牌”。</p>
      </div>
    </div>
  );
};

export default HandCards;
