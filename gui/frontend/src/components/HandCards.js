import Card from './Card';
import './HandCards.css';

// Card width and the widest the fan is allowed to get. The horizontal step
// between cards shrinks as the hand grows, but never below MIN_STEP so the
// rank in each card's top-left corner is always visible (the step stays
// wider than that number).
const CARD_WIDTH = 56;
const MAX_FAN_WIDTH = 760;
const MAX_STEP = 40; // most spacing (fewest cards)
const MIN_STEP = 28; // tightest spacing (many cards); > rank-number width

const HandCards = ({ cards, selectedCards, onCardSelect, isInteractive }) => {
  if (!cards || cards.length === 0) {
    return (
      <div className="hand-cards-container">
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
              className={`hand-card ${isSelected ? 'selected' : ''} ${isInteractive ? 'interactive' : ''}`}
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
        <p>
          {selectedCards.length > 0
            ? `已选择 ${selectedCards.length} 张`
            : '💡 点击卡牌选择 / 取消，凑齐有效牌型后点“出牌”。'}
        </p>
      </div>
    </div>
  );
};

export default HandCards;
