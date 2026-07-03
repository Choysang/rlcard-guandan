import { useRef } from 'react';
import Card from './Card';
import './HandCards.css';
import { applyCardSelection, dragModeForIndex } from '../utils/cardSelection';

// Card width and the widest the fan is allowed to get. The horizontal step
// between cards shrinks as the hand grows, but never below MIN_STEP so the
// rank in each card's top-left corner is always visible (the step stays
// wider than that number).
const CARD_WIDTH = 56;
const MAX_FAN_WIDTH = 760;
const MAX_STEP = 40; // most spacing (fewest cards)
const MIN_STEP = 28; // tightest spacing (many cards); > rank-number width

const HandCards = ({ cards, selectedCards, setSelected, isInteractive }) => {
  // Drag-to-select state: the mode (select/deselect) is fixed on press from
  // the first card, then applied to every card the pointer sweeps over.
  const dragModeRef = useRef(null);
  const processedRef = useRef(new Set());

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

  const applyCard = (index, mode) => {
    setSelected((prev) => applyCardSelection(prev, index, mode));
  };

  const cardIndexAtPoint = (x, y) => {
    const el = document.elementFromPoint(x, y);
    const slot = el && el.closest('[data-card-index]');
    return slot ? Number(slot.dataset.cardIndex) : null;
  };

  const handlePointerDown = (e) => {
    if (!isInteractive) return;
    const index = cardIndexAtPoint(e.clientX, e.clientY);
    if (index == null) return;
    e.preventDefault();
    // Capture so move/up keep firing on this element even if the pointer
    // slides off the edge of a card.
    try { e.currentTarget.setPointerCapture(e.pointerId); } catch { /* noop */ }
    const mode = dragModeForIndex(selectedCards, index);
    dragModeRef.current = mode;
    processedRef.current = new Set([index]);
    applyCard(index, mode);
  };

  const handlePointerMove = (e) => {
    if (!dragModeRef.current) return;
    const index = cardIndexAtPoint(e.clientX, e.clientY);
    if (index == null || processedRef.current.has(index)) return;
    processedRef.current.add(index);
    applyCard(index, dragModeRef.current);
  };

  const endDrag = () => {
    dragModeRef.current = null;
    processedRef.current = new Set();
  };

  return (
    <div className="hand-cards-container">
      <div
        className="hand-cards"
        data-card-count={count}
        style={{ width: `${totalWidth}px` }}
        onPointerDown={handlePointerDown}
        onPointerMove={handlePointerMove}
        onPointerUp={endDrag}
        onPointerCancel={endDrag}
      >
        {cards.map((cardString, index) => {
          const isSelected = selectedCards.includes(index);
          const leftPosition = index * step;
          return (
            <div
              key={`${cardString}-${index}`}
              data-card-index={index}
              className={`hand-card ${isSelected ? 'selected' : ''} ${isInteractive ? 'interactive' : ''}`}
              style={{
                left: `${leftPosition}px`,
                top: isSelected ? '-22px' : '0px',
                zIndex: isSelected ? 1000 + index : index,
              }}
              role="button"
              tabIndex={isInteractive ? 0 : -1}
              aria-pressed={isSelected}
              aria-label={`${cardString}${isSelected ? '（已选）' : ''}`}
              onKeyDown={(e) => {
                if (isInteractive && (e.key === 'Enter' || e.key === ' ')) {
                  e.preventDefault();
                  applyCard(index, isSelected ? 'deselect' : 'select');
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
            : '💡 点击或按住滑动选牌，凑齐有效牌型后点“出牌”。'}
        </p>
      </div>
    </div>
  );
};

export default HandCards;
