import { useState } from 'react';
import './GameStatus.css';
import {
  formatRecentPlays,
  formatRemainingCounts,
  hasUrgentRoundStatus,
} from '../utils/statusDock';

// Compact, corner-pinned table dock. It stays out of the player zones by
// default and opens into utility information instead of duplicating the
// center table's current-turn display.
const GameStatus = ({ gameState, currentPlayer, onRestart }) => {
  const [open, setOpen] = useState(false);

  if (!gameState) return null;

  const recentPlays = formatRecentPlays(gameState.recent_plays).slice(-4);
  const remainingCounts = formatRemainingCounts(gameState.num_cards_left);
  const urgent = hasUrgentRoundStatus(gameState);

  return (
    <div className={`game-status ${open ? 'open' : ''}`}>
      <button
        type="button"
        className="status-dock-button"
        onClick={() => setOpen((o) => !o)}
        aria-expanded={open}
        aria-label={`牌局工具面板，当前玩家${currentPlayer}`}
      >
        <span className="dock-caret">{open ? '▾' : '▸'}</span>
        <span>牌局</span>
        {urgent && (
          <span className={`dock-alert ${gameState.is_over ? 'game-over' : ''}`}>
            {gameState.is_over ? '结束' : '换轮'}
          </span>
        )}
      </button>

      {open && (
        <div className="status-details">
          <section className="status-section">
            <div className="status-section-title">最近出牌</div>
            <div className="recent-play-list">
              {recentPlays.length ? recentPlays.map((play) => (
                <div key={play.playerId} className={`recent-play-row ${play.pass ? 'pass' : ''}`}>
                  <span>{play.label}</span>
                  <strong>{play.text}</strong>
                </div>
              )) : (
                <div className="status-empty">暂无出牌</div>
              )}
            </div>
          </section>

          <section className="status-section">
            <div className="status-section-title">余牌</div>
            <div className="remaining-grid">
              {remainingCounts.map((item) => (
                <div key={item.playerId} className={`remaining-cell ${item.danger ? 'danger' : ''}`}>
                  <span>{item.label}</span>
                  <strong>{item.count}</strong>
                </div>
              ))}
            </div>
          </section>

          <button type="button" className="btn btn-danger status-restart" onClick={onRestart}>
            重新开始
          </button>
        </div>
      )}
    </div>
  );
};

export default GameStatus;
