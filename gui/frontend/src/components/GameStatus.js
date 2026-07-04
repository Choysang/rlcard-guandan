import { useState } from 'react';
import './GameStatus.css';
import {
  formatPlayHistory,
  formatRecentPlays,
  hasUrgentRoundStatus,
} from '../utils/statusDock';

// Compact, corner-pinned table dock. It stays out of the player zones by
// default and opens into utility information instead of duplicating the
// center table's current-turn display.
const GameStatus = ({ gameState, currentPlayer, onRestart }) => {
  const [open, setOpen] = useState(false);

  if (!gameState) return null;

  const playHistory = formatPlayHistory(gameState.play_history).slice().reverse();
  const recentPlays = playHistory.length
    ? playHistory
    : formatRecentPlays(gameState.recent_plays).slice(-4).reverse();
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
            <div className="recent-play-list" role="log" aria-label="本局出牌记录">
              {recentPlays.length ? recentPlays.map((play) => (
                <div key={play.id ?? play.playerId} className={`recent-play-row ${play.pass ? 'pass' : ''}`}>
                  <span className="play-order">{play.order ? `#${play.order}` : ''}</span>
                  <span>{play.label}</span>
                  <strong>{play.text}</strong>
                </div>
              )) : (
                <div className="status-empty">暂无出牌</div>
              )}
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
