import { useState } from 'react';
import './GameStatus.css';

const RANK_NAMES = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A'];

// Slim, collapsible status bar pinned to the top of the board. Collapsed by
// default; the details drop down as an overlay so opening it never pushes
// the table around.
const GameStatus = ({ gameState, currentPlayer, onRestart }) => {
  const [open, setOpen] = useState(false);

  if (!gameState) return null;

  const rank = gameState.current_rank || 0;
  const rankName = RANK_NAMES[rank] || '?';
  const progress = Number.isFinite(gameState.trace_length)
    ? gameState.trace_length
    : Array.isArray(gameState.trace) ? gameState.trace.length : 0;

  return (
    <div className={`game-status ${open ? 'open' : ''}`}>
      <div className="status-bar">
        <button
          type="button"
          className="status-toggle"
          onClick={() => setOpen((o) => !o)}
          aria-expanded={open}
        >
          <span className="caret">{open ? '▾' : '▸'}</span>
          游戏状态
        </button>

        <div className="status-quick">
          <span className="chip">级别 {rankName}</span>
          <span className="chip">当前 玩家{currentPlayer}</span>
          <span className="chip">出牌 {progress}</span>
          {gameState.is_over && <span className="chip chip-over">已结束</span>}
          {gameState.round_completed && !gameState.is_over && (
            <span className="chip chip-round">本轮结束</span>
          )}
        </div>

        <button type="button" className="btn btn-danger status-restart" onClick={onRestart}>
          重新开始
        </button>
      </div>

      {open && (
        <div className="status-details">
          <div className="detail-row"><span>当前级别</span><strong>{rankName}</strong></div>
          <div className="detail-row"><span>当前回合</span><strong>玩家{currentPlayer}</strong></div>
          <div className="detail-row"><span>累计出牌次数</span><strong>{progress}</strong></div>
          {gameState.is_over && (
            <div className="detail-row"><span>状态</span><strong>🏆 游戏结束</strong></div>
          )}
        </div>
      )}
    </div>
  );
};

export default GameStatus;
