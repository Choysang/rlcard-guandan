import './GameStatus.css';

const GameStatus = ({ gameState, currentPlayer, gameMode, onRestart }) => {
  if (!gameState) {
    return (
      <div className="game-status">
        <div className="status-loading">加载状态中...</div>
      </div>
    );
  }

  // 获取游戏基本信息
  const getCurrentRank = () => {
    return gameState.current_rank || 0;
  };

  const getRankName = (rank) => {
    const rankNames = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A'];
    return rankNames[rank] || '未知';
  };

  // 获取游戏进度
  const getGameProgress = () => {
    if (!gameState.trace) return 0;
    return gameState.trace.length;
  };

  const currentRank = getCurrentRank();
  const rankName = getRankName(currentRank);
  const gameProgress = getGameProgress();

  return (
    <div className="game-status">
      <div className="status-header">
        <h2>游戏状态</h2>
        <div className="status-actions">
          <button type="button" className="btn btn-danger" onClick={onRestart}>
            重新开始
          </button>
        </div>
      </div>

      <div className="status-grid">
        {/* 基本游戏信息 */}
        <div className="status-card basic-info">
          <h3>游戏信息</h3>
          <div className="info-list">
            <div className="info-item">
              <span className="info-label">游戏模式:</span>
              <span className="info-value mode-badge">
                {gameMode === 'pvp' ? '玩家对战' : '人机对战'}
              </span>
            </div>
            <div className="info-item">
              <span className="info-label">当前级别:</span>
              <span className="info-value rank-badge">{rankName}</span>
            </div>
            <div className="info-item">
              <span className="info-label">当前回合:</span>
              <span className="info-value player-badge">玩家{currentPlayer + 1}</span>
            </div>
            <div className="info-item">
              <span className="info-label">出牌次数:</span>
              <span className="info-value progress-badge">{gameProgress}</span>
            </div>
          </div>
        </div>
      </div>

      {/* 游戏状态指示器 */}
      <div className="status-indicators">
        {gameState.is_over && (
          <div className="indicator game-over">
            <span className="indicator-icon">🏆</span>
            <span>游戏结束</span>
          </div>
        )}
        
        {gameState.round_completed && (
          <div className="indicator round-completed">
            <span className="indicator-icon">✅</span>
            <span>回合完成</span>
          </div>
        )}
        
        {currentPlayer === 0 && (
          <div className="indicator your-turn">
            <span className="indicator-icon">👆</span>
            <span>您的回合</span>
          </div>
        )}
      </div>
    </div>
  );
};



export default GameStatus; 