import Card from './Card';
import './PlayArea.css';

const PlayArea = ({ gameState, currentPlayer, humanPlayerIds = [], thisPlayerId = null }) => {
  if (!gameState) {
    return (
      <div className="play-area">
        <div className="table-center loading">
          <div className="loading-indicator">🎴</div>
          <p>等待游戏开始...</p>
        </div>
      </div>
    );
  }

  // 获取游戏追踪信息
  const trace = gameState.trace || [];
  const currentRank = gameState.current_rank || 0;
  const greaterAction = gameState.greaterAction || null;
  const greaterPos = gameState.greaterPos || -1;

  // 获取最近的出牌记录（只显示最后一轮）
  const lastRoundPlays = trace.slice(-4); // 最近4次出牌

  // 获取当前最大的牌
  const getCurrentPlay = () => {
    if (greaterAction && greaterAction[0] !== 'PASS') {
      return {
        playerId: greaterPos,
        action: greaterAction
      };
    }
    return null;
  };

  const currentPlay = getCurrentPlay();

  // 格式化动作描述
  const formatAction = (action) => {
    if (!action || !Array.isArray(action)) return '未知动作';
    
    const [type, , cards] = action;
    
    const typeMap = {
      'Single': '单牌',
      'Pair': '对子', 
      'Trips': '三张',
      'ThreeWithTwo': '三带二',
      'Straight': '顺子',
      'StraightFlush': '同花顺',
      'Bomb': '炸弹',
      'ThreePair': '三连对',
      'TwoTrips': '飞机',
      'PASS': '过牌'
    };

    const typeName = typeMap[type] || type;
    
    if (type === 'PASS') {
      return '过牌';
    }
    
    if (type === 'Bomb' && cards) {
      return `${cards.length}张炸弹`;
    }
    
    return typeName;
  };

  // 获取玩家名称
  const getPlayerName = (playerId) => {
    if (playerId === null || playerId === undefined) {
      return '未知玩家';
    }

    const isHuman = Array.isArray(humanPlayerIds) && humanPlayerIds.includes(playerId);
    const isSelf = thisPlayerId !== null && playerId === thisPlayerId;

    if (isHuman && isSelf) {
      return '你';
    }

    // 按玩家编号显示：玩家0、玩家1、玩家2、玩家3
    return `玩家${playerId}`;
  };

  return (
    <div className="play-area">
      {/* 中央牌桌 */}
      <div className="table-center">
        <div className="table-surface">
          {/* 游戏级别指示器 */}
          <div className="game-level">
            <span className="level-icon">🏆</span>
            <span className="level-text">等级 {currentRank}</span>
          </div>

          {/* 当前最大的牌面 */}
          {currentPlay ? (
            <div className="current-cards-display">
              <div className="current-play-info">
                <span className="current-player-name">{getPlayerName(currentPlay.playerId)}</span>
                <span className="current-play-type">{formatAction(currentPlay.action)}</span>
              </div>
              <div className="current-cards">
                {currentPlay.action[2] && Array.isArray(currentPlay.action[2]) && 
                 currentPlay.action[2].map((cardStr, index) => (
                  <Card
                    key={`current-${cardStr}-${index}`}
                    cardString={cardStr}
                    size="normal"
                  />
                ))}
              </div>
            </div>
          ) : (
            <div className="empty-table">
              <div className="table-logo">🃏</div>
              <p>等待出牌</p>
            </div>
          )}

          {/* 当前回合指示器 */}
          <div className="turn-indicator-center">
            <div className="turn-arrow">▶</div>
            <span>{getPlayerName(currentPlayer)}的回合</span>
          </div>
        </div>

        {/* 最近出牌历史（紧凑显示） */}
        {lastRoundPlays.length > 0 && (
          <div className="mini-history">
            <div className="history-title">最近出牌</div>
            <div className="history-items">
              {lastRoundPlays.map((playRecord, index) => {
                const [playerId, action] = playRecord;
                const isPass = action[0] === 'PASS';

                return (
                  <div
                    key={`${trace.length - lastRoundPlays.length + index}-${playerId}`}
                    className={`mini-history-item ${isPass ? 'pass' : ''}`}
                  >
                    <span className="mini-player">{getPlayerName(playerId)}</span>
                    <span className="mini-action">{formatAction(action)}</span>
                  </div>
                );
              })}
            </div>
          </div>
        )}
      </div>

      {/* 游戏状态指示器 */}
      {(gameState.is_over || gameState.round_completed) && (
        <div className="game-status-overlay">
          {gameState.is_over && (
            <div className="status-badge game-over">
              <span className="status-icon">🎉</span>
              <span>游戏结束</span>
            </div>
          )}
          {gameState.round_completed && !gameState.is_over && (
            <div className="status-badge round-end">
              <span className="status-icon">🔄</span>
              <span>本轮结束</span>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default PlayArea; 