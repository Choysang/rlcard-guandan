import { useMemo, useRef, useState } from 'react';
import ActionPanel from './ActionPanel';
import './GameBoard.css';
import GameStatus from './GameStatus';
import HandCards from './HandCards';
import PlayArea from './PlayArea';

const sortedKey = (cards) => [...cards].sort().join(',');

const GameBoard = ({ gameState, currentPlayer, onAction, onRestart, humanPlayerIds, thisPlayerId }) => {
  const [selectedIndices, setSelectedIndices] = useState([]);
  // Hint cursor is only read inside the click handler, so a ref avoids
  // re-rendering the whole board on every hint.
  const hintIndexRef = useRef(0);

  // Reset the selection whenever the turn changes (new player or new trick).
  // Done with a render-time prev-key comparison (React's recommended pattern)
  // rather than an effect, so the selection never lags a turn behind.
  const turnKey = `${currentPlayer}:${gameState?.turn_count}`;
  const [prevTurnKey, setPrevTurnKey] = useState(turnKey);
  if (turnKey !== prevTurnKey) {
    setPrevTurnKey(turnKey);
    setSelectedIndices([]);
    hintIndexRef.current = 0;
  }

  const isPlayerTurn = currentPlayer === thisPlayerId;

  const playerHand = useMemo(() => {
    if (thisPlayerId !== null && gameState?.player_hands) {
      return gameState.player_hands[thisPlayerId] || [];
    }
    return [];
  }, [gameState, thisPlayerId]);

  const legalActions = useMemo(
    () => (isPlayerTurn ? gameState?.actions || [] : []),
    [gameState, isPlayerTurn],
  );

  // The action matching the current selection is derived, not stored, so
  // there is no effect chain feeding render state.
  const matchedAction = useMemo(() => {
    if (selectedIndices.length === 0 || legalActions.length === 0) return null;
    const selectedKey = sortedKey(selectedIndices.map((i) => playerHand[i]));
    return legalActions.find(
      (a) => Array.isArray(a[2]) && a[2].length === selectedIndices.length
        && sortedKey(a[2]) === selectedKey,
    ) || null;
  }, [selectedIndices, playerHand, legalActions]);

  const handleCardSelect = (cardIndex) => {
    if (!isPlayerTurn) return;
    setSelectedIndices((prev) => (prev.includes(cardIndex)
      ? prev.filter((index) => index !== cardIndex)
      : [...prev, cardIndex].sort((a, b) => a - b)));
  };

  const handleActionSelect = (action) => {
    if (!action || !isPlayerTurn) return;
    onAction(action);
    setSelectedIndices([]);
  };

  const handlePass = () => {
    if (!isPlayerTurn) return;
    const passAction = legalActions.find((a) => a[0] === 'PASS');
    if (passAction) handleActionSelect(passAction);
  };

  const handleHint = () => {
    if (!isPlayerTurn || !playerHand.length || !legalActions.length) return;
    const nonPassActions = legalActions.filter(
      (a) => a[0] !== 'PASS' && a[2] && a[2].length > 0,
    );
    if (nonPassActions.length === 0) return;

    const hintAction = nonPassActions[hintIndexRef.current % nonPassActions.length];
    hintIndexRef.current += 1;

    const indicesToSelect = [];
    const usedIndices = new Set();
    for (const cardToFind of hintAction[2]) {
      for (let i = 0; i < playerHand.length; i += 1) {
        if (!usedIndices.has(i) && playerHand[i] === cardToFind) {
          indicesToSelect.push(i);
          usedIndices.add(i);
          break;
        }
      }
    }
    setSelectedIndices(indicesToSelect.sort((a, b) => a - b));
  };

  const clearSelection = () => {
    setSelectedIndices([]);
    hintIndexRef.current = 0;
  };

  const getPlayerName = (playerId) => {
    if (playerId === null || playerId === undefined) return '未知玩家';
    const isHuman = Array.isArray(humanPlayerIds) && humanPlayerIds.includes(playerId);
    const isSelf = thisPlayerId !== null && playerId === thisPlayerId;
    if (isHuman && isSelf) return '你';
    return `玩家${playerId}`;
  };

  const getPlayerCardCount = (playerId) => gameState?.num_cards_left?.[playerId] ?? 0;

  if (!gameState) {
    return (
      <div className="game-board loading">
        <div className="loading-spinner">
          <div className="spinner"></div>
          <p>正在加载游戏...</p>
        </div>
      </div>
    );
  }

  const playerPositions = {
    bottom: thisPlayerId,
    right: (thisPlayerId + 1) % 4,
    top: (thisPlayerId + 2) % 4,
    left: (thisPlayerId + 3) % 4,
  };

  const renderPlayer = (position) => {
    const playerId = playerPositions[position];
    if (playerId === null) return null;
    const isCurrentActivePlayer = currentPlayer === playerId;

    if (position === 'bottom') {
      return (
        <div className="player-area bottom-player">
          <div className="current-player-info">
            <div className="player-info-card">
              <div className="player-avatar current"><span>😊</span></div>
              <div className="player-details">
                <div className="player-name">{getPlayerName(playerId)}</div>
                <div className="card-count">手牌: {playerHand.length} 张</div>
                {isCurrentActivePlayer && <div className="turn-indicator your-turn">你的回合</div>}
              </div>
            </div>
          </div>

          <div className="current-player-hand-and-actions">
            {isPlayerTurn && (
              <div className="action-area">
                <ActionPanel
                  onActionSelect={handleActionSelect}
                  onPass={handlePass}
                  onClearSelection={clearSelection}
                  onHint={handleHint}
                  isPlayerTurn={isPlayerTurn}
                  matchedAction={matchedAction}
                  legalActions={legalActions}
                />
              </div>
            )}

            <div className="current-player-cards">
              <HandCards
                cards={playerHand}
                selectedCards={selectedIndices}
                onCardSelect={handleCardSelect}
                isInteractive={isPlayerTurn}
              />
            </div>
          </div>
        </div>
      );
    }

    const backCount = Math.min(getPlayerCardCount(playerId), position === 'top' ? 13 : 10);
    return (
      <div className={`player-area ${position}-player`}>
        <div className="player-info-card">
          <div className="player-avatar">
            <span>{humanPlayerIds.includes(playerId) ? '👤' : '🤖'}</span>
          </div>
          <div className="player-details">
            <div className="player-name">{getPlayerName(playerId)}</div>
            <div className="card-count">手牌: {getPlayerCardCount(playerId)} 张</div>
            {isCurrentActivePlayer && <div className="turn-indicator">出牌中...</div>}
          </div>
        </div>
        <div className={`player-cards ${position === 'top' ? 'horizontal' : 'vertical'}`}>
          {Array.from({ length: backCount }).map((_, i) => (
            <div key={`back-${playerId}-${i}`} className={`card-back ${position !== 'top' ? 'vertical-card' : ''}`} />
          ))}
        </div>
      </div>
    );
  };

  const winnerIsMyTeam = gameState.winner_team === thisPlayerId % 2;

  return (
    <div className="game-board">
      <GameStatus gameState={gameState} currentPlayer={currentPlayer} onRestart={onRestart} />

      <div className="game-layout">
        {renderPlayer('top')}
        <div className="middle-area">
          {renderPlayer('left')}
          <div className="center-table">
            <PlayArea
              gameState={gameState}
              currentPlayer={currentPlayer}
              humanPlayerIds={humanPlayerIds}
              thisPlayerId={thisPlayerId}
            />
          </div>
          {renderPlayer('right')}
        </div>
        {renderPlayer('bottom')}
      </div>

      {gameState.is_over && (
        <div className="game-over-modal">
          <div className="modal-content">
            <h2>🎉 游戏结束</h2>
            <div className="winner-info">
              <p className="winner-team">
                {winnerIsMyTeam ? '🏆 你的队伍获胜！' : '😢 对方队伍获胜'}
              </p>
              <p className="team-details">
                获胜队伍: {gameState.winner_team === 0
                  ? `${getPlayerName(0)} & ${getPlayerName(2)}`
                  : `${getPlayerName(1)} & ${getPlayerName(3)}`}
              </p>
            </div>
            {gameState.finished_players && (
              <div className="finish-order">
                <p>完成顺序:</p>
                <div className="player-ranks">
                  {gameState.finished_players.map((playerId, idx) => (
                    <span key={playerId} className={`rank-badge rank-${idx + 1}`}>
                      {idx === 0 ? '🥇' : idx === 1 ? '🥈' : idx === 2 ? '🥉' : '4️⃣'}
                      {getPlayerName(playerId)}
                    </span>
                  ))}
                </div>
              </div>
            )}
            <div className="modal-actions">
              <button type="button" className="btn btn-primary" onClick={onRestart}>
                返回大厅
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default GameBoard;
