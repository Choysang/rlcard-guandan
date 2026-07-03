import { useEffect, useMemo, useRef, useState } from 'react';
import ActionPanel from './ActionPanel';
import DebugPanel from './DebugPanel';
import './GameBoard.css';
import GameStatus from './GameStatus';
import HandCards from './HandCards';
import LandscapeGuard from './LandscapeGuard';
import PlayArea from './PlayArea';
import PlayerSeat from './PlayerSeat';

const sortedKey = (cards) => [...cards].sort().join(',');

const GameBoard = ({
  gameState,
  debugState,
  currentPlayer,
  onAction,
  onRestart,
  humanPlayerIds,
  thisPlayerId,
  roomId,
  participantId,
  onSetAiSpeed = () => {},
  onSetDebugMode = () => {},
}) => {
  const [selectedIndices, setSelectedIndices] = useState([]);
  const hintIndexRef = useRef(0);

  const activePlayer = currentPlayer ?? gameState?.current_player;
  const turnKey = `${activePlayer}:${gameState?.turn_count}`;
  const prevTurnKeyRef = useRef(turnKey);

  useEffect(() => {
    if (turnKey !== prevTurnKeyRef.current) {
      prevTurnKeyRef.current = turnKey;
      setSelectedIndices([]);
      hintIndexRef.current = 0;
    }
  }, [turnKey]);

  const isPlayerTurn = activePlayer === thisPlayerId;
  const aiSpeed = gameState?.ai_speed || 'normal';
  const canControlRoom = Boolean(gameState?.viewer_is_host);
  const canUseDebug = Boolean(gameState?.viewer_can_debug);
  const debugOpen = canUseDebug && Boolean(gameState?.debug_enabled);

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

  const latestActions = gameState?.recent_plays || {};

  const matchedAction = useMemo(() => {
    if (selectedIndices.length === 0 || legalActions.length === 0) return null;
    const selectedKey = sortedKey(selectedIndices.map((i) => playerHand[i]));
    return legalActions.find(
      (a) => Array.isArray(a[2]) && a[2].length === selectedIndices.length
        && sortedKey(a[2]) === selectedKey,
    ) || null;
  }, [selectedIndices, playerHand, legalActions]);

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
    selectActionCards(hintAction);
  };

  const selectActionCards = (action) => {
    if (!action || action[0] === 'PASS') return;
    const indicesToSelect = [];
    const used = new Set();
    for (const card of action[2] || []) {
      const index = playerHand.findIndex((handCard, i) => (
        handCard === card && !used.has(i)
      ));
      if (index >= 0) {
        indicesToSelect.push(index);
        used.add(index);
      }
    }
    setSelectedIndices(indicesToSelect.sort((a, b) => a - b));
  };

  const handleDebugActionSelect = (action) => {
    if (!isPlayerTurn) return;
    selectActionCards(action);
  };

  const handleSpeedChange = (speed) => {
    if (!canControlRoom) return;
    onSetAiSpeed(speed);
  };

  const handleDebugToggle = () => {
    if (!canUseDebug) return;
    onSetDebugMode(!debugOpen);
  };

  const getPlayerName = (playerId) => {
    if (playerId === null || playerId === undefined) return '未知玩家';
    const isHuman = Array.isArray(humanPlayerIds) && humanPlayerIds.includes(playerId);
    const isSelf = thisPlayerId !== null && playerId === thisPlayerId;
    if (isHuman && isSelf) return '你';
    return `玩家${playerId}`;
  };

  const getPlayerCardCount = (playerId) => gameState?.num_cards_left?.[playerId] ?? 0;
  const isHumanSeat = (playerId) => Array.isArray(humanPlayerIds)
    && humanPlayerIds.includes(playerId);
  const debugHandFor = () => null;

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

  const baseSeat = thisPlayerId ?? 0;
  const playerPositions = {
    bottom: baseSeat,
    right: (baseSeat + 1) % 4,
    top: (baseSeat + 2) % 4,
    left: (baseSeat + 3) % 4,
  };
  const bottomPlayer = playerPositions.bottom;
  const winnerIsMyTeam = thisPlayerId !== null
    && gameState.winner_team === thisPlayerId % 2;

  return (
    <LandscapeGuard>
      <div className={`game-board-shell ${debugOpen ? 'debug-open' : ''}`}>
        <div className="game-board">
          <GameStatus
            gameState={gameState}
            currentPlayer={activePlayer}
            onRestart={onRestart}
          />

          <div className="board-toolbar">
            <div className="speed-segment" aria-label="AI 速度">
              {['fast', 'normal', 'slow'].map((speed) => (
                <button
                  key={speed}
                  type="button"
                  className={aiSpeed === speed ? 'active' : ''}
                  disabled={!canControlRoom}
                  onClick={() => handleSpeedChange(speed)}
                >
                  {speed === 'fast' ? '极速' : speed === 'slow' ? '慢速' : '正常'}
                </button>
              ))}
            </div>
            {canUseDebug && (
              <button
                type="button"
                className={`debug-toggle ${debugOpen ? 'active' : ''}`}
                onClick={handleDebugToggle}
              >
                调测
              </button>
            )}
          </div>

          <div className="top-player-zone">
            <PlayerSeat
              position="top"
              playerId={playerPositions.top}
              playerName={getPlayerName(playerPositions.top)}
              cardCount={getPlayerCardCount(playerPositions.top)}
              isActive={activePlayer === playerPositions.top}
              isHuman={isHumanSeat(playerPositions.top)}
              latestAction={latestActions[playerPositions.top]}
              debugHand={debugHandFor(playerPositions.top)}
            />
          </div>

          <div className="left-player-zone">
            <PlayerSeat
              position="left"
              playerId={playerPositions.left}
              playerName={getPlayerName(playerPositions.left)}
              cardCount={getPlayerCardCount(playerPositions.left)}
              isActive={activePlayer === playerPositions.left}
              isHuman={isHumanSeat(playerPositions.left)}
              latestAction={latestActions[playerPositions.left]}
              debugHand={debugHandFor(playerPositions.left)}
            />
          </div>

          <div className="center-table">
            <PlayArea
              gameState={gameState}
              currentPlayer={activePlayer}
              humanPlayerIds={humanPlayerIds}
              thisPlayerId={thisPlayerId}
            />
          </div>

          <div className="right-player-zone">
            <PlayerSeat
              position="right"
              playerId={playerPositions.right}
              playerName={getPlayerName(playerPositions.right)}
              cardCount={getPlayerCardCount(playerPositions.right)}
              isActive={activePlayer === playerPositions.right}
              isHuman={isHumanSeat(playerPositions.right)}
              latestAction={latestActions[playerPositions.right]}
              debugHand={debugHandFor(playerPositions.right)}
            />
          </div>

          <div className="bottom-player-zone">
            <div className="bottom-player-meta">
              <div className={`bottom-avatar ${activePlayer === bottomPlayer ? 'active-turn' : ''}`}>
                你
              </div>
              <div>
                <div className="bottom-player-name">{getPlayerName(bottomPlayer)}</div>
                <div className="bottom-player-detail">
                  手牌 {playerHand.length} 张
                  {roomId && <span> · 房间 {roomId}</span>}
                  {participantId && <span> · {participantId.slice(0, 18)}</span>}
                </div>
              </div>
            </div>

            <PlayerSeat
              position="bottom"
              playerId={bottomPlayer}
              playerName={getPlayerName(bottomPlayer)}
              cardCount={playerHand.length}
              isActive={activePlayer === bottomPlayer}
              isHuman={isHumanSeat(bottomPlayer)}
              latestAction={latestActions[bottomPlayer]}
              debugHand={null}
              showMeta={false}
              showCardBacks={false}
            />

            {isPlayerTurn && (
              <div className="action-area">
                <ActionPanel
                  onActionSelect={handleActionSelect}
                  onPass={handlePass}
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
                setSelected={setSelectedIndices}
                isInteractive={isPlayerTurn}
              />
            </div>
          </div>

          {gameState.is_over && (
            <div className="game-over-modal">
              <div className="modal-content">
                <h2>游戏结束</h2>
                <div className="winner-info">
                  <p className="winner-team">
                    {winnerIsMyTeam ? '你的队伍获胜' : '对方队伍获胜'}
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
                          {idx + 1}. {getPlayerName(playerId)}
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

        {debugOpen && debugState && (
          <DebugPanel
            debugState={debugState}
            currentPlayer={activePlayer}
            onSelectAction={handleDebugActionSelect}
            onClose={handleDebugToggle}
          />
        )}
      </div>
    </LandscapeGuard>
  );
};

export default GameBoard;
