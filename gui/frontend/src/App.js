import { useEffect, useState, useCallback } from 'react';
import './App.css';
import GameBoard from './components/GameBoard';
import Lobby from './components/Lobby'; // A new component for joining/creating rooms
import guandanService from './services/GuandanService'; // Import the instance

function App() {
  const [appState, setAppState] = useState('Lobby'); // Lobby, WaitingInRoom, InGame
  const [gameState, setGameState] = useState(null);
  const [debugState, setDebugState] = useState(null);
  const [currentPlayer, setCurrentPlayer] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [isConnected, setIsConnected] = useState(false);
  
  const [roomId, setRoomId] = useState('');
  const [playerId, setPlayerId] = useState(null);
  const [participantId, setParticipantId] = useState(null);
  const [humanPlayerIds, setHumanPlayerIds] = useState([]);

  // 设置socket事件监听器
  const setupSocketListeners = useCallback(() => {
    guandanService.on('room_created', ({
      roomId, playerId, participantId, sessionId, resumeToken, hostToken,
    }) => {
      guandanService.rememberSessionId(sessionId);
      guandanService.rememberRoomTokens(roomId, { resumeToken, hostToken });
      setRoomId(roomId);
      setPlayerId(playerId);
      setParticipantId(participantId || null);
      setAppState('WaitingInRoom');
      setLoading(false);
    });

    guandanService.on('joined_room', ({
      roomId, playerId, participantId, sessionId, resumeToken, hostToken,
    }) => {
      guandanService.rememberSessionId(sessionId);
      guandanService.rememberRoomTokens(roomId, { resumeToken, hostToken });
      setRoomId(roomId);
      setPlayerId(playerId);
      setParticipantId(participantId || null);
      setAppState('WaitingInRoom');
      setLoading(false);
    });

    guandanService.on('game_started', ({ state, debug_state, current_player }) => {
      setGameState(state);
      setDebugState(debug_state || null);
      setCurrentPlayer(current_player);
      setHumanPlayerIds(state.human_player_ids || []);
      setAppState('InGame');
      setLoading(false);
    });

    guandanService.on('update_state', ({ state, debug_state, current_player }) => {
      setGameState(state);
      setDebugState(debug_state || null);
      setCurrentPlayer(current_player);
      setHumanPlayerIds(state.human_player_ids || []);
    });
    
    guandanService.on('player_joined', () => {
      // 预留：可在此提示有玩家加入。
    });

    guandanService.on('error', (data) => {
      setError(data.message || '发生未知错误');
      setLoading(false);
    });
  }, []);

  useEffect(() => {
    // 连接到服务器
    guandanService.connect(
      () => {
        setIsConnected(true);
        setError(null);
      },
      (err) => {
        setIsConnected(false);
        setError(`连接服务器失败: ${err}`);
      }
    );

    // 设置事件监听器
    setupSocketListeners();

    // 定期检查连接状态
    const checkConnection = setInterval(() => {
      setIsConnected(guandanService.isConnected());
    }, 2000);

    return () => {
      clearInterval(checkConnection);
    };
  }, [setupSocketListeners]);

  // 重新连接后重新设置监听器
  const handleReconnect = useCallback(() => {
    setIsConnected(guandanService.isConnected());
    setupSocketListeners();
  }, [setupSocketListeners]);

  const handleCreateRoom = (config) => {
    setLoading(true);
    setError(null);
    guandanService.createRoom(config);
  };

  const handleJoinRoom = (roomIdToJoin) => {
    setLoading(true);
    setError(null);
    if (roomIdToJoin) {
      guandanService.joinRoom(roomIdToJoin);
    } else {
        setError("请输入房间号。");
        setLoading(false);
    }
  };

  const handleAction = async (action) => {
    if (!roomId || playerId === null) return;
    // The action is now sent without expecting a direct response
    guandanService.sendAction(roomId, playerId, action);
  };

  const handleRestart = () => {
    // Reset all state to go back to the lobby
    setAppState('Lobby');
    setGameState(null);
    setDebugState(null);
    setCurrentPlayer(null);
    setError(null);
    setRoomId('');
    setPlayerId(null);
    setParticipantId(null);
    setHumanPlayerIds([]);
  };

  const renderContent = () => {
    switch (appState) {
      case 'WaitingInRoom':
        return (
          <div className="waiting-room">
            <h2>等待玩家加入...</h2>
            <p>房间号: <strong className="room-id">{roomId}</strong></p>
            <p>分享房间号给你的朋友，让他们加入游戏！</p>
            <div className="loading-spinner"></div>
          </div>
        );
      case 'InGame':
        return (
          gameState && (
            <GameBoard
              gameState={gameState}
              debugState={debugState}
              humanPlayerIds={humanPlayerIds}
              thisPlayerId={playerId} // Pass this client's player ID
              participantId={participantId}
              roomId={roomId}
              currentPlayer={currentPlayer}
              onAction={handleAction}
              onRestart={handleRestart}
              onSetAiSpeed={(speed) => guandanService.setAiSpeed(roomId, speed)}
              onSetDebugMode={(enabled) => guandanService.setDebugMode(roomId, enabled)}
              loading={loading}
            />
          )
        );
      case 'Lobby':
      default:
        return (
          <Lobby 
            onJoinRoom={handleJoinRoom}
            onCreateRoom={handleCreateRoom}
            loading={loading}
            isConnected={isConnected}
            onReconnect={handleReconnect}
          />
        );
    }
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>🃏 掼蛋游戏 - 实时多人版</h1>
      </header>
      
      {error && (
        <div className="error-message">
          <span className="error-text">{error}</span>
          <button
            type="button"
            className="btn btn-secondary"
            onClick={() => setError(null)}
          >
            关闭
          </button>
        </div>
      )}
      
      {loading && appState === 'Lobby' && (
        <div className="loading-overlay">
          <div className="loading-spinner"></div>
          <span>请稍候...</span>
        </div>
      )}
      
      {renderContent()}
    </div>
  );
}

export default App; 
