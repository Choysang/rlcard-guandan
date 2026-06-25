import { useState } from 'react';
import GameSetup from './GameSetup';
import guandanService from '../services/GuandanService';
import './Lobby.css';

const Lobby = ({ onJoinRoom, onCreateRoom, loading, isConnected, onReconnect }) => {
  const [activeTab, setActiveTab] = useState('create'); // 'create' | 'join' | 'settings'
  const [roomIdInput, setRoomIdInput] = useState('');
  const [serverUrl, setServerUrl] = useState(() => guandanService.getServerUrl());
  const [connecting, setConnecting] = useState(false);
  const [connectionError, setConnectionError] = useState('');

  const handleJoin = () => {
    onJoinRoom(roomIdInput.trim().toUpperCase());
  };

  const handleServerConnect = () => {
    setConnecting(true);
    setConnectionError('');
    guandanService.reconnect(
      serverUrl,
      () => {
        setConnecting(false);
        setConnectionError('');
        if (onReconnect) onReconnect();
      },
      (error) => {
        setConnecting(false);
        setConnectionError(`连接失败: ${error}`);
      },
    );
  };

  return (
    <div className="lobby-container">
      <div className={`connection-status ${isConnected ? 'connected' : 'disconnected'}`}>
        <span className="status-dot"></span>
        <span className="status-text">{isConnected ? '已连接' : '未连接'}</span>
        <span className="server-url">{guandanService.getServerUrl()}</span>
      </div>

      <div className="lobby-tabs">
        <button
          type="button"
          className={`tab-btn ${activeTab === 'create' ? 'active' : ''}`}
          onClick={() => setActiveTab('create')}
          disabled={!isConnected}
        >
          创建房间
        </button>
        <button
          type="button"
          className={`tab-btn ${activeTab === 'join' ? 'active' : ''}`}
          onClick={() => setActiveTab('join')}
          disabled={!isConnected}
        >
          加入房间
        </button>
        <button
          type="button"
          className={`tab-btn ${activeTab === 'settings' ? 'active' : ''}`}
          onClick={() => setActiveTab('settings')}
        >
          ⚙️ 服务器
        </button>
      </div>

      <div className="lobby-content">
        {activeTab === 'create' && (
          isConnected ? (
            <GameSetup onGameStart={onCreateRoom} loading={loading} />
          ) : (
            <div className="not-connected-hint">
              <h3>⚠️ 未连接到服务器</h3>
              <p>请先在「服务器」标签页中连接到游戏服务器</p>
            </div>
          )
        )}

        {activeTab === 'join' && (
          isConnected ? (
            <div className="join-room-section">
              <h2>输入房间号加入</h2>
              <div className="join-input-group">
                <label className="sr-only" htmlFor="room-id-input">房间号</label>
                <input
                  id="room-id-input"
                  type="text"
                  className="room-id-input"
                  placeholder="输入 6 位房间号"
                  value={roomIdInput}
                  onChange={(e) => setRoomIdInput(e.target.value)}
                  maxLength="6"
                />
                <button
                  type="button"
                  className="btn btn-primary join-btn"
                  onClick={handleJoin}
                  disabled={loading || roomIdInput.length < 6}
                >
                  {loading ? '加入中...' : '确认加入'}
                </button>
              </div>
            </div>
          ) : (
            <div className="not-connected-hint">
              <h3>⚠️ 未连接到服务器</h3>
              <p>请先在「服务器」标签页中连接到游戏服务器</p>
            </div>
          )
        )}

        {activeTab === 'settings' && (
          <div className="server-settings-section">
            <h2>🌐 服务器设置</h2>
            <p className="settings-hint">
              连接到局域网内的游戏服务器。如果你是主机，其他玩家需要输入你的 IP 地址来连接。
            </p>

            <div className="server-input-group">
              <label htmlFor="server-url-input">服务器地址:</label>
              <input
                id="server-url-input"
                type="text"
                className="server-url-input"
                placeholder="例如: 192.168.1.100:5000"
                value={serverUrl}
                onChange={(e) => setServerUrl(e.target.value)}
              />
              <button
                type="button"
                className="btn btn-primary connect-btn"
                onClick={handleServerConnect}
                disabled={connecting}
              >
                {connecting ? '连接中...' : '连接'}
              </button>
            </div>

            {connectionError && (
              <div className="connection-error">{connectionError}</div>
            )}

            <div className="server-tips">
              <h4>💡 使用提示</h4>
              <ul>
                <li><strong>本机游戏:</strong> 使用 <code>localhost:5000</code> 或 <code>127.0.0.1:5000</code></li>
                <li><strong>局域网联机:</strong> 输入主机的局域网 IP（如 <code>192.168.x.x:5000</code>）</li>
                <li><strong>主机玩家:</strong> 启动服务器后，在命令行窗口查看你的局域网 IP</li>
              </ul>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default Lobby;
