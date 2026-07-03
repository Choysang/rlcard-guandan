import { io } from 'socket.io-client';

// 存储键名
const STORAGE_KEY = 'guandan_server_url';
const SESSION_STORAGE_KEY = 'guandan_session_id';
const ROOM_TOKENS_STORAGE_KEY = 'guandan_room_tokens';

// 获取默认服务器地址（优先使用当前页面的 host）
const getDefaultServerUrl = () => {
  const { protocol, hostname, port } = window.location;
  const isViteDev = import.meta.env.DEV;
  // 由后端托管的页面：使用同源地址。
  if (!isViteDev && hostname) {
    return `${protocol}//${hostname}${port ? ':' + port : ''}`;
  }
  // 开发模式（Vite dev server）默认连接本地后端。
  return import.meta.env.VITE_BACKEND_URL || `${protocol}//${hostname}:5000`;
};

class GuandanService {
  socket = null;
  serverUrl = null;
  connectionCallback = null;
  errorCallback = null;

  constructor() {
    // 从 localStorage 读取保存的服务器地址，或使用默认值
    this.serverUrl = localStorage.getItem(STORAGE_KEY) || getDefaultServerUrl();
  }

  // 获取当前服务器地址
  getServerUrl() {
    return this.serverUrl;
  }

  getSessionId() {
    return localStorage.getItem(SESSION_STORAGE_KEY) || null;
  }

  rememberSessionId(sessionId) {
    if (sessionId) {
      localStorage.setItem(SESSION_STORAGE_KEY, sessionId);
    }
  }

  getRoomTokens() {
    try {
      return JSON.parse(localStorage.getItem(ROOM_TOKENS_STORAGE_KEY) || '{}');
    } catch {
      return {};
    }
  }

  getRoomToken(roomId, key) {
    return this.getRoomTokens()[roomId]?.[key] || null;
  }

  rememberRoomTokens(roomId, tokens = {}) {
    if (!roomId) return;
    const allTokens = this.getRoomTokens();
    allTokens[roomId] = {
      ...(allTokens[roomId] || {}),
      ...Object.fromEntries(
        Object.entries(tokens).filter(([, value]) => Boolean(value)),
      ),
    };
    localStorage.setItem(ROOM_TOKENS_STORAGE_KEY, JSON.stringify(allTokens));
  }

  // 设置服务器地址并保存
  setServerUrl(url) {
    // 确保 URL 格式正确
    if (!url.startsWith('http://') && !url.startsWith('https://')) {
      url = 'http://' + url;
    }
    // 如果没有端口，添加默认端口
    if (!url.match(/:\d+$/)) {
      url = url + ':5000';
    }
    this.serverUrl = url;
    localStorage.setItem(STORAGE_KEY, url);
    return url;
  }

  // 断开当前连接
  disconnect() {
    if (this.socket) {
      this.socket.disconnect();
      this.socket = null;
    }
  }

  // 连接到服务器
  connect(callback, errorCallback) {
    this.connectionCallback = callback;
    this.errorCallback = errorCallback;

    // 已连接则直接回调。
    if (this.socket && this.socket.connected) {
      if (callback) callback();
      return;
    }

    // 如果有旧的 socket，先断开
    if (this.socket) {
      this.socket.disconnect();
    }

    // 创建新的 socket 连接
    this.socket = io(this.serverUrl, {
      reconnection: true,
      reconnectionAttempts: 5,
      reconnectionDelay: 1000,
      timeout: 10000,
    });

    this.socket.on('connect', () => {
      console.log('Socket.IO 已连接:', this.serverUrl, 'id:', this.socket.id);
      if (this.connectionCallback) this.connectionCallback();
    });

    this.socket.on('connect_error', (err) => {
      console.error('Socket.IO 连接失败:', err.message);
      if (this.errorCallback) this.errorCallback(err.message);
    });

    this.socket.on('disconnect', (reason) => {
      console.log('Socket.IO 断开连接:', reason);
    });
  }

  // 检查连接状态
  isConnected() {
    return !!(this.socket && this.socket.connected);
  }

  // 重新连接到新的服务器地址
  reconnect(newUrl, callback, errorCallback) {
    this.disconnect();
    this.setServerUrl(newUrl);
    this.connect(callback, errorCallback);
  }

  createRoom(playerConfig) {
    if (!this.socket) return;
    this.socket.emit('create_room', {
      player_config: playerConfig,
      sessionId: this.getSessionId(),
    });
  }

  joinRoom(roomId) {
    if (!this.socket) return;
    this.socket.emit('join_room', {
      roomId,
      sessionId: this.getSessionId(),
      resumeToken: this.getRoomToken(roomId, 'resumeToken'),
      hostToken: this.getRoomToken(roomId, 'hostToken'),
    });
  }

  sendAction(roomId, playerId, action) {
    if (!this.socket) return;
    this.socket.emit('player_action', { roomId, playerId, action });
  }

  setAiSpeed(roomId, speed) {
    if (!this.socket) return;
    this.socket.emit('set_ai_speed', {
      roomId,
      speed,
      hostToken: this.getRoomToken(roomId, 'hostToken'),
    });
  }

  setDebugMode(roomId, enabled) {
    if (!this.socket) return;
    this.socket.emit('set_debug_mode', {
      roomId,
      enabled,
      hostToken: this.getRoomToken(roomId, 'hostToken'),
    });
  }

  on(eventName, callback) {
    if (this.socket) {
      this.socket.off(eventName); // 避免重复注册造成回调多次触发
      this.socket.on(eventName, callback);
    }
  }

  off(eventName) {
    if (this.socket) {
      this.socket.off(eventName);
    }
  }
}

const guandanServiceInstance = new GuandanService();
export default guandanServiceInstance;
