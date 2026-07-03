import { useEffect, useState } from 'react';
import guandanService from '../services/GuandanService';
import './GameSetup.css';

// guandan_rlcard baseline registry -> friendly label. Agents needing
// downloaded weights are flagged so players know they may be unavailable.
const AGENT_LABELS = {
  random: { name: '随机 Random' },
  base1: { name: '规则 AI 1' },
  base2: { name: '规则 AI 2' },
  base3: { name: '规则 AI 3' },
  base4: { name: '规则 AI 4' },
  base5: { name: '规则 AI 5' },
  base6: { name: '规则 AI 6' },
  base7: { name: '规则 AI 7（较强）' },
  base8: { name: '规则 AI 8' },
  danzero: { name: 'DanZero', weights: true },
  danzero_plus: { name: 'DanZero+', weights: true },
  dmc: { name: 'DMC', weights: true },
  perfectdan: { name: 'PerfectDan', weights: true },
  llm: { name: '大模型 LLM', weights: true },
};

const DEFAULT_AGENTS = Object.keys(AGENT_LABELS);
const DEFAULT_AI = 'base7';

const GameSetup = ({ onGameStart, loading }) => {
  const [playerConfigs, setPlayerConfigs] = useState([
    { type: 'human', agent: DEFAULT_AI },
    { type: 'ai', agent: DEFAULT_AI },
    { type: 'ai', agent: DEFAULT_AI },
    { type: 'ai', agent: DEFAULT_AI },
  ]);
  const [agentNames, setAgentNames] = useState(DEFAULT_AGENTS);
  const [debugEnabled, setDebugEnabled] = useState(false);

  // 从后端获取可用的 AI 列表（失败则用内置列表）。
  useEffect(() => {
    let cancelled = false;
    fetch(`${guandanService.getServerUrl()}/api/agents`)
      .then((res) => (res.ok ? res.json() : null))
      .then((data) => {
        if (!cancelled && data?.agents?.length) {
          setAgentNames(data.agents);
        }
      })
      .catch(() => {
        /* 使用内置列表 */
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const humanPlayerCount = playerConfigs.filter((p) => p.type === 'human').length;

  const updateConfig = (index, changes) => {
    setPlayerConfigs((prev) =>
      prev.map((cfg, i) => (i === index ? { ...cfg, ...changes } : cfg)),
    );
  };

  const handlePlayerTypeChange = (index, newType) => {
    if (newType === 'ai' && humanPlayerCount <= 1) {
      alert('至少需要一名人类玩家。');
      return;
    }
    updateConfig(index, { type: newType });
  };

  const handleAgentTypeChange = (index, agentType) => {
    updateConfig(index, { agent: agentType });
  };

  const handleStartGame = () => {
    const agentTypes = {};
    const human_player_ids = [];

    playerConfigs.forEach((p, index) => {
      if (p.type === 'ai') {
        agentTypes[index.toString()] = p.agent;
      } else {
        human_player_ids.push(index);
      }
    });

    if (human_player_ids.length === 0) {
      alert('必须至少选择一名人类玩家！');
      return;
    }

    onGameStart({
      agentTypes,
      human_player_ids,
      debug_enabled: debugEnabled,
    });
  };

  const labelFor = (name) => AGENT_LABELS[name]?.name || name;

  return (
    <div className="game-setup">
      <div className="setup-container">
        <h2>创建你的牌局</h2>
        <p className="setup-subtitle">自由配置每个座位是人类玩家或 AI 对手</p>

        <div className="player-config-new">
          {playerConfigs.map((config, index) => (
            <div key={index} className={`player-slot-new ${config.type}`}>
              <div className="player-slot-header">
                <h4>座位 {index + 1}</h4>
              </div>
              <div className="player-type-toggle-large">
                <button
                  type="button"
                  className={`toggle-option ${config.type === 'human' ? 'active' : ''}`}
                  onClick={() => handlePlayerTypeChange(index, 'human')}
                >
                  <span className="toggle-icon" role="img" aria-label="human">👤</span>
                  <span className="toggle-label">人类玩家</span>
                </button>
                <button
                  type="button"
                  className={`toggle-option ${config.type === 'ai' ? 'active' : ''}`}
                  onClick={() => handlePlayerTypeChange(index, 'ai')}
                >
                  <span className="toggle-icon" role="img" aria-label="ai">🤖</span>
                  <span className="toggle-label">AI 玩家</span>
                </button>
              </div>

              {config.type === 'ai' && (
                <div className="agent-selector-new">
                  <label htmlFor={`agent-select-${index}`}>AI 类型:</label>
                  <select
                    id={`agent-select-${index}`}
                    value={config.agent}
                    onChange={(e) => handleAgentTypeChange(index, e.target.value)}
                    className="agent-select"
                  >
                    {agentNames.map((name) => (
                      <option key={name} value={name}>
                        {labelFor(name)}
                        {AGENT_LABELS[name]?.weights ? ' ⚙️' : ''}
                      </option>
                    ))}
                  </select>
                  {AGENT_LABELS[config.agent]?.weights && (
                    <p className="agent-description">
                      ⚙️ 该 AI 需要下载模型权重，详见 docs/gui_guide.md。
                    </p>
                  )}
                </div>
              )}
            </div>
          ))}
        </div>

        <div className="start-section">
          <div className="setup-extra-options">
            <label className="debug-room-toggle">
              <input
                type="checkbox"
                checked={debugEnabled}
                onChange={(e) => setDebugEnabled(e.target.checked)}
              />
              <span>创建调测房间（显示明牌和耗时）</span>
            </label>
          </div>
          <button
            className="btn btn-primary start-btn"
            onClick={handleStartGame}
            disabled={loading || humanPlayerCount === 0}
          >
            {loading ? '正在创建...' : `开始游戏 (${humanPlayerCount} 人局)`}
          </button>
        </div>
      </div>
    </div>
  );
};

export default GameSetup;
