import Card from './Card';
import './DebugPanel.css';
import { actionCards } from '../utils/actionCards';

const DebugPanel = ({ debugState, currentPlayer, onSelectAction, onClose }) => {
  if (!debugState) return null;

  const legalActions = debugState.legal_actions_by_player?.[currentPlayer] || [];
  const timings = debugState.timings || {};
  const roomConfig = debugState.room_config || {};
  const snapshot = debugState.state_snapshot || {};
  const lastActions = debugState.last_actions || {};

  return (
    <aside className="debug-panel">
      <header className="debug-panel-header">
        <div>
          <h2>调测</h2>
          <span>当前座位 {currentPlayer}</span>
        </div>
        <div className="debug-header-actions">
          <span className="debug-seed">seed {debugState.seed ?? '-'}</span>
          <button type="button" className="debug-close" onClick={onClose}>
            关闭
          </button>
        </div>
      </header>

      <section className="debug-section">
        <h3>耗时</h3>
        {Object.keys(timings).length === 0 && <p className="debug-empty">暂无数据</p>}
        {Object.entries(timings).map(([key, value]) => (
          <div key={key} className="debug-row">
            <span>{key}</span>
            <strong>{value} ms</strong>
          </div>
        ))}
      </section>

      <section className="debug-section">
        <h3>房间配置</h3>
        <pre>{JSON.stringify(roomConfig, null, 2)}</pre>
      </section>

      <section className="debug-section">
        <h3>四家手牌</h3>
        {Object.entries(debugState.all_player_hands || {}).map(([pid, hand]) => (
          <div key={pid} className="debug-hand-row">
            <span className="debug-seat-label">P{pid}</span>
            <div className="debug-hand-cards">
              {hand.map((card, index) => (
                <Card key={`${pid}-${card}-${index}`} cardString={card} size="small" />
              ))}
            </div>
          </div>
        ))}
      </section>

      <section className="debug-section">
        <h3>当前合法动作 {legalActions.length}</h3>
        <div className="debug-actions">
          {legalActions.map((action, index) => (
            <button
              key={`${action[0]}-${action[1]}-${index}`}
              type="button"
              onClick={() => onSelectAction(action)}
              className="debug-action-btn"
            >
              <span>{action[0]} {action[1]}</span>
              <small>{actionCards(action).join(' ')}</small>
            </button>
          ))}
        </div>
      </section>

      <section className="debug-section">
        <h3>最近动作</h3>
        <pre>{JSON.stringify(lastActions, null, 2)}</pre>
      </section>

      <section className="debug-section">
        <h3>状态快照</h3>
        <pre>{JSON.stringify(snapshot, null, 2)}</pre>
      </section>

      <section className="debug-section">
        <h3>Trace</h3>
        <pre>{JSON.stringify((debugState.trace || []).slice(-20), null, 2)}</pre>
      </section>
    </aside>
  );
};

export default DebugPanel;
