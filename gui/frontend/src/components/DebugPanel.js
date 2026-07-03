import Card from './Card';
import './DebugPanel.css';

const DebugPanel = ({ debugState, currentPlayer, onSelectAction }) => {
  if (!debugState) return null;

  const legalActions = debugState.legal_actions_by_player?.[currentPlayer] || [];
  const timings = debugState.timings || {};

  return (
    <aside className="debug-panel">
      <header className="debug-panel-header">
        <div>
          <h2>调测</h2>
          <span>当前座位 {currentPlayer}</span>
        </div>
        <span className="debug-seed">seed {debugState.seed ?? '-'}</span>
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
          {legalActions.slice(0, 80).map((action, index) => (
            <button
              key={`${action[0]}-${action[1]}-${index}`}
              type="button"
              onClick={() => onSelectAction(action)}
              className="debug-action-btn"
            >
              <span>{action[0]} {action[1]}</span>
              <small>{(action[2] || []).join(' ')}</small>
            </button>
          ))}
        </div>
      </section>

      <section className="debug-section">
        <h3>Trace</h3>
        <pre>{JSON.stringify((debugState.trace || []).slice(-20), null, 2)}</pre>
      </section>
    </aside>
  );
};

export default DebugPanel;
