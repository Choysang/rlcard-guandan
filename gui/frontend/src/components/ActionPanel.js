import './ActionPanel.css';

// The matched action is computed once in GameBoard and passed down; this
// panel only renders the controls and reports clicks.
const ActionPanel = ({
  onActionSelect,
  onPass,
  isPlayerTurn,
  legalActions,
  matchedAction,
  onHint,
}) => {
  const canPass = Array.isArray(legalActions)
    && legalActions.some((a) => a[0] === 'PASS');

  if (!isPlayerTurn) {
    return (
      <div className="action-panel">
        <div className="waiting-indicator">
          <div className="spinner-small"></div>
          <span>等待其他玩家...</span>
        </div>
      </div>
    );
  }

  return (
    <div className="action-panel-container">
      <button
        type="button"
        className="action-button hint-btn"
        onClick={onHint}
        title="提示可出的牌"
      >
        💡 提示
      </button>
      <button
        type="button"
        className="action-button pass-btn"
        onClick={onPass}
        disabled={!canPass}
        title={canPass ? '选择不出牌' : '本轮必须出牌'}
      >
        ⏭️ 不出
      </button>
      <button
        type="button"
        className="action-button play-btn"
        onClick={() => onActionSelect(matchedAction)}
        disabled={!matchedAction}
        title={matchedAction ? '出选中的牌' : '请先选择有效的牌型'}
      >
        🎴 出牌
      </button>
    </div>
  );
};

export default ActionPanel;
