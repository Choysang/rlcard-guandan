const ACTION_LABELS = {
  Single: '单张',
  Pair: '对子',
  Trips: '三张',
  ThreeWithTwo: '三带二',
  ThreePair: '三连对',
  TwoTrips: '钢板',
  Straight: '顺子',
  StraightFlush: '同花顺',
  Bomb: '炸弹',
  PASS: '不出',
};

export const hasUrgentRoundStatus = (gameState = {}) =>
  Boolean(gameState.is_over || gameState.round_completed);

export const formatActionText = (action) => {
  if (!Array.isArray(action) || action.length === 0) {
    return '';
  }
  const [type, rank, cards] = action;
  if (type === 'PASS') {
    return ACTION_LABELS.PASS;
  }
  const label = ACTION_LABELS[type] || type;
  const rankText = rank && rank !== 'PASS' ? ` ${rank}` : '';
  const countText = Array.isArray(cards) && cards.length > 1 ? ` · ${cards.length}张` : '';
  return `${label}${rankText}${countText}`;
};

export const formatRecentPlays = (recentPlays = {}) =>
  Object.entries(recentPlays)
    .map(([playerId, action]) => {
      const pass = Array.isArray(action) && action[0] === 'PASS';
      return {
        playerId: Number(playerId),
        label: `玩家${playerId}`,
        text: formatActionText(action),
        ...(pass ? { pass: true } : {}),
      };
    })
    .filter((item) => item.text);

export const formatPlayHistory = (history = []) =>
  history
    .map((entry, index) => {
      const playerId = entry?.player_id ?? entry?.playerId ?? entry?.[0];
      const action = entry?.action ?? entry?.[1];
      const pass = Array.isArray(action) && action[0] === 'PASS';
      return {
        id: index,
        order: index + 1,
        playerId: Number(playerId),
        label: `玩家${playerId}`,
        text: formatActionText(action),
        ...(pass ? { pass: true } : {}),
      };
    })
    .filter((item) => item.text);
