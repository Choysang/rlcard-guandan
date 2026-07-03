export const getCurrentTablePlay = (gameState) => {
  const greaterAction = gameState?.greaterAction;
  if (!Array.isArray(greaterAction) || greaterAction.length === 0) {
    return null;
  }
  if (greaterAction[0] === 'PASS') {
    return null;
  }
  return {
    playerId: gameState.greaterPos ?? -1,
    action: greaterAction,
  };
};
