export const debugSeatHandFor = ({
  debugOpen,
  debugState,
  playerId,
  bottomPlayerId,
}) => {
  if (!debugOpen || !debugState || playerId === bottomPlayerId) return null;

  const hands = debugState.all_player_hands || debugState.other_player_hands || {};
  const hand = hands[playerId] || hands[String(playerId)];
  return Array.isArray(hand) && hand.length > 0 ? hand : null;
};
