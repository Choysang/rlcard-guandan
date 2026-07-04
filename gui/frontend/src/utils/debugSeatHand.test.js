import assert from 'node:assert/strict';
import test from 'node:test';
import { debugSeatHandFor } from './debugSeatHand.js';

test('debug seat hand exposes opponent hands only when debug view is open', () => {
  const debugState = {
    all_player_hands: {
      0: ['S2'],
      1: ['H3', 'D3'],
    },
  };

  assert.deepEqual(debugSeatHandFor({
    debugOpen: true,
    debugState,
    playerId: 1,
    bottomPlayerId: 0,
  }), ['H3', 'D3']);
  assert.equal(debugSeatHandFor({
    debugOpen: false,
    debugState,
    playerId: 1,
    bottomPlayerId: 0,
  }), null);
  assert.equal(debugSeatHandFor({
    debugOpen: true,
    debugState,
    playerId: 0,
    bottomPlayerId: 0,
  }), null);
});
