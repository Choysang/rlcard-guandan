import assert from 'node:assert/strict';
import test from 'node:test';
import {
  formatPlayHistory,
  formatRecentPlays,
  hasUrgentRoundStatus,
} from './statusDock.js';

test('formats full play history as ordered table rows', () => {
  assert.deepEqual(formatPlayHistory([
    { player_id: 0, action: ['Single', 'A', ['HA']] },
    { player_id: 1, action: ['PASS', 'PASS', []] },
    { player_id: 3, action: ['Bomb', '5', ['H5', 'S5', 'C5', 'D5']] },
  ]), [
    { id: 0, order: 1, playerId: 0, label: '玩家0', text: '单张 A' },
    { id: 1, order: 2, playerId: 1, label: '玩家1', text: '不出', pass: true },
    { id: 2, order: 3, playerId: 3, label: '玩家3', text: '炸弹 5 · 4张' },
  ]);
});

test('formats recent plays and hides empty pass-only values', () => {
  assert.deepEqual(formatRecentPlays({
    0: ['Single', 'A', ['HA']],
    1: ['PASS', 'PASS', []],
    2: null,
    3: ['Bomb', '5', ['H5', 'S5', 'C5', 'D5']],
  }), [
    { playerId: 0, label: '玩家0', text: '单张 A' },
    { playerId: 1, label: '玩家1', text: '不出', pass: true },
    { playerId: 3, label: '玩家3', text: '炸弹 5 · 4张' },
  ]);
});

test('urgent status only appears for terminal or round transition states', () => {
  assert.equal(hasUrgentRoundStatus({ is_over: false, round_completed: false }), false);
  assert.equal(hasUrgentRoundStatus({ is_over: true }), true);
  assert.equal(hasUrgentRoundStatus({ round_completed: true }), true);
});
