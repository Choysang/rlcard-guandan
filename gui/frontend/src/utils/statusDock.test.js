import assert from 'node:assert/strict';
import test from 'node:test';
import {
  formatRemainingCounts,
  formatRecentPlays,
  hasUrgentRoundStatus,
} from './statusDock.js';

test('formats remaining card counts as compact recorder rows', () => {
  assert.deepEqual(formatRemainingCounts([12, 8, 3, 20]), [
    { playerId: 0, label: '玩家0', count: 12, danger: false },
    { playerId: 1, label: '玩家1', count: 8, danger: false },
    { playerId: 2, label: '玩家2', count: 3, danger: true },
    { playerId: 3, label: '玩家3', count: 20, danger: false },
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
