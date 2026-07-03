import assert from 'node:assert/strict';
import test from 'node:test';
import { getCurrentTablePlay } from './playAreaState.js';

test('empty greater action means the table is waiting for a play', () => {
  assert.equal(getCurrentTablePlay({
    greaterAction: [],
    greaterPos: -1,
  }), null);
});

test('player zero remains a valid holder of the current table play', () => {
  assert.deepEqual(getCurrentTablePlay({
    greaterAction: ['Single', '3', ['S3']],
    greaterPos: 0,
  }), {
    playerId: 0,
    action: ['Single', '3', ['S3']],
  });
});

test('pass action is not rendered as the current table play', () => {
  assert.equal(getCurrentTablePlay({
    greaterAction: ['PASS', 'PASS', []],
    greaterPos: 2,
  }), null);
});
