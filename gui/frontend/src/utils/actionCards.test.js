import assert from 'node:assert/strict';
import test from 'node:test';
import { actionCards } from './actionCards.js';

test('actionCards returns cards only when the action payload is an array', () => {
  assert.deepEqual(actionCards(['Pair', '3', ['H3', 'S3']]), ['H3', 'S3']);
  assert.deepEqual(actionCards(['PASS', 'PASS', []]), []);
  assert.deepEqual(actionCards(['Single', '3', 'H3']), []);
  assert.deepEqual(actionCards(null), []);
});
