import test from 'node:test';
import assert from 'node:assert/strict';
import {
  applyCardSelection,
  dragModeForIndex,
  sortedUnique,
} from './cardSelection.js';

test('dragModeForIndex selects when starting on unselected card', () => {
  assert.equal(dragModeForIndex([1, 3], 2), 'select');
});

test('dragModeForIndex deselects when starting on selected card', () => {
  assert.equal(dragModeForIndex([1, 3], 3), 'deselect');
});

test('applyCardSelection adds and sorts indexes', () => {
  assert.deepEqual(applyCardSelection([3], 1, 'select'), [1, 3]);
});

test('applyCardSelection removes an index', () => {
  assert.deepEqual(applyCardSelection([1, 3], 1, 'deselect'), [3]);
});

test('sortedUnique removes duplicates', () => {
  assert.deepEqual(sortedUnique([4, 2, 4, 1]), [1, 2, 4]);
});
