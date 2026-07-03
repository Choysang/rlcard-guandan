import assert from 'node:assert/strict';
import test from 'node:test';
import { handCardStyle } from './handLayout.js';

test('hand cards always stack left under right', () => {
  assert.equal(handCardStyle({ index: 0, step: 28, selected: false }).zIndex, 0);
  assert.equal(handCardStyle({ index: 1, step: 28, selected: false }).zIndex, 1);
  assert.equal(handCardStyle({ index: 0, step: 28, selected: true }).zIndex, 0);
  assert.equal(handCardStyle({ index: 1, step: 28, selected: true }).zIndex, 1);
});

test('selected hand cards rise without changing stacking order', () => {
  assert.deepEqual(handCardStyle({ index: 3, step: 32, selected: true }), {
    left: '96px',
    top: '-22px',
    zIndex: 3,
  });
});
