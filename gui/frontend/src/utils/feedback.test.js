import assert from 'node:assert/strict';
import test from 'node:test';
import { buildFeedbackPayload, trimFeedbackText } from './feedback.js';

test('trimFeedbackText trims whitespace and caps long input', () => {
  assert.equal(trimFeedbackText('  希望按钮更大  '), '希望按钮更大');
  assert.equal(trimFeedbackText('x'.repeat(1205)).length, 1200);
});

test('buildFeedbackPayload carries safe page and room context', () => {
  assert.deepEqual(buildFeedbackPayload({
    kind: 'suggestion',
    message: ' 增加历史滚动 ',
    page: 'in_game',
    roomId: 'ABC123',
    playerId: 0,
    context: {
      url: 'https://guandan.aiwatch.icu/',
      userAgent: 'Chrome',
      apiKey: 'sk-secret',
    },
  }), {
    kind: 'suggestion',
    message: '增加历史滚动',
    page: 'in_game',
    roomId: 'ABC123',
    playerId: 0,
    context: {
      url: 'https://guandan.aiwatch.icu/',
      userAgent: 'Chrome',
    },
  });
});
