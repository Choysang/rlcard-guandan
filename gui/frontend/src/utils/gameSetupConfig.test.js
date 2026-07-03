import assert from 'node:assert/strict';
import test from 'node:test';
import {
  buildRoomConfig,
  hasCompleteLlmConfig,
  isUnsafeLlmOrigin,
  needsLlmConfig,
} from './gameSetupConfig.js';

test('detects when a selected AI seat uses the LLM agent', () => {
  assert.equal(needsLlmConfig([
    { type: 'human', agent: 'base7' },
    { type: 'ai', agent: 'llm' },
  ]), true);
  assert.equal(needsLlmConfig([
    { type: 'human', agent: 'llm' },
    { type: 'ai', agent: 'base7' },
  ]), false);
});

test('LLM config is complete only with model, base URL and api key', () => {
  assert.equal(hasCompleteLlmConfig({
    apiKey: 'sk-test',
    baseUrl: 'https://api.example.com/v1',
    model: 'deepseek-chat',
  }), true);
  assert.equal(hasCompleteLlmConfig({
    apiKey: 'sk-test',
    baseUrl: 'https://api.example.com/v1',
    model: ' ',
  }), false);
  assert.equal(hasCompleteLlmConfig({
    apiKey: 'sk-test',
    baseUrl: ' ',
    model: 'deepseek-chat',
  }), false);
});

test('room config includes trimmed LLM settings only when needed', () => {
  const config = buildRoomConfig([
    { type: 'human', agent: 'base7' },
    { type: 'ai', agent: 'llm' },
    { type: 'ai', agent: 'dmc' },
    { type: 'human', agent: 'base7' },
  ], true, {
    apiKey: ' sk-test ',
    baseUrl: ' https://api.example.com/v1 ',
    model: ' deepseek-chat ',
  });

  assert.deepEqual(config, {
    agentTypes: {
      1: 'llm',
      2: 'dmc',
    },
    human_player_ids: [0, 3],
    debug_enabled: true,
    llmConfig: {
      apiKey: 'sk-test',
      baseUrl: 'https://api.example.com/v1',
      model: 'deepseek-chat',
    },
  });

  const noLlm = buildRoomConfig([
    { type: 'human', agent: 'base7' },
    { type: 'ai', agent: 'base7' },
  ], false, {
    apiKey: 'sk-test',
    model: 'deepseek-chat',
  });
  assert.equal('llmConfig' in noLlm, false);
});

test('public HTTP origins are unsafe for LLM API keys', () => {
  assert.equal(isUnsafeLlmOrigin({
    protocol: 'http:',
    hostname: '8.219.61.189',
  }), true);
  assert.equal(isUnsafeLlmOrigin({
    protocol: 'https:',
    hostname: 'guandan.aiwatch.icu',
  }), false);
  assert.equal(isUnsafeLlmOrigin({
    protocol: 'http:',
    hostname: 'localhost',
  }), false);
});
