import assert from 'node:assert/strict';
import test from 'node:test';
import { normalizeServerUrl } from './serverUrl.js';

const PAGE_ORIGIN = 'https://guandan.aiwatch.icu';

test('keeps a public HTTPS origin on its standard port', () => {
  assert.equal(
    normalizeServerUrl('https://guandan.aiwatch.icu', PAGE_ORIGIN),
    PAGE_ORIGIN,
  );
});

test('uses the current HTTPS origin for the current bare hostname', () => {
  assert.equal(normalizeServerUrl('guandan.aiwatch.icu', PAGE_ORIGIN), PAGE_ORIGIN);
});

test('migrates the stale production port 5000 URL to the current origin', () => {
  assert.equal(
    normalizeServerUrl('https://guandan.aiwatch.icu:5000', PAGE_ORIGIN),
    PAGE_ORIGIN,
  );
});

test('keeps local network addresses compatible with the backend default port', () => {
  assert.equal(
    normalizeServerUrl('192.168.1.100', PAGE_ORIGIN),
    'http://192.168.1.100:5000',
  );
  assert.equal(
    normalizeServerUrl('192.168.1.100:5100', PAGE_ORIGIN),
    'http://192.168.1.100:5100',
  );
});
