import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const here = dirname(fileURLToPath(import.meta.url));
const frontendRoot = join(here, '..', '..');

test('manifest declares app-like fullscreen launch metadata', () => {
  const manifest = JSON.parse(readFileSync(
    join(frontendRoot, 'public', 'manifest.webmanifest'),
    'utf8',
  ));

  assert.equal(manifest.name, '掼蛋算法调测牌桌');
  assert.equal(manifest.short_name, '掼蛋调测');
  assert.equal(manifest.start_url, '/');
  assert.equal(manifest.scope, '/');
  assert.equal(manifest.display, 'fullscreen');
  assert.equal(manifest.orientation, 'landscape');
  assert.ok(manifest.icons.some((icon) => icon.purpose.includes('maskable')));
});

test('html advertises the web app manifest for mobile browsers', () => {
  const html = readFileSync(join(frontendRoot, 'index.html'), 'utf8');

  assert.match(html, /<link rel="manifest" href="\/manifest\.webmanifest"/);
  assert.match(html, /<meta name="theme-color" content="#0f7f5f"/);
  assert.match(html, /<meta name="mobile-web-app-capable" content="yes"/);
  assert.match(html, /<meta name="apple-mobile-web-app-capable" content="yes"/);
});

test('service worker avoids caching game traffic', () => {
  const worker = readFileSync(join(frontendRoot, 'public', 'sw.js'), 'utf8');

  assert.doesNotMatch(worker, /fetch\s*\(/);
  assert.doesNotMatch(worker, /caches\./);
});

test('app registers the service worker only when supported', () => {
  const entry = readFileSync(join(frontendRoot, 'src', 'index.js'), 'utf8');

  assert.match(entry, /'serviceWorker' in navigator/);
  assert.match(entry, /navigator\.serviceWorker\.register\('\/sw\.js'\)/);
});
