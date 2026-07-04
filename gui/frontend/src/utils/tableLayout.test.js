import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import test from 'node:test';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const componentsRoot = join(here, '..', 'components');

const readComponentCss = (fileName) => readFileSync(join(componentsRoot, fileName), 'utf8');

test('right opponent played cards keep the same left-under-right stack as other seats', () => {
  const css = readComponentCss('PlayerSeat.css');

  assert.match(css, /\.seat-play-cards \.card \+ \.card\s*\{[^}]*margin-left:\s*-\d+px;/);
  assert.doesNotMatch(css, /\.player-seat-right \.seat-play-cards\s*\{[^}]*flex-direction:\s*row-reverse;/);
  assert.doesNotMatch(css, /\.player-seat-right \.seat-play-cards \.card \+ \.card\s*\{[^}]*margin-right:/);
});

test('mobile landscape turn controls start from the top of the bottom play zone', () => {
  const boardCss = readComponentCss('GameBoard.css');
  const handCss = readComponentCss('HandCards.css');

  assert.match(
    boardCss,
    /@media \(max-width: 900px\) and \(orientation: landscape\)[\s\S]*?\.bottom-player-zone\s*\{[\s\S]*?align-items:\s*start;[\s\S]*?align-content:\s*start;/,
  );
  assert.match(
    boardCss,
    /@media \(max-width: 900px\) and \(orientation: landscape\)[\s\S]*?\.action-area\s*\{[\s\S]*?align-self:\s*start;/,
  );
  assert.match(
    handCss,
    /@media \(max-width: 900px\) and \(orientation: landscape\)[\s\S]*?\.hand-cards-container\s*\{[\s\S]*?justify-content:\s*flex-start;/,
  );
});
