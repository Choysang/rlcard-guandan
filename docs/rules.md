# Ruleset details · 规则细节与修复记录

This document pins down the exact ruleset the engine implements, the
known simplifications, and the rule bugs fixed relative to the original
research code (each fix has a regression test in `tests/`).

## Implemented ruleset · 实现的规则

* Two full decks, 108 cards; 4 players; seats 0/2 vs seats 1/3.
* Levels run 2 → A per team; the team that wins a deal plays its own
  level next. Level gain: 3 (teammates finish 1st+2nd), 2 (1st+3rd),
  1 (1st+4th).
* **Passing level A**: at level A only a 1st+2nd or 1st+3rd finish wins
  the episode; a 1st+4th finish stays at A. Levels never jump past A.
* **Level card (级牌)**: ranks above A in non-sequence combos. The two
  red-heart level cards are wildcards (逢人配) substituting any card
  except jokers.
* **Combos**: Single, Pair, Trips, ThreeWithTwo (full house), ThreePair
  (three consecutive pairs, 6 cards), TwoTrips (plate, two consecutive
  trips, 6 cards), Straight (exactly 5), StraightFlush, Bomb (4-10 same
  rank), four-joker bomb.
* **Sequences** (Straight / ThreePair / TwoTrips) compare by natural
  rank; A may be low (A2345, AA2233, AAA222) or high (TJQKA, QQKKAA,
  KKKAAA); no wrap-around (KKAA22 is illegal). The level card inside a
  sequence counts only at its natural position.
* **Bomb ordering**: four-joker bomb > 8/7/6-card bomb > straight flush
  > 5-card bomb > 4-card bomb; same size compares by (level-aware) rank.
* **Deal end**: a deal ends as soon as one team finishes 1st+2nd
  (双上), otherwise when the third player finishes.
* **Tribute (进贡)**: after a deal, the loser(s) pay their biggest card
  (red-heart level cards exempt; with a joker in hand the joker is
  forced). Double-down: both losers pay; the bigger card goes to the
  first-out, who leads-or see ties below. Tribute receiver returns a
  card of rank ≤ 10 that is not a level card (还贡).
* **Counter-tribute (抗贡)**: tribute is cancelled when the paying side
  jointly holds both big jokers (single-down: the last player holds
  both); the previous first-out leads instead.
* **Wind-follow (接风)**: when a player goes out and the other three all
  pass on the final combo, the teammate gets a free lead.

## Simplifications · 已知简化

* Double-down tribute tie: the first-out player takes the last player's
  tribute card (no suit choice); the seat after the first-out leads.
* When a deal ends by 双上, the remaining two seats fill the 3rd/4th
  result slots in seat order (this only affects bookkeeping; both
  opponents pay tribute either way).
* `step_back` is not supported (the original implementation was broken
  and unused).
* `round_completed` / `last_trick_*` fields use a three-consecutive-pass
  heuristic that can mislabel trick boundaries once players have
  finished; they feed auxiliary features only, never legality.

## Rule fixes vs the original research code · 修复记录

Engine fixes (all covered by `tests/test_action_compare.py` /
`test_action_generation.py` / `test_tribute.py`):

1. **Four-joker bomb** could not beat bombs of 5+ cards or straight
   flushes (it was filtered by a length check before its special-case).
2. **Equal straight flushes** could beat each other; now strictly
   greater is required.
3. **TwoTrips (plate) comparison** used the level-promoted table, so a
   level-rank plate (e.g. 555666 at level 5) wrongly beat KKKAAA;
   plates now compare naturally like all sequences.
4. **AA2233** was never generated when the hand also held a K pair
   (a `break` instead of `continue` in the generator).
5. **AAA222** (ace-low plate) was never generated.
6. **Joker-pair full house** (e.g. 999+BB) was never generated.
7. **Duplicate pair actions** differing only in card order inflated the
   action list.
8. **Bomb size cap** raised from 8 to 10 (8 natural copies + 2
   wildcards).
9. **Tribute-back** could hand back a non-heart level card; level cards
   of every suit are now excluded (a level card counts above an ace).
10. **双上 ends the deal** immediately (play used to continue pointlessly
    until a third player finished).

Baseline fixes (internals otherwise ported verbatim):

* `base1/4/5` family: in `base5` the module-level `active()` called
  `self.getlist(...)`, raising a silently-swallowed NameError that
  degraded all leading plays to "first legal action".
* `base3`: evaluating a four-joker bomb crashed on `int('R')`;
  `get_num` now maps `'B'/'R'` to the joker slot.
* `base7/gen_agent`: had no tribute methods (crashed at the tribute
  phase) and wrote its dataset to a hardcoded developer path.
* Debug `print` calls across the rule baselines are silenced via a
  no-op `_debug_print` shim.

Infrastructure fixes relative to the original fork:

* `seed` in the env config actually seeds the game (reproducible runs).
* `env.run()` returns `(trajectories, payoffs)` per RLCard convention
  and `get_payoffs()` is implemented (the original returned win counters
  and had no payoff implementation).
* The engine no longer calls into agent training internals; agents may
  implement an optional `on_episode_end(deal_summary, winner_team)`
  hook that fires after every deal.
* No `sys.path` hacks, no hardcoded local paths, no stdout
  reconfiguration at import time, no matplotlib import in the engine.
