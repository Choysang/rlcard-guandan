# Guandan Debug Table Redesign

## Goal

Build the first iteration of a modern Guandan interaction and debugging table for `rlcard-guandan`: a clear horizontal card-table experience for play, plus an opt-in debug mode for algorithm inspection.

## Assumptions

- The first iteration optimizes the GUI play/debug loop, not large-scale training throughput.
- Desktop is the primary algorithm debugging surface.
- Mobile is primarily for public algorithm trials and data collection.
- Mobile game play should be landscape-only, modeled after mature Dou Dizhu products.
- Anonymous play is acceptable for first release; account support can be added in a future phase without changing logged game data.
- Existing uncommitted GUI changes are user work and must be preserved.

## Scope

In scope:

- Redesign the game table around a horizontal four-seat layout.
- Make each player's latest play appear in front of that player's seat.
- Keep the local player's hand large, readable, and easy to select.
- Support press-drag selection across multiple hand cards.
- Add a play mode and a debug mode in the same table.
- Add AI speed modes: fast, normal, and slow.
- Add lightweight timing instrumentation for GUI-facing actions.
- Split normal play state from debug state.
- In debug mode, show all players' hands, legal actions, trace, and timings.
- Add anonymous game/session logging to JSONL, with fields reserved for future accounts.

Out of scope for the first iteration:

- Full account system with login, password, rankings, and persistent history.
- Training/self-play throughput optimization as the main target.
- Full analytics dashboard.
- Rewriting the whole game engine.
- Mobile portrait game table.

## Product Shape

The table has two modes.

Play mode is the default. It prioritizes readable cards, clear turn flow, and fast human input. It does not expose hidden hands.

Debug mode is opt-in. On desktop, it opens a side panel or drawer with four-player hands, legal actions, trace, timings, seed, room config, and state snapshots. On mobile, debug information is hidden by default and should not interfere with public trial play.

Mobile game play is landscape-first. If a user enters the game table in portrait orientation, the app shows a lightweight rotate prompt instead of compressing the full table into portrait.

## Frontend Design

### Layout

`GameBoard` becomes the horizontal table shell. The layout is shared by desktop and mobile landscape:

- Bottom seat: current client, largest hand and controls.
- Left, top, right seats: opponent/team positions around the table.
- Each seat renders player info, remaining card count, active turn state, and latest play in front of that seat.
- The center area shows current table holder, current largest action, rank/level state, and round status.
- Debug UI is outside the primary play surface so it does not shrink cards unnecessarily.

### Hand Interaction

`HandCards` owns local hand display and selection behavior:

- Click toggles one card.
- Press-drag over cards selects or deselects multiple cards.
- Drag mode is decided by the first card: starting on an unselected card selects; starting on a selected card deselects.
- Selection is based on pointer position and card index, not only click events, so overlapping cards remain easy to hit.
- Selected cards lift enough to show state without hiding card rank/suit.
- Card spacing adapts to hand size while keeping rank corners visible.
- A valid selected action enables the play button and shows the matched combo name.
- Invalid selections keep the play button disabled without interruptive errors.

### Seat Plays

Every non-local player seat renders its latest action near the table-facing side. The local player's latest action renders above the hand area. `PASS` displays as a compact label. Multi-card actions use overlapping small cards.

The center remains reserved for the current winning action and global state. It should not be the only place to see who played what.

### Controls

The action controls stay close to the local hand:

- Hint
- Pass
- Play selected action
- AI speed: fast, normal, slow
- Debug mode toggle on desktop

Mobile controls should stay thumb-reachable in landscape.

### Debug Panel

`DebugPanel` shows:

- room id, seed, current player, turn count
- current legal action count
- full legal action list
- recent trace
- all four hands
- latest action per player
- timing metrics for state construction, AI decision, environment step, and broadcast
- compact JSON state snapshot for inspection

When it is the local player's turn, clicking a legal action in the panel selects the corresponding local hand cards.

## Backend Design

### State Contract

Split frontend state into two layers.

`play_state` is always safe for normal play:

- current player's visible hand for that client
- current player id
- human player ids
- card counts
- latest action per player
- current greater action and greater position
- rank list and current rank
- play team
- turn count
- round completion flag
- match over flag and result when available
- legal actions only for the acting human client

`debug_state` is sent only when debug is enabled and the viewer is authorized
by the host debug token:

- all player hands
- other player hands
- legal actions
- trace
- recent plays
- last actions
- bomb history
- finished players
- seed and room config
- timing metrics
- compact state snapshot

Normal public trial rooms must not receive hidden hands through `play_state`.

### AI Speed

The backend supports three AI advance modes.

- `fast`: advance AI/auto turns until the next human decision or match end, broadcasting only the final state plus summary timing.
- `normal`: broadcast each AI step with a moderate delay.
- `slow`: broadcast each AI step with a longer delay for observation.

The mode is room-scoped and can be changed during a game.

### Timing

`Game` records lightweight timings for GUI-facing steps:

- state construction
- AI `step`
- environment `step`
- broadcast preparation
- total AI advance loop

Timing data feeds `debug_state` and JSONL logs.

### Data Logging

The first release uses anonymous IDs:

- `participant_id`
- `session_id`
- `game_id`
- `account_id: null`

Free-form nickname/config fields are not persisted in the first release.

Write JSONL events for:

- room creation
- player join
- game start
- every human and AI action
- speed/debug mode changes
- disconnects
- match summary

Events include `schema_version: 1` for downstream import. Action events
include timestamp, room id, game id, player id, human/AI flag, action,
legal action count, current rank, remaining card counts, and timings such as
`state_ms`, `ai_decision_ms`, `env_step_ms`, `broadcast_ms`, and
`ai_advance_loop_ms` when available.

This is enough for algorithm strength analysis and user behavior review without requiring accounts. A future account system can bind anonymous participant ids to accounts.

## Files

Frontend:

- Modify `gui/frontend/src/components/GameBoard.js`
- Modify `gui/frontend/src/components/GameBoard.css`
- Modify `gui/frontend/src/components/HandCards.js`
- Modify `gui/frontend/src/components/HandCards.css`
- Create `gui/frontend/src/components/PlayerSeat.js`
- Create `gui/frontend/src/components/PlayerSeat.css`
- Create `gui/frontend/src/components/DebugPanel.js`
- Create `gui/frontend/src/components/DebugPanel.css`
- Create `gui/frontend/src/components/LandscapeGuard.js`
- Create `gui/frontend/src/components/LandscapeGuard.css`
- Create `gui/frontend/src/utils/cardSelection.js` if drag selection requires shared pure functions; otherwise keep the logic inside `HandCards.js`.

Backend:

- Modify `gui/backend/state_adapter.py`
- Modify `gui/backend/game_manager.py`
- Modify `gui/backend/server.py`
- Create `gui/backend/game_logger.py`

Tests:

- Create `tests/test_gui_state_adapter.py`
- Create `tests/test_gui_game_logger.py`
- Add frontend verification through `npm run build` and browser checks unless a frontend test harness is added separately.

## Verification

Required checks:

- `python -m pytest`
- `npm run build` from `gui/frontend`
- GUI state benchmark before and after backend state changes

Manual or browser verification:

- Desktop landscape table fits in one screen.
- Mobile portrait game view shows rotate prompt.
- Mobile landscape game view is playable.
- Press-drag selection selects multiple cards.
- Starting a drag on a selected card deselects swept cards.
- Each player's latest play appears in front of that player.
- Normal play mode does not expose other hands.
- Debug mode shows all hands.
- AI fast mode advances to the next human decision.
- AI normal and slow modes show step-by-step actions.
- JSONL logs contain action and summary events.

## Risks

- Exposing all hands in debug mode is useful for local algorithm work but would break public play if accidentally enabled. The implementation must keep hidden hands out of `play_state`.
- Landscape-only mobile play improves game quality but requires a clear portrait guard.
- Existing uncommitted GUI work overlaps with this design. Implementation must preserve or intentionally absorb that work without reverting it.
- Splitting play/debug state changes the Socket.IO contract. The frontend and backend must be updated together.

## Success Criteria

- Developers can use the desktop table to observe AI behavior, hidden hands, legal actions, trace, and timings.
- Public users can enter a mobile landscape table and play without seeing debug information.
- Human card selection is faster and less error-prone than click-only selection.
- GUI-facing state construction and AI advance timings are visible in debug mode.
- Existing game rule tests continue to pass.
- Frontend production build continues to pass.
