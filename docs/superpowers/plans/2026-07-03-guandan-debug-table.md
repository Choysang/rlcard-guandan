# Guandan Debug Table Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the first implementation of the horizontal Guandan play table with debug mode, per-client play state, debug hidden-hand visibility, AI speed control, GUI timings, and anonymous JSONL logging.

**Architecture:** Split the backend state contract into `play_state` and `debug_state`, emit viewer-specific play payloads per Socket.IO client, and keep debug data opt-in. The React table becomes a landscape four-seat surface composed from focused components: `PlayerSeat`, `HandCards`, `DebugPanel`, and `LandscapeGuard`.

**Tech Stack:** Python 3, pytest, Flask-SocketIO, React 18, Vite, Socket.IO client, Node built-in test runner for frontend pure selection logic.

---

## Workspace Notes

The repository currently has uncommitted GUI changes in:

- `gui/frontend/src/components/Card.css`
- `gui/frontend/src/components/Card.js`
- `gui/frontend/src/components/GameBoard.css`
- `gui/frontend/src/components/GameBoard.js`
- `gui/frontend/src/components/HandCards.css`
- `gui/frontend/src/components/HandCards.js`

Treat these as user-owned work. Do not revert them. Implementation should either preserve them or intentionally absorb them into the new table design.

Before editing a file with user changes, read the current file and diff:

```powershell
git status --short
git diff -- gui/frontend/src/components/GameBoard.js gui/frontend/src/components/HandCards.js
```

Expected: the six GUI files remain modified until frontend tasks intentionally edit them.

Run frontend `npm` commands from `D:\zuomian\guandan-rlcard\gui\frontend`. Run Python commands from `D:\zuomian\guandan-rlcard`.

## File Structure

Backend files:

- `gui/backend/state_adapter.py`: build viewer-safe `play_state` and opt-in `debug_state`.
- `gui/backend/game_logger.py`: append JSONL events and generate anonymous ids.
- `gui/backend/game_manager.py`: own room-level game metadata, speed mode, debug mode, timings, and logger calls.
- `gui/backend/server.py`: emit per-client state, handle speed/debug events, and pass anonymous participant metadata.

Frontend files:

- `gui/frontend/src/services/GuandanService.js`: add speed/debug events and preserve connection behavior.
- `gui/frontend/src/App.js`: store `debugState`, pass it to the board, and accept per-client state payloads.
- `gui/frontend/src/components/GameSetup.js`: add optional nickname and debug-room toggle for local testing.
- `gui/frontend/src/components/GameBoard.js`: horizontal table shell, speed/debug controls, selection/action orchestration.
- `gui/frontend/src/components/GameBoard.css`: landscape table layout.
- `gui/frontend/src/components/HandCards.js`: hand fan and drag-selection integration.
- `gui/frontend/src/components/HandCards.css`: stable hand sizing and pointer behavior.
- `gui/frontend/src/components/PlayerSeat.js`: seat info and latest play rendering.
- `gui/frontend/src/components/PlayerSeat.css`: seat positioning and latest-play presentation.
- `gui/frontend/src/components/DebugPanel.js`: hidden hands, legal actions, trace, timings.
- `gui/frontend/src/components/DebugPanel.css`: compact desktop debug drawer.
- `gui/frontend/src/components/LandscapeGuard.js`: portrait guard for mobile game view.
- `gui/frontend/src/components/LandscapeGuard.css`: orientation prompt styling.
- `gui/frontend/src/utils/cardSelection.js`: pure selection helpers for drag/click behavior.
- `gui/frontend/src/utils/cardSelection.test.js`: Node tests for selection helpers.
- `gui/frontend/package.json`: add `test:selection` script.

Test files:

- `tests/test_gui_state_adapter.py`
- `tests/test_gui_game_logger.py`
- `tests/test_gui_game_manager.py`

---

### Task 1: Backend State Contract

**Files:**

- Modify: `gui/backend/state_adapter.py`
- Create: `tests/test_gui_state_adapter.py`

- [ ] **Step 1: Write failing state adapter tests**

Create `tests/test_gui_state_adapter.py`:

```python
"""GUI state adapter contract tests."""

import numpy as np

import guandan_rlcard
from guandan_rlcard.baselines.random_agent import RandomAgent
from gui.backend.state_adapter import build_debug_state, build_play_state


def make_env(seed=17):
    env = guandan_rlcard.make({'seed': seed, 'perfect_info': True})
    rng = np.random.RandomState(seed)
    env.set_agents([RandomAgent(i, rng) for i in range(4)])
    env.reset()
    return env


def test_play_state_is_viewer_specific_and_hides_other_hands():
    env = make_env()
    state = build_play_state(env, human_player_ids=[0, 2], viewer_player_id=0)

    assert state['player_hands'].keys() == {0}
    assert state['player_hands'][0] == env.get_state(0)['current_hand']
    assert 'all_players_hands' not in state
    assert 'other_player_hands' not in state
    assert state['human_player_ids'] == [0, 2]


def test_play_state_only_includes_actions_for_acting_viewer():
    env = make_env()
    current = env.get_player_id()
    acting = build_play_state(env, [current], current)
    waiting = build_play_state(env, [current, (current + 1) % 4], (current + 1) % 4)

    assert acting['actions'] == env.get_state(current)['actions']
    assert waiting['actions'] == []


def test_debug_state_exposes_all_hands_and_timings():
    env = make_env()
    debug = build_debug_state(
        env,
        human_player_ids=[0],
        seed=123,
        room_config={'debug_enabled': True},
        timings={'state_ms': 1.5},
    )

    assert set(debug['all_player_hands'].keys()) == {0, 1, 2, 3}
    assert debug['other_player_hands'][0] == debug['all_player_hands'][0]
    assert debug['seed'] == 123
    assert debug['room_config']['debug_enabled'] is True
    assert debug['timings']['state_ms'] == 1.5
    assert isinstance(debug['legal_actions_by_player'], dict)
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```powershell
python -m pytest tests/test_gui_state_adapter.py -v
```

Expected: FAIL with import errors for `build_play_state` or `build_debug_state`.

- [ ] **Step 3: Implement state adapter functions**

Replace `gui/backend/state_adapter.py` with this contract-oriented implementation, keeping `build_frontend_state` as a compatibility wrapper:

```python
"""Translate engine state into GUI play/debug state contracts."""


def _recent_plays_from_trace(trace):
    recent = {pid: None for pid in range(4)}
    for pid, action in reversed(trace or []):
        if recent[pid] is None:
            recent[pid] = action
    return recent


def _result_fields(env):
    game = env.game
    if not env.is_over():
        return {}
    result = list(getattr(game.round, 'result', []))
    winner_team = game.winner_team
    if winner_team is None or winner_team < 0:
        winner_team = result[0] % 2 if result and result[0] >= 0 else 0
    return {
        'winner_team': winner_team,
        'finished_players': [p for p in result if p >= 0],
    }


def build_play_state(env, human_player_ids, viewer_player_id=None, timings=None):
    """Build the viewer-specific state safe for normal play."""
    game = env.game
    current_player = env.get_player_id()
    current_state = env.get_state(current_player)
    human_ids = list(human_player_ids)

    player_hands = {}
    if viewer_player_id in human_ids:
        player_hands[viewer_player_id] = env.get_state(
            viewer_player_id).get('current_hand', [])

    actions = []
    if viewer_player_id == current_player and viewer_player_id in human_ids:
        actions = current_state.get('actions', [])

    trace = current_state.get('trace', [])
    recent_plays = current_state.get('recent_plays') or _recent_plays_from_trace(trace)

    state = {
        'player_hands': player_hands,
        'actions': actions,
        'num_cards_left': current_state.get('num_cards_left', []),
        'trace_length': len(trace),
        'recent_plays': recent_plays,
        'last_actions': current_state.get('last_actions', {}),
        'greaterAction': current_state.get('greaterAction', []),
        'greaterPos': current_state.get('greaterPos', -1),
        'rank_list': current_state.get('rank_list', [0, 0]),
        'play_team': current_state.get('play_team', 0),
        'current_rank': getattr(game, 'cur_rank', 0),
        'turn_count': current_state.get('global_turn_count', 0),
        'round_completed': current_state.get('round_completed', False),
        'human_player_ids': human_ids,
        'viewer_player_id': viewer_player_id,
        'current_player': current_player,
        'is_over': env.is_over(),
        'timings': dict(timings or {}),
    }
    state.update(_result_fields(env))
    return state


def build_debug_state(env, human_player_ids, seed=None, room_config=None,
                      timings=None):
    """Build opt-in debugging state. This intentionally exposes all hands."""
    current_player = env.get_player_id()
    current_state = env.get_state(current_player)
    all_hands = {
        pid: env.get_state(pid).get('current_hand', [])
        for pid in range(env.num_players)
    }
    legal_actions_by_player = {
        pid: env.get_state(pid).get('actions', [])
        for pid in range(env.num_players)
    }

    keys = (
        'greaterAction', 'greaterPos', 'rank_list', 'play_team',
        'num_cards_left', 'recent_plays', 'last_actions', 'bomb_history',
        'finished_players', 'round_completed', 'global_turn_count',
    )
    snapshot = {key: current_state.get(key) for key in keys}

    return {
        'seed': seed,
        'room_config': dict(room_config or {}),
        'human_player_ids': list(human_player_ids),
        'current_player': current_player,
        'all_player_hands': all_hands,
        'other_player_hands': all_hands,
        'legal_actions_by_player': legal_actions_by_player,
        'trace': current_state.get('trace', []),
        'recent_plays': current_state.get('recent_plays', {}),
        'last_actions': current_state.get('last_actions', {}),
        'bomb_history': current_state.get('bomb_history', []),
        'timings': dict(timings or {}),
        'state_snapshot': snapshot,
    }


def build_frontend_state(env, human_player_ids):
    """Compatibility wrapper for older callers.

    The first human seat is used as the viewer. New Socket.IO code should
    call ``build_play_state`` per connected client.
    """
    viewer = human_player_ids[0] if human_player_ids else None
    return build_play_state(env, human_player_ids, viewer)
```

- [ ] **Step 4: Run tests and verify they pass**

Run:

```powershell
python -m pytest tests/test_gui_state_adapter.py -v
```

Expected: PASS.

- [ ] **Step 5: Run existing rule tests**

Run:

```powershell
python -m pytest
```

Expected: all existing tests pass.

- [ ] **Step 6: Commit**

```powershell
git add gui/backend/state_adapter.py tests/test_gui_state_adapter.py
git commit -m "feat(gui): split play and debug state"
```

---

### Task 2: Anonymous JSONL Game Logger

**Files:**

- Create: `gui/backend/game_logger.py`
- Create: `tests/test_gui_game_logger.py`

- [ ] **Step 1: Write failing logger tests**

Create `tests/test_gui_game_logger.py`:

```python
"""Tests for GUI JSONL logging."""

import json

from gui.backend.game_logger import GameLogger


def read_lines(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]


def test_logger_writes_event_with_ids_and_payload(tmp_path):
    logger = GameLogger(log_dir=tmp_path, clock=lambda: '2026-07-03T00:00:00Z')
    logger.write_event('game_start', {
        'room_id': 'ABC123',
        'game_id': 'game-1',
        'account_id': None,
    })

    lines = read_lines(tmp_path / 'guandan_gui.jsonl')
    assert lines == [{
        'event_type': 'game_start',
        'timestamp': '2026-07-03T00:00:00Z',
        'room_id': 'ABC123',
        'game_id': 'game-1',
        'account_id': None,
    }]


def test_logger_action_event_keeps_action_and_timings(tmp_path):
    logger = GameLogger(log_dir=tmp_path, clock=lambda: '2026-07-03T00:00:00Z')
    logger.log_action(
        room_id='ROOM01',
        game_id='game-2',
        participant_id='p-1',
        player_id=0,
        is_human=True,
        action=['Single', '3', ['S3']],
        legal_action_count=9,
        current_rank=0,
        num_cards_left=[26, 27, 27, 27],
        timings={'env_step_ms': 0.2},
        account_id=None,
    )

    line = read_lines(tmp_path / 'guandan_gui.jsonl')[0]
    assert line['event_type'] == 'action'
    assert line['participant_id'] == 'p-1'
    assert line['action'] == ['Single', '3', ['S3']]
    assert line['timings']['env_step_ms'] == 0.2


def test_id_helpers_generate_prefixed_values(tmp_path):
    logger = GameLogger(log_dir=tmp_path)

    assert logger.new_participant_id().startswith('participant_')
    assert logger.new_session_id().startswith('session_')
    assert logger.new_game_id().startswith('game_')
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```powershell
python -m pytest tests/test_gui_game_logger.py -v
```

Expected: FAIL with `ModuleNotFoundError` for `gui.backend.game_logger`.

- [ ] **Step 3: Implement `GameLogger`**

Create `gui/backend/game_logger.py`:

```python
"""Append-only JSONL logging for GUI games."""

import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_LOG_DIR = os.path.join('logs', 'gui')


def utc_now():
    return datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')


class GameLogger:
    """Small JSONL writer used by the Socket.IO GUI backend."""

    def __init__(self, log_dir=None, clock=utc_now):
        self.log_dir = Path(log_dir or os.environ.get(
            'GUANDAN_GUI_LOG_DIR', DEFAULT_LOG_DIR))
        self.clock = clock
        self.path = self.log_dir / 'guandan_gui.jsonl'

    @staticmethod
    def _new_id(prefix):
        return f'{prefix}_{uuid.uuid4().hex}'

    def new_participant_id(self):
        return self._new_id('participant')

    def new_session_id(self):
        return self._new_id('session')

    def new_game_id(self):
        return self._new_id('game')

    def write_event(self, event_type, payload):
        self.log_dir.mkdir(parents=True, exist_ok=True)
        event = {'event_type': event_type, 'timestamp': self.clock()}
        event.update(payload)
        with self.path.open('a', encoding='utf-8') as fh:
            fh.write(json.dumps(event, ensure_ascii=False, sort_keys=True))
            fh.write('\n')

    def log_action(self, room_id, game_id, participant_id, player_id,
                   is_human, action, legal_action_count, current_rank,
                   num_cards_left, timings, account_id=None):
        self.write_event('action', {
            'room_id': room_id,
            'game_id': game_id,
            'participant_id': participant_id,
            'account_id': account_id,
            'player_id': player_id,
            'is_human': bool(is_human),
            'action': action,
            'legal_action_count': legal_action_count,
            'current_rank': current_rank,
            'num_cards_left': list(num_cards_left),
            'timings': dict(timings or {}),
        })
```

- [ ] **Step 4: Run logger tests**

Run:

```powershell
python -m pytest tests/test_gui_game_logger.py -v
```

Expected: PASS.

- [ ] **Step 5: Run backend contract tests**

Run:

```powershell
python -m pytest tests/test_gui_state_adapter.py tests/test_gui_game_logger.py -v
```

Expected: PASS.

- [ ] **Step 6: Commit**

```powershell
git add gui/backend/game_logger.py tests/test_gui_game_logger.py
git commit -m "feat(gui): add anonymous game logger"
```

---

### Task 3: Game Manager Modes and Timings

**Files:**

- Modify: `gui/backend/game_manager.py`
- Create: `tests/test_gui_game_manager.py`

- [ ] **Step 1: Write failing game manager tests**

Create `tests/test_gui_game_manager.py`:

```python
"""GUI game manager metadata and mode tests."""

import pytest

from gui.backend.game_manager import Game


def test_game_mode_defaults_and_setters():
    game = Game({'human_player_ids': [0]}, seed=99)

    assert game.ai_speed == 'normal'
    assert game.debug_enabled is False
    assert game.seed == 99

    game.set_ai_speed('fast')
    game.set_debug_enabled(True)

    assert game.ai_speed == 'fast'
    assert game.debug_enabled is True


def test_game_rejects_unknown_ai_speed():
    game = Game({'human_player_ids': [0]}, seed=99)

    with pytest.raises(ValueError, match='Unsupported AI speed'):
        game.set_ai_speed('instant')


def test_frontend_payload_contains_play_and_optional_debug_state():
    game = Game({'human_player_ids': [0], 'debug_enabled': True}, seed=7)
    game.init_game()

    payload = game.frontend_payload(viewer_player_id=0)

    assert set(payload.keys()) == {'play_state', 'debug_state'}
    assert payload['play_state']['viewer_player_id'] == 0
    assert payload['debug_state']['seed'] == 7
    assert set(payload['debug_state']['all_player_hands'].keys()) == {0, 1, 2, 3}
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```powershell
python -m pytest tests/test_gui_game_manager.py -v
```

Expected: FAIL because `ai_speed`, `set_ai_speed`, `set_debug_enabled`, or `frontend_payload` do not exist.

- [ ] **Step 3: Add modes, timing helpers, and payload construction**

Modify `gui/backend/game_manager.py`:

```python
import time
import uuid
```

Change imports:

```python
from .state_adapter import build_debug_state, build_play_state
```

Add constants:

```python
AI_SPEEDS = {'fast', 'normal', 'slow'}
```

Extend `Game.__init__`:

```python
self.debug_enabled = bool(player_config.get('debug_enabled', False))
self.ai_speed = player_config.get('ai_speed', 'normal')
if self.ai_speed not in AI_SPEEDS:
    self.ai_speed = 'normal'
self.game_id = player_config.get('game_id') or f'game_{uuid.uuid4().hex}'
self.last_timings = {}
self.last_action_meta = None
```

Add methods to `Game`:

```python
def set_ai_speed(self, speed):
    if speed not in AI_SPEEDS:
        raise ValueError(f'Unsupported AI speed: {speed}')
    self.ai_speed = speed


def set_debug_enabled(self, enabled):
    self.debug_enabled = bool(enabled)


def _time_call(self, name, fn):
    start = time.perf_counter()
    result = fn()
    self.last_timings[name] = round((time.perf_counter() - start) * 1000, 3)
    return result


def frontend_payload(self, viewer_player_id=None, include_debug=None):
    include_debug = self.debug_enabled if include_debug is None else include_debug

    def build_play():
        return build_play_state(
            self.env,
            self.human_player_ids,
            viewer_player_id,
            timings=self.last_timings,
        )

    play_state = self._time_call('state_ms', build_play)
    debug_state = None
    if include_debug:
        debug_state = build_debug_state(
            self.env,
            self.human_player_ids,
            seed=self.seed,
            room_config=self.player_config,
            timings=self.last_timings,
        )
    return {'play_state': play_state, 'debug_state': debug_state}


def frontend_state(self):
    viewer = self.human_player_ids[0] if self.human_player_ids else None
    return self.frontend_payload(viewer)['play_state']
```

Update `step_one_ai` to time AI decision and environment step:

```python
state = self.env.get_state(pid)
actions = state.get('actions')

def choose_action():
    return self.agents[pid].step(state) if actions else []

action = self._time_call('ai_decision_ms', choose_action)
self._time_call('env_step_ms', lambda: self.env.step(action))
self.last_action_meta = {
    'player_id': pid,
    'is_human': False,
    'action': action,
    'legal_action_count': len(actions or []),
}
return True
```

Update `perform_action` after matching the action:

```python
self._time_call('env_step_ms', lambda: self.env.step(action))
self.last_action_meta = {
    'player_id': player_id,
    'is_human': True,
    'action': action,
    'legal_action_count': len(legal_actions),
}
```

Do not keep the old direct `self.env.step(action)` line after adding the timed call.

- [ ] **Step 4: Run game manager tests**

Run:

```powershell
python -m pytest tests/test_gui_game_manager.py -v
```

Expected: PASS.

- [ ] **Step 5: Run all backend tests**

Run:

```powershell
python -m pytest tests/test_gui_state_adapter.py tests/test_gui_game_logger.py tests/test_gui_game_manager.py -v
```

Expected: PASS.

- [ ] **Step 6: Commit**

```powershell
git add gui/backend/game_manager.py tests/test_gui_game_manager.py
git commit -m "feat(gui): add debug and speed modes"
```

---

### Task 4: Socket.IO Per-Client State and Logging

**Files:**

- Modify: `gui/backend/server.py`
- Modify: `gui/backend/game_manager.py` if logger integration needs one helper

- [ ] **Step 1: Read server state flow before editing**

Run:

```powershell
Get-Content -Raw gui/backend/server.py
```

Confirm current flow:

- `_broadcast_state` emits one room-wide `state`.
- `rooms[room_id]['players']` maps Socket.IO sid to seat.
- `_drive_ai` controls AI animation.

- [ ] **Step 2: Replace room-wide state emission with per-client emission**

In `gui/backend/server.py`, import the logger:

```python
from .game_logger import GameLogger
```

Add constants and logger near the existing globals:

```python
AI_SPEED_DELAYS = {
    'fast': 0.0,
    'normal': AI_TURN_DELAY,
    'slow': float(os.environ.get('GUANDAN_GUI_AI_SLOW_DELAY', '1.6')),
}
game_logger = GameLogger()
```

Replace `_broadcast_state` with `_emit_state_to_room`:

```python
def _emit_state_to_room(room_id, event='update_state'):
    room = rooms.get(room_id)
    if not room:
        return
    game = room['game']
    for sid, seat in list(room['players'].items()):
        payload = game.frontend_payload(viewer_player_id=seat)
        socketio.emit(event, {
            'state': payload['play_state'],
            'debug_state': payload['debug_state'],
            'current_player': game.current_player(),
            'viewer_player_id': seat,
            'room_id': room_id,
        }, to=sid)
```

Change every `_broadcast_state(room_id, ...)` call to `_emit_state_to_room(room_id, ...)`.

- [ ] **Step 3: Add AI speed behavior to `_drive_ai`**

Replace `_drive_ai` body with:

```python
def _drive_ai(room_id):
    """Drive AI/auto turns according to the room speed mode."""
    room = rooms.get(room_id)
    if not room:
        return
    game = room['game']

    if game.ai_speed == 'fast':
        while not game.is_over() and not game.is_waiting_for_human():
            if room_id not in rooms:
                return
            if not game.step_one_ai():
                break
            _log_last_action(room_id)
        _emit_state_to_room(room_id)
        return

    delay = AI_SPEED_DELAYS.get(game.ai_speed, AI_TURN_DELAY)
    while not game.is_over() and not game.is_waiting_for_human():
        socketio.sleep(delay)
        if room_id not in rooms:
            return
        if not game.step_one_ai():
            break
        _log_last_action(room_id)
        _emit_state_to_room(room_id)
```

- [ ] **Step 4: Add logging helper**

Add helper in `server.py`:

```python
def _participant_for(room, player_id):
    for sid, seat in room['players'].items():
        if seat == player_id:
            return room['participants'].get(sid)
    return None


def _log_last_action(room_id):
    room = rooms.get(room_id)
    if not room:
        return
    game = room['game']
    meta = game.last_action_meta
    if not meta:
        return
    player_id = meta['player_id']
    participant_id = _participant_for(room, player_id)
    if participant_id is None:
        participant_id = f'ai_{player_id}'
    state = game.env.get_state(game.current_player())
    game_logger.log_action(
        room_id=room_id,
        game_id=game.game_id,
        participant_id=participant_id,
        player_id=player_id,
        is_human=meta['is_human'],
        action=meta['action'],
        legal_action_count=meta['legal_action_count'],
        current_rank=getattr(game.env.game, 'cur_rank', 0),
        num_cards_left=state.get('num_cards_left', []),
        timings=game.last_timings,
    )
```

- [ ] **Step 5: Store participant IDs when players create/join rooms**

In `handle_create_room`, add:

```python
participant_id = game_logger.new_participant_id()
```

In the `rooms[room_id]` dict, add:

```python
'participants': {request.sid: participant_id},
```

After room creation, log:

```python
game_logger.write_event('room_created', {
    'room_id': room_id,
    'game_id': game.game_id,
    'participant_id': participant_id,
    'player_id': creator_seat,
    'account_id': None,
    'player_config': config,
})
```

Emit `participantId` to the creator:

```python
emit('room_created', {
    'roomId': room_id,
    'playerId': creator_seat,
    'participantId': participant_id,
})
```

In `handle_join_room`, generate and store a participant ID:

```python
participant_id = game_logger.new_participant_id()
room['participants'][request.sid] = participant_id
```

Emit it:

```python
emit('joined_room', {
    'roomId': room_id,
    'playerId': seat,
    'participantId': participant_id,
})
```

Log join:

```python
game_logger.write_event('player_joined', {
    'room_id': room_id,
    'game_id': room['game'].game_id,
    'participant_id': participant_id,
    'player_id': seat,
    'account_id': None,
})
```

In disconnect cleanup, remove `room['participants'][request.sid]`.

- [ ] **Step 6: Add speed/debug Socket.IO events**

Add handlers:

```python
@socketio.on('set_ai_speed')
def handle_set_ai_speed(data):
    data = data or {}
    room = rooms.get(data.get('roomId'))
    if not room:
        emit('error', {'message': '游戏房间未找到。'})
        return
    try:
        room['game'].set_ai_speed(data.get('speed'))
    except ValueError as exc:
        emit('error', {'message': str(exc)})
        return
    game_logger.write_event('ai_speed_changed', {
        'room_id': data.get('roomId'),
        'game_id': room['game'].game_id,
        'speed': room['game'].ai_speed,
        'account_id': None,
    })
    _emit_state_to_room(data.get('roomId'))


@socketio.on('set_debug_mode')
def handle_set_debug_mode(data):
    data = data or {}
    room = rooms.get(data.get('roomId'))
    if not room:
        emit('error', {'message': '游戏房间未找到。'})
        return
    room['game'].set_debug_enabled(bool(data.get('enabled')))
    game_logger.write_event('debug_mode_changed', {
        'room_id': data.get('roomId'),
        'game_id': room['game'].game_id,
        'debug_enabled': room['game'].debug_enabled,
        'account_id': None,
    })
    _emit_state_to_room(data.get('roomId'))
```

- [ ] **Step 7: Log human actions and summaries**

After `game.perform_action(...)` succeeds in `handle_player_action`, call:

```python
_log_last_action(room_id)
```

When `game.is_over()` is true after a human action or after `_drive_ai`, write a `match_summary` event once. Use a room flag:

```python
if game.is_over() and not room.get('summary_logged'):
    room['summary_logged'] = True
    game_logger.write_event('match_summary', {
        'room_id': room_id,
        'game_id': game.game_id,
        'winner_team': game.env.game.winner_team,
        'account_id': None,
    })
```

- [ ] **Step 8: Run backend tests**

Run:

```powershell
python -m pytest tests/test_gui_state_adapter.py tests/test_gui_game_logger.py tests/test_gui_game_manager.py -v
python -m pytest
```

Expected: all tests pass.

- [ ] **Step 9: Commit**

```powershell
git add gui/backend/server.py gui/backend/game_manager.py
git commit -m "feat(gui): emit viewer-specific state"
```

---

### Task 5: Frontend Selection Utilities

**Files:**

- Create: `gui/frontend/src/utils/cardSelection.js`
- Create: `gui/frontend/src/utils/cardSelection.test.js`
- Modify: `gui/frontend/package.json`

- [ ] **Step 1: Add failing pure selection tests**

Create `gui/frontend/src/utils/cardSelection.test.js`:

```javascript
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
```

Add script to `gui/frontend/package.json`:

```json
"test:selection": "node --test src/utils/cardSelection.test.js"
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```powershell
npm run test:selection
```

Expected: FAIL because `cardSelection.js` does not exist.

- [ ] **Step 3: Implement selection helpers**

Create `gui/frontend/src/utils/cardSelection.js`:

```javascript
export const sortedUnique = (indexes) => (
  [...new Set(indexes)].sort((a, b) => a - b)
);

export const dragModeForIndex = (selectedIndexes, index) => (
  selectedIndexes.includes(index) ? 'deselect' : 'select'
);

export const applyCardSelection = (selectedIndexes, index, mode) => {
  if (mode === 'select') {
    return sortedUnique([...selectedIndexes, index]);
  }
  if (mode === 'deselect') {
    return selectedIndexes.filter((i) => i !== index);
  }
  return sortedUnique(selectedIndexes);
};
```

- [ ] **Step 4: Run selection tests**

Run:

```powershell
npm run test:selection
```

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add gui/frontend/package.json gui/frontend/src/utils/cardSelection.js gui/frontend/src/utils/cardSelection.test.js
git commit -m "test(gui): cover card drag selection helpers"
```

---

### Task 6: Frontend Socket Data Plumbing

**Files:**

- Modify: `gui/frontend/src/services/GuandanService.js`
- Modify: `gui/frontend/src/App.js`
- Modify: `gui/frontend/src/components/GameSetup.js`

- [ ] **Step 1: Read current frontend data flow and dirty diff**

Run:

```powershell
git diff -- gui/frontend/src/components/GameBoard.js gui/frontend/src/components/HandCards.js
Get-Content -Raw gui/frontend/src/App.js
Get-Content -Raw gui/frontend/src/services/GuandanService.js
```

Confirm `App` currently stores only `gameState` and `currentPlayer`.

- [ ] **Step 2: Add service methods**

In `GuandanService.js`, add methods:

```javascript
setAiSpeed(roomId, speed) {
  if (!this.socket) return;
  this.socket.emit('set_ai_speed', { roomId, speed });
}

setDebugMode(roomId, enabled) {
  if (!this.socket) return;
  this.socket.emit('set_debug_mode', { roomId, enabled });
}
```

- [ ] **Step 3: Store debug state and participant id in `App`**

In `App.js`, add state:

```javascript
const [debugState, setDebugState] = useState(null);
const [participantId, setParticipantId] = useState(null);
```

Update `room_created` and `joined_room` listeners:

```javascript
guandanService.on('room_created', ({ roomId, playerId, participantId }) => {
  setRoomId(roomId);
  setPlayerId(playerId);
  setParticipantId(participantId || null);
  setAppState('WaitingInRoom');
  setLoading(false);
});
```

Use the same `setParticipantId(participantId || null)` in `joined_room`.

Update `game_started` and `update_state` listeners:

```javascript
guandanService.on('game_started', ({ state, debug_state, current_player }) => {
  setGameState(state);
  setDebugState(debug_state || null);
  setCurrentPlayer(current_player);
  setHumanPlayerIds(state.human_player_ids || []);
  setAppState('InGame');
  setLoading(false);
});

guandanService.on('update_state', ({ state, debug_state, current_player }) => {
  setGameState(state);
  setDebugState(debug_state || null);
  setCurrentPlayer(current_player);
  setHumanPlayerIds(state.human_player_ids || []);
});
```

Pass new props to `GameBoard`:

```javascript
debugState={debugState}
roomId={roomId}
participantId={participantId}
onSetAiSpeed={(speed) => guandanService.setAiSpeed(roomId, speed)}
onSetDebugMode={(enabled) => guandanService.setDebugMode(roomId, enabled)}
```

In `handleRestart`, reset `debugState` and `participantId`.

- [ ] **Step 4: Add setup fields**

In `GameSetup.js`, add local state:

```javascript
const [nickname, setNickname] = useState('');
const [debugEnabled, setDebugEnabled] = useState(false);
```

In `handleStartGame`, include:

```javascript
onGameStart({
  agentTypes,
  human_player_ids,
  nickname: nickname.trim(),
  debug_enabled: debugEnabled,
});
```

Add simple controls above the start button:

```jsx
<div className="setup-extra-options">
  <label htmlFor="nickname-input">昵称（可选）</label>
  <input
    id="nickname-input"
    type="text"
    value={nickname}
    onChange={(e) => setNickname(e.target.value)}
    maxLength="24"
    placeholder="调测玩家"
  />
  <label className="debug-room-toggle">
    <input
      type="checkbox"
      checked={debugEnabled}
      onChange={(e) => setDebugEnabled(e.target.checked)}
    />
    <span>创建调测房间（显示明牌和耗时）</span>
  </label>
</div>
```

- [ ] **Step 5: Build frontend**

Run:

```powershell
npm run build
```

Expected: PASS.

- [ ] **Step 6: Commit**

```powershell
git add gui/frontend/src/services/GuandanService.js gui/frontend/src/App.js gui/frontend/src/components/GameSetup.js
git commit -m "feat(gui): plumb debug state to frontend"
```

---

### Task 7: Landscape Table Components

**Files:**

- Create: `gui/frontend/src/components/PlayerSeat.js`
- Create: `gui/frontend/src/components/PlayerSeat.css`
- Create: `gui/frontend/src/components/DebugPanel.js`
- Create: `gui/frontend/src/components/DebugPanel.css`
- Create: `gui/frontend/src/components/LandscapeGuard.js`
- Create: `gui/frontend/src/components/LandscapeGuard.css`
- Modify: `gui/frontend/src/components/GameBoard.js`
- Modify: `gui/frontend/src/components/GameBoard.css`
- Modify: `gui/frontend/src/components/PlayArea.js` if the center table needs to shrink to current-holder state only

- [ ] **Step 1: Create `PlayerSeat`**

Create `gui/frontend/src/components/PlayerSeat.js`:

```javascript
import Card from './Card';
import './PlayerSeat.css';

const renderActionCards = (action) => {
  if (!action) return null;
  if (action[0] === 'PASS') {
    return <span className="seat-pass-chip">不出</span>;
  }
  return (
    <div className="seat-play-cards">
      {(action[2] || []).map((card, index) => (
        <Card key={`${card}-${index}`} cardString={card} size="small" />
      ))}
    </div>
  );
};

const PlayerSeat = ({
  position,
  playerId,
  playerName,
  cardCount,
  isActive,
  isHuman,
  latestAction,
  debugHand,
}) => (
  <section className={`player-seat player-seat-${position} ${isActive ? 'is-active' : ''}`}>
    <div className="seat-meta">
      <div className="seat-avatar">{isHuman ? '人' : 'AI'}</div>
      <div>
        <div className="seat-name">{playerName}</div>
        <div className="seat-count">{cardCount} 张</div>
      </div>
    </div>
    <div className="seat-latest-play">{renderActionCards(latestAction)}</div>
    {debugHand && (
      <div className="seat-debug-hand">
        {debugHand.map((card, index) => (
          <Card key={`debug-${playerId}-${card}-${index}`} cardString={card} size="small" />
        ))}
      </div>
    )}
  </section>
);

export default PlayerSeat;
```

Create `PlayerSeat.css` with stable dimensions:

```css
.player-seat {
  position: relative;
  color: #fff;
  min-width: 104px;
  min-height: 72px;
}

.seat-meta {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 6px 8px;
  background: rgba(0, 0, 0, 0.26);
  border: 1px solid rgba(255, 255, 255, 0.18);
  border-radius: 8px;
}

.seat-avatar {
  width: 30px;
  height: 30px;
  display: grid;
  place-items: center;
  border-radius: 50%;
  background: #1f7a5c;
  font-size: 12px;
  font-weight: 700;
}

.player-seat.is-active .seat-meta {
  border-color: #ffd166;
  box-shadow: 0 0 0 2px rgba(255, 209, 102, 0.4);
}

.seat-name {
  font-size: 13px;
  font-weight: 700;
}

.seat-count {
  font-size: 12px;
  opacity: 0.8;
}

.seat-latest-play {
  position: absolute;
  display: flex;
  align-items: center;
  min-height: 42px;
  pointer-events: none;
}

.seat-play-cards {
  display: flex;
  filter: drop-shadow(0 3px 6px rgba(0, 0, 0, 0.35));
}

.seat-play-cards .card + .card {
  margin-left: -18px;
}

.seat-pass-chip {
  padding: 4px 10px;
  border-radius: 999px;
  background: rgba(0, 0, 0, 0.55);
  border: 1px solid rgba(255, 255, 255, 0.2);
  font-size: 12px;
  font-weight: 700;
}

.player-seat-top .seat-latest-play {
  left: 50%;
  bottom: -42px;
  transform: translateX(-50%);
}

.player-seat-left .seat-latest-play {
  left: 100%;
  top: 50%;
  transform: translate(10px, -50%);
}

.player-seat-right .seat-latest-play {
  right: 100%;
  top: 50%;
  transform: translate(-10px, -50%);
}

.player-seat-bottom .seat-latest-play {
  left: 50%;
  top: -48px;
  transform: translateX(-50%);
}

.seat-debug-hand {
  display: flex;
  max-width: 240px;
  overflow: hidden;
  margin-top: 6px;
}

.seat-debug-hand .card + .card {
  margin-left: -22px;
}
```

- [ ] **Step 2: Create `LandscapeGuard`**

Create `LandscapeGuard.js`:

```javascript
import './LandscapeGuard.css';

const LandscapeGuard = ({ children }) => (
  <>
    <div className="landscape-guard">
      <div className="rotate-device-icon">↻</div>
      <div className="rotate-title">请横屏游戏</div>
      <div className="rotate-copy">横屏可以看清手牌和每家出牌。</div>
    </div>
    <div className="landscape-content">{children}</div>
  </>
);

export default LandscapeGuard;
```

Create `LandscapeGuard.css`:

```css
.landscape-guard {
  display: none;
}

.landscape-content {
  min-height: 0;
}

@media (max-width: 900px) and (orientation: portrait) {
  .landscape-content {
    display: none;
  }

  .landscape-guard {
    min-height: 100vh;
    display: grid;
    place-content: center;
    gap: 10px;
    padding: 24px;
    text-align: center;
    color: #fff;
    background: #123528;
  }

  .rotate-device-icon {
    font-size: 44px;
    line-height: 1;
  }

  .rotate-title {
    font-size: 22px;
    font-weight: 800;
  }

  .rotate-copy {
    font-size: 14px;
    opacity: 0.82;
  }
}
```

- [ ] **Step 3: Create `DebugPanel`**

Create `DebugPanel.js`:

```javascript
import Card from './Card';
import './DebugPanel.css';

const DebugPanel = ({ debugState, currentPlayer, onSelectAction }) => {
  if (!debugState) return null;
  const legalActions = debugState.legal_actions_by_player?.[currentPlayer] || [];

  return (
    <aside className="debug-panel">
      <header className="debug-panel-header">
        <h2>调测</h2>
        <span>Seat {currentPlayer}</span>
      </header>

      <section className="debug-section">
        <h3>耗时</h3>
        {Object.entries(debugState.timings || {}).map(([key, value]) => (
          <div key={key} className="debug-row">
            <span>{key}</span>
            <strong>{value} ms</strong>
          </div>
        ))}
      </section>

      <section className="debug-section">
        <h3>四家手牌</h3>
        {Object.entries(debugState.all_player_hands || {}).map(([pid, hand]) => (
          <div key={pid} className="debug-hand-row">
            <span className="debug-seat-label">P{pid}</span>
            <div className="debug-hand-cards">
              {hand.map((card, index) => (
                <Card key={`${pid}-${card}-${index}`} cardString={card} size="small" />
              ))}
            </div>
          </div>
        ))}
      </section>

      <section className="debug-section">
        <h3>当前合法动作 {legalActions.length}</h3>
        <div className="debug-actions">
          {legalActions.slice(0, 80).map((action, index) => (
            <button
              key={`${action[0]}-${action[1]}-${index}`}
              type="button"
              onClick={() => onSelectAction(action)}
              className="debug-action-btn"
            >
              {action[0]} {action[1]} {(action[2] || []).join(' ')}
            </button>
          ))}
        </div>
      </section>

      <section className="debug-section">
        <h3>Trace</h3>
        <pre>{JSON.stringify((debugState.trace || []).slice(-20), null, 2)}</pre>
      </section>
    </aside>
  );
};

export default DebugPanel;
```

Create `DebugPanel.css` with a fixed desktop drawer:

```css
.debug-panel {
  width: min(360px, 32vw);
  height: 100%;
  overflow: auto;
  padding: 12px;
  background: #111a1f;
  color: #e9f2ef;
  border-left: 1px solid rgba(255, 255, 255, 0.12);
}

.debug-panel-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 12px;
}

.debug-panel h2,
.debug-panel h3 {
  margin: 0;
}

.debug-panel h3 {
  font-size: 13px;
  margin-bottom: 8px;
}

.debug-section {
  padding: 10px 0;
  border-top: 1px solid rgba(255, 255, 255, 0.1);
}

.debug-row {
  display: flex;
  justify-content: space-between;
  gap: 12px;
  font-size: 12px;
}

.debug-hand-row {
  display: grid;
  grid-template-columns: 28px 1fr;
  gap: 6px;
  align-items: center;
  margin-bottom: 6px;
}

.debug-hand-cards {
  display: flex;
  overflow: hidden;
}

.debug-hand-cards .card + .card {
  margin-left: -24px;
}

.debug-actions {
  display: grid;
  gap: 4px;
}

.debug-action-btn {
  min-height: 28px;
  text-align: left;
  border: 1px solid rgba(255, 255, 255, 0.12);
  border-radius: 6px;
  color: #e9f2ef;
  background: rgba(255, 255, 255, 0.06);
  cursor: pointer;
}

.debug-panel pre {
  margin: 0;
  white-space: pre-wrap;
  font-size: 11px;
}

@media (max-width: 900px) {
  .debug-panel {
    display: none;
  }
}
```

- [ ] **Step 4: Integrate components in `GameBoard`**

Change imports:

```javascript
import DebugPanel from './DebugPanel';
import LandscapeGuard from './LandscapeGuard';
import PlayerSeat from './PlayerSeat';
```

Use `gameState.recent_plays` instead of rebuilding latest actions from full trace:

```javascript
const latestActions = gameState?.recent_plays || {};
```

Add props:

```javascript
debugState,
roomId,
participantId,
onSetAiSpeed,
onSetDebugMode,
```

Add local UI state:

```javascript
const [debugOpen, setDebugOpen] = useState(Boolean(debugState));
const [aiSpeed, setAiSpeed] = useState('normal');
```

Add handlers:

```javascript
const handleSpeedChange = (speed) => {
  setAiSpeed(speed);
  onSetAiSpeed(speed);
};

const handleDebugToggle = () => {
  const next = !debugOpen;
  setDebugOpen(next);
  onSetDebugMode(next);
};
```

Replace opponent rendering with `PlayerSeat` calls for top, left, and right. Render the bottom seat's latest play above `HandCards`.

Wrap the returned table:

```jsx
<LandscapeGuard>
  <div className={`game-board-shell ${debugOpen ? 'debug-open' : ''}`}>
    <div className="game-board">
      {/* horizontal table */}
    </div>
    {debugOpen && (
      <DebugPanel
        debugState={debugState}
        currentPlayer={currentPlayer}
        onSelectAction={handleDebugActionSelect}
      />
    )}
  </div>
</LandscapeGuard>
```

Add `handleDebugActionSelect`:

```javascript
const handleDebugActionSelect = (action) => {
  if (!isPlayerTurn || !action || action[0] === 'PASS') return;
  const indicesToSelect = [];
  const used = new Set();
  for (const card of action[2] || []) {
    const index = playerHand.findIndex((handCard, i) => handCard === card && !used.has(i));
    if (index >= 0) {
      indicesToSelect.push(index);
      used.add(index);
    }
  }
  setSelectedIndices(indicesToSelect.sort((a, b) => a - b));
};
```

- [ ] **Step 5: Rework `GameBoard.css`**

Keep the existing card styles, then replace the board layout with these stable regions:

```css
.game-board-shell {
  width: 100%;
  min-height: 0;
  flex: 1;
  display: flex;
  background: #123528;
}

.game-board {
  flex: 1;
  min-width: 0;
  min-height: 0;
  display: grid;
  grid-template-rows: 96px minmax(160px, 1fr) 170px;
  grid-template-columns: 150px minmax(420px, 1fr) 150px;
  grid-template-areas:
    ". top ."
    "left center right"
    "bottom bottom bottom";
  gap: 10px;
  padding: 10px 14px;
  overflow: hidden;
}

.top-player-zone { grid-area: top; justify-self: center; }
.left-player-zone { grid-area: left; align-self: center; }
.right-player-zone { grid-area: right; align-self: center; justify-self: end; }
.bottom-player-zone { grid-area: bottom; position: relative; }
.center-table { grid-area: center; min-height: 0; }

.board-toolbar {
  position: absolute;
  top: 10px;
  right: 14px;
  display: flex;
  gap: 6px;
  z-index: 20;
}

.speed-segment {
  display: flex;
  border: 1px solid rgba(255, 255, 255, 0.2);
  border-radius: 8px;
  overflow: hidden;
}

.speed-segment button,
.debug-toggle {
  min-height: 30px;
  border: 0;
  padding: 0 10px;
  color: #fff;
  background: rgba(0, 0, 0, 0.35);
  cursor: pointer;
}

.speed-segment button.active,
.debug-toggle.active {
  background: #1f7a5c;
}

@media (max-width: 900px) and (orientation: landscape) {
  .game-board {
    grid-template-rows: 68px minmax(110px, 1fr) 138px;
    grid-template-columns: 104px minmax(280px, 1fr) 104px;
    gap: 6px;
    padding: 6px 8px;
  }
}
```

Adjust class names in JSX to match these zones.

- [ ] **Step 6: Build frontend**

Run:

```powershell
npm run build
```

Expected: PASS.

- [ ] **Step 7: Commit**

```powershell
git add gui/frontend/src/components/PlayerSeat.js gui/frontend/src/components/PlayerSeat.css gui/frontend/src/components/DebugPanel.js gui/frontend/src/components/DebugPanel.css gui/frontend/src/components/LandscapeGuard.js gui/frontend/src/components/LandscapeGuard.css gui/frontend/src/components/GameBoard.js gui/frontend/src/components/GameBoard.css gui/frontend/src/components/PlayArea.js
git commit -m "feat(gui): add landscape debug table"
```

---

### Task 8: HandCards Drag Selection Integration

**Files:**

- Modify: `gui/frontend/src/components/HandCards.js`
- Modify: `gui/frontend/src/components/HandCards.css`

- [ ] **Step 1: Inspect existing dirty implementation**

Run:

```powershell
git diff -- gui/frontend/src/components/HandCards.js gui/frontend/src/components/HandCards.css
```

Confirm whether the existing pointer-drag implementation is still present. Keep the behavior if it already matches the design.

- [ ] **Step 2: Integrate pure helpers**

In `HandCards.js`, import:

```javascript
import { applyCardSelection, dragModeForIndex } from '../utils/cardSelection';
```

Use helpers in pointer handlers:

```javascript
const applyCard = (index, mode) => {
  setSelected((prev) => applyCardSelection(prev, index, mode));
};

const handlePointerDown = (e) => {
  if (!isInteractive) return;
  const index = cardIndexAtPoint(e.clientX, e.clientY);
  if (index == null) return;
  e.preventDefault();
  try { e.currentTarget.setPointerCapture(e.pointerId); } catch { /* pointer capture can fail in tests */ }
  const mode = dragModeForIndex(selectedCards, index);
  dragModeRef.current = mode;
  processedRef.current = new Set([index]);
  applyCard(index, mode);
};
```

In keyboard handling:

```javascript
const mode = isSelected ? 'deselect' : 'select';
applyCard(index, mode);
```

- [ ] **Step 3: Keep stable CSS**

Ensure `HandCards.css` includes:

```css
.hand-cards {
  touch-action: none;
  user-select: none;
  -webkit-user-select: none;
}

.hand-card {
  touch-action: none;
}
```

Keep selected cards lifted by 20 to 24 px and keep card rank corners visible.

- [ ] **Step 4: Run frontend tests and build**

Run:

```powershell
npm run test:selection
npm run build
```

Expected: both pass.

- [ ] **Step 5: Commit**

```powershell
git add gui/frontend/src/components/HandCards.js gui/frontend/src/components/HandCards.css
git commit -m "feat(gui): integrate drag hand selection"
```

---

### Task 9: Full Verification and Baseline Comparison

**Files:**

- No required source edits unless verification finds a regression.

- [ ] **Step 1: Run Python tests**

Run:

```powershell
python -m pytest
```

Expected: all tests pass.

- [ ] **Step 2: Run frontend tests and build**

Run:

```powershell
npm run test:selection
npm run build
```

Expected: both pass.

- [ ] **Step 3: Run GUI state benchmark**

Run:

```powershell
@'
import time
import numpy as np
import guandan_rlcard
from guandan_rlcard.baselines.random_agent import RandomAgent
from gui.backend.state_adapter import build_debug_state, build_play_state

env = guandan_rlcard.make({'seed': 0, 'perfect_info': True})
rng = np.random.RandomState(0)
env.set_agents([RandomAgent(i, rng) for i in range(4)])
state, pid = env.reset()
play_times = []
debug_times = []
for _ in range(300):
    if env.is_over():
        break
    t0 = time.perf_counter()
    build_play_state(env, [0], 0)
    play_times.append(time.perf_counter() - t0)
    t0 = time.perf_counter()
    build_debug_state(env, [0], seed=0, room_config={}, timings={})
    debug_times.append(time.perf_counter() - t0)
    action = env.agents[pid].step(state)
    state, pid = env.step(action)
print(f'play_state_avg_ms={sum(play_times)/len(play_times)*1000:.3f}')
print(f'debug_state_avg_ms={sum(debug_times)/len(debug_times)*1000:.3f}')
'@ | python -
```

Expected: prints average state construction times. Record these numbers in the final response.

- [ ] **Step 4: Run local GUI server**

Build frontend first:

```powershell
npm run build
```

From repo root:

```powershell
$env:GUANDAN_GUI_AI_DELAY='0.3'
python -m gui.backend.server
```

Expected: server starts on `http://localhost:5000`.

- [ ] **Step 5: Browser checks**

Open `http://localhost:5000` and verify:

- Desktop table is landscape and fits in one screen.
- Creating a normal room does not expose other hands.
- Creating a debug room exposes all hands in `DebugPanel`.
- AI speed buttons switch fast, normal, and slow without disconnecting.
- Each player's latest play appears in front of that seat.
- Press-drag selection selects multiple cards.
- Starting drag on a selected card deselects swept cards.

- [ ] **Step 6: Mobile orientation check**

Use browser device emulation:

- Portrait width below 900 px shows rotate prompt.
- Landscape width below 900 px shows the table.

- [ ] **Step 7: Log check**

After a short game sequence, inspect:

```powershell
Get-Content logs/gui/guandan_gui.jsonl -Tail 5
```

Expected: recent JSONL lines include `action` events with `participant_id`, `game_id`, `action`, `legal_action_count`, and `timings`.

- [ ] **Step 8: Final status**

Run:

```powershell
git status --short
```

Expected: no unexpected unstaged changes. If the original GUI dirty changes were absorbed, they should be committed through the frontend tasks.

---

## Execution Order

Execute tasks in this order:

1. Backend state contract.
2. JSONL logger.
3. Game manager modes and timings.
4. Socket.IO per-client state and logging.
5. Frontend selection utilities.
6. Frontend socket data plumbing.
7. Landscape table components.
8. HandCards drag-selection integration.
9. Full verification.

Backend tasks come first because frontend debug UI depends on the new payload shape. The pure selection helper task comes before React integration so drag behavior has a small automated test.
