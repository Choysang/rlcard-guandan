# Human Expert Data Collection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Capture human GUI decisions, join them with match outcomes, and export winning human decisions against strong agents as `question`/`sentence` JSONL.

**Architecture:** Add a small pure helper module for expert-data serialization and export logic, extend the existing append-only `GameLogger`, and connect the GUI server to log one pre-step snapshot per human action plus one outcome event per completed match. Public exports use only visible state fields and never include debug hidden hands.

**Tech Stack:** Python 3, Flask-Socket.IO backend, existing `guandan_rlcard` environment state dicts, JSONL logs, pytest.

---

## File Map

- Create `gui/backend/expert_data.py`: pure functions for snapshot building, expert filtering, prompt/rationale generation, and JSONL export.
- Modify `gui/backend/game_logger.py`: add typed logger methods for `decision_snapshot` and `game_outcome`.
- Modify `gui/backend/game_manager.py`: capture the pre-step decision snapshot for human moves before `env.step(action)`.
- Modify `gui/backend/server.py`: attach participant/session identity and write decision snapshots and outcome events.
- Create `examples/export_human_expert_dataset.py`: command-line exporter from GUI JSONL logs to training JSONL.
- Modify `docs/gui_guide.md`: document where human expert logs live and how to export them.
- Modify `README.md`: add a short link to the expert-data export command.
- Create `tests/test_expert_data.py`: unit tests for snapshot construction, filtering, and prompt generation.
- Modify `tests/test_gui_game_logger.py`: test new logger methods.
- Modify `tests/test_gui_game_manager.py`: test pre-step human snapshot capture.
- Modify `tests/test_gui_server.py`: integration-test that a human action writes a `decision_snapshot` and completed games write `game_outcome`.

## Task 1: Expert Data Helper Module

**Files:**
- Create: `gui/backend/expert_data.py`
- Create: `tests/test_expert_data.py`

- [ ] **Step 1: Write failing tests for public snapshot construction**

Add this to `tests/test_expert_data.py`:

```python
import json
from pathlib import Path

import numpy as np

import guandan_rlcard
from guandan_rlcard.baselines import get_agent_class
from gui.backend.expert_data import (
    STRONG_AGENT_TYPES,
    build_decision_snapshot,
    export_expert_samples,
    human_team_won_against_strong_model,
    render_training_sample,
)


def make_env(seed=7):
    env = guandan_rlcard.make({'seed': seed, 'perfect_info': True})
    Base7 = get_agent_class('base7')
    env.set_agents([
        Base7(0, np.random.RandomState(seed)),
        Base7(1, np.random.RandomState(seed + 1)),
        Base7(2, np.random.RandomState(seed + 2)),
        Base7(3, np.random.RandomState(seed + 3)),
    ])
    env.reset()
    return env


def test_build_decision_snapshot_uses_public_fields_only():
    env = make_env()
    player_id = env.get_player_id()
    state = env.get_state(player_id)
    action = state['actions'][0]

    snapshot = build_decision_snapshot(
        env=env,
        player_id=player_id,
        legal_actions=state['actions'],
        chosen_action=action,
        agent_types={'0': 'human', '1': 'perfectdan', '2': 'base7', '3': 'danzero_plus'},
        decision_id='decision_1',
    )

    assert snapshot['event_type'] == 'decision_snapshot'
    assert snapshot['schema_version'] == 2
    assert snapshot['player_id'] == player_id
    assert snapshot['team_id'] == player_id % 2
    assert snapshot['teammate_id'] == (player_id + 2) % 4
    assert snapshot['chosen_action'] == action
    assert snapshot['chosen_action_index'] == 0
    assert snapshot['visibility'] == 'public'
    assert 'current_hand' in snapshot['state']
    assert 'all_players_hands' not in json.dumps(snapshot, ensure_ascii=False)
    assert 'other_player_hands' not in json.dumps(snapshot, ensure_ascii=False)
```

Run: `python -m pytest tests/test_expert_data.py::test_build_decision_snapshot_uses_public_fields_only -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'gui.backend.expert_data'`.

- [ ] **Step 2: Implement minimal snapshot builder**

Create `gui/backend/expert_data.py`:

```python
"""Human expert decision data helpers for GUI logs and exports."""

import json
from pathlib import Path

from guandan_rlcard.constants import CARD_RANK

STRONG_AGENT_TYPES = {'danzero_plus', 'perfectdan'}
PUBLIC_STATE_KEYS = (
    'trace',
    'rank_list',
    'play_team',
    'greaterPos',
    'greaterAction',
    'current_hand',
    'num_cards_left',
    'last_actions',
    'bomb_history',
    'finished_players',
)


def _rank_name(rank_index):
    try:
        return CARD_RANK[int(rank_index)]
    except (TypeError, ValueError, IndexError):
        return str(rank_index)


def _normalize_agent_types(agent_types):
    return {
        str(player_id): str(agent_type)
        for player_id, agent_type in dict(agent_types or {}).items()
    }


def _action_index(legal_actions, chosen_action):
    for index, action in enumerate(legal_actions or []):
        if action == chosen_action:
            return index
    return -1


def build_decision_snapshot(env, player_id, legal_actions, chosen_action,
                            agent_types, decision_id):
    """Build a public-information snapshot for one human decision."""
    state = env.get_state(player_id)
    rank_list = state.get('rank_list', [])
    play_team = state.get('play_team', 0)
    current_rank_index = rank_list[play_team] if play_team < len(rank_list) else None
    public_state = {
        key: state.get(key)
        for key in PUBLIC_STATE_KEYS
        if key in state
    }
    public_state.update({
        'player_ids': [0, 1, 2, 3],
        'current_player': player_id,
        'rank_list': [_rank_name(rank) for rank in rank_list],
        'current_rank': _rank_name(current_rank_index),
        'wildcard': 'H' + _rank_name(current_rank_index),
        'recent_trace': state.get('trace', [])[-12:],
    })
    return {
        'event_type': 'decision_snapshot',
        'schema_version': 2,
        'decision_id': decision_id,
        'player_id': player_id,
        'team_id': player_id % 2,
        'teammate_id': (player_id + 2) % 4,
        'is_human': True,
        'agent_types': _normalize_agent_types(agent_types),
        'state': public_state,
        'legal_actions': list(legal_actions or []),
        'chosen_action_index': _action_index(legal_actions, chosen_action),
        'chosen_action': chosen_action,
        'visibility': 'public',
    }
```

Run: `python -m pytest tests/test_expert_data.py::test_build_decision_snapshot_uses_public_fields_only -v`

Expected: PASS.

- [ ] **Step 3: Write failing tests for filtering and sample rendering**

Append to `tests/test_expert_data.py`:

```python
def test_human_team_filter_requires_win_and_strong_opponent():
    snapshot = {
        'event_type': 'decision_snapshot',
        'player_id': 0,
        'team_id': 0,
        'is_human': True,
        'agent_types': {'0': 'human', '1': 'perfectdan', '2': 'base7', '3': 'danzero_plus'},
        'visibility': 'public',
        'legal_actions': [['PASS', 'PASS', 'PASS']],
        'chosen_action': ['PASS', 'PASS', 'PASS'],
        'chosen_action_index': 0,
    }
    outcome = {'winner_team': 0}

    assert STRONG_AGENT_TYPES == {'danzero_plus', 'perfectdan'}
    assert human_team_won_against_strong_model(snapshot, outcome) is True

    losing_outcome = {'winner_team': 1}
    assert human_team_won_against_strong_model(snapshot, losing_outcome) is False

    no_strong = dict(snapshot, agent_types={'0': 'human', '1': 'base7', '2': 'base7', '3': 'random'})
    assert human_team_won_against_strong_model(no_strong, outcome) is False


def test_render_training_sample_keeps_legal_action_indexes():
    snapshot = {
        'event_type': 'decision_snapshot',
        'schema_version': 2,
        'game_id': 'game_1',
        'decision_id': 'decision_1',
        'player_id': 0,
        'team_id': 0,
        'teammate_id': 2,
        'agent_types': {'0': 'human', '1': 'perfectdan', '2': 'base7', '3': 'danzero_plus'},
        'visibility': 'public',
        'state': {
            'rank_list': ['2', '3'],
            'play_team': 0,
            'current_rank': '2',
            'wildcard': 'H2',
            'current_hand': ['SA', 'HA', 'S6', 'H6', 'C6', 'D6'],
            'num_cards_left': [6, 5, 12, 2],
            'greaterPos': 3,
            'greaterAction': ['Pair', 'K', ['SK', 'HK']],
            'recent_trace': [[3, ['Pair', 'K', ['SK', 'HK']]]],
        },
        'legal_actions': [
            ['PASS', 'PASS', 'PASS'],
            ['Pair', 'A', ['SA', 'HA']],
            ['Bomb', '6', ['S6', 'H6', 'C6', 'D6']],
        ],
        'chosen_action_index': 1,
        'chosen_action': ['Pair', 'A', ['SA', 'HA']],
    }
    outcome = {'winner_team': 0, 'finished_players': [0, 2, 1, 3]}

    sample = render_training_sample(snapshot, outcome)

    assert '0: ' in sample['question']
    assert '1: ' in sample['question']
    assert '2: ' in sample['question']
    assert sample['chosen_action_index'] == 1
    assert sample['chosen_action'] == ['Pair', 'A', ['SA', 'HA']]
    assert sample['source']['type'] == 'gui_human_win_vs_strong_model'
    assert sample['quality']['human_team_won'] is True
    assert 'all_players_hands' not in sample['question']
```

Run: `python -m pytest tests/test_expert_data.py -v`

Expected: FAIL because `human_team_won_against_strong_model` and `render_training_sample` are missing.

- [ ] **Step 4: Implement filter and renderer**

Append this code to `gui/backend/expert_data.py`:

```python
def _team_of(player_id):
    return int(player_id) % 2


def _opponent_agent_types(snapshot):
    player_team = snapshot.get('team_id', _team_of(snapshot.get('player_id', 0)))
    agents = _normalize_agent_types(snapshot.get('agent_types', {}))
    return [
        agent_type
        for player_id, agent_type in agents.items()
        if _team_of(player_id) != player_team
    ]


def human_team_won_against_strong_model(snapshot, outcome,
                                        strong_agents=STRONG_AGENT_TYPES):
    """Whether a snapshot qualifies as expert human data."""
    if snapshot.get('event_type') != 'decision_snapshot':
        return False
    if not snapshot.get('is_human'):
        return False
    if snapshot.get('visibility') != 'public':
        return False
    if snapshot.get('chosen_action_index', -1) < 0:
        return False
    if snapshot.get('team_id') != outcome.get('winner_team'):
        return False
    opponent_agents = set(_opponent_agent_types(snapshot))
    return bool(opponent_agents & set(strong_agents))


def _format_action_list(actions):
    return '; '.join(
        f'{index}: {action}'
        for index, action in enumerate(actions or [])
    )


def _criticality(snapshot):
    counts = snapshot.get('state', {}).get('num_cards_left') or []
    player_team = snapshot.get('team_id', _team_of(snapshot.get('player_id', 0)))
    opponent_counts = [
        count for player_id, count in enumerate(counts)
        if _team_of(player_id) != player_team
    ]
    if any(count <= 3 for count in opponent_counts):
        return 'opponent_near_finish'
    if any(action and action[0] in ('Bomb', 'StraightFlush') for action in snapshot.get('legal_actions', [])):
        return 'bomb_available'
    return 'standard_decision'


def render_training_sample(snapshot, outcome):
    """Render one filtered snapshot as a question/sentence training row."""
    state = snapshot.get('state', {})
    legal_actions = snapshot.get('legal_actions', [])
    chosen_index = snapshot.get('chosen_action_index')
    chosen_action = snapshot.get('chosen_action')
    opponent_agents = sorted(set(_opponent_agent_types(snapshot)) & STRONG_AGENT_TYPES)
    question = (
        f"Here is a Guandan card game. You are player {snapshot.get('player_id')} "
        f"on team {snapshot.get('team_id')}, and your teammate is player "
        f"{snapshot.get('teammate_id')}. Team levels are {state.get('rank_list')}, "
        f"the current play team is {state.get('play_team')}, and the current "
        f"level card is {state.get('wildcard')}. Your current hand is "
        f"{state.get('current_hand')}. The remaining card counts for players "
        f"[0, 1, 2, 3] are {state.get('num_cards_left')}. The current greatest "
        f"action is from player {state.get('greaterPos')}: "
        f"{state.get('greaterAction')}. Recent actions are "
        f"{state.get('recent_trace')}. Your legal actions are indexed as: "
        f"{_format_action_list(legal_actions)}. First, select several "
        f"reasonable candidate actions. Then consider the risk from opponents "
        f"and your teammate's position. Finally, provide the best action."
    )
    candidates = [
        f"action {index}"
        for index, action in enumerate(legal_actions)
        if action == chosen_action or (action and action[0] in ('Bomb', 'StraightFlush'))
    ][:3]
    if not candidates:
        candidates = [f"action {chosen_index}"]
    sentence = (
        f"Reasonable candidate actions are {', '.join(candidates)}. "
        f"The selected action was legal in this public state and came from a "
        f"human team that won the game against strong agents. Therefore, I "
        f"choose action {chosen_index}: {chosen_action}."
    )
    return {
        'question': question,
        'sentence': sentence,
        'chosen_action': chosen_action,
        'chosen_action_index': chosen_index,
        'legal_actions': legal_actions,
        'source': {
            'type': 'gui_human_win_vs_strong_model',
            'game_id': snapshot.get('game_id'),
            'decision_id': snapshot.get('decision_id'),
            'visibility': snapshot.get('visibility', 'public'),
        },
        'quality': {
            'human_team_won': snapshot.get('team_id') == outcome.get('winner_team'),
            'opponent_agents': opponent_agents,
            'criticality': _criticality(snapshot),
        },
    }
```

Run: `python -m pytest tests/test_expert_data.py -v`

Expected: PASS for the current tests.

- [ ] **Step 5: Write failing test for JSONL export**

Append this test to `tests/test_expert_data.py`:

```python
def test_export_expert_samples_joins_snapshots_to_outcomes(tmp_path):
    input_path = tmp_path / 'guandan_gui.jsonl'
    output_path = tmp_path / 'expert.jsonl'
    snapshot = {
        'event_type': 'decision_snapshot',
        'schema_version': 2,
        'timestamp': '2026-07-04T00:00:00Z',
        'room_id': 'ROOM01',
        'game_id': 'game_1',
        'decision_id': 'decision_1',
        'player_id': 0,
        'team_id': 0,
        'teammate_id': 2,
        'is_human': True,
        'agent_types': {'0': 'human', '1': 'perfectdan', '2': 'base7', '3': 'danzero_plus'},
        'visibility': 'public',
        'state': {
            'rank_list': ['2', '3'],
            'play_team': 0,
            'wildcard': 'H2',
            'current_hand': ['SA', 'HA'],
            'num_cards_left': [2, 5, 12, 2],
            'greaterPos': 3,
            'greaterAction': ['Pair', 'K', ['SK', 'HK']],
            'recent_trace': [],
        },
        'legal_actions': [['Pair', 'A', ['SA', 'HA']]],
        'chosen_action_index': 0,
        'chosen_action': ['Pair', 'A', ['SA', 'HA']],
    }
    outcome = {
        'event_type': 'game_outcome',
        'schema_version': 2,
        'game_id': 'game_1',
        'winner_team': 0,
        'finished_players': [0, 2, 1, 3],
    }
    input_path.write_text(
        json.dumps(snapshot, ensure_ascii=False) + '\n'
        + json.dumps(outcome, ensure_ascii=False) + '\n',
        encoding='utf-8',
    )

    summary = export_expert_samples(input_path, output_path)

    assert summary == {'read': 2, 'written': 1, 'skipped': 0}
    rows = [json.loads(line) for line in output_path.read_text(encoding='utf-8').splitlines()]
    assert len(rows) == 1
    assert rows[0]['chosen_action'] == ['Pair', 'A', ['SA', 'HA']]
```

Run: `python -m pytest tests/test_expert_data.py::test_export_expert_samples_joins_snapshots_to_outcomes -v`

Expected: FAIL because `export_expert_samples` has no implementation.

- [ ] **Step 6: Implement JSONL export**

Append this code to `gui/backend/expert_data.py`:

```python
def _read_jsonl(path):
    for line in Path(path).read_text(encoding='utf-8').splitlines():
        if not line.strip():
            continue
        yield json.loads(line)


def export_expert_samples(input_path, output_path):
    """Export filtered expert samples from a GUI JSONL log."""
    events = list(_read_jsonl(input_path))
    outcomes = {
        event.get('game_id'): event
        for event in events
        if event.get('event_type') == 'game_outcome'
    }
    read = len(events)
    written = 0
    skipped = 0
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with Path(output_path).open('w', encoding='utf-8') as fh:
        for event in events:
            if event.get('event_type') != 'decision_snapshot':
                continue
            outcome = outcomes.get(event.get('game_id'))
            if not outcome:
                skipped += 1
                continue
            if not human_team_won_against_strong_model(event, outcome):
                skipped += 1
                continue
            sample = render_training_sample(event, outcome)
            fh.write(json.dumps(sample, ensure_ascii=False, sort_keys=True) + '\n')
            written += 1
    return {'read': read, 'written': written, 'skipped': skipped}
```

Run: `python -m pytest tests/test_expert_data.py -v`

Expected: PASS.

## Task 2: Logger Methods

**Files:**
- Modify: `gui/backend/game_logger.py`
- Modify: `tests/test_gui_game_logger.py`

- [ ] **Step 1: Write failing logger tests**

Append to `tests/test_gui_game_logger.py`:

```python
def test_logger_decision_snapshot_uses_schema_v2(tmp_path):
    logger = GameLogger(log_dir=tmp_path,
                        clock=lambda: '2026-07-04T00:00:00Z')

    logger.log_decision_snapshot({
        'room_id': 'ROOM01',
        'game_id': 'game-1',
        'decision_id': 'decision-1',
        'player_id': 0,
        'team_id': 0,
        'teammate_id': 2,
        'is_human': True,
        'agent_types': {'0': 'human', '1': 'perfectdan'},
        'state': {'current_hand': ['S3']},
        'legal_actions': [['Single', '3', ['S3']]],
        'chosen_action_index': 0,
        'chosen_action': ['Single', '3', ['S3']],
        'visibility': 'public',
    })

    line = read_lines(tmp_path / 'guandan_gui.jsonl')[0]
    assert line['event_type'] == 'decision_snapshot'
    assert line['schema_version'] == 2
    assert line['timestamp'] == '2026-07-04T00:00:00Z'
    assert line['chosen_action'] == ['Single', '3', ['S3']]


def test_logger_game_outcome_uses_schema_v2(tmp_path):
    logger = GameLogger(log_dir=tmp_path,
                        clock=lambda: '2026-07-04T00:00:00Z')

    logger.log_game_outcome(
        room_id='ROOM01',
        game_id='game-1',
        winner_team=0,
        finished_players=[0, 2, 1, 3],
        agent_types={'0': 'human', '1': 'perfectdan'},
    )

    line = read_lines(tmp_path / 'guandan_gui.jsonl')[0]
    assert line['event_type'] == 'game_outcome'
    assert line['schema_version'] == 2
    assert line['winner_team'] == 0
    assert line['finished_players'] == [0, 2, 1, 3]
```

Run: `python -m pytest tests/test_gui_game_logger.py::test_logger_decision_snapshot_uses_schema_v2 tests/test_gui_game_logger.py::test_logger_game_outcome_uses_schema_v2 -v`

Expected: FAIL because the new methods do not exist.

- [ ] **Step 2: Add schema override support and methods**

Modify `gui/backend/game_logger.py`:

```python
    def write_event(self, event_type, payload, schema_version=1):
        event = {
            'event_type': event_type,
            'schema_version': schema_version,
            'timestamp': self.clock(),
        }
        event.update(payload)
        line = json.dumps(event, ensure_ascii=False, sort_keys=True) + '\n'
        with self._lock:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            with self.path.open('a', encoding='utf-8') as fh:
                fh.write(line)
```

Add these methods below `log_action`:

```python
    def log_decision_snapshot(self, snapshot):
        payload = dict(snapshot)
        payload.pop('event_type', None)
        payload.pop('schema_version', None)
        payload.pop('timestamp', None)
        self.write_event('decision_snapshot', payload, schema_version=2)

    def log_game_outcome(self, room_id, game_id, winner_team,
                         finished_players, agent_types):
        self.write_event('game_outcome', {
            'room_id': room_id,
            'game_id': game_id,
            'winner_team': winner_team,
            'finished_players': list(finished_players or []),
            'agent_types': {
                str(player_id): str(agent_type)
                for player_id, agent_type in dict(agent_types or {}).items()
            },
        }, schema_version=2)
```

Run: `python -m pytest tests/test_gui_game_logger.py -v`

Expected: PASS and existing schema version 1 tests still pass.

## Task 3: Capture Human Pre-Step Snapshots

**Files:**
- Modify: `gui/backend/game_manager.py`
- Modify: `tests/test_gui_game_manager.py`

- [ ] **Step 1: Write failing game-manager test**

Append to `tests/test_gui_game_manager.py`:

```python
def test_perform_action_captures_human_decision_snapshot():
    game = Game({
        'human_player_ids': [0],
        'agentTypes': {'1': 'perfectdan', '2': 'base7', '3': 'danzero_plus'},
    }, seed=7)
    game.init_game()
    while not game.is_waiting_for_human():
        assert game.step_one_ai() is True

    player_id = game.current_player()
    legal_action = game.env.get_state(player_id)['actions'][0]

    game.perform_action(player_id, legal_action)

    snapshot = game.last_decision_snapshot
    assert snapshot['event_type'] == 'decision_snapshot'
    assert snapshot['player_id'] == player_id
    assert snapshot['chosen_action'] == legal_action
    assert snapshot['agent_types']['0'] == 'human'
    assert snapshot['agent_types']['1'] == 'perfectdan'
```

Run: `python -m pytest tests/test_gui_game_manager.py::test_perform_action_captures_human_decision_snapshot -v`

Expected: FAIL because `last_decision_snapshot` does not exist.

- [ ] **Step 2: Capture snapshot before `env.step`**

Modify `gui/backend/game_manager.py` imports:

```python
from .expert_data import build_decision_snapshot
```

In `Game.__init__`, add:

```python
        agent_types = self.player_config.get('agentTypes', {})
        for human_id in self.human_player_ids:
            agent_types[str(human_id)] = 'human'
        self.agent_types = {
            str(player_id): str(agent_type)
            for player_id, agent_type in dict(agent_types or {}).items()
        }
        self.last_decision_snapshot = None
```

In `perform_action`, immediately before `env.step(action)`:

```python
        self.last_decision_snapshot = build_decision_snapshot(
            env=self.env,
            player_id=player_id,
            legal_actions=legal_actions,
            chosen_action=action,
            agent_types=self.agent_types,
            decision_id=f'decision_{uuid.uuid4().hex}',
        )
```

Run: `python -m pytest tests/test_gui_game_manager.py -v`

Expected: PASS.

## Task 4: Wire GUI Server Logging

**Files:**
- Modify: `gui/backend/server.py`
- Modify: `tests/test_gui_server.py`

- [ ] **Step 1: Write failing integration tests**

Append to `tests/test_gui_server.py`:

```python
def test_human_action_logs_decision_snapshot(isolated_server):
    room_id, clients = create_started_room()
    host = clients[0]
    current, action = current_legal_action(room_id)
    assert current == 0

    host.emit('player_action', {
        'roomId': room_id,
        'playerId': current,
        'action': action,
    })

    events = read_log_events(isolated_server)
    snapshots = [
        event for event in events
        if event['event_type'] == 'decision_snapshot'
    ]
    assert len(snapshots) == 1
    assert snapshots[0]['schema_version'] == 2
    assert snapshots[0]['participant_id'].startswith('participant_')
    assert snapshots[0]['chosen_action'] == action
    assert snapshots[0]['agent_types']['0'] == 'human'


def test_completed_match_logs_game_outcome(monkeypatch, isolated_server):
    room_id, clients = create_started_room()
    room = server.rooms[room_id]
    room['game'].env.game.winner_team = 0
    room['game'].env.game.round.result = [0, 2, 1, 3]
    monkeypatch.setattr(room['game'], 'is_over', lambda: True)

    server._log_match_summary(room_id)

    events = read_log_events(isolated_server)
    outcomes = [
        event for event in events
        if event['event_type'] == 'game_outcome'
    ]
    assert len(outcomes) == 1
    assert outcomes[0]['schema_version'] == 2
    assert outcomes[0]['winner_team'] == 0
    assert outcomes[0]['finished_players'] == [0, 2, 1, 3]
```

Run: `python -m pytest tests/test_gui_server.py::test_human_action_logs_decision_snapshot tests/test_gui_server.py::test_completed_match_logs_game_outcome -v`

Expected: FAIL because the server does not write these event types.

- [ ] **Step 2: Add `_log_last_decision_snapshot`**

Modify `gui/backend/server.py` near `_log_last_action`:

```python
def _log_last_decision_snapshot(room_id):
    room = rooms.get(room_id)
    if not room:
        return
    game = room['game']
    snapshot = game.last_decision_snapshot
    if not snapshot:
        return
    player_id = snapshot['player_id']
    participant_id, session_id = _identity_for_player(room, player_id)
    snapshot = dict(snapshot)
    snapshot.update({
        'room_id': room_id,
        'game_id': game.game_id,
        'participant_id': participant_id,
        'session_id': session_id,
        'account_id': None,
    })
    game_logger.log_decision_snapshot(snapshot)
    game.last_decision_snapshot = None
```

In `handle_player_action`, call the new function before `_log_last_action`:

```python
        _log_last_decision_snapshot(room_id)
        _log_last_action(room_id)
```

Run: `python -m pytest tests/test_gui_server.py::test_human_action_logs_decision_snapshot -v`

Expected: PASS.

- [ ] **Step 3: Add game outcome logging**

Modify `_log_match_summary` in `gui/backend/server.py` so it calls both the
existing summary and the new outcome method:

```python
    agent_types = dict(getattr(game, 'agent_types', {}) or {})
    game_logger.write_event('match_summary', {
        'room_id': room_id,
        'game_id': game.game_id,
        'winner_team': game.env.game.winner_team,
        'finished_players': [p for p in result if p >= 0],
        'participants': _participants_snapshot(room),
        'account_id': None,
    })
    game_logger.log_game_outcome(
        room_id=room_id,
        game_id=game.game_id,
        winner_team=game.env.game.winner_team,
        finished_players=[p for p in result if p >= 0],
        agent_types=agent_types,
    )
```

Run: `python -m pytest tests/test_gui_server.py::test_completed_match_logs_game_outcome -v`

Expected: PASS.

- [ ] **Step 4: Run focused server tests**

Run: `python -m pytest tests/test_gui_server.py tests/test_gui_game_manager.py tests/test_gui_game_logger.py tests/test_expert_data.py -q`

Expected: PASS.

## Task 5: CLI Exporter

**Files:**
- Create: `examples/export_human_expert_dataset.py`

- [ ] **Step 1: Add CLI wrapper**

Create `examples/export_human_expert_dataset.py`:

```python
"""Export human expert GUI decisions to question/sentence JSONL.

Example:
    python examples/export_human_expert_dataset.py \
        --input logs/gui/guandan_gui.jsonl \
        --output dataset/human_expert.jsonl
"""

import argparse

from gui.backend.expert_data import export_expert_samples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', default='logs/gui/guandan_gui.jsonl')
    parser.add_argument('--output', default='dataset/human_expert.jsonl')
    args = parser.parse_args()

    summary = export_expert_samples(args.input, args.output)
    print(
        f"read={summary['read']} written={summary['written']} "
        f"skipped={summary['skipped']} output={args.output}"
    )


if __name__ == '__main__':
    main()
```

Run: `python examples/export_human_expert_dataset.py --help`

Expected: command prints `--input` and `--output`.

## Task 6: Documentation

**Files:**
- Modify: `docs/gui_guide.md`
- Modify: `README.md`

- [ ] **Step 1: Document expert-data export in GUI guide**

Add a short section to `docs/gui_guide.md` after the existing logging section:

```markdown
### Human expert dataset export

The GUI writes one `decision_snapshot` event for each validated human action
and one `game_outcome` event when a match completes. To export winning human
decisions against `danzero_plus` or `perfectdan`:

```bash
python examples/export_human_expert_dataset.py \
  --input logs/gui/guandan_gui.jsonl \
  --output dataset/human_expert.jsonl
```

The public exporter uses only the acting player's visible state, legal actions
and game outcome. Debug-only hidden hands are not included in `question`.
```

Run: `python -m pytest tests/test_expert_data.py -q`

Expected: PASS; this confirms docs-only changes did not affect code.

- [ ] **Step 2: Add README pointer**

Add one bullet under the README logs/data area:

```markdown
导出人类专家胜局样本：

```bash
python examples/export_human_expert_dataset.py \
  --input logs/gui/guandan_gui.jsonl \
  --output dataset/human_expert.jsonl
```
```

Run: `git diff --check`

Expected: no whitespace errors.

## Task 7: Final Verification

**Files:**
- Verify all changed files.

- [ ] **Step 1: Run Python tests**

Run: `python -m pytest -q`

Expected: all tests pass.

- [ ] **Step 2: Run frontend tests**

Run: `npm run test:ui` in `gui/frontend`.

Expected: all frontend utility tests pass.

- [ ] **Step 3: Build frontend**

Run: `npm run build` in `gui/frontend`.

Expected: Vite build succeeds.

- [ ] **Step 4: Confirm git state**

Run: `git status --short`

Expected: only intended files are modified or untracked.

- [ ] **Step 5: Commit implementation**

Run:

```bash
git add README.md docs/gui_guide.md examples/export_human_expert_dataset.py gui/backend/expert_data.py gui/backend/game_logger.py gui/backend/game_manager.py gui/backend/server.py tests/test_expert_data.py tests/test_gui_game_logger.py tests/test_gui_game_manager.py tests/test_gui_server.py
git commit -m "feat: export human expert GUI decisions"
```

Expected: one feature commit containing code, tests, and docs.
