"""Tests for GUI JSONL logging."""

import json
import threading

from gui.backend.game_logger import GameLogger


def read_lines(path):
    return [
        json.loads(line)
        for line in path.read_text(encoding='utf-8').splitlines()
    ]


def test_logger_writes_event_with_ids_and_payload(tmp_path):
    logger = GameLogger(log_dir=tmp_path,
                        clock=lambda: '2026-07-03T00:00:00Z')
    logger.write_event('game_start', {
        'room_id': 'ABC123',
        'game_id': 'game-1',
        'account_id': None,
    })

    lines = read_lines(tmp_path / 'guandan_gui.jsonl')
    assert lines == [{
        'event_type': 'game_start',
        'schema_version': 1,
        'timestamp': '2026-07-03T00:00:00Z',
        'room_id': 'ABC123',
        'game_id': 'game-1',
        'account_id': None,
    }]


def test_logger_action_event_keeps_action_and_timings(tmp_path):
    logger = GameLogger(log_dir=tmp_path,
                        clock=lambda: '2026-07-03T00:00:00Z')
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
    assert line['schema_version'] == 1
    assert line['participant_id'] == 'p-1'
    assert line['action'] == ['Single', '3', ['S3']]
    assert line['timings']['env_step_ms'] == 0.2


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


def test_id_helpers_generate_prefixed_values(tmp_path):
    logger = GameLogger(log_dir=tmp_path)

    assert logger.new_participant_id().startswith('participant_')
    assert logger.new_session_id().startswith('session_')
    assert logger.new_game_id().startswith('game_')


def test_logger_writes_each_event_as_one_serialized_line(tmp_path):
    logger = GameLogger(log_dir=tmp_path,
                        clock=lambda: '2026-07-03T00:00:00Z')
    calls = []

    class RecordingHandle:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def write(self, text):
            calls.append(text)

    class RecordingPath:
        def open(self, *args, **kwargs):
            return RecordingHandle()

    logger.path = RecordingPath()

    logger.write_event('event', {'room_id': 'ROOM01'})

    assert len(calls) == 1
    assert calls[0].endswith('\n')
    assert json.loads(calls[0])['event_type'] == 'event'


def test_logger_concurrent_writes_remain_valid_jsonl(tmp_path):
    logger = GameLogger(log_dir=tmp_path,
                        clock=lambda: '2026-07-03T00:00:00Z')

    def write_many(worker):
        for index in range(100):
            logger.write_event('event', {
                'worker': worker,
                'index': index,
                'account_id': None,
            })

    threads = [threading.Thread(target=write_many, args=(i,))
               for i in range(6)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    lines = read_lines(tmp_path / 'guandan_gui.jsonl')
    assert len(lines) == 600
    assert {line['worker'] for line in lines} == set(range(6))
