"""Tests for GUI JSONL logging."""

import json

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
    assert line['participant_id'] == 'p-1'
    assert line['action'] == ['Single', '3', ['S3']]
    assert line['timings']['env_step_ms'] == 0.2


def test_id_helpers_generate_prefixed_values(tmp_path):
    logger = GameLogger(log_dir=tmp_path)

    assert logger.new_participant_id().startswith('participant_')
    assert logger.new_session_id().startswith('session_')
    assert logger.new_game_id().startswith('game_')
