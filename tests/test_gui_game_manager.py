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

    payload = game.frontend_payload(viewer_player_id=0, include_debug=True)

    assert set(payload.keys()) == {'play_state', 'debug_state'}
    assert payload['play_state']['viewer_player_id'] == 0
    assert payload['debug_state']['seed'] == 7
    assert set(payload['debug_state']['all_player_hands'].keys()) == {
        0, 1, 2, 3}


def test_frontend_payload_exposes_room_modes():
    game = Game({
        'human_player_ids': [0],
        'debug_enabled': True,
        'ai_speed': 'slow',
    }, seed=7)
    game.init_game()

    payload = game.frontend_payload(viewer_player_id=0, include_debug=False)

    assert payload['play_state']['debug_enabled'] is True
    assert payload['play_state']['ai_speed'] == 'slow'


def test_init_game_leaves_auto_drive_to_server(monkeypatch):
    called = False

    def record_auto_advance(self):
        nonlocal called
        called = True

    monkeypatch.setattr(Game, '_auto_advance', record_auto_advance)
    game = Game({'human_player_ids': [0]}, seed=7)

    game.init_game()

    assert called is False


def test_agent_types_for_logging_do_not_mutate_public_config():
    config = {
        'human_player_ids': [0],
        'agentTypes': {'1': 'ppo', '2': 'base7'},
    }
    game = Game(config, seed=7)

    assert game.player_config['agentTypes'] == {'1': 'ppo', '2': 'base7'}
    assert game.agent_types == {
        '0': 'human',
        '1': 'perfectdan',
        '2': 'base7',
        '3': 'random',
    }


def test_perform_action_captures_human_decision_snapshot():
    game = Game({
        'human_player_ids': [0],
        'agentTypes': {'1': 'random', '2': 'random', '3': 'random'},
    }, seed=7)
    game.init_game()
    while not game.is_waiting_for_human():
        assert game.step_one_ai() is True

    player_id = game.current_player()
    legal_action = game.env.get_state(player_id)['actions'][0]

    game.perform_action(player_id, legal_action)

    snapshot = game.last_decision_snapshot
    assert snapshot['event_type'] == 'decision_snapshot'
    assert snapshot['schema_version'] == 2
    assert snapshot['player_id'] == player_id
    assert snapshot['chosen_action'] == legal_action
    assert snapshot['agent_types']['0'] == 'human'
    assert snapshot['agent_types']['1'] == 'random'
