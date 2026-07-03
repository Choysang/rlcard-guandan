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
    assert set(payload['debug_state']['all_player_hands'].keys()) == {
        0, 1, 2, 3}
