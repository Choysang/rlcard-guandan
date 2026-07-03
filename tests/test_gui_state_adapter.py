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
    assert 'all_player_hands' not in state
    assert 'other_player_hands' not in state
    assert state['human_player_ids'] == [0, 2]


def test_play_state_only_includes_actions_for_acting_viewer():
    env = make_env()
    current = env.get_player_id()
    acting = build_play_state(env, [current], current)
    waiting = build_play_state(
        env, [current, (current + 1) % 4], (current + 1) % 4)

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


def test_debug_state_sanitizes_room_config():
    env = make_env()
    debug = build_debug_state(
        env,
        human_player_ids=[0],
        room_config={
            'human_player_ids': [0],
            'agentTypes': {'1': 'random'},
            'debug_enabled': True,
            'nickname': 'alice@example.com',
            'token': 'secret',
        },
    )

    assert debug['room_config'] == {
        'human_player_ids': [0],
        'agentTypes': {'1': 'random'},
        'debug_enabled': True,
    }
