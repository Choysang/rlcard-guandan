"""Tests for GUI human expert dataset helpers."""

import json

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
        agent_types={
            '0': 'human',
            '1': 'perfectdan',
            '2': 'base7',
            '3': 'danzero_plus',
        },
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
    serialized = json.dumps(snapshot, ensure_ascii=False)
    for forbidden in (
        'all_players_hands',
        'others_hands',
        'other_player_hands',
        'other_hand_cards',
        'other_legal_actions',
        'min_steps_estimation',
        'debug_state',
    ):
        assert forbidden not in serialized


def test_human_team_filter_requires_win_and_strong_opponent():
    snapshot = {
        'event_type': 'decision_snapshot',
        'schema_version': 2,
        'player_id': 0,
        'team_id': 0,
        'is_human': True,
        'agent_types': {
            '0': 'human',
            '1': 'perfectdan',
            '2': 'base7',
            '3': 'danzero_plus',
        },
        'visibility': 'public',
        'legal_actions': [['PASS', 'PASS', 'PASS']],
        'chosen_action': ['PASS', 'PASS', 'PASS'],
        'chosen_action_index': 0,
    }
    outcome = {'event_type': 'game_outcome', 'schema_version': 2,
               'winner_team': 0}

    assert STRONG_AGENT_TYPES == {'danzero_plus', 'perfectdan'}
    assert human_team_won_against_strong_model(snapshot, outcome) is True

    losing_outcome = {'event_type': 'game_outcome', 'schema_version': 2,
                      'winner_team': 1}
    assert human_team_won_against_strong_model(
        snapshot, losing_outcome) is False

    no_strong = dict(snapshot, agent_types={
        '0': 'human',
        '1': 'base7',
        '2': 'base7',
        '3': 'random',
    })
    assert human_team_won_against_strong_model(no_strong, outcome) is False


def test_human_team_filter_requires_chosen_action_to_match_legal_index():
    snapshot = {
        'event_type': 'decision_snapshot',
        'schema_version': 2,
        'player_id': 0,
        'team_id': 0,
        'is_human': True,
        'agent_types': {
            '0': 'human',
            '1': 'perfectdan',
            '2': 'base7',
            '3': 'danzero_plus',
        },
        'visibility': 'public',
        'legal_actions': [
            ['PASS', 'PASS', 'PASS'],
            ['Bomb', '6', ['S6', 'H6', 'C6', 'D6']],
        ],
        'chosen_action': ['Bomb', '6', ['S6', 'H6', 'C6', 'D6']],
        'chosen_action_index': 0,
    }
    outcome = {'event_type': 'game_outcome', 'schema_version': 2,
               'winner_team': 0}

    assert human_team_won_against_strong_model(snapshot, outcome) is False

    snapshot['chosen_action_index'] = 1
    assert human_team_won_against_strong_model(snapshot, outcome) is True


def test_render_training_sample_keeps_legal_action_indexes():
    snapshot = {
        'event_type': 'decision_snapshot',
        'schema_version': 2,
        'game_id': 'game_1',
        'decision_id': 'decision_1',
        'player_id': 0,
        'team_id': 0,
        'teammate_id': 2,
        'agent_types': {
            '0': 'human',
            '1': 'perfectdan',
            '2': 'base7',
            '3': 'danzero_plus',
        },
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
    assert sample['source']['rationale_type'] == 'generated_heuristic'
    assert sample['quality']['human_team_won'] is True
    assert sample['quality']['rationale_generated'] is True
    assert 'all_players_hands' not in sample['question']


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
        'agent_types': {
            '0': 'human',
            '1': 'perfectdan',
            '2': 'base7',
            '3': 'danzero_plus',
        },
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
        'room_id': 'ROOM01',
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
    rows = [
        json.loads(line)
        for line in output_path.read_text(encoding='utf-8').splitlines()
    ]
    assert len(rows) == 1
    assert rows[0]['chosen_action'] == ['Pair', 'A', ['SA', 'HA']]


def test_export_expert_samples_requires_schema_v2_and_matching_room(tmp_path):
    input_path = tmp_path / 'guandan_gui.jsonl'
    output_path = tmp_path / 'expert.jsonl'
    snapshot = {
        'event_type': 'decision_snapshot',
        'schema_version': 2,
        'room_id': 'ROOM_A',
        'game_id': 'game_1',
        'decision_id': 'decision_1',
        'player_id': 0,
        'team_id': 0,
        'teammate_id': 2,
        'is_human': True,
        'agent_types': {
            '0': 'human',
            '1': 'perfectdan',
            '2': 'base7',
            '3': 'danzero_plus',
        },
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
    wrong_room_outcome = {
        'event_type': 'game_outcome',
        'schema_version': 2,
        'room_id': 'ROOM_B',
        'game_id': 'game_1',
        'winner_team': 0,
    }
    old_schema_outcome = {
        'event_type': 'game_outcome',
        'schema_version': 1,
        'room_id': 'ROOM_A',
        'game_id': 'game_1',
        'winner_team': 0,
    }
    input_path.write_text(
        '\n'.join(json.dumps(event, ensure_ascii=False) for event in (
            snapshot, wrong_room_outcome, old_schema_outcome,
        )) + '\n',
        encoding='utf-8',
    )

    summary = export_expert_samples(input_path, output_path)

    assert summary == {'read': 3, 'written': 0, 'skipped': 1}
    assert output_path.read_text(encoding='utf-8') == ''
