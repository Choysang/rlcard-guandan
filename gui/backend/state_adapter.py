"""Translate engine state into GUI play/debug state contracts.

``play_state`` is viewer-specific and safe for normal play. ``debug_state``
is opt-in and intentionally exposes hidden hands for algorithm inspection.
"""

from guandan_rlcard.game.card_utils import cards2str

SAFE_ROOM_CONFIG_KEYS = ('human_player_ids', 'agentTypes',
                         'debug_enabled', 'ai_speed')


def sanitize_room_config(room_config):
    """Return the non-identifying room config safe for logs/debug payloads."""
    source = room_config or {}
    safe = {}
    for key in SAFE_ROOM_CONFIG_KEYS:
        if key not in source:
            continue
        value = source[key]
        if key == 'human_player_ids':
            safe[key] = list(value or [])
        elif key == 'agentTypes':
            safe[key] = {str(seat): str(agent)
                         for seat, agent in dict(value or {}).items()}
        else:
            safe[key] = value
    return safe


def _recent_plays_from_trace(trace):
    recent = {pid: None for pid in range(4)}
    for pid, action in reversed(trace or []):
        if recent[pid] is None:
            recent[pid] = action
    return recent


def _play_history_from_trace(trace):
    return [
        {
            'player_id': int(pid),
            'action': action,
        }
        for pid, action in (trace or [])
    ]


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


def _hand_for(env, player_id):
    return cards2str(env.game.players[player_id].current_hand)


def build_play_state(env, human_player_ids, viewer_player_id=None,
                     timings=None):
    """Build the viewer-specific state safe for normal play."""
    game = env.game
    current_player = env.get_player_id()
    current_state = env.get_state(current_player)
    human_ids = list(human_player_ids)

    player_hands = {}
    if viewer_player_id in human_ids:
        player_hands[viewer_player_id] = _hand_for(env, viewer_player_id)

    actions = []
    if viewer_player_id == current_player and viewer_player_id in human_ids:
        actions = current_state.get('actions', [])

    trace = current_state.get('trace', [])
    recent_plays = current_state.get('recent_plays') or \
        _recent_plays_from_trace(trace)

    state = {
        'player_hands': player_hands,
        'actions': actions,
        'num_cards_left': current_state.get('num_cards_left', []),
        'trace_length': len(trace),
        'play_history': _play_history_from_trace(trace),
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
        pid: _hand_for(env, pid)
        for pid in range(env.num_players)
    }
    legal_actions_by_player = {
        pid: env.get_state(pid).get('actions', [])
        for pid in range(env.num_players)
    }

    snapshot_keys = (
        'greaterAction', 'greaterPos', 'rank_list', 'play_team',
        'num_cards_left', 'recent_plays', 'last_actions', 'bomb_history',
        'finished_players', 'round_completed', 'global_turn_count',
    )
    snapshot = {key: current_state.get(key) for key in snapshot_keys}

    return {
        'seed': seed,
        'room_config': sanitize_room_config(room_config),
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
    """Compatibility wrapper for older callers."""
    viewer = human_player_ids[0] if human_player_ids else None
    return build_play_state(env, human_player_ids, viewer)
