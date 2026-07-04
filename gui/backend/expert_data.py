"""Human expert decision data helpers for GUI logs and exports."""

import copy
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
    current_rank_index = rank_list[play_team] \
        if isinstance(play_team, int) and play_team < len(rank_list) else None
    public_state = {
        key: copy.deepcopy(state.get(key))
        for key in PUBLIC_STATE_KEYS
        if key in state
    }
    current_rank = _rank_name(current_rank_index)
    public_state.update({
        'player_ids': [0, 1, 2, 3],
        'current_player': player_id,
        'rank_list': [_rank_name(rank) for rank in rank_list],
        'current_rank': current_rank,
        'wildcard': 'H' + current_rank,
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
        'legal_actions': copy.deepcopy(list(legal_actions or [])),
        'chosen_action_index': _action_index(legal_actions, chosen_action),
        'chosen_action': copy.deepcopy(chosen_action),
        'visibility': 'public',
    }


def _team_of(player_id):
    return int(player_id) % 2


def _opponent_agent_types(snapshot):
    player_team = snapshot.get('team_id')
    if player_team is None:
        player_team = _team_of(snapshot.get('player_id', 0))
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
    if snapshot.get('schema_version') != 2:
        return False
    if outcome.get('schema_version') != 2:
        return False
    if not snapshot.get('is_human'):
        return False
    if snapshot.get('visibility') != 'public':
        return False
    legal_actions = snapshot.get('legal_actions') or []
    chosen_index = snapshot.get('chosen_action_index', -1)
    if not isinstance(chosen_index, int):
        return False
    if chosen_index < 0 or chosen_index >= len(legal_actions):
        return False
    if legal_actions[chosen_index] != snapshot.get('chosen_action'):
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
    player_team = snapshot.get('team_id')
    if player_team is None:
        player_team = _team_of(snapshot.get('player_id', 0))
    opponent_counts = [
        count for player_id, count in enumerate(counts)
        if _team_of(player_id) != player_team
    ]
    if any(count <= 3 for count in opponent_counts):
        return 'opponent_near_finish'
    if any(action and action[0] in ('Bomb', 'StraightFlush')
           for action in snapshot.get('legal_actions', [])):
        return 'bomb_available'
    return 'standard_decision'


def render_training_sample(snapshot, outcome):
    """Render one filtered snapshot as a question/sentence training row."""
    state = snapshot.get('state', {})
    legal_actions = snapshot.get('legal_actions', [])
    chosen_index = snapshot.get('chosen_action_index')
    chosen_action = snapshot.get('chosen_action')
    opponent_agents = sorted(
        set(_opponent_agent_types(snapshot)) & STRONG_AGENT_TYPES)
    question = (
        f"Here is a Guandan card game. You are player "
        f"{snapshot.get('player_id')} on team {snapshot.get('team_id')}, "
        f"and your teammate is player {snapshot.get('teammate_id')}. Team "
        f"levels are {state.get('rank_list')}, the current play team is "
        f"{state.get('play_team')}, and the current level card is "
        f"{state.get('wildcard')}. Your current hand is "
        f"{state.get('current_hand')}. The remaining card counts for players "
        f"[0, 1, 2, 3] are {state.get('num_cards_left')}. The current "
        f"greatest action is from player {state.get('greaterPos')}: "
        f"{state.get('greaterAction')}. Recent actions are "
        f"{state.get('recent_trace')}. Your legal actions are indexed as: "
        f"{_format_action_list(legal_actions)}. First, select several "
        f"reasonable candidate actions. Then consider the risk from opponents "
        f"and your teammate's position. Finally, provide the best action."
    )
    candidates = [
        f"action {index}"
        for index, action in enumerate(legal_actions)
        if action == chosen_action
        or (action and action[0] in ('Bomb', 'StraightFlush'))
    ][:3]
    if not candidates:
        candidates = [f"action {chosen_index}"]
    sentence = (
        f"Reasonable candidate actions are {', '.join(candidates)}. The "
        f"logged human action was legal in this public state and came from a "
        f"human team that won the game against strong agents. The final "
        f"recorded decision is action {chosen_index}: {chosen_action}."
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
            'rationale_type': 'generated_heuristic',
        },
        'quality': {
            'human_team_won': snapshot.get('team_id') == outcome.get(
                'winner_team'),
            'opponent_agents': opponent_agents,
            'criticality': _criticality(snapshot),
            'rationale_generated': True,
        },
    }


def _read_jsonl(path):
    for line in Path(path).read_text(encoding='utf-8').splitlines():
        if line.strip():
            yield json.loads(line)


def export_expert_samples(input_path, output_path):
    """Export filtered expert samples from a GUI JSONL log."""
    events = list(_read_jsonl(input_path))
    outcomes = {
        (event.get('room_id'), event.get('game_id')): event
        for event in events
        if event.get('event_type') == 'game_outcome'
        and event.get('schema_version') == 2
    }
    read = len(events)
    written = 0
    skipped = 0
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with Path(output_path).open('w', encoding='utf-8') as fh:
        for event in events:
            if event.get('event_type') != 'decision_snapshot':
                continue
            outcome = outcomes.get((event.get('room_id'), event.get('game_id')))
            if not outcome:
                skipped += 1
                continue
            if not human_team_won_against_strong_model(event, outcome):
                skipped += 1
                continue
            sample = render_training_sample(event, outcome)
            fh.write(json.dumps(
                sample, ensure_ascii=False, sort_keys=True) + '\n')
            written += 1
    return {'read': read, 'written': written, 'skipped': skipped}
