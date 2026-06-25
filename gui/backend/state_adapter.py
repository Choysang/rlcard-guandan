"""Translate an engine state into the JSON contract the frontend uses.

The React client expects a flat dict with these keys (see
``gui/frontend/src``):

    player_hands       {seat: [card_str, ...]}  - hands of the human seats
    actions            [[type, key, [cards]], ...] - current player's legal
                       actions (only consumed when it is the client's turn)
    num_cards_left     [n0, n1, n2, n3]
    trace              [[seat, action], ...]
    greaterAction      the combo currently holding the table (or [])
    greaterPos         seat holding the table (or -1)
    current_rank       index of the current level card in CARD_RANK
    rank_list          [team0_rank, team1_rank]
    play_team          team currently leading
    turn_count         trick counter (used by the client to reset its
                       card selection between turns)
    round_completed    True right after a trick is decided
    human_player_ids   [...]
    is_over            whole match finished (a team passed level A)
    winner_team        0 or 1 when is_over, else absent
    finished_players   finish order of the last deal when is_over
"""


def build_frontend_state(env, human_player_ids):
    """Assemble the broadcast state for a room.

    Args:
        env (GuandanEnv): the running environment.
        human_player_ids (list[int]): seats controlled by humans.

    Returns:
        dict: the JSON-serialisable state described in the module docstring.
    """
    game = env.game
    current_player = env.get_player_id()
    current_state = env.get_state(current_player)

    state = {
        'player_hands': {
            seat: env.get_state(seat).get('current_hand', [])
            for seat in human_player_ids
        },
        'actions': current_state.get('actions', []),
        'num_cards_left': current_state.get('num_cards_left', []),
        'trace': current_state.get('trace', []),
        'greaterAction': current_state.get('greaterAction', []),
        'greaterPos': current_state.get('greaterPos', -1),
        'rank_list': current_state.get('rank_list', [0, 0]),
        'play_team': current_state.get('play_team', 0),
        'current_rank': getattr(game, 'cur_rank', 0),
        'turn_count': current_state.get('global_turn_count', 0),
        'round_completed': current_state.get('round_completed', False),
        'human_player_ids': list(human_player_ids),
        'is_over': env.is_over(),
    }

    if state['is_over']:
        result = list(getattr(game.round, 'result', []))
        winner_team = game.winner_team
        if winner_team is None or winner_team < 0:
            # Fall back to the finish order of the last deal.
            winner_team = result[0] % 2 if result and result[0] >= 0 else 0
        state['winner_team'] = winner_team
        state['finished_players'] = [p for p in result if p >= 0]

    return state
