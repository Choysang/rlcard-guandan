"""Per-room game manager: drives the engine for human-vs-AI play.

The :mod:`guandan_rlcard` engine already handles deal rollover, the
tribute phase (via each agent's ``tribute_act``/``back_act``) and the
wind-follow (接风) bookkeeping inside ``env.step``. This manager only has
to:

* play AI seats automatically until a human has a real decision, and
* apply a validated human action and then auto-advance again.

Human tribute is resolved automatically with the rule-abiding greedy
default (see :class:`gui.backend.agents.HumanAgent`); the synchronous
engine cannot pause mid-deal to wait for a websocket. ``is_over`` is the
whole-match result (a team passing level A), so play continues across
deals just like a real game.
"""

import logging

import numpy as np

import guandan_rlcard
from .agents import build_agents
from .state_adapter import build_frontend_state

logger = logging.getLogger('guandan.gui')


class Game:
    """Owns one environment instance and its seat configuration."""

    def __init__(self, player_config, seed=None):
        self.player_config = player_config
        self.human_player_ids = list(player_config.get('human_player_ids', [0]))
        self.seed = seed
        self.env = None
        self.agents = None

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def init_game(self):
        """Create the env, build agents, reset and auto-play opening AI turns."""
        np_random = np.random.RandomState(self.seed)
        self.env = guandan_rlcard.make({'seed': self.seed, 'perfect_info': True})
        self.agents = build_agents(self.player_config, np_random)
        self.env.set_agents(self.agents)
        self.env.reset()
        logger.info('Game initialised; humans=%s', self.human_player_ids)
        self._auto_advance()

    # ------------------------------------------------------------------
    # Turn flow
    # ------------------------------------------------------------------

    def _needs_human_input(self):
        """Whether the game is waiting on a human's real decision."""
        if self.env.is_over():
            return False
        pid = self.env.get_player_id()
        if pid not in self.human_player_ids:
            return False
        # A finished human only gets empty-action wind-follow queries,
        # which the backend auto-steps; no input needed there.
        return bool(self.env.get_state(pid).get('actions'))

    def _auto_advance(self):
        """Play AI/auto turns to the first human decision (used at game
        start so the opening hand is ready in one shot)."""
        guard = 0
        while self.step_one_ai():
            guard += 1
            if guard > 100000:
                logger.error('Auto-advance guard tripped; aborting loop.')
                break

    def step_one_ai(self):
        """Play exactly one AI (or empty wind-follow) turn.

        Returns:
            bool: True if a turn was played; False if the match is over or
            it is now a human's turn to decide. Lets the server animate AI
            turns one at a time.
        """
        if self.env.is_over() or self._needs_human_input():
            return False
        pid = self.env.get_player_id()
        state = self.env.get_state(pid)
        actions = state.get('actions')
        action = self.agents[pid].step(state) if actions else []
        self.env.step(action)
        return True

    def is_waiting_for_human(self):
        return self._needs_human_input()

    def is_over(self):
        return self.env.is_over()

    def current_player(self):
        return self.env.get_player_id()

    def perform_action(self, player_id, action_intent):
        """Validate and apply a human action, then auto-advance.

        Args:
            player_id (int): seat claiming the action.
            action_intent (list): ``[type, key, [cards]]`` from the client.

        Raises:
            ValueError: if the seat is not human, not on turn, or the
                action is not currently legal.
        """
        if player_id not in self.human_player_ids:
            raise ValueError(f'玩家 {player_id} 不是人类玩家。')
        current = self.env.get_player_id()
        if current != player_id:
            raise ValueError(f'现在不是玩家 {player_id} 的回合。')

        legal_actions = self.env.get_state(current).get('actions', [])
        action = self._match_action(action_intent, legal_actions)
        if action is None:
            raise ValueError('动作不合法或已过期，请重新选择。')

        logger.info('Human seat %s plays %s', player_id, action)
        self.env.step(action)

    @staticmethod
    def _match_action(intent, legal_actions):
        """Find the legal action matching a client intent (order-insensitive
        on the card list)."""
        if not isinstance(intent, (list, tuple)) or len(intent) < 1:
            return None
        for action in legal_actions:
            if action[0] != intent[0]:
                continue
            if action[0] == 'PASS':
                return action
            if len(intent) >= 3 and sorted(action[2]) == sorted(intent[2]):
                return action
        return None

    # ------------------------------------------------------------------
    # Views
    # ------------------------------------------------------------------

    def frontend_state(self):
        return build_frontend_state(self.env, self.human_player_ids)
