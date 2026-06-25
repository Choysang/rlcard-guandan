"""Agent construction for a GUI game.

Seats marked human get a :class:`HumanAgent` placeholder (it never
chooses a play - the backend waits for the browser instead); AI seats get
a baseline from :data:`guandan_rlcard.baselines.AGENT_REGISTRY`.

The frontend should send the registry names directly (``random``,
``base1``..``base8``, ``danzero``, ``dmc``, ``danzero_plus``,
``perfectdan``, ``llm``). A small alias map keeps older GUI names
(``baseline_random``, ``ai1``..``ai8``, ``ppo``) working.
"""

import logging

import numpy as np

from guandan_rlcard.agents import GuandanAgent
from guandan_rlcard.baselines import AGENT_REGISTRY, get_agent_class

logger = logging.getLogger('guandan.gui')

NUM_SEATS = 4

# Legacy GUI names -> registry names.
AGENT_ALIASES = {
    'baseline_random': 'random',
    'ai1': 'base1', 'ai2': 'base2', 'ai3': 'base3', 'ai4': 'base4',
    'ai5': 'base5', 'ai6': 'base6', 'ai7': 'base7', 'ai8': 'base8',
    'ppo': 'perfectdan',
}

# Baselines that work with no extra weights/dependencies. Used for a safe
# fallback and to tell the frontend which agents are ready out of the box.
DEPENDENCY_FREE_AGENTS = ('random', 'base1', 'base2', 'base3', 'base4',
                          'base5', 'base6', 'base7', 'base8')


class HumanAgent(GuandanAgent):
    """Placeholder for a human seat.

    The backend never calls :meth:`step` for a human that has a real
    decision to make (it waits for the websocket instead), so this just
    returns a safe no-op. Tribute is auto-resolved with the rule-abiding
    greedy default inherited from :class:`GuandanAgent`.
    """

    def step(self, state):
        return []


def normalize_agent_name(name):
    """Map a (possibly legacy) agent name to a registry key."""
    return AGENT_ALIASES.get(name, name)


def available_agents():
    """Registry names known to this backend, dependency-free first."""
    extras = [n for n in sorted(AGENT_REGISTRY) if n not in DEPENDENCY_FREE_AGENTS]
    return list(DEPENDENCY_FREE_AGENTS) + extras


def build_agents(player_config, np_random=None):
    """Build the four seat agents from a frontend player config.

    Args:
        player_config (dict): ``{'human_player_ids': [...],
            'agentTypes': {'1': 'base7', ...}}``.
        np_random (np.random.RandomState): shared RNG for the agents.

    Returns:
        list[GuandanAgent]: four agents indexed by seat.

    Raises:
        ValueError: if an AI seat names an agent that cannot be built
            (e.g. a learning agent whose weights are not installed).
    """
    np_random = np_random or np.random.RandomState()
    human_ids = set(player_config.get('human_player_ids', [0]))
    agent_types = player_config.get('agentTypes', {})

    agents = []
    for seat in range(NUM_SEATS):
        if seat in human_ids:
            agents.append(HumanAgent(player_id=seat, np_random=np_random))
            continue
        raw_name = agent_types.get(str(seat), 'random')
        name = normalize_agent_name(raw_name)
        agents.append(_build_ai_agent(name, seat, np_random))
    return agents


def _build_ai_agent(name, seat, np_random):
    try:
        agent_class = get_agent_class(name)
    except KeyError as exc:
        raise ValueError(f'未知的 AI 类型 "{name}"。') from exc
    try:
        return agent_class(player_id=seat, np_random=np_random)
    except Exception as exc:  # noqa: BLE001 - surface a friendly message
        logger.exception('Failed to build agent %s for seat %s', name, seat)
        raise ValueError(
            f'无法加载 AI "{name}"：{exc}. 该智能体可能需要下载模型权重，'
            f'请参考 docs/gui_guide.md。') from exc
