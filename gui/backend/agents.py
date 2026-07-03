"""Agent construction for a GUI game.

Seats marked human get a :class:`HumanAgent` placeholder (it never
chooses a play - the backend waits for the browser instead); AI seats get
a baseline from :data:`guandan_rlcard.baselines.AGENT_REGISTRY`.

The frontend should send the registry names directly (``random``,
``base1``..``base8``, ``danzero``, ``dmc``, ``danzero_plus``,
``perfectdan``, ``llm``). A small alias map keeps older GUI names
(``baseline_random``, ``ai1``..``ai8``, ``ppo``) working.
"""

from functools import lru_cache
import logging
import os
from pathlib import Path

import numpy as np

from guandan_rlcard.agents import GuandanAgent
from guandan_rlcard.baselines import AGENT_REGISTRY, get_agent_class
from guandan_rlcard.baselines.llm.client import LLMClient
from guandan_rlcard.baselines.llm.llm_agent import LLMAgent

logger = logging.getLogger('guandan.gui')

NUM_SEATS = 4
MODEL_DIR_ENV = 'GUANDAN_MODEL_DIR'
DEFAULT_MODEL_DIR = '/data/weights'

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

WEIGHTED_AGENT_CONFIG = {
    'dmc': ('GUANDAN_DMC_MODEL_TAR', 'dmc/model.tar'),
    'danzero_plus': (
        'GUANDAN_DANZERO_PLUS_MODEL_TAR',
        'danzero_plus/model.tar',
    ),
    'perfectdan': ('GUANDAN_PERFECTDAN_MODEL', 'perfectdan/models_v0.pt'),
}


def danzero_weight_path():
    """Return the DanZero checkpoint path used by the original baseline."""
    explicit = os.environ.get('GUANDAN_DANZERO_CKPT')
    if explicit:
        return explicit
    import guandan_rlcard.baselines.danzero as danzero_pkg
    return str(Path(danzero_pkg.__file__).with_name('q_network.ckpt'))


def _file_status(path):
    exists = os.path.exists(path)
    return {
        'path': path,
        'available': exists,
        **({'bytes': os.path.getsize(path)} if exists else {}),
    }


def agent_runtime_status():
    """Return non-secret runtime readiness metadata for GUI AI choices."""
    status = {
        name: {'available': True, 'requires': []}
        for name in DEPENDENCY_FREE_AGENTS
    }
    status['danzero'] = {
        **_file_status(danzero_weight_path()),
        'requires': ['torch', 'checkpoint'],
    }
    for name in WEIGHTED_AGENT_CONFIG:
        status[name] = {
            **_file_status(configured_weight_path(name)),
            'requires': ['torch', 'checkpoint'],
        }
    status['llm'] = {
        'available': True,
        'requires': ['model', 'base_url', 'api_key'],
        'runtime_config_required': True,
    }
    for name in available_agents():
        status.setdefault(name, {'available': True, 'requires': []})
    return status


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


def configured_weight_path(agent_name):
    """Return the configured model file path for a weighted GUI baseline."""
    if agent_name not in WEIGHTED_AGENT_CONFIG:
        raise KeyError(f'No external weight path is configured for {agent_name}')
    env_name, relative_path = WEIGHTED_AGENT_CONFIG[agent_name]
    explicit = os.environ.get(env_name)
    if explicit:
        return explicit
    root = os.environ.get(MODEL_DIR_ENV, DEFAULT_MODEL_DIR)
    return str(Path(root) / relative_path)


def _require_weight_path(agent_name):
    path = configured_weight_path(agent_name)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f'{path} 不存在。请把 {agent_name} 权重放到该路径，或设置 '
            f'{WEIGHTED_AGENT_CONFIG[agent_name][0]}；详见 docs/gui_guide.md。')
    return path


def _load_torch_checkpoint(path, device='cpu'):
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            '未安装 PyTorch，无法加载学习型 AI。请安装 pip install -e ".[ppo]"。'
        ) from exc
    try:
        return torch.load(path, map_location=device)
    except Exception:
        return torch.load(path, map_location=device, weights_only=False)


def _seat_state_dict(checkpoint, seat):
    state_dicts = checkpoint.get('model_state_dict', checkpoint)
    if not isinstance(state_dicts, dict):
        raise ValueError('权重文件格式错误：缺少 model_state_dict。')
    seat_key = str(seat)
    if seat_key not in state_dicts:
        raise ValueError(f'权重文件缺少座位 {seat_key} 的模型。')
    return state_dicts[seat_key]


@lru_cache(maxsize=16)
def _load_dmc_model(model_path, seat):
    from guandan_rlcard.baselines.dmc.models import PlayerModel

    checkpoint = _load_torch_checkpoint(model_path, device='cpu')
    model = PlayerModel()
    model.load_state_dict(_seat_state_dict(checkpoint, seat))
    model.eval()
    return model


@lru_cache(maxsize=16)
def _load_danzero_plus_model(model_path, seat):
    from guandan_rlcard.baselines.danzero_plus.models import PlayerModel

    checkpoint = _load_torch_checkpoint(model_path, device='cpu')
    model = PlayerModel()
    model.load_state_dict(_seat_state_dict(checkpoint, seat))
    model.eval()
    return model


@lru_cache(maxsize=4)
def _load_perfectdan_checkpoint(model_path):
    return _load_torch_checkpoint(model_path, device='cpu')


def _build_weighted_agent(name, seat, np_random):
    model_path = _require_weight_path(name)
    if name == 'dmc':
        from guandan_rlcard.baselines.dmc.dmc_agent import DMCAgent
        return DMCAgent(
            player_id=seat,
            np_random=np_random,
            model=_load_dmc_model(model_path, seat),
            device='cpu',
        )
    if name == 'danzero_plus':
        from guandan_rlcard.baselines.danzero_plus.ppo_agent import PPOAgent
        return PPOAgent(
            player_id=seat,
            np_random=np_random,
            model=_load_danzero_plus_model(model_path, seat),
            device='cpu',
        )
    if name == 'perfectdan':
        from guandan_rlcard.baselines.perfectdan.ppo_agent import PPOGuandanAgent

        agent = PPOGuandanAgent(
            player_id=seat,
            np_random=np_random,
            device='cpu',
            preserve_rnn_state=True,
        )
        checkpoint = _load_perfectdan_checkpoint(model_path)
        if 'policy_network_state_dict' in checkpoint:
            agent.load_model(model_path)
        else:
            policy_key = f'policy_player{seat}'
            value_key = f'value_player{seat}'
            if policy_key not in checkpoint or value_key not in checkpoint:
                raise ValueError(
                    f'PerfectDan 权重缺少 {policy_key}/{value_key}。')
            agent.policy_network.load_state_dict(checkpoint[policy_key])
            agent.value_network.load_state_dict(checkpoint[value_key])
        agent.policy_network.eval()
        agent.value_network.eval()
        agent.train_mode = False
        return agent
    raise KeyError(f'No weighted builder for {name}')


def _trim_config_value(value):
    return value.strip() if isinstance(value, str) else ''


def _llm_config(player_config):
    source = (player_config or {}).get('llmConfig') or {}
    return {
        'api_key': _trim_config_value(
            source.get('apiKey') or source.get('api_key')),
        'base_url': _trim_config_value(
            source.get('baseUrl') or source.get('base_url')),
        'model': _trim_config_value(source.get('model')),
    }


def _build_llm_agent(seat, np_random, player_config):
    llm_config = _llm_config(player_config)
    if (not llm_config['api_key'] or not llm_config['base_url'] or
            not llm_config['model']):
        raise ValueError(
            'LLM AI 需要在创建房间时填写模型名称、Base URL 和 API Key。')
    client = LLMClient(**llm_config)
    return LLMAgent(player_id=seat, np_random=np_random, client=client)


def _build_ai_agent(name, seat, np_random, player_config=None):
    try:
        if name == 'llm':
            return _build_llm_agent(seat, np_random, player_config)
        if name in WEIGHTED_AGENT_CONFIG:
            return _build_weighted_agent(name, seat, np_random)
        agent_class = get_agent_class(name)
        return agent_class(player_id=seat, np_random=np_random)
    except KeyError as exc:
        raise ValueError(f'未知的 AI 类型 "{name}"。') from exc
    except ValueError:
        raise
    except Exception as exc:  # noqa: BLE001 - surface a friendly message
        logger.exception('Failed to build agent %s for seat %s', name, seat)
        raise ValueError(
            f'无法加载 AI "{name}"：{exc}. 该智能体可能需要下载模型权重，'
            f'请参考 docs/gui_guide.md。') from exc


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
        agents.append(_build_ai_agent(name, seat, np_random, player_config))
    return agents
