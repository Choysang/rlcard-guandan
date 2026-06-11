"""Baseline agents for the Guandan environment.

The collection gathers the publicly available Guandan agents in one
open, debuggable environment:

    random_agent  - uniform random baseline (no extra dependencies)
    rule_based    - top-8 rule agents of the 1st China AI Guandan
                    Competition (base1..base8)
    danzero       - DanZero paper agent with pretrained value net (torch)
    dmc           - DanZero-style Deep Monte-Carlo training framework
    danzero_plus  - DanZero+ (PPO) framework building on danzero (torch)
    perfectdan    - the author's own PPO self-play algorithm (torch)
    llm           - LLM-driven agent template (openai, key via env vars)

Heavy optional dependencies are only imported when the corresponding
submodule is imported.
"""

import importlib

from guandan_rlcard.baselines.random_agent import RandomAgent

# name -> (module, class). Modules are imported lazily so optional
# dependencies (torch / openai) stay optional.
AGENT_REGISTRY = {
    'random': ('guandan_rlcard.baselines.random_agent', 'RandomAgent'),
    'base1': ('guandan_rlcard.baselines.rule_based.base1', 'Base1Agent'),
    'base2': ('guandan_rlcard.baselines.rule_based.base2.base2_agent',
              'Base2Agent'),
    'base3': ('guandan_rlcard.baselines.rule_based.base3', 'Base3Agent'),
    'base4': ('guandan_rlcard.baselines.rule_based.base4', 'Base4Agent'),
    'base5': ('guandan_rlcard.baselines.rule_based.base5', 'Base5Agent'),
    'base6': ('guandan_rlcard.baselines.rule_based.base6', 'Base6Agent'),
    'base7': ('guandan_rlcard.baselines.rule_based.base7', 'Base7Agent'),
    'base8': ('guandan_rlcard.baselines.rule_based.base8', 'Base8Agent'),
    'danzero': ('guandan_rlcard.baselines.danzero.danzero_agent',
                'DanzeroAgent'),
    'dmc': ('guandan_rlcard.baselines.dmc.dmc_agent', 'DMCAgent'),
    'danzero_plus': ('guandan_rlcard.baselines.danzero_plus.ppo_agent',
                     'PPOAgent'),
    'perfectdan': ('guandan_rlcard.baselines.perfectdan.ppo_agent',
                   'PPOGuandanAgent'),
    'llm': ('guandan_rlcard.baselines.llm.llm_agent', 'LLMAgent'),
}


def get_agent_class(name):
    """Resolve a baseline name (see AGENT_REGISTRY) to its agent class."""
    if name not in AGENT_REGISTRY:
        raise KeyError(f'Unknown baseline {name!r}; choose from '
                       f'{sorted(AGENT_REGISTRY)}')
    module_name, class_name = AGENT_REGISTRY[name]
    return getattr(importlib.import_module(module_name), class_name)


__all__ = ['RandomAgent', 'AGENT_REGISTRY', 'get_agent_class']
