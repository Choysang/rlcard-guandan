"""Pit an LLM (team 1) against a rule baseline (team 0).

Set the credentials first - keys are read from the environment only::

    export GUANDAN_LLM_API_KEY=sk-...
    export GUANDAN_LLM_BASE_URL=https://api.deepseek.com/
    export GUANDAN_LLM_MODEL=deepseek-chat

    python examples/run_llm_match.py --opponent base5 --episodes 1
"""

import argparse

import numpy as np

import guandan_rlcard
from guandan_rlcard.baselines import AGENT_REGISTRY, get_agent_class
from guandan_rlcard.baselines.llm.client import LLMClient
from guandan_rlcard.baselines.llm.llm_agent import LLMAgent
from guandan_rlcard.baselines.llm.prompts import COMPACT_PROMPT, GUIDED_PROMPT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--opponent', default='base5',
                        choices=sorted(k for k in AGENT_REGISTRY
                                       if k not in ('llm',)))
    parser.add_argument('--episodes', type=int, default=1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--model', default=None,
                        help='overrides GUANDAN_LLM_MODEL')
    parser.add_argument('--prompt', choices=['compact', 'guided'],
                        default='compact')
    parser.add_argument('--record-dialogue', action='store_true',
                        help='keep (prompt, reply, action) triples on the '
                             'agents for dataset building')
    args = parser.parse_args()

    template = COMPACT_PROMPT if args.prompt == 'compact' else GUIDED_PROMPT
    client = LLMClient(model=args.model)
    opponent_cls = get_agent_class(args.opponent)

    env = guandan_rlcard.make({'seed': args.seed})
    env.set_agents([
        opponent_cls(0, np.random.RandomState(args.seed)),
        LLMAgent(1, np.random.RandomState(args.seed + 1), client=client,
                 template=template, record_dialogue=args.record_dialogue),
        opponent_cls(2, np.random.RandomState(args.seed + 2)),
        LLMAgent(3, np.random.RandomState(args.seed + 3), client=client,
                 template=template, record_dialogue=args.record_dialogue),
    ])

    wins = [0, 0]
    for episode in range(args.episodes):
        _, payoffs = env.run()
        wins[env.game.winner_team] += 1
        fallbacks = sum(a.fallback_count for a in env.agents
                        if isinstance(a, LLMAgent))
        print(f'episode {episode}: winner team {env.game.winner_team}, '
              f'deals {env.game.gwin}, payoffs {payoffs}, '
              f'LLM fallbacks so far {fallbacks}')

    print(f'\nLLM (team 1) won {wins[1]}/{args.episodes} episodes '
          f'vs {args.opponent} (team 0)')


if __name__ == '__main__':
    main()
