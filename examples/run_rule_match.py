"""Play baseline-vs-baseline Guandan matches.

Team 0 holds seats 0/2 and team 1 holds seats 1/3, e.g.::

    python examples/run_rule_match.py --team0 base5 --team1 random \
        --episodes 10 --seed 42
"""

import argparse
import time

import numpy as np

import guandan_rlcard
from guandan_rlcard.baselines import AGENT_REGISTRY, get_agent_class


def build_agents(team0_name, team1_name, seed):
    team0_cls = get_agent_class(team0_name)
    team1_cls = get_agent_class(team1_name)
    return [
        team0_cls(0, np.random.RandomState(seed)),
        team1_cls(1, np.random.RandomState(seed + 1)),
        team0_cls(2, np.random.RandomState(seed + 2)),
        team1_cls(3, np.random.RandomState(seed + 3)),
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--team0', default='base5',
                        choices=sorted(AGENT_REGISTRY))
    parser.add_argument('--team1', default='random',
                        choices=sorted(AGENT_REGISTRY))
    parser.add_argument('--episodes', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    env = guandan_rlcard.make({'seed': args.seed})
    env.set_agents(build_agents(args.team0, args.team1, args.seed))

    episode_wins = [0, 0]
    deal_wins = [0, 0]
    deals = 0
    start = time.time()

    for episode in range(args.episodes):
        _, payoffs = env.run()
        winner = env.game.winner_team
        episode_wins[winner] += 1
        deal_wins[0] += env.game.gwin[0]
        deal_wins[1] += env.game.gwin[1]
        deals += sum(env.game.gwin)
        print(f'episode {episode}: winner team {winner}, '
              f'deals {env.game.gwin}, payoffs {payoffs}')

    elapsed = time.time() - start
    print(f'\n{args.episodes} episodes ({deals} deals) in {elapsed:.1f}s')
    print(f'team0 ({args.team0}) episode win rate: '
          f'{episode_wins[0] / args.episodes:.2%}, '
          f'deal win rate: {deal_wins[0] / max(deals, 1):.2%}')
    print(f'team1 ({args.team1}) episode win rate: '
          f'{episode_wins[1] / args.episodes:.2%}, '
          f'deal win rate: {deal_wins[1] / max(deals, 1):.2%}')


if __name__ == '__main__':
    main()
