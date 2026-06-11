"""Generate an LLM fine-tuning dataset from rule-agent play.

GenAgent plays with the Base7 policy while logging one JSONL record per
decision: the rendered Chinese prompt, the chosen action and the legal
action list. Useful for supervised fine-tuning of card-playing LLMs::

    python examples/generate_dataset.py --episodes 10 \
        --output dataset/guandan_base7.jsonl
"""

import argparse
from pathlib import Path

import numpy as np

import guandan_rlcard
from guandan_rlcard.baselines.rule_based.base7.base7_agent import Base7Agent
from guandan_rlcard.baselines.rule_based.base7.gen_agent import GenAgent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--episodes', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', default='guandan_dataset.jsonl')
    args = parser.parse_args()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    env = guandan_rlcard.make({'seed': args.seed})
    env.set_agents([
        Base7Agent(0, np.random.RandomState(args.seed)),
        GenAgent(1, np.random.RandomState(args.seed + 1),
                 output_path=args.output),
        Base7Agent(2, np.random.RandomState(args.seed + 2)),
        GenAgent(3, np.random.RandomState(args.seed + 3),
                 output_path=args.output),
    ])

    for episode in range(args.episodes):
        env.run()
        print(f'episode {episode}: winner team {env.game.winner_team}, '
              f'deals {env.game.gwin}')

    lines = sum(1 for _ in open(args.output, encoding='utf-8'))
    print(f'\ndataset written to {args.output} ({lines} records)')


if __name__ == '__main__':
    main()
