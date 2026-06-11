# guandan-rlcard

**An open Guandan (掼蛋) environment for reinforcement learning and LLM agents, built on [RLCard](https://github.com/datamllab/rlcard).**

掼蛋开放式强化学习环境 —— 完整暴露发牌、进还贡与每一步交互，方便调试、训练与 LLM 接入。

Guandan is a four-player, two-team climbing card game played with two
full decks (108 cards), hugely popular in China. Teams race from level 2
to level A; each deal ends with a tribute phase that feeds into the
next. The hidden information, the wildcard (red-heart level card) and
the team play make it a challenging benchmark for multi-agent RL and
for LLM decision making.

## Why "open"? · 为什么是开放式环境

Unlike standard RLCard environments that hand agents a fixed tensor
observation, this environment exposes the **entire interaction as plain
Python data**:

* States are readable dicts (hand, trace, public counters, legal actions).
* Actions are readable lists like `['Pair', '3', ['H3', 'D3']]`.
* The agents ARE the players: an agent subclasses `GuandanPlayer`, so the
  tribute phase (进贡/还贡) is also driven by agent decisions and fully
  observable.
* With `perfect_info=True` (default) every state additionally carries
  **all four hands** and per-hand step estimates.

> ⚠️ **Perfect information warning** · 完美信息提醒
> By default `state['all_players_hands']` exposes every hand. This is by
> design, for debugging and oracle-style training. For honest
> imperfect-information experiments create the env with
> `{'perfect_info': False}` and make sure your agent only reads public
> fields.

This design trades drop-in compatibility with RLCard's built-in RL
agents (DQN/NFSP expect tensor states) for transparency: every decision
an agent makes can be logged, replayed and turned into an LLM prompt.

## Installation · 安装

```bash
git clone https://github.com/Choysang/guandan-rlcard.git
cd guandan-rlcard
pip install -e .            # core (rlcard + numpy)
pip install -e ".[ppo]"     # + torch, for the PPO baseline
pip install -e ".[llm]"     # + openai, for the LLM agent
pip install -e ".[dev]"     # + pytest
```

Python >= 3.8.

## Quick start · 快速上手

```python
import numpy as np
import guandan_rlcard
from guandan_rlcard.baselines import get_agent_class

env = guandan_rlcard.make({'seed': 42})          # or rlcard.make('guandan')

Base5 = get_agent_class('base5')
Random = get_agent_class('random')
env.set_agents([
    Base5(0, np.random.RandomState(0)),          # seats 0/2 = team 0
    Random(1, np.random.RandomState(1)),         # seats 1/3 = team 1
    Base5(2, np.random.RandomState(2)),
    Random(3, np.random.RandomState(3)),
])

trajectories, payoffs = env.run()                # one full episode (to level A)
print(payoffs)                                   # e.g. [1.0, -1.0, 1.0, -1.0]
print(env.game.gwin, env.game.winner_team)       # per-deal wins, episode winner
```

Or use the ready-made scripts:

```bash
python examples/run_rule_match.py --team0 base7 --team1 base5 --episodes 10
python examples/run_llm_match.py  --opponent base5 --episodes 1   # needs API key env vars
python examples/generate_dataset.py --episodes 10 --output dataset/base7.jsonl
```

## State and action format · 状态与动作格式

An **action** is `[combo_type, key_rank, cards]`:

| combo_type | example | notes |
|---|---|---|
| `Single` / `Pair` / `Trips` | `['Pair', '3', ['H3', 'D3']]` | key = natural rank |
| `ThreeWithTwo` | `['ThreeWithTwo', '9', ['S9','H9','C9','S2','H2']]` | full house |
| `ThreePair` | `['ThreePair', 'A', ['SA','HA','S2','H2','S3','H3']]` | key = lowest rank (A low) |
| `TwoTrips` | `['TwoTrips', 'K', ['SK','HK','CK','SA','HA','CA']]` | plate, key = lowest rank |
| `Straight` / `StraightFlush` | `['Straight', 'A', ['SA','H2','S3','H4','S5']]` | exactly 5 cards, key = lowest rank |
| `Bomb` | `['Bomb', 'R', ['HR','HR','SB','SB']]` | 4-10 cards; key `'R'` = four-joker bomb |
| pass | `['PASS', 'PASS', 'PASS']` | |

Cards are 2-char strings `<suit><rank>` (`'H3'`, `'ST'`), jokers are
`'SB'` (small) and `'HR'` (big).

The **state** dict contains (selection): `current_hand`, `actions`
(legal actions; empty when you already finished and are only queried for
the wind-follow turn - return `[]`), `trace`, `played_cards`,
`remain_cards`, `rank_list`, `play_team`, `greaterPos`/`greaterAction`,
`num_cards_left`, `last_oracle`/`current_oracle` (team step-estimate
difference, refreshed per trick), and with perfect info also
`all_players_hands` and `min_steps_estimation`.

**Payoffs** follow the RLCard convention: `env.run()` returns
`(trajectories, payoffs)` with `+1` per player on the team that passed
level A and `-1` on the other; per-deal wins are in `env.game.gwin`.

## Writing an agent · 编写自己的智能体

```python
from guandan_rlcard.agents import GuandanAgent

class MyAgent(GuandanAgent):
    def step(self, state):
        actions = state['actions']
        if not actions:          # finished hand, wind-follow query
            return []
        return actions[0]        # your policy here
```

`GuandanAgent` provides rule-following defaults for the tribute phase
(`tribute_act` pays the forced biggest card, `back_act` returns the
smallest legal card); override them for smarter tribute play.

## Baselines · 基线智能体

| name | class | type | notes |
|---|---|---|---|
| `random` | `RandomAgent` | random | weakest reference |
| `base1`..`base8` | `Base1Agent`.. | rule-based | eight hand-crafted heuristics, see each package's docstring; `base7` was the strongest in our runs, `base2` is strong but slow |
| `ppo` | `PPOGuandanAgent` | RL (torch) | PPO self-play with LSTM policy/value nets; training script in `guandan_rlcard/baselines/ppo/train.py` |
| `llm` | `LLMAgent` | LLM | prompts any OpenAI-compatible model for an action index; keys via `GUANDAN_LLM_API_KEY` / `GUANDAN_LLM_BASE_URL` / `GUANDAN_LLM_MODEL` |

All baselines are addressable by name through
`guandan_rlcard.baselines.get_agent_class(name)`.

## Implemented rules · 规则实现说明

Standard competitive Guandan: 108 cards, levels 2→A, red-heart level
card as wildcard (逢人配, never a joker substitute), tribute and
counter-tribute (抗贡: payers jointly hold both big jokers), wind-follow
(接风), bomb ordering `joker bomb > 6+-card bomb > straight flush >
5-card bomb > 4-card bomb`, sequences compare by natural rank (A may be
low), passing level A requires finishing 1st+2nd or 1st+3rd.

Documented simplifications (see [docs/rules.md](docs/rules.md)):

* When both tribute cards tie in a double-down, the first-out player
  takes the last player's card (no suit choice) and the player after the
  first-out leads.
* `step_back` is not supported.

This codebase fixes a number of rule bugs found in the original research
code (four-joker bomb being unbeatable but also unable to beat 5+-card
bombs, equal straight flushes beating each other, level-promoted plate
comparison, missing AA2233/AAA222/joker-pair full houses, and more);
the full list with tests is in [docs/rules.md](docs/rules.md).

## Repository layout · 目录结构

```
guandan_rlcard/
├── constants.py          # card/action constants, value tables
├── game/                 # game engine
│   ├── card_utils.py     #   card conversion & single-card comparison
│   ├── action_compare.py #   which actions beat the table action
│   ├── hand_heuristics.py#   greedy hand decomposition & step estimate
│   ├── dealer.py / judger.py / player.py / round.py / game.py
├── envs/guandan_env.py   # RLCard-style env (registered as 'guandan')
├── agents/base_agent.py  # GuandanAgent base class
└── baselines/            # random / rule_based(base1-8) / ppo / llm
examples/                 # match runners & dataset generation
tests/                    # pytest suite for rules and game flow
docs/rules.md             # ruleset details & changelog vs original code
```

## Testing · 测试

```bash
pip install -e ".[dev]"
pytest
```

The suite covers action generation, action comparison (every fixed rule
bug has a regression test), the tribute phase and full game flow
including seeding determinism.

## Contributing · 贡献

Issues and PRs are welcome - see [CONTRIBUTING.md](CONTRIBUTING.md).
Particularly appreciated: rule-edge-case reports with a failing test,
new baselines, and an optional fixed-size state encoder for tensor-based
RL agents.

## Acknowledgements · 致谢

* Built on [RLCard](https://github.com/datamllab/rlcard) (MIT).
* The rule baselines originate from earlier Guandan AI research code;
  `base5` carries an original author credit (Duofeng Wu, 2020).

## License

MIT - see [LICENSE](LICENSE).
