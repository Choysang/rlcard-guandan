# rlcard-guandan

An open Guandan (掼蛋) environment for reinforcement learning, rule agents,
LLM agents and browser-based algorithm testing. It is built on
[RLCard](https://github.com/datamllab/rlcard), exposes readable Python game
state, and now includes a real-time Web GUI for human-vs-AI evaluation.

掼蛋开放式强化学习环境，面向规则 AI、强化学习、LLM 决策和网页交互调测。项目
提供同一套底层引擎、命令行实验入口、网页牌桌 GUI、数据日志和可部署容器。

## What This Project Provides

| Area | What you get |
|---|---|
| Open Guandan engine | 108-card Guandan, tribute/back tribute, wind-follow, level-up rules, readable state/action data and deterministic seeding. |
| Web GUI | React + Flask/Socket.IO browser table with human/AI seats, LAN rooms, mobile landscape support, press-drag card selection, host AI speed, debug rooms, play history and feedback/error logging. |
| Command-line experiments | Scripts for baseline-vs-baseline matches, LLM matches and dataset generation. Good for quick regression checks and algorithm bake-offs. |
| Baseline agents | Random, competition rule agents `base1`..`base8`, DanZero, DMC, DanZero+, PerfectDan and an OpenAI-compatible LLM agent. |
| Debug and data | Perfect-information debug mode for developers, pytest regression suite, JSONL match/action/feedback logs for later analysis. |
| Deployment | Dockerfile, GitHub Actions image build, GHCR image tags and `deploy/docker-compose.yml` for low-cost server deployment. |

## Choose Your Entry Point

| Goal | Start here |
|---|---|
| Try the browser table locally | [Web GUI quick start](#web-gui-quick-start) |
| Run AI matches in a terminal | [CLI quick start](#cli-quick-start) |
| Use the environment in Python code | [Python API quick start](#python-api-quick-start) |
| Deploy the GUI to a server | [Docker / server deployment](#docker--server-deployment) |
| Configure LLM agents | [docs/llm_guide.md](docs/llm_guide.md) |
| Understand rule details | [docs/rules.md](docs/rules.md) |
| Full GUI manual | [docs/gui_guide.md](docs/gui_guide.md) |

Python >= 3.8 is required. The Web GUI frontend requires Node.js >= 18.

## Install

```bash
git clone https://github.com/Choysang/rlcard-guandan.git
cd rlcard-guandan

# Core engine
pip install -e .

# Optional learned baselines and LLM support
pip install -e ".[ppo,llm]"

# Optional test dependencies
pip install -e ".[dev]"
```

## Web GUI Quick Start

The GUI drives the same `guandan_rlcard` engine as the Python API. It is not a
separate game implementation.

```bash
# Backend dependencies
pip install -r gui/backend/requirements.txt

# Build the React app once; Flask will serve gui/frontend/dist
cd gui/frontend
npm install
npm run build

# Back to repo root
cd ../..
python -m gui.backend.server
```

Open `http://localhost:5000`. Other players on the same LAN can open the LAN
address printed by the server.

GUI capabilities:

- Create a room and configure each seat as human or AI.
- Use `random`, `base1`..`base8`, DanZero, DMC, DanZero+, PerfectDan or LLM seats.
- Press and drag across the hand to select or deselect multiple cards.
- Mobile landscape table layout for user testing.
- Host-controlled AI speed: fast, normal, slow.
- Optional debug rooms for developers: host can view hidden hands, legal actions,
  state snapshots, trace and timing fields.
- Compact table dock with scrollable full play history for the current game.
- User feedback entry in the lobby and in-game table.
- Browser error reporting through the same JSONL feedback log.

For hot reload development:

```bash
# terminal 1
python -m gui.backend.server

# terminal 2
cd gui/frontend
npm run dev
```

Open `http://localhost:5173`. See [docs/gui_guide.md](docs/gui_guide.md) for
LAN setup, environment variables, learned model weights and troubleshooting.

## CLI Quick Start

Run baseline matches from the terminal:

```bash
python examples/run_rule_match.py --team0 base7 --team1 base5 --episodes 10
python examples/run_rule_match.py --team0 danzero --team1 base7 --episodes 5
```

Run one LLM match after setting provider credentials:

```bash
python examples/run_llm_match.py --opponent base5 --episodes 1
```

Generate supervised fine-tuning data from rule-agent play:

```bash
python examples/generate_dataset.py --episodes 10 --output dataset/base7.jsonl
```

The CLI path is the fastest way to test agents, compare win rates, reproduce
rule bugs and generate datasets without opening the browser.

## Python API Quick Start

```python
import numpy as np
import guandan_rlcard
from guandan_rlcard.baselines import get_agent_class

env = guandan_rlcard.make({'seed': 42})

Base7 = get_agent_class('base7')
Base5 = get_agent_class('base5')

env.set_agents([
    Base7(0, np.random.RandomState(0)),  # seats 0/2 = team 0
    Base5(1, np.random.RandomState(1)),  # seats 1/3 = team 1
    Base7(2, np.random.RandomState(2)),
    Base5(3, np.random.RandomState(3)),
])

trajectories, payoffs = env.run()
print(payoffs)
print(env.game.gwin, env.game.winner_team)
```

By default this project favors transparent debugging. With `perfect_info=True`
the state includes all four hands and oracle-style fields. For honest
imperfect-information experiments, create the env with:

```python
env = guandan_rlcard.make({'seed': 42, 'perfect_info': False})
```

## Docker / Server Deployment

The recommended low-cost deployment path is GitHub Container Registry plus one
Docker container on your server. GitHub Actions builds the frontend and backend
into one image.

Published image names:

```text
ghcr.io/choysang/rlcard-guandan-gui:latest
ghcr.io/choysang/rlcard-guandan-gui:<branch-name>
ghcr.io/choysang/rlcard-guandan-gui:<tag>
ghcr.io/choysang/rlcard-guandan-gui:sha-<commit>
```

Server quick start:

```bash
mkdir -p /opt/guandan-gui
cd /opt/guandan-gui
mkdir -p weights/dmc weights/danzero_plus weights/perfectdan

# Copy deploy/docker-compose.yml here first.
GUANDAN_GUI_PUBLIC_PORT=5080 docker compose up -d
curl -fsS http://127.0.0.1:5080/healthz
```

The compose file binds to `127.0.0.1` by default so Caddy/Nginx can publish it
over HTTPS without exposing the container port directly. Full instructions:
[deploy/README.md](deploy/README.md).

## Agents And Model Weights

All agents are selected by name through
`guandan_rlcard.baselines.get_agent_class(name)` and through the Web GUI seat
dropdown.

| name | type | notes |
|---|---|---|
| `random` | random | Weak reference baseline. |
| `base1`..`base8` | rule-based | Top-8 entries of the 1st China AI Guandan Competition, re-interfaced for this environment. `base7` is a strong default opponent. |
| `danzero` | learned RL | DanZero-style agent. The repository includes `guandan_rlcard/baselines/danzero/q_network.ckpt`. Requires PyTorch. |
| `dmc` | learned RL framework | DanZero-style DMC training/runtime. GUI use requires `dmc/model.tar`. |
| `danzero_plus` | learned RL framework | DanZero+ PPO-style framework. GUI use requires `danzero_plus/model.tar`. |
| `perfectdan` | learned PPO agent | Project PPO/self-play agent. GUI use requires `perfectdan/models_v0.pt`. |
| `llm` | LLM agent | Uses OpenAI-compatible chat APIs. Users provide model, Base URL and API Key at runtime. |

Default GUI weight layout:

```text
/data/weights/
├── dmc/model.tar
├── danzero_plus/model.tar
└── perfectdan/models_v0.pt
```

For local development, set `GUANDAN_MODEL_DIR` to a folder with the same layout
or use the per-agent environment variables documented in
[docs/gui_guide.md](docs/gui_guide.md). Weights and API keys are not committed
to git and are not baked into the Docker image.

LLM setup is documented in [docs/llm_guide.md](docs/llm_guide.md). The Web GUI
blocks public HTTP origins from creating LLM rooms, because API keys should only
be entered through HTTPS.

## Logs, Feedback And Daily Maintenance

The GUI backend writes append-only JSONL logs. Locally the default path is:

```text
logs/gui/guandan_gui.jsonl
```

In Docker the default path inside the container volume is:

```text
/data/logs/gui/guandan_gui.jsonl
```

Logged event types include room creation, joins, player actions, disconnects,
match summaries, user feedback and browser client errors. Feedback from the
lobby and in-game table uses:

```text
event_type = "feedback"
kind = "suggestion" | "bug" | "client_error"
```

Before each product iteration, review recent feedback first. Example:

```bash
# local
python - <<'PY'
import json
from pathlib import Path
path = Path('logs/gui/guandan_gui.jsonl')
for line in path.read_text(encoding='utf-8').splitlines():
    event = json.loads(line)
    if event.get('event_type') == 'feedback':
        print(event['timestamp'], event.get('kind'), event.get('page'), event.get('message'))
PY
```

The logger intentionally avoids browser tokens, host tokens, resume tokens, API
keys and arbitrary free-form profile fields.

## State And Action Format

An action is `[combo_type, key_rank, cards]`:

| combo_type | example |
|---|---|
| `Single` / `Pair` / `Trips` | `['Pair', '3', ['H3', 'D3']]` |
| `ThreeWithTwo` | `['ThreeWithTwo', '9', ['S9', 'H9', 'C9', 'S2', 'H2']]` |
| `ThreePair` | `['ThreePair', 'A', ['SA', 'HA', 'S2', 'H2', 'S3', 'H3']]` |
| `TwoTrips` | `['TwoTrips', 'K', ['SK', 'HK', 'CK', 'SA', 'HA', 'CA']]` |
| `Straight` / `StraightFlush` | `['Straight', 'A', ['SA', 'H2', 'S3', 'H4', 'S5']]` |
| `Bomb` | `['Bomb', 'R', ['HR', 'HR', 'SB', 'SB']]` |
| pass | `['PASS', 'PASS', 'PASS']` |

Cards are 2-character strings `<suit><rank>` such as `H3` or `ST`; jokers are
`SB` and `HR`.

The state dict includes fields such as `current_hand`, `actions`, `trace`,
`played_cards`, `remain_cards`, `rank_list`, `play_team`,
`greaterPos`/`greaterAction`, `num_cards_left`, and, when perfect information
is enabled, all hands and per-hand step estimates.

## Rules

Standard competitive Guandan is implemented: two decks, teams 0/2 vs 1/3,
levels 2 to A, red-heart level-card wildcard, tribute and counter-tribute,
wind-follow, bomb ordering and level-up rules. The project also includes
regression tests for many edge cases fixed during development. Full details:
[docs/rules.md](docs/rules.md).

## Repository Layout

```text
guandan_rlcard/          Python engine, env, agents and baselines
examples/                CLI match runners and dataset generation
gui/backend/             Flask + Socket.IO GUI backend
gui/frontend/            React + Vite GUI frontend
deploy/                  Docker Compose and server deployment notes
docs/                    GUI, LLM and rule guides
tests/                   Pytest regression suite
.github/workflows/       GHCR image build workflow
```

## Testing

```bash
pip install -e ".[dev]"
python -m pytest

cd gui/frontend
npm install
npm run test:ui
npm run build
```

The Python suite covers action generation, action comparison, tribute behavior,
game flow, GUI backend contracts, state adaptation and JSONL logging. The
frontend tests cover hand selection, table state formatting, PWA metadata,
game setup config, status history and feedback payloads.

## Contributing

Issues and pull requests are welcome. Useful contributions include:

- Rule-edge-case reports with a failing test.
- New or stronger baselines.
- Better state encoders for tensor-based RL agents.
- GUI usability feedback backed by screenshots or recorded logs.

See [CONTRIBUTING.md](CONTRIBUTING.md).

## Acknowledgements

- Built on [RLCard](https://github.com/datamllab/rlcard) (MIT).
- `base1`..`base8` are re-interfaced top-8 rule entries from the 1st China AI
  Guandan Competition.
- DanZero and DanZero+ follow the papers
  [arXiv:2210.17087](https://arxiv.org/abs/2210.17087) and
  [arXiv:2312.02561](https://arxiv.org/abs/2312.02561), adapted to this
  environment.

## License

MIT. See [LICENSE](LICENSE).
