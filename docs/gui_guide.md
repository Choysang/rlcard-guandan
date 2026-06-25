# Web GUI · 网页图形界面使用指南

A browser-based, real-time multiplayer front end for the `guandan_rlcard`
engine. One player hosts; humans and AI baselines share the four seats and
play in the browser over a LAN.

基于 `guandan_rlcard` 引擎的实时多人网页界面：一人开服建房，人类玩家与
AI 基线共用四个座位，在浏览器中对战（局域网联机）。

```
gui/
├── backend/          # Flask + Socket.IO server (drives the engine)
│   ├── server.py     #   HTTP/WebSocket entry point
│   ├── game_manager.py  # one env per room, AI/human turn flow
│   ├── agents.py     #   builds seat agents from the baseline registry
│   ├── state_adapter.py # engine state -> frontend JSON contract
│   └── requirements.txt
└── frontend/         # React + Vite single-page app
    ├── src/          #   components, Socket.IO client service
    └── package.json
```

The backend reuses the same engine and baselines as the rest of the repo
(`guandan_rlcard.make(...)`, `guandan_rlcard.baselines.get_agent_class`),
so there is **one** game implementation, not a separate copy.

## 1. Install · 安装

```bash
# from the repository root
pip install -e .                       # the guandan_rlcard engine
pip install -r gui/backend/requirements.txt   # flask, flask-socketio, ...
# optional: torch baselines for AI seats (danzero / perfectdan / ...)
pip install -e ".[ppo]"

# frontend
cd gui/frontend
npm install
```

Requirements: Python >= 3.8, Node.js >= 18.

## 2. Run · 运行

There are two ways to run, depending on whether you want hot-reloading.

### A. One server (production-style) · 单服务

Build the frontend once; the backend then serves it at the same origin:

```bash
cd gui/frontend && npm run build      # outputs gui/frontend/dist
cd ../.. && python -m gui.backend.server
```

Open `http://localhost:5000`. Other players on the same network open
`http://<your-LAN-ip>:5000` (the address is printed on startup).

### B. Two servers (development) · 开发模式（热更新）

```bash
# terminal 1 - backend
python -m gui.backend.server

# terminal 2 - frontend dev server with hot reload
cd gui/frontend && npm run dev
```

Open `http://localhost:5173`. The dev page connects to the backend on
`http://localhost:5000`; change that in `gui/frontend/.env`
(`VITE_BACKEND_URL=...`) if needed.

### Environment variables · 环境变量

Backend (all optional):

| variable | meaning | default |
|---|---|---|
| `GUANDAN_GUI_PORT` | backend port | `5000` |
| `GUANDAN_GUI_AI_DELAY` | seconds between animated AI turns | `0.8` |
| `GUANDAN_GUI_CORS` | allowed origins (`*` or comma list) | `*` |
| `GUANDAN_GUI_OPEN_BROWSER` | auto-open a browser on start | `false` |

Frontend: `VITE_BACKEND_URL` in `gui/frontend/.env` (see `.env.example`).

## 3. Play · 怎么玩

1. **Create a room** (创建房间): on the host, configure each of the four
   seats as a human or an AI, pick the AI type, and start. A 6-character
   room id is shown.
2. **Join** (加入房间): other humans open the LAN address, go to the
   "加入房间" tab and enter the room id. The match starts once every human
   seat is filled (an all-AI-but-one room starts immediately).
3. **Your turn** (你的回合): click cards to select them; when they form a
   legal combo the **出牌** button lights up. Use **提示** to cycle through
   legal combos and **不出** to pass.
4. The match runs across multiple deals until one team passes level A;
   the tribute phase between deals is resolved automatically.

## 4. AI opponents · AI 对手

The seat dropdown lists every baseline from the engine registry
(see the [main README](../README.md#baselines--基线智能体)):

| GUI option | registry name | needs weights? |
|---|---|---|
| 随机 Random | `random` | no |
| 规则 AI 1–8 | `base1`..`base8` | no |
| DanZero | `danzero` | yes (ships a checkpoint) |
| DanZero+ / DMC / PerfectDan | `danzero_plus` / `dmc` / `perfectdan` | yes |
| 大模型 LLM | `llm` | API key, see [llm_guide.md](llm_guide.md) |

`random` and `base1`..`base8` are pure rule agents and always work out of
the box. **Options marked ⚙️ load a learned model** and need PyTorch
(`pip install -e ".[ppo]"`) plus their weights. Weights are **not**
committed to git (they are large); place them where each baseline expects
them under `guandan_rlcard/baselines/<name>/` — see the main README's
Baselines table and the per-baseline source for the exact path
(`danzero` ships its checkpoint; `perfectdan` weights are on GitHub
Releases). If an AI cannot load its weights the server returns a clear
error and you can pick another seat type.

## 5. How it works · 实现说明

* **One engine.** Each room owns a `GuandanEnv`; AI seats use registry
  agents and human seats use a thin placeholder. The backend auto-plays
  AI turns and pauses only when a human has a real decision.
* **Tribute is automatic.** The engine resolves 进贡/还贡 synchronously
  between deals using each agent's rule-abiding default, so human seats do
  not block the server waiting on a websocket. This is a deliberate
  simplification of the GUI (the engine and CLI still expose full tribute
  control to programmatic agents).
* **State contract.** `state_adapter.build_frontend_state` maps the engine
  dict to the flat JSON the React client consumes (`player_hands`,
  `actions`, `num_cards_left`, `greaterAction`, `current_rank`, `is_over`,
  `winner_team`, ...). The action format is the same as the engine:
  `[combo_type, key_rank, [cards]]`.

## 6. Troubleshooting · 常见问题

* **Other players cannot connect.** Make sure the host firewall allows the
  backend port (default 5000) and everyone is on the same network.
* **"该 AI 需要下载模型权重".** The chosen AI needs weights/torch; install
  `".[ppo]"` and place the weights, or pick a rule AI.
* **Blank page in production mode.** Run `npm run build` in
  `gui/frontend` first so `dist/` exists for the backend to serve.
