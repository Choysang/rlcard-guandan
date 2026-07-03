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
# optional: learned baselines and OpenAI-compatible LLM seats
pip install -e ".[ppo,llm]"

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

Open `http://localhost:5173`. In dev mode the page connects to the
backend on the same hostname at port `5000` (for example a phone opening
`http://192.168.1.8:5173` connects to `http://192.168.1.8:5000`). Override
that in `gui/frontend/.env` (`VITE_BACKEND_URL=...`) if needed.

### Environment variables · 环境变量

Backend (all optional):

| variable | meaning | default |
|---|---|---|
| `GUANDAN_GUI_PORT` | backend port | `5000` |
| `GUANDAN_GUI_AI_DELAY` | seconds between animated AI turns | `0.8` |
| `GUANDAN_GUI_AI_SLOW_DELAY` | slow-mode AI turn delay | `1.6` |
| `GUANDAN_GUI_CORS` | allowed origins (`*` or comma list) | `*` |
| `GUANDAN_GUI_LOG_DIR` | JSONL event log directory | `logs/gui` |
| `GUANDAN_GUI_OPEN_BROWSER` | auto-open a browser on start | `false` |
| `GUANDAN_MODEL_DIR` | root directory for GUI model weights | `/data/weights` |
| `GUANDAN_DMC_MODEL_TAR` | DMC checkpoint file | `$GUANDAN_MODEL_DIR/dmc/model.tar` |
| `GUANDAN_DANZERO_PLUS_MODEL_TAR` | DanZero+ checkpoint file | `$GUANDAN_MODEL_DIR/danzero_plus/model.tar` |
| `GUANDAN_PERFECTDAN_MODEL` | PerfectDan checkpoint file | `$GUANDAN_MODEL_DIR/perfectdan/models_v0.pt` |

Frontend: `VITE_BACKEND_URL` in `gui/frontend/.env` (see `.env.example`).

### Container deployment · 容器部署

For public testing, prefer the GHCR container image flow in
[`deploy/README.md`](../deploy/README.md). GitHub Actions builds the React
frontend and Python backend into one image, pushes it to
`ghcr.io/choysang/rlcard-guandan-gui`, and the server only pulls and runs the
image. This avoids keeping the source tree, `node_modules`, and build caches
on the server. Model weights are not baked into the image; mount them into
`/data/weights` with `GUANDAN_GUI_WEIGHTS_DIR`. Use
`GUANDAN_GUI_PUBLIC_PORT` in `deploy/docker-compose.yml` to bind an unused
host port so existing services are not disturbed.

## 3. Play · 怎么玩

1. **Create a room** (创建房间): on the host, configure each of the four
   seats as a human or an AI, pick the AI type, and start. A 6-character
   room id is shown.
2. **Join** (加入房间): other humans open the LAN address, go to the
   "加入房间" tab and enter the room id. The match starts once every human
   seat is filled (an all-AI-but-one room starts immediately).
3. **Your turn** (你的回合): press and drag across your hand to select or
   deselect multiple cards continuously. Clicking still toggles one card.
   When the selected cards form a legal combo the **出牌** button lights up.
   Use **提示** to cycle through legal combos and **不出** to pass.
4. The match runs across multiple deals until one team passes level A;
   the tribute phase between deals is resolved automatically.

### Debug rooms · 调测房

The create-room form has an optional debug switch for developer testing.
Only a room created with that switch can use debug mode, and only the host
browser that holds the server-issued `hostToken` can toggle it or receive
`debug_state`. Normal joined players never receive hidden hands. The debug
drawer shows all hands, current legal actions, recent actions, room config,
state snapshot, trace and timing fields.

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
the box. **Options marked ⚙️ load a learned model** and need PyTorch plus
their weights. Weights are **not** committed to git and should not be copied
into the Docker image.

Default GUI weight layout:

```text
/data/weights/
├── dmc/model.tar
├── danzero_plus/model.tar
└── perfectdan/models_v0.pt
```

For local development, set `GUANDAN_MODEL_DIR` to a folder with the same
layout or set the per-agent variables in the table above. `danzero` ships
`q_network.ckpt` in the Python package; DMC, DanZero+ and PerfectDan require
external files. If an AI cannot load its weights the server returns a clear
error and you can pick another seat type.

When selecting **大模型 LLM** in the create-room form, fill in the model name,
Base URL and API Key. The API Key is sent only to the backend for that room and
is excluded from GUI logs and debug room config. The Web GUI blocks LLM room
creation from public HTTP origins; use an HTTPS domain before entering keys.

## 5. How it works · 实现说明

* **One engine.** Each room owns a `GuandanEnv`; AI seats use registry
  agents and human seats use a thin placeholder. The backend auto-plays
  AI turns through a per-room serialized driver and pauses only when a
  human has a real decision.
* **Tribute is automatic.** The engine resolves 进贡/还贡 synchronously
  between deals using each agent's rule-abiding default, so human seats do
  not block the server waiting on a websocket. This is a deliberate
  simplification of the GUI (the engine and CLI still expose full tribute
  control to programmatic agents).
* **Viewer-specific state.** Every Socket.IO `game_started` / `update_state`
  payload is shaped as `{state, debug_state, current_player, viewer_player_id,
  room_id}`. `state` is safe for normal play: it only includes the viewer's
  own hand and only includes `actions` when that viewer is the acting human.
  It also carries room/viewer modes such as `ai_speed`, `debug_enabled`,
  `debug_allowed` (room can debug), `viewer_can_debug` (this browser may
  see debug data) and `viewer_is_host`.
* **Debug state.** `debug_state` is `null` unless the viewer is authorized.
  When present it intentionally exposes `all_player_hands`,
  `legal_actions_by_player`, trace, timing and sanitized room config for
  algorithm debugging.
* **Action format.** The action format is the same as the engine:
  `[combo_type, key_rank, [cards]]`.
* **Anonymous logging.** The backend appends JSONL events to
  `logs/gui/guandan_gui.jsonl` by default. Every event carries
  `schema_version: 1`. Player-scoped events carry generated
  `participant_id`, server-shaped `session_id`, `game_id` and
  `account_id: null`; room-scoped events carry a `participants` list instead
  of pretending to belong to one player. Timing fields include state
  construction, AI decision, environment step, broadcast, and AI-loop timings
  when available. Browser-held `resumeToken` and `hostToken` are not logged.
  Free-form fields such as nicknames are not persisted.
* **Deployment boundary.** Rooms are stored in process memory. The current
  server is suitable for local/LAN testing and single-process deployment.
  Multi-worker or multi-instance production needs external room/session
  state plus sticky Socket.IO sessions.

## 6. Troubleshooting · 常见问题

* **Other players cannot connect.** Make sure the host firewall allows the
  backend port (default 5000) and everyone is on the same network.
* **Phone opened the dev page but cannot connect.** Set
  `VITE_BACKEND_URL=http://<host-LAN-ip>:5000`, or open the Vite page using
  the host LAN IP instead of `localhost`.
* **"该 AI 需要下载模型权重".** The chosen AI needs weights/torch; install
  `".[ppo,llm]"` and place the weights, or pick a rule AI. Check
  `/api/agents/status` on the running server to confirm mounted weight paths.
* **Blank page in production mode.** Run `npm run build` in
  `gui/frontend` first so `dist/` exists for the backend to serve.
