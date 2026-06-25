# Guandan Web GUI · 网页对战界面

A React + Vite front end and a Flask + Socket.IO backend that let humans
play Guandan against the `guandan_rlcard` baselines in the browser.

Full instructions are in **[../docs/gui_guide.md](../docs/gui_guide.md)**.

## Quick start · 快速开始

```bash
# from the repository root
pip install -e .
pip install -r gui/backend/requirements.txt

cd gui/frontend
npm install
npm run build          # build the SPA the backend will serve

cd ../..
python -m gui.backend.server     # open http://localhost:5000
```

For hot-reload development run the backend and `npm run dev` (port 5173)
in two terminals instead. See the guide for LAN play, AI options and
environment variables.

## Layout · 结构

```
gui/
├── backend/    Flask + Socket.IO server (server.py is the entry point)
└── frontend/   React + Vite single-page app (src/, npm run dev|build)
```
