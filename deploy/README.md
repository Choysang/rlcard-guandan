# Guandan GUI container deployment

The recommended free-first deployment is GitHub Container Registry (GHCR)
plus one Docker container on the server. GitHub Actions builds the frontend
and Python runtime, pushes an image to GHCR, and the server only pulls that
image. The server does not need the source tree, `node_modules`, or build
caches.

## Publish image

Push to `main`, `master`, `feat/**`, or run the `GUI Container Image`
workflow manually. The workflow publishes:

- `ghcr.io/choysang/rlcard-guandan-gui:latest` from the default branch
- `ghcr.io/choysang/rlcard-guandan-gui:feat-web-gui` from this deployment branch
- `ghcr.io/choysang/rlcard-guandan-gui:<branch-or-tag>`
- `ghcr.io/choysang/rlcard-guandan-gui:sha-<commit>`

If the package is private, log in on the server with a GitHub token that has
`read:packages` permission:

```bash
echo '<TOKEN>' | docker login ghcr.io -u '<GITHUB_USER>' --password-stdin
```

## Run on server

Copy only `deploy/docker-compose.yml` to the server, then run:

```bash
mkdir -p /opt/guandan-gui
cd /opt/guandan-gui
mkdir -p weights/dmc weights/danzero_plus weights/perfectdan
docker compose pull
GUANDAN_GUI_PUBLIC_PORT=5080 docker compose up -d
docker compose ps
curl -fsS http://127.0.0.1:5080/healthz
curl -fsS http://127.0.0.1:5080/api/agents/status
```

Use a port that is not already occupied. The container listens on port `5000`
internally; `GUANDAN_GUI_PUBLIC_PORT` controls the host port. By default the
host port binds to `127.0.0.1` so a reverse proxy such as Caddy can publish it
without exposing the container port directly. Set `GUANDAN_GUI_BIND=0.0.0.0`
only when you intentionally want direct public port access. Override
`GUANDAN_GUI_IMAGE` to pin a specific tag or digest.

Learned-agent weights are mounted read-only from
`/opt/guandan-gui/weights` by default:

```text
/opt/guandan-gui/weights/
├── dmc/model.tar
├── danzero_plus/model.tar
└── perfectdan/models_v0.pt
```

Override the host folder with `GUANDAN_GUI_WEIGHTS_DIR`. Do not commit these
files and do not bake them into the image; uploading them once to the server is
enough, and subsequent image pulls reuse the same mounted directory.

Example Caddy route for the HTTPS public entry:

```caddyfile
guandan.aiwatch.icu {
    encode gzip zstd
    reverse_proxy 127.0.0.1:5080
}
```

## Update

```bash
cd /opt/guandan-gui
docker compose pull
docker compose up -d
docker image prune -f
```

`docker image prune -f` removes unreferenced old layers and keeps server disk
usage bounded.
