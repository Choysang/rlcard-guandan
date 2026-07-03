FROM node:22-bookworm-slim AS frontend

WORKDIR /app/gui/frontend
COPY gui/frontend/package.json gui/frontend/package-lock.json ./
RUN npm ci
COPY gui/frontend/ ./
RUN npm run build

FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    GUANDAN_GUI_PORT=5000 \
    GUANDAN_GUI_LOG_DIR=/data/logs/gui \
    GUANDAN_MODEL_DIR=/data/weights \
    GUANDAN_GUI_OPEN_BROWSER=false

WORKDIR /app
RUN useradd --create-home --shell /usr/sbin/nologin guandan \
    && mkdir -p /data/logs/gui /data/weights \
    && chown -R guandan:guandan /data
COPY pyproject.toml README.md LICENSE ./
COPY guandan_rlcard ./guandan_rlcard
COPY gui/backend ./gui/backend
COPY gui/frontend/package.json ./gui/frontend/package.json
COPY --from=frontend /app/gui/frontend/dist ./gui/frontend/dist
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch \
    && pip install --no-cache-dir -e ".[llm]" \
    && pip install --no-cache-dir -r gui/backend/requirements.txt

VOLUME ["/data"]
EXPOSE 5000
USER guandan

CMD ["python", "-m", "gui.backend.server"]
