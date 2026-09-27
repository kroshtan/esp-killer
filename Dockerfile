# One image, three roles (docker/start.sh, ESPK_ROLE): the ingest API (default), the background worker, or both
# in one container (for hosts like Render where only one service can mount the database disk). The same image runs
# the weekly leakage-model trainer with the command `python -m trainer train` (a Render Cron Job; see NOTES.md).
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.11.11 /uv /usr/local/bin/uv

# LightGBM (the leakage model, server/training/leakage.py) needs the OpenMP runtime, which the slim image lacks.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Dependencies (cached layer — only reruns when the lock file changes) ─────
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-cache

# ── Source ───────────────────────────────────────────────────────────────────
# The server imports server/, shared/ and game/. The trainer also needs trainer/ and the simulator in tools/ (its
# promotion gate benchmarks every candidate model on simulated servers). agent/ is not needed.
COPY server/ ./server/
COPY shared/ ./shared/
COPY game/ ./game/
COPY tools/ ./tools/
COPY trainer/ ./trainer/
COPY docker/start.sh ./start.sh

# A fixed uid, so a bind-mounted data directory can be chowned to match (`chown 10001:10001 ./data`). /app stays
# owned by root: the app never writes there, and chowning the venv would copy it into another layer.
RUN addgroup --system --gid 10001 app && adduser --system --uid 10001 --ingroup app app \
    && mkdir /data && chown app:app /data
USER app

# /data holds the SQLite database (with its -wal/-shm files) and config.yaml. It must be a directory mount, not a
# single-file mount: the CLI rewrites config.yaml via a temp file and a rename in the same directory.
# The git commit this image was built from, reported by /healthz so CI can tell when a deploy is live.
ARG REVISION=unknown
ENV ESPK_REVISION=$REVISION \
    PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    ESPK_DATABASE_PATH=/data/espk.db \
    ESPK_CONFIG_PATH=/data/config.yaml \
    MPLCONFIGDIR=/tmp/matplotlib
VOLUME ["/data"]

EXPOSE 8000

# slim has no curl. The worker has no HTTP port, so docker-compose.yaml disables this check for it.
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD ["python", "-c", "import os, urllib.request; urllib.request.urlopen(f\"http://127.0.0.1:{os.environ.get('PORT', '8000')}/healthz\", timeout=4)"]

# Access logs are off so client IPs and request details are not logged.
CMD ["/app/start.sh"]
