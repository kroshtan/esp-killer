# One image, two roles: the ingest API (default CMD) and the background worker (`python -m server.worker`).
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.11.11 /uv /usr/local/bin/uv

WORKDIR /app

# ── Dependencies (cached layer — only reruns when the lock file changes) ─────
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-cache

# ── Source ───────────────────────────────────────────────────────────────────
# The server imports only server/ and shared/. agent/ and tools/ are not needed at runtime.
COPY server/ ./server/
COPY shared/ ./shared/

# A fixed uid, so a bind-mounted data directory can be chowned to match (`chown 10001:10001 ./data`). /app stays
# owned by root: the app never writes there, and chowning the venv would copy it into another layer.
RUN addgroup --system --gid 10001 app && adduser --system --uid 10001 --ingroup app app \
    && mkdir /data && chown app:app /data
USER app

# /data holds the SQLite database (with its -wal/-shm files) and config.yaml. It must be a directory mount, not a
# single-file mount: the CLI rewrites config.yaml via a temp file and a rename in the same directory.
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    ESPK_DATABASE_PATH=/data/espk.db \
    ESPK_CONFIG_PATH=/data/config.yaml \
    MPLCONFIGDIR=/tmp/matplotlib
VOLUME ["/data"]

EXPOSE 8000

# slim has no curl. The worker has no HTTP port, so docker-compose.yaml disables this check for it.
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=4)"]

# Access logs are off so client IPs and request details are not logged. For proxy headers behind a TLS reverse
# proxy, see docker-compose.caddy.yaml.
CMD ["uvicorn", "--factory", "server.app:create_app", "--host", "0.0.0.0", "--port", "8000", "--no-access-log", "--no-server-header"]
