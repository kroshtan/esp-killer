#!/bin/bash
# Container entrypoint. ESPK_ROLE picks what runs:
#   api     the ingest API (default; docker-compose runs the worker as a second service)
#   worker  the background worker only
#   all     both, in one container. For hosts where the database disk can attach to only one service (Render):
#           the API and the worker must share the SQLite file. If either process exits, the other is stopped
#           and the container exits non-zero, so the platform restarts it.
# PORT is honoured because Render sets it.
set -euo pipefail

port="${PORT:-8000}"
api=(uvicorn --factory server.app:create_app --host 0.0.0.0 --port "$port" --no-access-log --no-server-header)
# Behind a TLS-terminating proxy (Render, Caddy) the scheme comes from X-Forwarded-Proto. Nothing here uses the
# client address (auth and rate limits key on the API key), so trusting the proxy's headers is safe.
if [[ "${ESPK_BEHIND_PROXY:-false}" == "true" ]]; then
    api+=(--proxy-headers --forwarded-allow-ips='*')
fi

data_dir="$(dirname "${ESPK_DATABASE_PATH:-/data/espk.db}")"
if [[ ! -w "$data_dir" ]]; then
    echo "error: $data_dir is not writable by uid $(id -u); give the data volume/disk to this user" >&2
    exit 1
fi

case "${ESPK_ROLE:-api}" in
    api)
        exec "${api[@]}"
        ;;
    worker)
        exec python -m server.worker
        ;;
    all)
        python -m server.worker &
        worker=$!
        "${api[@]}" &
        server=$!
        trap 'kill -TERM "$worker" "$server" 2>/dev/null' TERM INT
        # Whichever exits first (or a signal) ends the container; take the other one down with it.
        set +e
        wait -n "$worker" "$server"
        status=$?
        kill -TERM "$worker" "$server" 2>/dev/null
        wait
        exit "$status"
        ;;
    *)
        echo "error: unknown ESPK_ROLE '${ESPK_ROLE}' (api, worker or all)" >&2
        exit 2
        ;;
esac
