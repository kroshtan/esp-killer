# esp-killer

Server-side detection of likely **ESP (wallhack) users on The Isle: Evrima** servers, from player movement
alone.

Evrima has no modding API and ships with Easy Anti-Cheat, so this never touches the game client. A small,
read-only agent next to the game server polls player positions over RCON and uploads them. The backend looks
for movement that only makes sense if the player can see through walls (heading straight for players far out of
sight, arriving implausibly fast, waiting on paths others later cross), and alerts the server's admins on
Discord or by email. Scores are evidence for human review only. Nothing here kicks or bans anyone.

> **Status:** the agent, ingest API, scoring and alerts are done. The RCON response format is inferred from
> other clients and still has to be verified on a live server (see [NOTES.md](NOTES.md#rcon-format-evidence)).

```
game server host                          backend (Docker: api + worker, SQLite)
+-----------------+    HTTPS, API key     +-------------------------------------+
| Evrima  <-RCON- | espk-agent ---------> | api:    POST /v1/ingest, /healthz   |
|  (read-only)    |  (on-disk queue)      | worker: scoring, alerts, retention  | --> Discord / email
+-----------------+                       +-------------------------------------+
```

```
agent/     runs next to the game server: RCON client (read-only), poller, on-disk queue, uploader
shared/    ingest payload models, RCON wire format, player-data parser (used by agent and server)
server/    FastAPI ingest API, scoring (server/scoring/), background worker, CLI, database layer
tools/     fake Evrima RCON server, movement simulator with honest and cheating players, evaluation
tests/     unit, integration and end-to-end tests
```

## How scoring works

The guiding principle: **a false flag punishes a fair player, a missed cheater is caught later.**

Three behaviours are counted per player: **beelines** (lining up on someone out of sight and staying lined up
until they come into range), **ambushes** (a wait during which someone who was out of range arrives and is killed)
and **time to contact** after spawning. An approach does not count if anything else explains it: the player saw
the target recently, **someone independent had it in view shortly before** (a clanmate, a scout, a friend on voice
chat; not the target's own companions, so raiding a group is not excused by its members seeing each other), a
clanmate was already hunting it, or it ended in a peaceful reunion. What remains is heading straight for someone
nobody independent could see, such as a lone clan member picked off far from anyone. Clans, mixed species included, are
inferred from behaviour (players who keep meeting up or tipping each other off), because the game exposes no group
data.

Each behaviour is compared with a *time-shifted null*: the same statistic computed against every other player's
trajectory shifted by 10–30 minutes. That keeps where people go (waterholes, trails, bases) and removes only where
they are right now. Evidence is counted per 2-hour window and summed over a week, so it grows with a cheater's
playing time and not with an honest player's.

On simulated servers (24 servers × 6 hours, including servers with rival clans hunting on each other's calls;
seeds not used for tuning), **no honest player was flagged** (0 of 616), while full-time cheaters clearly stood out
(AUC 0.88–1.00; 50–100% already flagged after 6 hours). Simulation is not reality: real thresholds need tuning on
real servers with admin feedback. [NOTES.md](NOTES.md#scoring) has the method, the evaluation and the limitations.

## Server owner guide

You run an Evrima server and want to be alerted about likely ESP users. You run the **agent** on the game server
host; someone (possibly you, see the [operator guide](#operator-guide)) runs the backend and gives you an API key.

### Prerequisites

- **RCON enabled** on the game server. In `Game.ini` this is typically:

  ```ini
  bRconEnabled=true
  RconPassword=a-long-random-password
  RconPort=8888
  ```

  Section names and file locations differ between hosts and game versions, so check your host's documentation.
  The agent should run on the same machine (or network) as the game server; RCON never needs to be exposed to
  the internet.
- **An API key and the backend URL** from the operator of the backend. Keys look like `espk_...` and are shown
  once; if yours is lost, ask for a new one.

### Install the agent

Download the agent from [GitHub Releases](https://github.com/kroshtan/esp-killer/releases) (tags `agent-v*`):
`espk-agent` for Linux, `espk-agent.exe` for Windows. It is a single self-contained binary; no Python needed.

Create `agent.toml` next to it (full example with comments: [agent.example.toml](agent.example.toml)):

```toml
[rcon]
host = "127.0.0.1"
port = 8888                  # RconPort from Game.ini
password = "a-long-random-password"

[backend]
url = "https://espk.example.org"
api_key = "espk_..."
```

Every setting can also come from an environment variable instead of the file, for example
`ESPK_RCON__PASSWORD` and `ESPK_BACKEND__API_KEY`. The file holds two secrets: make it readable by the agent's
user only (`chmod 600 agent.toml`).

Check the setup, then run it:

```bash
./espk-agent check -c agent.toml    # RCON login, parses one player-data response, backend reachable
./espk-agent run -c agent.toml      # polls every 3 s, uploads every 15 s, until stopped
```

`check` confirms the backend answers, not that the key is valid: a wrong key shows up in the `run` log as
401 errors.

### Run it as a service

**Linux (systemd).** Assuming the binary and config are in `/opt/espk-agent`:

```ini
# /etc/systemd/system/espk-agent.service
[Unit]
Description=ESP detector agent (read-only Evrima RCON)
After=network-online.target
Wants=network-online.target

[Service]
User=espk
WorkingDirectory=/opt/espk-agent
ExecStart=/opt/espk-agent/espk-agent run -c /opt/espk-agent/agent.toml
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

```bash
sudo useradd --system --home /opt/espk-agent espk && sudo chown -R espk: /opt/espk-agent
sudo systemctl daemon-reload && sudo systemctl enable --now espk-agent
journalctl -u espk-agent -f
```

The working directory matters: the upload queue (`esp-agent-queue.db`, see `queue_path`) is created there.

**Windows.** Either create a Task Scheduler task triggered "At startup" that runs
`espk-agent.exe run -c C:\espk-agent\agent.toml` with "Start in" set to `C:\espk-agent` and "Run whether user is
logged on or not", or wrap the same command as a service with [NSSM](https://nssm.cc/).

### What leaves your machine

Per poll, for each player on the server: Steam/EOS id, player name, position (x, y, z), dinosaur class and
growth, plus a timestamp. Nothing else. Health, stamina, hunger and thirst are dropped in the agent, and the RCON
password is only used to log in locally and is never sent anywhere.

The agent is **read-only**: it can only send the player-data, player-list and server-details RCON commands, and
has no way to send anything else. It never kicks, bans or announces. If the backend is unreachable, snapshots
wait in the local queue file (capped at 100 MiB, oldest dropped first) and are uploaded later.

### When it doesn't work

1. Run `espk-agent check -c agent.toml`. It says which half fails: `RCON FAILED` (host, port, password, RCON not
   enabled, firewall) or `backend FAILED` (URL, DNS, TLS, the backend is down).
2. Read the log of `run` (stderr, or `journalctl -u espk-agent` under systemd). Set `log_level = "DEBUG"` in
   `agent.toml` for more detail.
3. `backend.url must use https`: the agent refuses plain HTTP to anything but localhost. Use the HTTPS URL the
   operator gave you.
4. `check` reports unparseable lines, or players are missing: the game's RCON output format may differ from what
   the parser expects. `espk-agent capture -c agent.toml` saves a few raw responses to `captures/` for a bug
   report. **Those files contain player names, ids and positions**: look at them, replace names and ids before
   sharing, and delete them afterwards.

## Operator guide

Running the backend: an ingest API and a background worker (scoring, alerts, retention), one Docker image, one
SQLite database on a shared volume.

### Quick start

```bash
git clone https://github.com/kroshtan/esp-killer.git && cd esp-killer
cp .env.example .env               # optional: SMTP for email alerts, retention, log level
docker compose up -d --build       # api on 127.0.0.1:8000, worker in the background
curl http://127.0.0.1:8000/healthz
```

`/data` in the containers holds `espk.db` and `config.yaml` (a named volume, `espk-data`). It must be a local
volume or directory, never NFS/SMB: two processes share the SQLite file in WAL mode, which relies on local file
locking. Mount a directory, not a single `config.yaml`, because the CLI replaces that file atomically with a
rename. To use a host directory, replace the volume with `./data:/data` and `chown 10001:10001 ./data` (the
container runs as uid 10001).

### HTTPS

The agent refuses plain HTTP to anything but localhost, so the API must be served over HTTPS. The API port is
published on `127.0.0.1` only; put a TLS reverse proxy in front. The repository includes one: with a DNS name
pointing at the host and ports 80/443 open,

```bash
echo "ESPK_DOMAIN=espk.example.org" >> .env
docker compose -f docker-compose.yaml -f docker-compose.caddy.yaml up -d --build
```

runs [Caddy](https://caddyserver.com/) with automatic Let's Encrypt certificates ([Caddyfile](Caddyfile)). In
that setup uvicorn trusts `X-Forwarded-*` headers from any peer (`--forwarded-allow-ips=*`), which is fine only
because the API is reachable through Caddy and from the host itself. The API authenticates and rate-limits by
key, not by client IP, so nothing security-relevant depends on those headers. Access logs are off in both
uvicorn and the Caddyfile, so client IPs are not logged.

### Deploying on Render

CI builds the image on every push, pushes it to GHCR from `main` (`ghcr.io/kroshtan/esp-killer:latest` and
`:sha-<commit>`), and then calls a Render deploy hook with the exact digest. [render.yaml](render.yaml) is the
Blueprint:

- **One web service** runs both the API and the worker (`ESPK_ROLE=all`), because they share one SQLite file and
  a Render disk attaches to a single service. If either process dies, the container exits and Render restarts it.
- **A persistent disk** at `/data` holds `espk.db` and `config.yaml`. Disks need a paid instance type, and a
  service with a disk has no zero-downtime deploys. Each deploy pauses ingest for a few seconds, and agents
  queue and resend, so nothing is lost.
- Render terminates TLS, so there is no Caddy; `ESPK_BEHIND_PROXY=true` makes uvicorn trust its forwarded headers.
- Region `frankfurt`, because this is EU personal data.

To set it up, create the service from the Blueprint, make the GHCR package public (or give Render registry
credentials), add the service's deploy hook URL as the `RENDER_DEPLOY_HOOK_URL` repository secret and its
`/healthz` URL as the `ESPK_HEALTH_URL` repository variable, and enter the `ESPK_SMTP_*` values in the dashboard
if you want email alerts. Manage orgs and keys from the service's shell with
the CLI below. Without the secret, the CI `deploy` job skips with a warning.

**Agent releases follow the backend.** When `agent/__init__.py` has a version that has not been released yet, CI
publishes it as a GitHub Release (Linux and Windows binaries, `SHA256SUMS`, tag `agent-vX.Y.Z`) right after the
backend deploy has gone live, so a released agent never talks to an older backend. Changing `agent/`, `shared/`
or `packaging/` without bumping that version fails CI.

The image's roles also work anywhere else: `ESPK_ROLE=api` (default), `worker`, or `all`; `PORT` is honoured.

### Orgs, servers and keys

An **org** is one community (its admins get the alerts); it has one or more game **servers**, each with its own
API key. Manage them with the CLI inside the container; the running API picks up changes to `config.yaml`
automatically.

```bash
docker compose exec api python -m server add-org my-community \
    --discord-webhook https://discord.com/api/webhooks/... --email admins@example.org
docker compose exec api python -m server add-server my-community main-1          # prints the key once
docker compose exec api python -m server add-server my-community main-1 --rotate # replace a lost key
docker compose exec api python -m server revoke-key my-community main-1          # agent rejected from now on
```

Only a SHA-256 hash of each key is stored. Send the key and the HTTPS URL to the server owner over a private
channel. [config.example.yaml](config.example.yaml) shows the file format, including optional `scoring:`
thresholds.

### Alerts

When a player is flagged, the worker alerts the org's destinations with the behaviours behind the score and a
map of the flagged path:

- **Discord:** the org's `--discord-webhook` (Server Settings → Integrations → Webhooks in Discord).
- **Email:** the org's `--email` address, sent through one SMTP server for all orgs, configured in `.env`:

  | Variable | Default | |
  |---|---|---|
  | `ESPK_SMTP_HOST` | (none) | email alerts are off without a host and a from address |
  | `ESPK_SMTP_PORT` | `587` | |
  | `ESPK_SMTP_USERNAME`, `ESPK_SMTP_PASSWORD` | (none) | |
  | `ESPK_SMTP_FROM_ADDRESS` | (none) | |
  | `ESPK_SMTP_STARTTLS` | `true` | STARTTLS on the submission port |
  | `ESPK_SMTP_USE_SSL` | `false` | implicit TLS instead (usually port 465) |

  See [.env.example](.env.example); restart the containers after changing `.env`.

### Reviewing flags

```bash
docker compose exec api python -m server list-flags                    # open flags; --org, --status all
docker compose exec api python -m server mark-false-positive 12 --note "streamer, verified with admins"
```

A player marked as a false positive is not flagged again for 30 days (`false_positive_suppress_days`).

### Retention and logs

Raw positions are personal data and are kept for 14 days by default (`ESPK_RETENTION_DAYS`); the worker's daily
retention job deletes older ones, and the API rejects snapshots that would already be past retention. Scores and
flags contain no positions. `docker compose logs -f` shows both processes; request bodies are never logged.

## Development

Requires Python 3.12+ and `make`.

```bash
make install     # venv (with uv inside it), dependencies, pre-commit hook
make test        # pytest with coverage
make fix         # pre-commit on all files: ruff, ruff-format, mypy, pydoclint
```

Run the whole pipeline locally against the fake game server:

```bash
source .venv/bin/activate
python -m server add-org demo              # creates config.yaml (format: config.example.yaml)
python -m server add-server demo gateway-1 # prints the API key once

make fake-rcon   # simulated players on localhost:8888, one of them a scripted cheater
make api         # ingest API on localhost:8000
make worker      # scoring every 5 minutes

cp agent.example.toml agent.toml        # set port 8888, password "devpassword",
                                        # url http://127.0.0.1:8000 and the key from above
python -m agent check -c agent.toml
make agent
```

Positions land in `data/espk.db`. The scoring evaluation on simulated servers:

```bash
python -m tools.sim.evaluate --seeds 12 --first-seed 500 --hours 6
```

`docker build -t espk .` builds the backend image; CI builds it on every run and pushes it to GHCR from `main`.

## Privacy

Player names, ids and positions are personal data under the GDPR. The design tries to collect and keep as little
as possible:

- **Minimisation:** only id, name, position, class and growth are collected; vitals are dropped on the game
  server host. The agent cannot run admin commands.
- **Retention:** raw positions are deleted after 14 days by default. Scoring keeps per-window counts, not paths.
- **No payload logging:** request bodies are never logged, validation errors don't echo input, access logs
  (client IPs) are off.
- **Human review:** a score is evidence for admins to look at, not an automated decision; nothing kicks or bans.

Server owners who run the agent decide to collect this data about their players and are responsible for it
(typically as controller, with the backend operator processing it on their behalf). They should tell their
players, for example in the server rules or on Discord, what is collected, why and for how long. This is not
legal advice.

## Game mechanics: help wanted

What counts as suspicious depends on what a player could legitimately know: how far each species can see, smell
and hear. Those assumptions live in [game/evrima.yaml](game/evrima.yaml), in plain YAML, and most are educated
guesses. If you know the game well, corrections are very welcome as an issue or pull request, ideally saying how
you checked. Modded servers can get their own profile file that overrides only what differs.

## License

[Apache-2.0](LICENSE).
