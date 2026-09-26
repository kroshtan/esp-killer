# esp-killer

Server-side detection of likely **ESP (wallhack) users on The Isle: Evrima** servers, from player movement
alone.

Evrima has no modding API and ships with Easy Anti-Cheat, so this never touches the game client. A small,
read-only agent next to the game server polls player positions over RCON and uploads them. The backend looks
for movement that only makes sense if the player can see through walls (heading straight for players far out of
sight, arriving implausibly fast, waiting on paths others later cross), and alerts the server's admins on
Discord or by email. Scores are evidence for human review only. Nothing here kicks or bans anyone.

> **Status:** early development. The agent, ingest API and scoring are done; alerting and packaging are next. The RCON response format is inferred from other clients and still has to be
> verified on a live server (see [NOTES.md](NOTES.md)).

```
agent/     runs next to the game server: RCON client (read-only), poller, on-disk queue, uploader
shared/    ingest payload models, RCON wire format, player-data parser (used by agent and server)
server/    FastAPI ingest API, scoring (server/scoring/), background worker, CLI, database layer
tools/     fake Evrima RCON server, movement simulator with honest and cheating players, evaluation
tests/     unit, integration and end-to-end tests
```

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

cp agent.example.toml agent.toml        # set port 8888, password "devpassword",
                                        # url http://127.0.0.1:8000 and the key from above
python -m agent check -c agent.toml
make agent
```

Positions land in `data/espk.db`.

## How scoring works

Each behaviour is compared with a *time-shifted null*: the same statistic computed against every other player's
trajectory shifted by 10–30 minutes. That keeps where people go (waterholes, trails) and removes only where they
are right now, which is exactly what an honest player can't know about someone out of sight. Evidence is counted
per 2-hour window and summed over a week, so it grows with a cheater's playing time and not with an honest
player's. [NOTES.md](NOTES.md#scoring-phase-2) has the details and an evaluation on simulated servers:

```bash
python -m tools.sim.evaluate --seeds 12 --first-seed 500 --hours 6
```

Flags are listed and resolved from the CLI:

```bash
python -m server list-flags
python -m server mark-false-positive 12 --note "streamer, verified with admins"
make worker      # runs scoring every 5 minutes
```

## License

[Apache-2.0](LICENSE).
