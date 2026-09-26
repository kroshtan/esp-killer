# esp-killer

Server-side detection of likely **ESP (wallhack) users on The Isle: Evrima** servers, from player movement
alone.

Evrima has no modding API and ships with Easy Anti-Cheat, so this never touches the game client. A small,
read-only agent next to the game server polls player positions over RCON and uploads them. The backend looks
for movement that only makes sense if the player can see through walls (heading straight for players far out of
sight, arriving implausibly fast, waiting on paths others later cross), and alerts the server's admins on
Discord or by email. Scores are evidence for human review only. Nothing here kicks or bans anyone.

> **Status:** early development. Phase 1 (agent, RCON client, ingest API, key management) is done. Scoring,
> alerting and packaging are next. The RCON response format is inferred from other clients and still has to be
> verified on a live server (see [NOTES.md](NOTES.md)).

```
agent/     runs next to the game server: RCON client (read-only), poller, on-disk queue, uploader
shared/    ingest payload models, RCON wire format, player-data parser (used by agent and server)
server/    FastAPI ingest API, key-management CLI, database layer
tools/     fake Evrima RCON server and movement simulator, for development and tests
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

## License

[Apache-2.0](LICENSE).
