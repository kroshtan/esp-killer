# Notes

Decisions, assumptions that still need checking against a real server, and known limitations. Newest context
at the top of each section.

## Verify on a real server

Run `python -m agent capture -c agent.toml` on a machine with a live Evrima server. It writes raw responses to
`captures/`. Those files contain player names, ids and positions: check them, then delete them. Never commit
them unless names and ids have been replaced.

| # | Assumption | Where it matters | Source |
|---|---|---|---|
| 1 | Auth is `0x01 + password + 0x00`, the reply contains `Password Accepted` | `shared/rcon_protocol.py` | all three reference clients |
| 2 | Commands are `0x02 + opcode + params + 0x00`. The Go client omits the trailing NUL and we send it | `shared/rcon_protocol.py` | TS and Python clients |
| 3 | Player data is opcode `0x77`, player list `0x40`, server details `0x12` | `ReadOnlyCommand` | all three |
| 4 | Responses have no length prefix. We don't know whether they end with a NUL, or how big responses fragment | `agent/rcon.py` read-until-idle | none document it; references do a single `read()` |
| 5 | The player data format is `[YYYY.MM.DD-HH.MM.SS] PlayerData` then one `Name: …, PlayerID: …, Location: X=… Y=… Z=…, Class: BP_…_C, Growth: …, Health: …, …` line per player | `shared/playerdata.py` | butt4cak3/theislercon parser only |
| 6 | Names may contain commas, so we take the *last* `, PlayerID:` | parser | inferred from the Go parser |
| 7 | Player ids are Steam64 digits or EOS hex. We accept any `[A-Za-z0-9_-]{1,64}` | `shared/models.py` | the Go client takes digits only |
| 8 | Growth is a 0–1 fraction (the Go comment says 0.75 = adult in the current version) | parser | Go client |
| 9 | Coordinates are Unreal units (cm) and the map is roughly ±400 000 uu | scoring distances (phase 2) | Unreal convention, unverified |
| 10 | Player data excludes players still in class selection | rejoin detection (phase 3) | Go client doc comment |
| 11 | The header timestamp is naive server-local time. We ignore it and stamp snapshots with the agent's UTC clock | `agent/poller.py` | Go client |

If the format differs, update `shared/playerdata.py`, `tools/rcon_format.py` and `tests/fixtures/rcon/` together.
The parser logs unparseable lines as reasons plus field *keys* (never values), which is usually enough to see
what changed.

## Decisions

- **Clean-room protocol implementation.** The reference clients were read as protocol documentation only. No
  code was copied (theislercon is AGPL, and gamercon-async has no clear license).
- **Read until idle, not a single read.** The references treat one `read()` (1–10 KB) as the whole response,
  which truncates full servers and leaves the tail on the socket to be mistaken for the next response. We read
  until the socket is quiet for `idle_timeout_s` (250 ms), a NUL arrives, or 1 MiB. Stale bytes are drained
  and warned about before each command.
- **The agent can't send admin commands.** `ReadOnlyCommand` has three members and there is no raw-exec method.
  The fake server records opcodes and the e2e test asserts that only `0x77` was sent.
- **Vitals are dropped in the agent.** Health, stamina, hunger and thirst never leave the game server host,
  because `PlayerSample` has no fields for them.
- **Snapshot-level idempotency.** Each poll gets a UUID. The backend dedupes on (org, server, snapshot_id), so
  retries are safe however the agent regroups snapshots into batches (e.g. after a 413 halves the batch size).
- **Empty snapshots are uploaded too.** "Server up, nobody on" is information for rejoin detection and gap
  handling.
- **Auth before body.** Unauthenticated or rate-limited requests are rejected before the body is read. gzip is
  decompressed with an output cap (zip-bomb safe).
- **Validation errors don't echo input.** Pydantic errors are returned with `include_input=False`, because
  the input is personal data.
- **Unsalted SHA-256 for API keys.** Keys have 256 bits of entropy, so a slow or salted hash buys nothing, and
  it keeps the lookup a dict access.
- **config.yaml reload** keys on (mtime, inode, size). An invalid file is logged and ignored, and the previous
  config stays active. The CLI rewrites the file without its comments.
- **Single uv project** (`package = false`) for the monorepo. The agent never imports `server`, `tools`,
  numpy, pandas, matplotlib, sqlalchemy or fastapi, and a test enforces this so the PyInstaller binary stays
  small.
- **Schema** is created with `metadata.create_all`. Add alembic before the first schema change in production.
- **License:** Apache-2.0.

### Packaging (agent)

- **One-file PyInstaller binary** per OS, so server admins download one file: `make agent-build` builds
  `dist/espk-agent` (Linux) or `dist/espk-agent.exe` (Windows) from `packaging/espk-agent.spec`; PyInstaller
  is in the `build` dependency group only. `make agent-smoke` runs `packaging/smoke_test.py`, which drives the
  built binary (`--version`, `--help`, `check`, 10 s of `run`) against the fake RCON server and a local API
  and checks that rows reached the database.
- **Size:** about 20 MB on Linux (mostly libpython and pydantic-core). The spec explicitly excludes `server`,
  `tools`, `tests`, numpy, pandas, matplotlib, PIL, fastapi, starlette, uvicorn, sqlalchemy, yaml, the dev
  tools, setuptools and unused stdlib parts (tkinter, unittest, pydoc, ...). pygments stays in: rich uses it
  for tracebacks.
- **Releases:** pushing a tag `agent-vX.Y.Z` (must match `agent/__init__.py`) runs
  `.github/workflows/release-agent.yaml`, which builds on ubuntu-latest and windows-latest and publishes
  `espk-agent-X.Y.Z-{linux,windows}-x86_64[.exe]` plus `SHA256SUMS` as a GitHub Release.
- **Windows is built in CI only** and has not been run by hand. Its smoke test runs in CI but is informational
  (`continue-on-error`) until it has proven stable.
- **Unsigned executables.** The Windows exe is not code-signed, so SmartScreen warns on first start and some
  antivirus products flag PyInstaller binaries. UPX is off to reduce false positives. Code signing is a known
  gap.

## Known limitations

- Rate limiting is in process memory, which is correct only for a single API process.
- The agent's SQLite queue calls run on the event loop. Each is sub-millisecond, which is fine at 1 poll per
  second or slower.
- A batch rejected with a 4xx other than 401/403/413/429 is dropped (logged) so it can't block the queue.
- The player list (`0x40`) format is unknown and isn't parsed. `capture` saves it for inspection.

## Future directions

- **Label-free model: information leakage.** Train a self-supervised next-movement model on what a player could
  legitimately know (own history, map context, players within awareness range), and a second model that also
  sees players beyond awareness range. A player's evidence is how much the out-of-range players improve the
  prediction of *their* movement (a conditional-mutual-information or Granger-style test). Honest players
  should gain about nothing. Confounders are the same as for the heuristics: friends on voice chat, popular
  destinations, stream sniping. Weak labels come from owner false-positive marks and bans, and recall can be
  measured with synthetic cheating segments injected into real honest trajectories.
- A pseudonymised training export (keyed-hash ids, no names, relative time) with its own retention, if data
  from several orgs is ever pooled.
