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

## Scoring (phase 2)

### Method

Positions are resampled per server onto a 5 s grid (`server/scoring/trajectories.py`). Three behaviours are
turned into **counts** per player (`server/scoring/features.py`):

- **Beeline:** the player lines up on someone who is out of sight (beyond `awareness_m`, and not within it for the
  last 5 minutes) and stays lined up until that player comes into range. Evidence stops at sighting, because
  whatever happens after that (a charge, a fight) is legitimate. Excluded: the target coming to the player (head-on
  meetings, being hunted), the player already heading that way before the target was in their path (walking into
  an ambush, or up to someone resting on their route), and group mates (pairs who spent 10+ minutes together).
- **Ambush:** a wait of at least a minute that ends with the arrival of someone who was out of range when it began.
- **Time to contact:** from each spawn to first contact, ranked against the org's other spawns of the same class.

Beeline and ambush counts are compared with a **time-shifted null**: the same statistic computed against every
other trajectory shifted by 10, 20 and 30 minutes. The shift keeps where people go (waterholes, trails) and breaks
only where they are *now*, which is the one thing an honest player cannot know about someone out of sight. The
evidence is the count z-score `(observed − null) / √(null + 1)`. For time to contact it is the z-score of the mean
rank, which is uniform under the null.

Counts are **additive**. Each scoring run processes complete 2-hour windows (with 30 minutes of leading context
that is not counted), stores each player's counts, and scores from the sum over the last 7 days. A cheater's excess
over the null grows with playing time and an honest player's does not, which is where the power comes from. Each
z becomes a sub-score that rises from 0 at z = 2 to 1 at z = 6 (3.5 for time to contact). They are combined with a
weighted noisy-OR: beeline 1.0, ambush 0.9, time to contact 0.35. So time to contact alone (fast, aggressive
honest players) can never cross the 0.6 flag threshold.

### Evaluation on simulated servers

`python -m tools.sim.evaluate --seeds 12 --first-seed 500 --hours 6`: 12 servers × 6 hours, each with 23 honest
players (roamers, waterhole regulars, campers, hunters who chase anyone they see, groups who regroup over voice
chat) and three cheaters. The thresholds were tuned on seeds 100–107; these seeds were not used for tuning.

| | first 2 h window | accumulated over 6 h |
|---|---|---|
| beeline cheater: AUC vs honest / flagged | 0.93 / 33% | **1.00 / 100%** |
| ambush cheater | 0.99 / 75% | **1.00 / 100%** |
| part-time cheater (ESP 40% of the time) | 0.83 / 8% | **0.97 / 67%** |
| honest players flagged | 1 / 276 | **1 / 276** |

Two things drove the design and are worth knowing when reading flags:

- **Cheaters create beelines for their victims.** Honest players "arrive" at a beeline cheater (it comes to them)
  and walk straight into an ambusher (it stood on their route). Hence the target-approach and turn-onto-target
  rules. Before those rules, a victim's beeline z grew as fast as some cheaters'.
- **Legitimate reactions can't be reproduced by the null.** Nobody reacts to a time-shifted phantom, so anything
  that follows a sighting (a hunter's charge) must not count as evidence. Hence evidence stops at sighting range.

### Limitations

- **Simulation is not reality.** The honest archetypes are my guess at the hard cases. Real thresholds should be
  tuned on real servers with admin feedback (`mark-false-positive`, and bans once there is a way to record them).
- **Latency.** A window is scored only once it is complete, plus a 5 minute lag, so a flag needs at least one full
  2-hour window of play. Positions that arrive after their window was processed (an agent that was offline for
  longer than the lag) are stored but never scored.
- **Respawn detection** relies on gaps, teleports and class changes. A respawn inside a gap shorter than
  `max_gap_s` (30 s) is interpolated over and missed.
- **Awareness range is one number.** In the game it depends on class, terrain, calls and scent. Per-class values
  are a natural next step.
- **Streamers.** Following a streamer's broadcast position is information leakage too, and it will look the same.
- **Scoring config is global** (the `scoring:` section of config.yaml), not per org.

## Alerts and retention (phase 3)

- **Outbox.** The worker queues alerts in the `alerts` table (one row per alert per channel) and delivers them in
  the same run. A transient failure (5xx, network, Discord 429) is retried with exponential backoff, or after
  Discord's `retry_after`, up to 8 attempts. A permanent one (deleted webhook, refused address) is marked failed
  at once. Destinations are read from config.yaml at send time and never stored in the database.
- **When alerts fire.** A new flag alerts on every configured channel, at most once per player per org per
  24 hours. A flagged player seen again after 10+ minutes away gets a *rejoin* alert naming the server; presence
  is tracked per flag in `flags.last_seen_at`, so each return alerts once. Flags marked as false positives go quiet.
  Rejoin latency is one worker interval (5 minutes by default).
- **Evidence image.** New-flag alerts carry a PNG of the player's last 30 minutes on the server where they spent
  most of them: the path shaded by time, the beeline episodes behind the evidence (with where the target was, and
  the awareness radius), and up to 8 players who came near. Coordinates go through a per-server `map_transform`
  (config.yaml, identity by default) so a calibrated map background can be added later. If rendering fails, the
  alert is sent without an image.
- **Secrets in logs.** httpx logs request URLs, and a Discord webhook URL is a credential: the worker sets the
  httpx logger to WARNING and installs a filter that redacts webhook URLs anyway. SMTP passwords are never logged.
- **Retention** runs daily in the worker, in 5,000-row batches: raw positions and ingest dedupe records after
  `ESPK_RETENTION_DAYS` (14), evidence one day after the scoring horizon, scores and alerts after 90 days, flags
  365 days after their last update (and only once no alert refers to them).
- **Migrations** take the write lock and re-read `user_version` first, so the API and the worker can start on a
  fresh database at the same time.

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
  numpy, pandas, matplotlib or fastapi, and a test enforces this so the PyInstaller binary stays
  small.
- **No ORM.** The database is SQLite through the standard library with plain SQL (`server/db/`). Each operation
  opens a short-lived connection (thread-safe for the API's threadpool), writes use `BEGIN IMMEDIATE` so
  read-then-write transactions cannot race, and the schema is a list of migrations tracked in
  `PRAGMA user_version`. Timestamps are fixed-width UTC ISO text. A Postgres port is a second implementation of
  the two repositories (placeholders differ; the SQL itself is portable, including `RETURNING` and
  `ON CONFLICT`), except `latest_names`, which relies on SQLite's documented bare-column-with-MAX behaviour.
- **License:** Apache-2.0.

### Deployment

- **One image, two roles.** The Dockerfile's default command is the API; compose runs the same image as
  `python -m server.worker`. The image holds only `server/` and `shared/` (no agent, tools or tests), runs as uid
  10001, and keeps `espk.db` and `config.yaml` in one `/data` volume. It is a directory mount because the CLI
  replaces `config.yaml` with a rename in the same directory, which fails on a single-file bind mount.
- **The worker starts after the API is healthy** (`depends_on: condition: service_healthy`). Migrations are
  race-safe on their own (they take the write lock and re-read `user_version`), so this is only tidiness.
- **API port on 127.0.0.1 only.** Agents need HTTPS, so a TLS proxy is always in front. The optional Caddy
  override trusts forwarded headers from any peer, which is acceptable only because the API is not reachable
  from outside except through Caddy, and because auth and rate limits key on the API key, not the client IP.
- **No access logs** (uvicorn `--no-access-log`, no `log` in the Caddyfile): client IPs are personal data too.
- **SQLite on a local volume only.** WAL mode needs working file locks and shared memory; NFS/SMB break both.

### Packaging (agent)

- **One-file PyInstaller binary** per OS, so server admins download one file: `make agent-build` builds
  `dist/espk-agent` (Linux) or `dist/espk-agent.exe` (Windows) from `packaging/espk-agent.spec`; PyInstaller
  is in the `build` dependency group only. `make agent-smoke` runs `packaging/smoke_test.py`, which drives the
  built binary (`--version`, `--help`, `check`, 10 s of `run`) against the fake RCON server and a local API
  and checks that rows reached the database.
- **Size:** about 20 MB on Linux (mostly libpython and pydantic-core). The spec explicitly excludes `server`,
  `tools`, `tests`, numpy, pandas, matplotlib, PIL, fastapi, starlette, uvicorn, yaml, the dev
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

- Per-class awareness ranges, and a map-aware null (e.g. shifting players within the same region).

- **Label-free model: information leakage.** Train a self-supervised next-movement model on what a player could
  legitimately know (own history, map context, players within awareness range), and a second model that also
  sees players beyond awareness range. A player's evidence is how much the out-of-range players improve the
  prediction of *their* movement (a conditional-mutual-information or Granger-style test). Honest players
  should gain about nothing. Confounders are the same as for the heuristics: friends on voice chat, popular
  destinations, stream sniping. Weak labels come from owner false-positive marks and bans, and recall can be
  measured with synthetic cheating segments injected into real honest trajectories.
- A pseudonymised training export (keyed-hash ids, no names, relative time) with its own retention, if data
  from several orgs is ever pooled.
