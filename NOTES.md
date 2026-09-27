# Notes

Decisions, assumptions that still need checking against a real server, and known limitations. Newest context
at the top of each section.

## RCON format evidence

Nobody has published a complete raw player-data response from a live server. The format is pieced together from
the parsers, regexes, test fixtures and logs of about a dozen independent Evrima tools (researched September 2026).
Evidence types: **(a)** real output (logs, hex captures), **(b)** parser code or fixtures that imply the format,
**(c)** documentation.

| Question | Answer | Confidence | Evidence |
|---|---|---|---|
| Wire format | Auth `0x01 + password + 0x00` answered by `Password Accepted`; commands `0x02 + opcode + 0x00`; player data `0x77`, player list `0x40`, server details `0x12` | high | the developers' protocol document (v0.17.54, Oct 2024) and every client |
| Framing | No length prefix. Player data arrives in several TCP segments (1.4–5.8 KB for a few players in one production log) and ends with a `PlayerDataEnd` line on current builds. No source has seen a NUL terminator | high / medium | (a) production bot log; (b) several clients read until `PlayerDataEnd` |
| Player data lines | One line per *spawned* player, fields `Key: value` joined by `, `. The name runs to `, PlayerID:` (names may contain commas) | high | (b) many parsers and regexes |
| Header | `[YYYY.MM.DD-HH.MM.SS] PlayerData`. The timestamp is sometimes missing. 2025 builds glue the first player onto it (`PlayerDataName: ...`); 2026 builds put it on its own line | high that both occur | (b) 2025 regexes vs 2026 parsers |
| Fields | `PlayerID`, `Location: X= Y= Z=` (3 decimals), `Class`, `Growth`, `Health`, `Stamina`, `Hunger`, `Thirst`. 2026 builds add `Gender` (after PlayerID), `MutationSlots: [1=None,...]` (commas inside brackets), `ParentMutationSlots`, `ElderMutationSlotsA/B`, `PrimeElder` | high | (b) |
| Class | `BP_Carnotaurus_C` in 2025, bare `Tyrannosaurus` in 2026 | high | (a) 2026 log; (b) 2025 regexes |
| Player ids | Steam64 (17 digits) since ~0.16; EOS ids before that | high | (c) protocol changelog; (b) `\d{17}` regexes |
| Growth | 0–1 fraction; adult is 1.0 now (0.75 in older builds) | medium | (b) |
| Coordinates | Unreal units (cm). Gateway spans roughly ±600 000 uu. RCON's X/Y are the in-game map's Long/Lat, i.e. swapped relative to the in-game Lat/Long display | medium-high | (a) one tool checked against a live position; (b) map tools |
| Server details | One line, first key glued on: `[ts] ServerDetailsServerName: ..., ServerPassword: <plaintext>, ServerMap: ...` | high | (b) four clients, one strict regex |
| Player list | `PlayerList`, then a line of ids and a line of names, each item followed by a comma; empty server: `PlayerList\n\n` | high | (a) hex capture |

What the code does about it:

- The parser (`shared/playerdata.py`) accepts both generations, stops at `PlayerDataEnd`, ignores and reports
  unknown fields, and salvages lines it cannot parse if they still hold one id-shaped token and an X/Y/Z triple.
  `tests/fixtures/rcon/` has 2025 and 2026 fixtures written to the evidence (fake names and ids), and the fake
  RCON server speaks both (`--format`).
- The RCON client reads player data until `PlayerDataEnd`. While the marker is still missing it waits up to
  `marker_idle_timeout_s` (1 s) for more, because 2025-era servers and possibly empty servers never send it.
- Agents report **parse health** with every upload: counts, error reasons and unknown field *names*, never
  values. The API logs a warning when a server first reports problems, and `python -m server servers` shows each
  server's agent version, last upload and parse health. A format change after a game update shows up there first.
- `capture` redacts `ServerPassword` from the server details it saves.

To settle what remains, run `espk-agent capture -c agent.toml` against a live server (a local dedicated server
via SteamCMD works). It writes raw responses to `captures/`. Those contain player names, ids and positions:
check them, add anonymised copies to `tests/fixtures/rcon/`, then delete them.

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
| beeline cheater: AUC vs honest / flagged | 1.00 / 67% | **1.00 / 100%** |
| ambush cheater | 0.95 / 83% | **1.00 / 100%** |
| part-time cheater (ESP 40% of the time) | 0.61 / 8% | **0.82 / 25%** |
| honest players flagged | 0 / 276 | **0 / 276** |

The part-time cheater's numbers swing a lot between simulator versions (an earlier version gave 0.97 / 67% on the
same seeds), because they depend on when its random ESP phases fall. Treat part-time cheating as detectable over
longer play, not within a few hours.

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

## Game mechanics profiles

`game/evrima.yaml` holds what the detector assumes about the game: how far each class can notice other players
(the larger of sight and scent is the player's awareness range), and, for later, calls and in-game groups. It is
public so players can correct it. Servers pick a profile in config.yaml (`game_profile`, default `evrima`); a
modded server gets its own file that `extends: evrima` and overrides what differs.

- **Only the Pteranodon differs** (sees ~900 m from the air). On identical simulated servers, per-class guesses
  for ground classes (250-350 m) performed no better than one 300 m range for everyone, so they were dropped.
- **Groups and calls are documented but not used.** Neither RCON nor the server log reports group membership or
  calls. In-game groups are same-species only, so a Pteranodon scouting for ground carnivores is informal teaming,
  which is for admins to judge; the detector does not excuse it. The server log's chat lines carry a
  `[GROUP-<id>]` tag that looks like the sender's group id (unconfirmed); if it holds up, group members could share
  awareness.
- **Modded servers matter.** Asura's companion platform, for example, shows a live map and lets friends teleport,
  which changes what a player can legitimately know. Such a server needs its own profile.

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
- **Render.** CI pushes the image to GHCR and calls a deploy hook with its digest (`render.yaml`). One service
  runs both roles (`ESPK_ROLE=all` in `docker/start.sh`), because a Render disk attaches to one service and both
  processes need the SQLite file. The start script exits if either process dies, so Render restarts the pair.
  Moving to Render Postgres would allow separate services; that is a second repository implementation (plain SQL).
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
- **Releases follow the backend.** CI's `agent-version` job fails when `agent/`, `shared/` or `packaging/`
  changed since the last `agent-v*` release without a version bump in `agent/__init__.py`. When the version is
  new, `release-agent` runs after the deploy job has seen the new build live (`/healthz` reports the image's git
  commit as `revision`) and publishes the release, creating the tag. The backend deploys first because it must
  accept anything the new agent sends; the agent's payload changes are always backwards compatible for that
  reason. Pushing an `agent-v*` tag by hand still works.
- **Windows is built and smoke-tested in CI** (the same test as Linux, blocking the release). A PyInstaller
  one-file exe on Windows is a launcher plus a child process, so the smoke test stops it with `taskkill /T`.
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
