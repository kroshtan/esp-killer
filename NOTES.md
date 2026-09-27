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

## Scoring

### Priority

**A false flag punishes a fair player; a missed cheater is caught later.** Every rule below leans that way: when
something could be legitimate, it does not count as evidence. Cheaters then need more playing time to be flagged,
which the week-long evidence horizon provides.

### Method

Positions are resampled per server onto a 5 s grid (`server/scoring/trajectories.py`), with each player's
awareness range at each moment from their class (`game/evrima.yaml`). Three behaviours become **counts** per
player (`server/scoring/features.py`):

- **Beeline:** the player lines up on someone out of their range, and stays lined up until that player comes into
  range. Evidence stops at sighting: whatever happens after (a charge, a fight) is legitimate. It does not count
  if the approach is explained some other way:
  - the target was within the player's range in the last 10 minutes (following someone you saw, a long chase);
  - **someone independent had the target within their own range shortly before** (3 minutes): a clanmate, a
    scout, a friend on voice chat, whether or not we know they are a team. Independent means watching from a
    distance (over 150 m) and not the target's own companion or clanmate: a group's members always see each other,
    and that must not excuse ten players raiding it. What remains is heading straight for players nobody
    independent could see, such as a lone clan member or a pair far from anyone;
  - a clanmate was already heading for the target (joining a hunt; the evidence stays with whoever started it);
  - the player was already heading that way before (the target stepped into their path), or the target came to
    the player (head-on meetings, being hunted);
  - it ended in a **peaceful reunion** (together within 150 m for a minute soon after, nobody dying): walking
    straight to a friend is meeting up, not hunting.
- **Ambush:** a wait of at least a minute during which someone who was out of range arrives **and dies shortly
  after**. An ambush is an attack; arrivals without a kill (regroups, passers-by) say nothing.
- **Time to contact:** from each spawn to first contact, ranked against the org's other spawns of the same class.

Beeline and ambush counts are compared with a **time-shifted null**: the same statistic computed with every other
trajectory shifted by 10, 20 and 30 minutes. That keeps where people go (waterholes, trails, bases) and breaks only
where they are *now*, which is what an honest player cannot know about someone out of sight. The evidence is the
count z-score `(observed − null) / √(null + 1)`; for time to contact, the z-score of the mean rank.

**Clans** (`server/scoring/teams.py`) are inferred from behaviour, since the game exposes no group data: pairs of
players who keep meeting up (separate meetings, not ending in a death) or keep heading for players the other had
just spotted, far more often than the null explains, are linked; connected players form a clan (any species, any
size, spread out or not). Clanmates are never beeline targets, and a clanmate already hunting a target makes
joining in a relay. Linking needs clear, repeated evidence (3+ meetups or tips, z ≥ 3): too lenient a rule chains
strangers into one giant clan and excuses everyone in it. With "anyone's sighting explains" doing most of the work,
clan inference only needs to be right about who never counts as a target.

Counts are **additive**: each scoring run processes complete 2-hour windows (plus 30 minutes of context), stores
per-player and per-pair counts, and scores from the sum over the last 7 days. A cheater's excess over the null
grows with playing time; an honest player's does not. Sub-scores rise from 0 at z = 3 to 1 at z = 7 for beelines
(legitimate hard cases reach z 4-5 on simulated clan servers, full-time ESP users 8-12), 4 to 8 for ambushes (without kill logs we cannot tell who killed the arriving player, so defending a base looks
like an ambush at z up to ~5; real ambushers are at 20+), and 1 to 3.5 for time to contact. They are combined with
a weighted noisy-OR (beeline 1.0, ambush 0.9, time to contact 0.35), so time to contact alone never flags anyone.

### Evaluation on simulated servers

`python -m tools.sim.evaluate --seeds 12 --first-seed 500 --hours 6` (mixed servers: 23 honest players of five
kinds, plus three cheaters) and `--clans --seeds 12 --first-seed 700` (two rival clans of 6-10 mixed-species
members who spread out, call sightings over "voice", hunt on each other's calls from 1-2 km away and regroup at a
base; one clan has a Pteranodon spotter, the other an ESP member; plus solo players and a solo cheater). Evidence
accumulated over 6 hours per server:

| | mixed servers | clan servers |
|---|---|---|
| honest players flagged | **0 / 276** (highest score 0.35) | **0 / 340** (highest 0.51) |
| full-time beeline cheater: flagged / AUC | 50% / 0.97 | 58% / 0.88 |
| ambush cheater | 100% / 1.00 | |
| ESP member inside a clan | | 8% / 0.73 |
| part-time cheater (ESP 40% of the time) | 8% / 0.62 | |

Without "anyone's sighting explains" and the reunion and kill rules, the same clan servers flagged 13 of 173 honest
players, almost all clan hunters answering calls; the cheaters were caught more often (83-100%). That trade was
made on purpose. Cheaters not flagged within 6 hours keep accumulating evidence over the week.

Things worth knowing when reading flags:

- **Cheaters create beelines for their victims.** Honest players "arrive" at a cheater who comes to them, or walk
  into an ambusher on their route; the target-approach and turn-onto-target rules handle that.
- **Legitimate reactions can't be reproduced by the null.** Nobody reacts to a time-shifted phantom, so anything
  after a sighting (a charge) is not evidence; hence evidence stops at sighting range.
- **Honest players acting on an ESP user's calls** (a clan with a cheater relaying positions) do accumulate
  evidence, because nobody could see those targets. The ESP user is usually the stronger signal; admins judge the
  rest.

### Limitations

- **Simulation is not reality.** The honest archetypes, and the clans in particular, are guesses at how people play.
  Real thresholds should be tuned with admin feedback (`mark-false-positive`).
- **Crowded servers hide some cheating.** On a busy server a target is often within someone's view, which excuses a
  cheater's approach to them; detection there relies on lone targets and more playing time.
- **No kill attribution.** Deaths are inferred from players disappearing or respawning; who killed whom would make
  the ambush and reunion rules sharper (the server log has kill lines; see "Game mechanics profiles").
- **Latency.** A window is scored once complete plus a 5 minute lag, so a flag needs at least one full 2-hour
  window of play. Positions arriving after their window was processed are stored but never scored.
- **Respawn detection** relies on gaps, teleports and class changes; a respawn inside a gap shorter than 30 s is
  interpolated over and missed.
- **Streamers.** Following a streamer's broadcast position is information leakage too, and looks the same.
- **Scoring config is global** (the `scoring:` section of config.yaml), not per org.

## Leakage model (self-supervised)

A label-free complement to the rules: how much a player's moves follow players **nobody could have told them
about**. Code: `server/training/leakage.py` (features and scoring, shared with the backend) and `trainer/`
(training, evaluation, promotion). Data and models live in the private store (`server/training/store.py`), never in
the repository.

### Method

- **Samples.** Every 15 s, per player continuously alive: the direction of their next 20 s of movement relative to
  the heading they last moved in, in 12 bins of 30°, or "stays put". Standing players are sampled too: where
  someone sets off to after a rest or a fight is one of the most telling decisions.
- **What is known.** Exactly the beeline rule's definition: visible (awareness range from the game profile, plus a
  50 m margin), seen by the player in the last 10 minutes, a friend (time spent together) or clanmate, within range
  of a clanmate or an **independent spotter** in the last 3 minutes, next to (150 m) any such player, or coming
  into view before the predicted move is over (the move may be a reaction to them). Everyone else within 3 km is
  **hidden**. Each of the last three rules removed a signal that simulated honest hunters had (prey at 310 m, the
  prey's companions, prey about to appear); the last one also costs about half of the signal from ESP users,
  whose final approach it excuses, as the beeline rule does.
- **Two models** (LightGBM, CPU). Model A sees the player's own movement, map position and absolute direction (map
  structure: waterholes, trails), time since spawn, class, and the nearest visible, friendly and known players
  (distance, bearing, speed, their heading relative to the player). Model B is A plus a residual that sees *only*
  the hidden players (nearest three, how many, how steadily the player has been heading for the nearest). Both are
  **step-selection models**: they score each candidate move, so "the move towards a hidden player is taken" is one
  pattern for all directions. A multiclass model needed a separate pattern per direction and learned almost nothing
  from the few ESP users in the data.
- **Against the time-shifted null.** A player's per-sample term is `log p_B(move) - mean over shifts of
  log p_B(move | hidden players shifted by 10/20/30 min)`. For an honest player real and phantom hidden players are
  interchangeable (same places, other times), so the terms average zero *whatever B learned*; the raw gain
  `log p_B - log p_A` did not (B's extra confidence and map effects gave everyone small positive or negative
  gains). Per player: z = sum of terms / standard deviation from 2-minute blocks (never below the independent-sample
  value). The statistics are additive, so the backend stores them per window and sums them over the week.
- **Training without labels.** Cross-fitted by player (3 folds by a hash of the pseudonymised id; the same player is
  in the same fold on every server): every number used for calibration and the gate comes from models that never
  saw that player. B's residual is also shown **synthetic ESP users**: every player's real track with hunting
  phases (half of all 10-minute phases) replaced by pursuits of the nearest hidden player at the player's own speed;
  the rest of the world stays real. Without that, B learns only from the ESP users already in the data, which on a
  small or clean dataset is too little (on 4 simulated servers x 3 h, beeline AUC 0.35-0.67 without, 0.8-0.9
  with). The
  null keeps this safe for honest players. The final model is refit on everything.
- **Calibration.** The sub-score ramps linearly over 4 z units and reaches the flag level (0.6) at the larger of
  z = 5 and the 99.5th percentile of the out-of-sample real z (players with at least 120 samples, about 30 min).
  At most 0.5 % of real players can be flagged by it.

### Promotion gate (no labels)

`python -m trainer train` trains a candidate on the last 28 days and promotes it (writes `models/<version>/` then
`models/current.json`) only if:

1. synthetic ESP users from held-out folds rank above real players: AUC >= 0.7;
2. on fresh simulated servers (mixed and clan worlds, seeds never used for training, 4 h each), no honest player
   reaches the flag level, and beeline cheaters rank above honest players: AUC >= 0.85;
3. the out-of-sample real flag rate is within 0.5 %;
4. if a model is promoted already: the candidate is not worse on (1) and (2) by more than 0.03 (the current model is
   re-run on the same simulated servers), and its highest honest simulated z is not more than 1 above the current
   model's.

The AUC thresholds sit just below what the method reaches on the simulated development data (0.74 and 0.92): they
stop a model that learned nothing (about 0.5; `tests/integration/test_trainer.py` trains one on shuffled hidden
columns) or regressed badly. They are not a claim that these values are good enough, and need revisiting on real
data.

Every candidate's metrics, benchmark and gate result are written to `models/candidates/<version>.json`, promoted
or not; the promoted model's `metadata.json` has the same. A rejected candidate is not an error (exit code 0).

### Evaluation on simulated data

`python -m trainer devdata --store ./devdata --seeds 1-6 --hours 4` (12 servers: 6 mixed, 6 clan worlds; 1.5 M
position rows, 340 players) then `python -m trainer train --store ./devdata --bench-seeds 9001-9003`:

| | |
|---|---|
| synthetic ESP users vs real players (held out) | AUC 0.74 |
| cross-fitted z on the training servers, beeline cheaters vs honest | AUC 0.87; mean z 1.4, honest archetypes -0.2 to 0.3 |
| benchmark: 6 fresh servers x 4 h, AUC vs honest | beeline 0.92, part-time 0.91, ambush 0.75, ESP clan member 0.66 |
| highest honest z on that benchmark | 2.7 (clan hunters 1.1, clan members 1.3, spotters 1.0, hunters 2.0) |
| benchmark: 4 fresh servers x 12 h | beeline mean z 3.3 (max 4.0), part-time 3.0, AUC 1.0; honest max 2.0 |
| flag level | z 5.0 (the floor; the highest real z was 3.6), so nobody is flagged yet |
| training (24 cores) | 2 min 7 s including sample building and benchmark; model 1.3 MB |
| scoring a 2.5 h window | 0.5 s for 26 players, 5.5 s for 93 |

The runs are deterministic (same data, same model). This is a slow, cumulative signal: after 4 hours no cheater is
near z 5, after 12 hours the best are at 4. z grows with the square root of playing time for a consistent ESP user
and stays put for honest players (hunters: mean 0.1 at 4 h, 0.2 at 12 h), so over the week's horizon full-time ESP
users should cross it; part-time and in-clan ESP use mostly will not. It ranks well long before it flags, which is
what the shadow and corroborating phases use.

### Limitations

- **Power depends on data.** B must learn what following hidden players looks like; the synthetic ESP users teach
  it one style (straight pursuit). ESP use that looks different (ambushes, relaying to a clan) is caught only as far
  as it resembles that, or as the real data shows it.
- **Simulation is not reality**, and neither are synthetic pursuits of real tracks: the gate's thresholds are set
  from simulated data and will need revisiting on real servers.
- **Honest confounders the null does not remove**: anything that correlates a player's moves with where unseen
  players are *right now* (a streamer's position, a friend on voice chat not detected as a friend, the game's own
  sounds beyond the awareness range). The margin and the known-player rules cover the ones seen in simulation.
- **Retraining changes scores.** Stored per-window statistics come from the model current at the time; after a
  promotion, older windows are not rescored.
- **The trainer needs the scoring config** (`--config` with the backend's config.yaml) to use the operator's
  thresholds for what counts as known; without it, the defaults.

### Running it

The trainer is a batch job: `python -m trainer train --store "$ESPK_DATA_URL"` exits 0 whether or not it promoted
anything. `--min-new-rows N` makes it exit early unless N position rows were exported since the current model was
trained. Environment: `ESPK_DATA_URL` (`s3://bucket/prefix` or a path), and for S3-compatible stores
`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` and `ESPK_S3_ENDPOINT_URL`. Schedule it weekly, either as

- a **Render Cron Job** running the backend image (`ghcr.io/kroshtan/esp-killer`) with the command
  `python -m trainer train`, schedule `0 4 * * 1` (Mondays 04:00 UTC), with the variables above set in the
  dashboard (see `render.yaml`); or
- a **crontab** line on any host with Docker:
  `0 4 * * 1 docker run --rm --env-file /etc/espk-trainer.env ghcr.io/kroshtan/esp-killer python -m trainer train`.

The trainer uses the scoring config the worker writes to the store (`config/scoring.json`), so the operator's
private thresholds apply without the trainer reading `config.yaml`; `--config` overrides it.

### Shadow mode

The worker loads the promoted model (`models/current.json`, checked every run) and scores each scoring window
within the evidence horizon with it, storing each player's additive statistics per window and model version. A
player's model score is their statistics summed over the horizon. It is shown in alerts and `list-flags` next to
the rule-based score and never opens a flag. A newly promoted model scores the horizon's windows again (as long as
their positions are still retained), so it catches up within a few runs. Any failure of the model is logged and
leaves scoring and alerts untouched.

**Do not run it in GitHub Actions**: this repository is public and so are its Actions logs, which would expose the
model metrics and the store's layout. Training needs a few GB of RAM for a few weeks of a busy org; it uses all cores.

## Game mechanics profiles

`game/evrima.yaml` holds what the detector assumes about the game: how far each class can notice other players
(the larger of sight and scent is the player's awareness range), and, for later, calls and in-game groups. It is
public so players can correct it. Servers pick a profile in config.yaml (`game_profile`, default `evrima`); a
modded server gets its own file that `extends: evrima` and overrides what differs.

- **Only the Pteranodon differs** (sees ~900 m from the air; how scouts actually play is still being asked). On
  identical simulated servers, per-class guesses for ground classes performed no better than one 300 m range.
- **Groups and calls are not read from the game.** Neither RCON nor the server log reports group membership or
  calls, so clans are inferred from behaviour (see Scoring), mixed species included. The server log's chat lines
  carry a `[GROUP-<id>]` tag that looks like the sender's in-game group id (unconfirmed), and kill lines name killer
  and victim; both could sharpen the inference later.
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

- **Label-free model: information leakage.** Built as stage 1 (see "Leakage model"). Next: weak labels from owner
  false-positive marks and bans, more ESP styles for the synthetic users (ambushes, relaying to a clan), and
  rescoring stored windows after a promotion.
- A pseudonymised training export (keyed-hash ids, no names, relative time) with its own retention, if data
  from several orgs is ever pooled.
