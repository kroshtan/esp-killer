# Configuration reference

Every setting, where it comes from, and its default. The README's [server owner guide](../README.md#server-owner-guide)
and [operator guide](../README.md#operator-guide) explain how to get started; this page is the full list.
`tests/unit/test_configuration_docs.py` fails if a setting exists in the code but not here.

## Agent (`agent.toml`)

The agent reads a TOML file (`-c agent.toml`) and environment variables; **environment variables win**. Nested
keys use a double underscore: `[rcon] password` is `ESPK_RCON__PASSWORD`. A commented example is in
[agent.example.toml](../agent.example.toml).

### Top level

| Setting | Environment variable | Default | Meaning |
|---|---|---|---|
| `poll_interval_s` | `ESPK_POLL_INTERVAL_S` | `3.0` | Seconds between RCON polls (minimum 0.5). |
| `upload_interval_s` | `ESPK_UPLOAD_INTERVAL_S` | `15.0` | Seconds between uploads when there is no backlog. |
| `max_snapshots_per_batch` | `ESPK_MAX_SNAPSHOTS_PER_BATCH` | `200` | Snapshots per upload (halved automatically if the backend says too large). |
| `queue_path` | `ESPK_QUEUE_PATH` | `esp-agent-queue.db` | The on-disk upload queue, relative to the working directory. |
| `queue_max_bytes` | `ESPK_QUEUE_MAX_BYTES` | `104857600` (100 MiB) | Queue size cap; beyond it the oldest snapshots are dropped. |
| `log_level` | `ESPK_LOG_LEVEL` | `INFO` | `DEBUG` shows unparseable RCON lines (field names only). |

### `[rcon]`

| Setting | Environment variable | Default | Meaning |
|---|---|---|---|
| `host` | `ESPK_RCON__HOST` | `127.0.0.1` | Game server RCON address. |
| `port` | `ESPK_RCON__PORT` | `8888` | `RconPort` from the game's `Game.ini`. |
| `password` | `ESPK_RCON__PASSWORD` | required | `RconPassword`. Only used to log in locally; never sent anywhere. |
| `connect_timeout_s` | `ESPK_RCON__CONNECT_TIMEOUT_S` | `5.0` | Giving up on connecting. |
| `response_timeout_s` | `ESPK_RCON__RESPONSE_TIMEOUT_S` | `5.0` | Waiting for the first bytes of a response. |
| `idle_timeout_s` | `ESPK_RCON__IDLE_TIMEOUT_S` | `0.25` | A response is complete once the socket is quiet this long (RCON has no length prefix). |
| `marker_idle_timeout_s` | `ESPK_RCON__MARKER_IDLE_TIMEOUT_S` | `1.0` | The same, while an end marker (`PlayerDataEnd`) is still expected. |
| `max_response_bytes` | `ESPK_RCON__MAX_RESPONSE_BYTES` | `1048576` | Larger responses are refused. |

### `[backend]`

| Setting | Environment variable | Default | Meaning |
|---|---|---|---|
| `url` | `ESPK_BACKEND__URL` | `https://espk.example.org` | The backend the operator gave you. Must be HTTPS unless it is on this machine. |
| `api_key` | `ESPK_BACKEND__API_KEY` | required | Your server's key (`espk_...`). |
| `timeout_s` | `ESPK_BACKEND__TIMEOUT_S` | `15.0` | Per upload request. |
| `allow_insecure_http` | `ESPK_BACKEND__ALLOW_INSECURE_HTTP` | `false` | Allow plain HTTP to a remote backend. For local testing only. |

## Backend (environment variables)

Set on the API/worker container (Render dashboard, `.env` for docker compose).

### Storage and roles

| Environment variable | Default | Meaning |
|---|---|---|
| `ESPK_ROLE` | `api` | What the container runs: `api`, `worker`, or `all` (both; for Render). |
| `PORT` | `8000` | HTTP port (Render sets it). |
| `ESPK_BEHIND_PROXY` | `false` | Trust `X-Forwarded-*` from a TLS proxy in front (Render, Caddy). |
| `ESPK_CONFIG_PATH` | `config.yaml` (`/data/config.yaml` in the image) | Orgs, servers, key hashes, alert destinations, scoring overrides. |
| `ESPK_DATABASE_PATH` | `data/espk.db` (`/data/espk.db` in the image) | The SQLite database. |
| `ESPK_LOG_LEVEL` | `INFO` | Log level. |

### Ingest limits

| Environment variable | Default | Meaning |
|---|---|---|
| `ESPK_MAX_BODY_BYTES` | `2097152` | Largest accepted upload (compressed). |
| `ESPK_MAX_DECOMPRESSED_BYTES` | `16777216` | Largest accepted upload after gzip (zip-bomb guard). |
| `ESPK_RATE_LIMIT_PER_S` | `1.0` | Sustained uploads per second per key. |
| `ESPK_RATE_LIMIT_BURST` | `20` | Burst size per key. |
| `ESPK_MAX_CLOCK_SKEW_S` | `300.0` | Snapshots further in the future than this are rejected (agent clock wrong). |

### Worker: scoring, alerts, retention

| Environment variable | Default | Meaning |
|---|---|---|
| `ESPK_SCORING_INTERVAL_S` | `300.0` | How often the worker runs. |
| `ESPK_SCORING_LAG_S` | `300.0` | A window is scored this long after it ends (late uploads). |
| `ESPK_ALERT_COOLDOWN_H` | `24.0` | At most one flag alert per player per org in this period. |
| `ESPK_REJOIN_GAP_S` | `600.0` | A flagged player seen again after this long away triggers a rejoin alert. |
| `ESPK_ALERT_MAX_ATTEMPTS` | `8` | Delivery attempts before an alert is marked failed. |
| `ESPK_ALERT_IMAGE_MINUTES` | `30.0` | How much of the player's path the alert image shows. |
| `ESPK_RETENTION_DAYS` | `14` | Raw positions are deleted after this many days. |
| `ESPK_SCORE_RETENTION_DAYS` | `90` | Score history. |
| `ESPK_ALERT_RETENTION_DAYS` | `90` | Delivered and failed alerts. |
| `ESPK_FLAG_RETENTION_DAYS` | `365` | Flags, counted from their last update. |

### Email alerts (optional)

Email is off until both `ESPK_SMTP_HOST` and `ESPK_SMTP_FROM_ADDRESS` are set.

| Environment variable | Default | Meaning |
|---|---|---|
| `ESPK_SMTP_HOST` | none | SMTP server. |
| `ESPK_SMTP_PORT` | `587` | SMTP port. |
| `ESPK_SMTP_USERNAME` | none | Login. |
| `ESPK_SMTP_PASSWORD` | none | Password. |
| `ESPK_SMTP_FROM_ADDRESS` | none | Sender address. |
| `ESPK_SMTP_STARTTLS` | `true` | Upgrade the connection with STARTTLS. |
| `ESPK_SMTP_USE_SSL` | `false` | Use implicit TLS (port 465) instead. |
| `ESPK_SMTP_TIMEOUT_S` | `30.0` | Per connection. |

### Training data and model (optional)

Export to the private training dataset is off until both `ESPK_DATA_URL` and `ESPK_EXPORT_KEY` are set.

| Environment variable | Default | Meaning |
|---|---|---|
| `ESPK_DATA_URL` | none | Private dataset and model store: `s3://bucket/prefix` or a path. |
| `ESPK_EXPORT_KEY` | none | Secret for pseudonymising ids in the dataset. Keep it stable, or histories stop linking up. |
| `ESPK_S3_ENDPOINT_URL` | none | For S3-compatible storage that is not AWS (MinIO, a storage box). |
| `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` | none | Bucket credentials (standard names, also for non-AWS storage). |

## Orgs, servers and alert destinations (`config.yaml`)

Managed with the CLI (`python -m server --help`); format in [config.example.yaml](../config.example.yaml). Per org:
`alert.discord_webhook`, `alert.email`, and `servers`, each with `key_hash` (written by `add-server`), optional
`game_profile` (default `evrima`, see [game/](../game/)) and optional `map_transform` (how alert images draw the
map).

## Scoring thresholds (`scoring:` in `config.yaml`)

Optional; leave it out to use the defaults below. These are the detector's cut-offs: operators may tune their own
values privately, in their `config.yaml`, rather than in the public defaults. [NOTES.md](../NOTES.md#scoring)
explains the method they belong to.

| Setting | Default | Meaning |
|---|---|---|
| `units_per_metre` | `100.0` | Game units per metre. |
| `window_minutes` | `120.0` | Evidence is extracted per window of this length. |
| `context_minutes` | `30.0` | Context read before each window (not counted). |
| `horizon_days` | `7.0` | Scores use the evidence of this many days. |
| `resample_s` | `5.0` | Trajectories are resampled to this grid. |
| `max_gap_s` | `30.0` | Longer gaps between samples mean "not present". |
| `awareness_m` | `300.0` | Fallback awareness range when no game profile applies. |
| `associate_radius_m`, `associate_min_s` | `100.0`, `600.0` | Pairs this close for this long are travelling companions. |
| `null_shifts_s` | `(600.0, 1200.0, 1800.0)` | Time shifts of the null. |
| `team_meet_radius_m`, `team_meet_min_s`, `team_apart_m` | `150.0`, `60.0`, `500.0` | What counts as a meetup (clan inference, reunions). |
| `team_fight_grace_s` | `120.0` | A meeting followed by a death within this long was a fight. |
| `team_min_meets`, `team_link_z` | `3`, `3.0` | Evidence needed to link two players into a clan. |
| `team_shared_awareness_s` | `180.0` | How long a sighting by someone else explains an approach. |
| `explain_by_any_spotter` | `true` | Independent spotters (not just clanmates) explain approaches. |
| `reunion_window_s` | `300.0` | A beeline ending in a peaceful meetup within this long is a reunion. |
| `heading_window_s`, `min_speed_mps` | `15.0`, `1.5` | Heading and "moving" definitions. |
| `beeline_cos` | `0.95` | Lined up means within about 18 degrees. |
| `beeline_min_duration_s`, `beeline_bridge_s` | `60.0`, `15.0` | Minimum lined-up time out of range; wobbles tolerated. |
| `beeline_max_start_m`, `beeline_arrive_m`, `beeline_arrive_grace_s` | `3000.0`, `300.0`, `15.0` | Episode start and arrival. |
| `beeline_turn_lookback_s` | `60.0` | Already heading there this long before means the target was in the way. |
| `beeline_max_target_approach` | `0.3` | The target coming this much of the way to the player explains the meeting. |
| `near_lookback_s` | `600.0` | Following someone seen this recently is legitimate. |
| `beeline_min_moving_s` | `600.0` | Moving time needed before beelines are scored. |
| `beeline_z_floor`, `beeline_z_full` | `3.0`, `7.0` | Beeline sub-score from 0 to 1 over this z range. |
| `contact_m`, `respawn_jump_m`, `ttc_min_observed_s` | `50.0`, `300.0`, `60.0` | Time-to-contact episodes. |
| `ttc_min_episodes`, `ttc_min_baseline` | `2`, `10` | Evidence needed for time to contact. |
| `ttc_z_floor`, `ttc_z_full` | `1.0`, `3.5` | Time-to-contact sub-score range. |
| `stationary_speed_mps`, `ambush_min_wait_s`, `ambush_radius_m`, `ambush_grace_s` | `0.5`, `60.0`, `50.0`, `30.0` | Ambush waits and arrivals. |
| `ambush_min_waits` | `3` | Waits needed before ambushes are scored. |
| `ambush_z_floor`, `ambush_z_full` | `4.0`, `8.0` | Ambush sub-score range. |
| `weight_beeline`, `weight_ambush`, `weight_ttc` | `1.0`, `0.9`, `0.35` | Weights in the combined score. |
| `flag_threshold` | `0.6` | Scores at or above this open a flag. |
| `false_positive_suppress_days` | `30.0` | After `mark-false-positive`, the player is not flagged again for this long. |
