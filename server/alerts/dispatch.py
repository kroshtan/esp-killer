"""
Deliver queued alerts: build the message (and, for new flags, the evidence image) and send it.

Every outbox row is one alert on one channel. A delivery that fails transiently is retried with backoff (or when
Discord says to), up to ``max_attempts``; a permanent failure (a deleted webhook, a refused address) is marked
failed straight away. Destinations are read from config.yaml at send time, so fixing a webhook URL takes effect
for alerts still in the queue.
"""

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta

import httpx
import numpy as np

from server.alerts.discord import send_discord
from server.alerts.email import send_email_async
from server.alerts.errors import DeliveryError
from server.alerts.message import Alert
from server.alerts.render import Highlight, PathImage, RenderOptions, Track, render_path_png, select_nearby
from server.alerts.settings import SmtpSettings
from server.db.alerts import AlertRepository, OutboxItem
from server.db.leakage import LeakageRepository
from server.db.scoring import Flag, ScoringRepository
from server.orgconfig import OrgConfig
from server.scoring.config import ScoringConfig
from server.scoring.features import beeline_episodes
from server.scoring.job import epoch_seconds, to_trajectories
from server.scoring.trajectories import Trajectories

logger = logging.getLogger(__name__)

NEARBY_RADIUS_M = 1500.0
NEARBY_MAX = 8


@dataclass(frozen=True)
class DispatchSettings:
    max_attempts: int = 8
    image_minutes: float = 30.0
    base_backoff_s: float = 60.0
    max_backoff_s: float = 6 * 3600.0


async def deliver_due(
    alerts: AlertRepository,
    scoring: ScoringRepository,
    config: OrgConfig,
    *,
    smtp: SmtpSettings,
    http: httpx.AsyncClient,
    now: datetime,
    settings: DispatchSettings | None = None,
) -> dict[str, int]:
    """
    Try every alert that is due.

    :param alerts: alert repository (the outbox)
    :param scoring: scoring repository (positions for the image)
    :param config: org config (destinations, scoring thresholds, map transforms)
    :param smtp: SMTP settings
    :param http: HTTP client for Discord
    :param now: current time
    :param settings: retry and image settings
    :return: counts of sent, retried and failed deliveries
    """
    settings = settings or DispatchSettings()
    leakage = LeakageRepository(scoring.db)
    counts = {"sent": 0, "retry": 0, "failed": 0}
    images: dict[tuple[int, str], bytes | None] = {}
    for item in alerts.due(now):
        flag = alerts.get_flag(item.flag_id)
        if flag is None:
            alerts.mark_failed(item.id, "flag no longer exists")
            counts["failed"] += 1
            continue
        # Only new-flag alerts carry an image; render it once for all of that alert's channels.
        key = (flag.id, item.kind)
        if key not in images:
            images[key] = None
            if item.kind == "flag":
                images[key] = await asyncio.to_thread(evidence_image, scoring, config, flag, item, settings)
        shadow = leakage.latest(flag.org_id, flag.player_id)
        alert = build_alert(flag, item, images[key], model_line=shadow.line() if shadow else None)
        try:
            await _send(item, alert, config, smtp, http)
        except DeliveryError as e:
            attempts = item.attempts + 1
            if e.permanent or attempts >= settings.max_attempts:
                alerts.mark_failed(item.id, str(e))
                counts["failed"] += 1
                logger.error("alert #%d (%s) failed permanently: %s", item.id, item.channel, e)
            else:
                backoff = min(settings.max_backoff_s, settings.base_backoff_s * 2 ** (attempts - 1))
                delay = max(backoff, e.retry_after_s or 0.0)
                alerts.mark_retry(item.id, now + timedelta(seconds=delay), str(e))
                counts["retry"] += 1
                logger.warning("alert #%d (%s) failed, retrying in %.0fs: %s", item.id, item.channel, delay, e)
            continue
        alerts.mark_sent(item.id, now)
        counts["sent"] += 1
        logger.info("alert #%d sent: %s %s via %s", item.id, item.org_id, item.kind, item.channel)
    return counts


def build_alert(flag: Flag, item: OutboxItem, image_png: bytes | None, model_line: str | None = None) -> Alert:
    """
    The message content for an outbox row.

    :param flag: the flag the alert is about
    :param item: the outbox row
    :param image_png: evidence image, if any
    :param model_line: the leakage model's opinion (shadow mode), if it has scored the player
    :return: the alert
    """
    servers = (item.server_id,) if item.server_id else tuple(flag.details.get("servers", ()))
    return Alert(
        kind="rejoin" if item.kind == "rejoin" else "flag",
        org_id=flag.org_id,
        player_id=flag.player_id,
        player_name=flag.player_name,
        server_ids=servers,
        score=flag.score,
        details=flag.details,
        created_at=item.created_at,
        flag_id=flag.id,
        image_png=image_png,
        model_line=model_line,
    )


async def _send(
    item: OutboxItem, alert: Alert, config: OrgConfig, smtp: SmtpSettings, http: httpx.AsyncClient
) -> None:
    org = config.orgs.get(item.org_id)
    if item.channel == "discord":
        if org is None or org.alert.discord_webhook is None:
            raise DeliveryError("no Discord webhook configured for this org", permanent=True)
        # One quick in-process retry; longer waits are the outbox's job (the next worker runs).
        await send_discord(http, str(org.alert.discord_webhook), alert, max_attempts=2)
    elif item.channel == "email":
        if org is None or org.alert.email is None:
            raise DeliveryError("no email address configured for this org", permanent=True)
        await send_email_async(smtp, org.alert.email, alert)
    else:
        raise DeliveryError(f"unknown channel {item.channel!r}", permanent=True)


def present_last(tr: Trajectories, player_id: str) -> int:
    """
    Grid index of a player's last sighting in ``tr``.

    :param tr: trajectories
    :param player_id: the player
    :return: the index (0 if never present)
    """
    seen = np.flatnonzero(tr.present[:, tr.player_ids.index(player_id)])
    return int(seen[-1]) if len(seen) else 0


def evidence_image(
    scoring: ScoringRepository, config: OrgConfig, flag: Flag, item: OutboxItem, settings: DispatchSettings
) -> bytes | None:
    """
    Render the flagged player's last minutes on the server where they spent most of them.

    Any failure is logged and the alert goes out without an image: the image helps a reviewer, it must never stop
    the alert.

    :param scoring: scoring repository (positions)
    :param config: org config (scoring thresholds, map transforms)
    :param flag: the flag
    :param item: the outbox row
    :param settings: image window
    :return: PNG bytes, or None if there is nothing to draw or rendering failed
    """
    try:
        return _render(scoring, config, flag, item, settings)
    except Exception:
        logger.exception("could not render the evidence image for flag #%d", flag.id)
        return None


def _render(
    scoring: ScoringRepository, config: OrgConfig, flag: Flag, item: OutboxItem, settings: DispatchSettings
) -> bytes | None:
    cfg: ScoringConfig = config.scoring_config
    end = flag.last_seen_at or item.created_at
    start = end - timedelta(minutes=settings.image_minutes)
    frame = scoring.load_positions(flag.org_id, start, end + timedelta(seconds=1))
    mine = frame[frame["player_id"] == flag.player_id]
    if mine.empty:
        return None
    server_id = str(mine["server_id"].value_counts().idxmax())
    frame = frame[frame["server_id"] == server_id]

    tracks = []
    for player_id, rows in frame.assign(t=epoch_seconds(frame["server_ts"])).sort_values("t").groupby("player_id"):
        name = str(rows["player_name"].iloc[-1])
        tracks.append(
            Track(
                player_id=str(player_id),
                label=name,
                t=rows["t"].to_numpy(dtype=float),
                x=rows["x"].to_numpy(dtype=float),
                y=rows["y"].to_numpy(dtype=float),
            )
        )

    profiles = config.game_profiles(flag.org_id)
    (tr,) = [t for t in to_trajectories(frame, cfg, profiles) if t.server_id == server_id]
    highlights = [
        Highlight(float(tr.t[e.start]), float(tr.t[e.arrival]), tr.player_ids[e.target])
        for e in beeline_episodes(tr, cfg, flag.player_id)
    ]

    targets = {h.target_player_id for h in highlights if h.target_player_id}
    nearby = select_nearby(
        tracks,
        flag.player_id,
        NEARBY_RADIUS_M,
        NEARBY_MAX,
        units_per_metre=cfg.units_per_metre,
        always_include=targets,
    )
    flagged = next(t for t in tracks if t.player_id == flag.player_id)
    org = config.orgs.get(flag.org_id)
    server = org.servers.get(server_id) if org is not None else None
    options = RenderOptions(units_per_metre=cfg.units_per_metre, max_gap_s=cfg.max_gap_s)
    if server is not None and server.map_transform is not None:
        options = options.model_copy(update={"transform": server.map_transform})
    image = PathImage(
        flagged_player_id=flag.player_id,
        flagged_name=flag.player_name,
        tracks=[flagged, *nearby],
        window_start_t=start.timestamp(),
        window_end_t=end.timestamp(),
        highlights=highlights,
        # The flagged player's own range, for their latest class.
        awareness_m=float(tr.awareness[:, tr.player_ids.index(flag.player_id)][present_last(tr, flag.player_id)]),
        subtitle=f"{flag.org_id}/{server_id} · score {flag.score:.2f}",
    )
    return render_path_png(image, options)
