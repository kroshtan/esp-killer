"""
Deciding which alerts to send: new flags, and flagged players coming back.

This module only queues alerts in the outbox (``server/db/alerts.py``); the worker delivers them. Rules:

* a new flag queues a ``flag`` alert on every configured channel, unless a ``flag`` alert about the same player
  was queued within ``cooldown`` (at most one per player per org per day);
* a flagged player seen again after an absence of at least ``rejoin_gap`` queues a ``rejoin`` alert, naming the
  server they joined. Presence is tracked per flag in ``flags.last_seen_at``, so each return alerts once.

Only open flags alert: a flag marked as a false positive goes quiet.
"""

import logging
from collections.abc import Iterable
from datetime import datetime, timedelta

from server.db.alerts import AlertRepository
from server.db.scoring import Flag
from server.orgconfig import AlertDestinations, OrgConfig

logger = logging.getLogger(__name__)


def channels_for(destinations: AlertDestinations, email_enabled: bool) -> list[str]:
    """
    The channels an org's alerts go to.

    :param destinations: the org's alert settings
    :param email_enabled: whether SMTP is configured on this backend
    :return: ``discord`` and/or ``email``; empty if neither is set up
    """
    channels = []
    if destinations.discord_webhook is not None:
        channels.append("discord")
    if destinations.email is not None and email_enabled:
        channels.append("email")
    return channels


def enqueue_flag_alerts(
    repo: AlertRepository,
    config: OrgConfig,
    flag_ids: Iterable[int],
    now: datetime,
    *,
    email_enabled: bool,
    cooldown: timedelta = timedelta(hours=24),
) -> list[int]:
    """
    Queue alerts for newly opened flags.

    :param repo: alert repository
    :param config: org config (alert destinations)
    :param flag_ids: flags opened by the scoring run
    :param now: current time
    :param email_enabled: whether SMTP is configured
    :param cooldown: minimum time between flag alerts about the same player
    :return: ids of queued outbox rows
    """
    queued: list[int] = []
    for flag_id in flag_ids:
        flag = repo.get_flag(flag_id)
        if flag is None:
            continue
        org = config.orgs.get(flag.org_id)
        channels = channels_for(org.alert, email_enabled) if org is not None else []
        if not channels:
            logger.warning("%s: flag #%d has no alert destination configured", flag.org_id, flag.id)
            continue
        last = repo.last_alert_at(flag.org_id, flag.player_id, "flag")
        if last is not None and now - last < cooldown:
            continue
        queued += repo.enqueue(flag, "flag", channels, now)
        _mark_seen_now(repo, flag)
    return queued


def detect_rejoins(
    repo: AlertRepository,
    config: OrgConfig,
    now: datetime,
    *,
    email_enabled: bool,
    rejoin_gap: timedelta = timedelta(minutes=10),
) -> list[int]:
    """
    Queue a rejoin alert for every open-flagged player who came back after an absence.

    :param repo: alert repository
    :param config: org config
    :param now: current time
    :param email_enabled: whether SMTP is configured
    :param rejoin_gap: an absence at least this long, followed by a sighting, is a rejoin
    :return: ids of queued outbox rows
    """
    queued: list[int] = []
    for org_id, org in config.orgs.items():
        channels = channels_for(org.alert, email_enabled)
        for flag in repo.open_flags(org_id):
            if flag.last_seen_at is None:
                # Never tracked (e.g. flagged while no destination was configured): start tracking now, no alert.
                _mark_seen_now(repo, flag)
                continue
            sightings = repo.sightings(org_id, flag.player_id, after=flag.last_seen_at)
            if not sightings:
                continue
            previous = flag.last_seen_at
            for seen_at, server_id in sightings:
                if seen_at - previous >= rejoin_gap and channels:
                    queued += repo.enqueue(flag, "rejoin", channels, now, server_id=server_id)
                previous = seen_at
            repo.set_last_seen(flag.id, previous)
    return queued


def _mark_seen_now(repo: AlertRepository, flag: Flag) -> None:
    """Start presence tracking from the player's latest sighting, so their current session is not a rejoin."""
    latest = repo.latest_sighting(flag.org_id, flag.player_id)
    if latest is not None and (flag.last_seen_at is None or latest > flag.last_seen_at):
        repo.set_last_seen(flag.id, latest)
    elif flag.last_seen_at is None:
        repo.set_last_seen(flag.id, flag.created_at)
