"""
The training dataset: layout, table schemas and pseudonymisation.

This is the contract between the backend (which exports) and the trainer (which reads). Training is
unsupervised: there are no labels, only movement.

The code is public; the dataset and the models are not. They live in a private store (see ``store.py``)::

    dataset/v1/positions/date=YYYY-MM-DD/<org>-<window_end>.parquet   one row per player per poll
    dataset/v1/evidence/date=YYYY-MM-DD/<org>-<window_end>.parquet    per player per scoring window
    dataset/v1/windows/<org>-<window_end>.json                        one per exported window (row counts)
    models/<version>/model_a.txt, model_b.txt, metadata.json          saved models (see ``leakage.py``)
    models/candidates/<version>.json                                  every candidate's gate metrics
    models/current.json                                               {"version": ...} of the promoted model
    config/scoring.json                                               the backend's scoring config, for the trainer

Identifiers are pseudonymised with a keyed hash (HMAC-SHA256 with ``ESPK_EXPORT_KEY``): the same player gets the
same id in every export, so their history links up, but ids cannot be turned back into Steam ids or names without
the key. Player names are not exported at all.
"""

import hashlib
import hmac
from datetime import datetime

import pyarrow as pa

VERSION = "v1"
DATASET = f"dataset/{VERSION}"
MODELS = "models"
CURRENT_MODEL = f"{MODELS}/current.json"
# Written by the worker, so the trainer (which cannot read the backend's config.yaml) uses the same thresholds.
SCORING_CONFIG = "config/scoring.json"

POSITIONS = pa.schema(
    [
        ("org", pa.string()),  # pseudonymised
        ("server", pa.string()),  # pseudonymised
        ("player", pa.string()),  # pseudonymised
        ("t", pa.float64()),  # epoch seconds (UTC)
        ("x", pa.float32()),  # game units (cm)
        ("y", pa.float32()),
        ("z", pa.float32()),
        ("dino_class", pa.dictionary(pa.int16(), pa.string())),
        ("growth", pa.float32()),
    ]
)

EVIDENCE = pa.schema(
    [
        ("org", pa.string()),
        ("player", pa.string()),
        ("window_end", pa.float64()),  # epoch seconds
        ("moving_s", pa.float64()),
        ("beeline_episodes", pa.int32()),
        ("beeline_null", pa.float64()),
        ("ambush_waits", pa.int32()),
        ("ambush_hits", pa.int32()),
        ("ambush_null", pa.float64()),
        ("spawn_episodes", pa.int32()),
        ("clan", pa.int32()),  # inferred clan number within the window's org, -1 for none
    ]
)


class Pseudonymiser:
    """Keyed hashing of identifiers; the key never leaves the backend and the trainer."""

    def __init__(self, key: str) -> None:
        if len(key) < 16:  # noqa: PLR2004
            raise ValueError("ESPK_EXPORT_KEY must be at least 16 characters")
        self._key = key.encode("utf-8")

    def __call__(self, kind: str, value: str) -> str:
        """
        Pseudonymise an identifier.

        :param kind: what it is (``player``, ``server``, ``org``), so equal strings of different kinds differ
        :param value: the identifier
        :return: 20 hex characters
        """
        return hmac.new(self._key, f"{kind}:{value}".encode(), hashlib.sha256).hexdigest()[:20]


def window_name(org: str, window_end: datetime) -> str:
    """
    File stem for one exported window.

    :param org: pseudonymised org id
    :param window_end: end of the scoring window
    :return: e.g. ``1a2b...-20260927T140000Z``
    """
    return f"{org}-{window_end:%Y%m%dT%H%M%SZ}"


def partition(window_end: datetime) -> str:
    """
    The date partition of a window.

    :param window_end: end of the scoring window
    :return: e.g. ``date=2026-09-27``
    """
    return f"date={window_end:%Y-%m-%d}"
