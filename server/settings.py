"""Process-level server settings from ``ESPK_*`` environment variables. Org and server config lives in config.yaml."""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ServerSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="ESPK_", extra="ignore")

    config_path: Path = Path("config.yaml")
    database_url: str = "sqlite:///data/espk.db"

    # Request limits. Compressed and decompressed sizes are capped separately so a small gzip bomb cannot
    # expand into memory.
    max_body_bytes: int = Field(default=2 * 1024 * 1024, ge=1024)
    max_decompressed_bytes: int = Field(default=16 * 1024 * 1024, ge=1024)

    # Per-key token bucket: sustained requests per second, and burst size.
    rate_limit_per_s: float = Field(default=1.0, gt=0)
    rate_limit_burst: int = Field(default=20, ge=1)

    # Snapshots stamped further in the future than this (agent clock skew) are rejected, as are ones older than
    # the retention period, which would be deleted straight away.
    max_clock_skew_s: float = Field(default=300.0, ge=0)
    retention_days: int = Field(default=14, ge=1)

    # Worker: how often the scoring job runs, and how long after a window ends before it is scored (uploads can
    # be a little late).
    scoring_interval_s: float = Field(default=300.0, gt=0)
    scoring_lag_s: float = Field(default=300.0, ge=0)

    log_level: str = "INFO"
