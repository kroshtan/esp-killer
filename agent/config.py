"""
Agent settings: a TOML file, overridable by ``ESPK_*`` environment variables.

Nested keys use a double underscore, e.g. ``ESPK_RCON__PASSWORD`` or ``ESPK_BACKEND__API_KEY``. The RCON password
is only ever used to log in to the local game server; it is not part of anything sent to the backend.
"""

from pathlib import Path
from typing import Self

from pydantic import BaseModel, Field, HttpUrl, SecretStr, model_validator
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    TomlConfigSettingsSource,
)

_LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1"}


class RconSettings(BaseModel):
    host: str = "127.0.0.1"
    port: int = Field(default=8888, ge=1, le=65535)
    password: SecretStr
    connect_timeout_s: float = Field(default=5.0, gt=0)
    response_timeout_s: float = Field(default=5.0, gt=0)
    # A response is complete once the socket has been quiet this long (the protocol has no length prefix).
    idle_timeout_s: float = Field(default=0.25, gt=0)
    # ...or, while an end marker the server may send (PlayerDataEnd) has not arrived yet, this long.
    marker_idle_timeout_s: float = Field(default=1.0, gt=0)
    max_response_bytes: int = Field(default=1024 * 1024, ge=1024)


class BackendSettings(BaseModel):
    url: HttpUrl = HttpUrl("https://espk.example.org")
    api_key: SecretStr
    timeout_s: float = Field(default=15.0, gt=0)
    # Plain HTTP is refused unless the backend is on this machine or this is explicitly set (never in production).
    allow_insecure_http: bool = False

    @model_validator(mode="after")
    def _require_https(self) -> Self:
        if self.url.scheme != "https" and self.url.host not in _LOCAL_HOSTS and not self.allow_insecure_http:
            raise ValueError("backend.url must use https (set allow_insecure_http only for local testing)")
        return self

    @property
    def ingest_url(self) -> str:
        """
        The ingest endpoint.

        :return: the full URL of ``/v1/ingest``
        """
        return str(self.url).rstrip("/") + "/v1/ingest"


class AgentSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="ESPK_", env_nested_delimiter="__", extra="forbid")

    rcon: RconSettings
    backend: BackendSettings
    poll_interval_s: float = Field(default=3.0, ge=0.5)
    upload_interval_s: float = Field(default=15.0, ge=1.0)
    max_snapshots_per_batch: int = Field(default=200, ge=1, le=2000)
    queue_path: Path = Path("esp-agent-queue.db")
    # When the backend is unreachable the queue grows to this size, then the oldest snapshots are dropped.
    queue_max_bytes: int = Field(default=100 * 1024 * 1024, ge=1024 * 1024)
    log_level: str = "INFO"


def load_settings(config_path: Path | None) -> AgentSettings:
    """
    Load settings from ``config_path`` (if given) with environment variables taking precedence.

    :param config_path: path to a TOML file, or None to use the environment only
    :return: the validated settings
    :raises FileNotFoundError: if ``config_path`` is given but does not exist
    """

    class _FileAndEnvSettings(AgentSettings):
        @classmethod
        def settings_customise_sources(
            cls,
            settings_cls: type[BaseSettings],
            init_settings: PydanticBaseSettingsSource,
            env_settings: PydanticBaseSettingsSource,
            dotenv_settings: PydanticBaseSettingsSource,
            file_secret_settings: PydanticBaseSettingsSource,
        ) -> tuple[PydanticBaseSettingsSource, ...]:
            sources: list[PydanticBaseSettingsSource] = [init_settings, env_settings]
            if config_path is not None:
                sources.append(TomlConfigSettingsSource(settings_cls, toml_file=config_path))
            return tuple(sources)

    if config_path is not None and not config_path.is_file():
        raise FileNotFoundError(f"agent config not found: {config_path}")
    return _FileAndEnvSettings()
