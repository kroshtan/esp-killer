"""
Organisations, their servers, API key hashes and alert destinations, stored in config.yaml.

The API process reloads the file whenever it changes, so keys added or revoked with the CLI
take effect without a restart. Writes are atomic (temp file + rename), so a reader never sees half a file.

config.yaml contains webhook URLs and is gitignored; config.example.yaml shows the format.
"""

import logging
import os
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, HttpUrl, StringConstraints

from server.keys import KEY_HASH_PATTERN
from server.scoring.config import ScoringConfig

logger = logging.getLogger(__name__)

Slug = Annotated[str, StringConstraints(pattern=r"^[a-z0-9][a-z0-9-]{0,62}$")]


class AlertDestinations(BaseModel):
    model_config = ConfigDict(extra="forbid")

    discord_webhook: HttpUrl | None = None
    email: Annotated[str, StringConstraints(pattern=r"^[^@\s]+@[^@\s]+\.[^@\s]+$")] | None = None


class ServerEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # None means the key was revoked; the server stays listed so its history keeps its name.
    key_hash: Annotated[str, StringConstraints(pattern=KEY_HASH_PATTERN)] | None = None


class OrgEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    alert: AlertDestinations = Field(default_factory=AlertDestinations)
    servers: dict[Slug, ServerEntry] = Field(default_factory=dict)


class OrgConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    orgs: dict[Slug, OrgEntry] = Field(default_factory=dict)
    # Thresholds for every org. Omitted means the public defaults; operators tune their own here, privately.
    scoring: ScoringConfig | None = None

    @property
    def scoring_config(self) -> ScoringConfig:
        """
        The effective scoring config.

        :return: the configured one, or the defaults
        """
        return self.scoring or ScoringConfig()

    def key_index(self) -> dict[str, "ServerIdentity"]:
        """
        Map every active key hash to the server it belongs to.

        :return: key hash -> identity
        :raises ValueError: if two servers share a key hash
        """
        index: dict[str, ServerIdentity] = {}
        for org_id, org in self.orgs.items():
            for server_id, server in org.servers.items():
                if server.key_hash is None:
                    continue
                if server.key_hash in index:
                    raise ValueError(f"duplicate key_hash for {org_id}/{server_id}")
                index[server.key_hash] = ServerIdentity(org_id=org_id, server_id=server_id)
        return index


@dataclass(frozen=True, slots=True)
class ServerIdentity:
    org_id: str
    server_id: str


def load_config(path: Path) -> OrgConfig:
    """
    Read and validate config.yaml. A missing file is an empty config.

    :param path: path to the file
    :return: the config
    """
    if not path.exists():
        return OrgConfig()
    data: Any = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return OrgConfig.model_validate(data)


def save_config(path: Path, config: OrgConfig) -> None:
    """
    Write config.yaml atomically, readable by the owner only (it contains webhook URLs).

    Comments in the existing file are not preserved.

    :param path: path to the file
    :param config: the config to write
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    data = config.model_dump(mode="json", exclude_none=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, sort_keys=False)
        os.chmod(tmp, 0o600)
        os.replace(tmp, path)
    finally:
        # Only still there if something failed before the rename.
        Path(tmp).unlink(missing_ok=True)


class ConfigStore:
    """Thread-safe, auto-reloading view of config.yaml for the API process."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()
        # (mtime, inode, size): an atomic replace changes the inode even within one mtime tick.
        self._stamp: tuple[int, int, int] | None = None
        self._config = OrgConfig()
        self._index: dict[str, ServerIdentity] = {}
        self.reload_if_changed()

    @property
    def config(self) -> OrgConfig:
        """
        The current config, reloaded first if the file changed.

        :return: the config
        """
        self.reload_if_changed()
        return self._config

    def identify(self, key_hash: str) -> ServerIdentity | None:
        """
        Find the server a key belongs to.

        :param key_hash: ``sha256:<hex>`` of the presented key
        :return: the server, or None if the key is unknown or revoked
        """
        self.reload_if_changed()
        return self._index.get(key_hash)

    def reload_if_changed(self) -> None:
        """
        Reload the file if it changed.

        A file that fails validation is logged and ignored, and the previous config stays in effect, so a bad
        hand edit cannot lock every agent out.
        """
        try:
            st = self.path.stat()
            stamp: tuple[int, int, int] | None = (st.st_mtime_ns, st.st_ino, st.st_size)
        except FileNotFoundError:
            stamp = None
        if stamp == self._stamp:
            return
        with self._lock:
            if stamp == self._stamp:
                return
            try:
                config = load_config(self.path)
                index = config.key_index()
            except (ValueError, yaml.YAMLError) as e:
                logger.error("ignoring invalid %s, keeping the previous config: %s", self.path, e)
                self._stamp = stamp
                return
            self._config, self._index, self._stamp = config, index, stamp
            logger.info("loaded %s: %d org(s), %d active key(s)", self.path, len(config.orgs), len(index))
