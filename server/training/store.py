"""
Where training data and models live: a private object store, never the repository.

Two implementations with the same small interface: a local directory (development, tests, or a trainer on the same
machine) and any S3-compatible bucket (self-hosted MinIO, a storage box, a small provider). Configured by one URL:

* ``file:///srv/espk-data`` or a plain path;
* ``s3://bucket/prefix``, with credentials from the standard ``AWS_ACCESS_KEY_ID`` / ``AWS_SECRET_ACCESS_KEY``
  environment variables and, for anything that is not AWS itself, ``ESPK_S3_ENDPOINT_URL``.

Keys are ``/``-separated relative paths (``dataset/v1/positions/date=2026-09-27/x.parquet``).
"""

import os
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import urlparse


class ObjectStore(Protocol):
    def put(self, key: str, data: bytes) -> None:
        """
        Write an object, replacing any existing one.

        :param key: object key
        :param data: contents
        """
        ...

    def get(self, key: str) -> bytes:
        """
        Read an object. Implementations raise ``KeyError`` if it does not exist.

        :param key: object key
        :return: contents
        """
        ...

    def list(self, prefix: str) -> list[str]:
        """
        Keys under a prefix, sorted.

        :param prefix: key prefix
        :return: the keys
        """
        ...

    def exists(self, key: str) -> bool:
        """
        Whether an object exists.

        :param key: object key
        :return: True if it exists
        """
        ...


class LocalStore:
    def __init__(self, root: Path) -> None:
        self.root = root

    def _path(self, key: str) -> Path:
        path = (self.root / key).resolve()
        if not path.is_relative_to(self.root.resolve()):
            raise ValueError(f"key escapes the store: {key!r}")
        return path

    def put(self, key: str, data: bytes) -> None:  # noqa: D102
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f".{path.name}.tmp")
        tmp.write_bytes(data)
        os.replace(tmp, path)  # readers never see half an object

    def get(self, key: str) -> bytes:  # noqa: D102
        path = self._path(key)
        if not path.is_file():
            raise KeyError(key)
        return path.read_bytes()

    def list(self, prefix: str) -> list[str]:  # noqa: D102
        base = self.root.resolve()
        if not base.exists():
            return []
        keys = (p.relative_to(base).as_posix() for p in base.rglob("*") if p.is_file() and not p.name.startswith("."))
        return sorted(k for k in keys if k.startswith(prefix))

    def exists(self, key: str) -> bool:  # noqa: D102
        return self._path(key).is_file()


class S3Store:
    def __init__(self, bucket: str, prefix: str = "", client: Any = None) -> None:
        import boto3  # noqa: PLC0415 - only needed when a bucket is configured

        self.bucket = bucket
        self.prefix = prefix.strip("/")
        self.client = client or boto3.client("s3", endpoint_url=os.environ.get("ESPK_S3_ENDPOINT_URL") or None)

    def _key(self, key: str) -> str:
        return f"{self.prefix}/{key}" if self.prefix else key

    def put(self, key: str, data: bytes) -> None:  # noqa: D102
        self.client.put_object(Bucket=self.bucket, Key=self._key(key), Body=data)

    def get(self, key: str) -> bytes:  # noqa: D102
        try:
            response = self.client.get_object(Bucket=self.bucket, Key=self._key(key))
        except self.client.exceptions.NoSuchKey as e:
            raise KeyError(key) from e
        body: bytes = response["Body"].read()
        return body

    def list(self, prefix: str) -> list[str]:  # noqa: D102
        keys: list[str] = []
        strip = len(self.prefix) + 1 if self.prefix else 0
        for page in self.client.get_paginator("list_objects_v2").paginate(
            Bucket=self.bucket, Prefix=self._key(prefix)
        ):
            keys.extend(obj["Key"][strip:] for obj in page.get("Contents", []))
        return sorted(keys)

    def exists(self, key: str) -> bool:  # noqa: D102
        try:
            self.client.head_object(Bucket=self.bucket, Key=self._key(key))
        except self.client.exceptions.ClientError:
            return False
        return True


def open_store(url: str) -> ObjectStore:
    """
    Open the store a URL names.

    :param url: ``s3://bucket/prefix``, ``file:///path`` or a plain path
    :return: the store
    :raises ValueError: for an unsupported scheme
    """
    parsed = urlparse(url)
    if parsed.scheme == "s3":
        return S3Store(parsed.netloc, parsed.path)
    if parsed.scheme in ("", "file"):
        return LocalStore(Path(parsed.path if parsed.scheme == "file" else url))
    raise ValueError(f"unsupported store URL {url!r} (use s3://bucket/prefix or a path)")
