"""
API keys.

A key is ``espk_`` plus 32 random bytes (base64url). Only its SHA-256 is stored. A fast unsalted hash is right
here, unlike for passwords: with 256 bits of entropy there is nothing to brute-force, and it makes the lookup a
plain dictionary access on the hash.
"""

import hashlib
import re
import secrets

KEY_PREFIX = "espk_"
HASH_PREFIX = "sha256:"
KEY_HASH_PATTERN = r"^sha256:[0-9a-f]{64}$"
_KEY_SHAPE = re.compile(r"^espk_[A-Za-z0-9_-]{43}$")


def generate_key() -> str:
    """
    Generate a new API key.

    :return: the key, to be shown to the operator exactly once
    """
    return KEY_PREFIX + secrets.token_urlsafe(32)


def hash_key(key: str) -> str:
    """
    Hash a key for storage and lookup.

    :param key: the API key
    :return: ``sha256:<hex digest>``
    """
    return HASH_PREFIX + hashlib.sha256(key.encode("utf-8")).hexdigest()


def looks_like_key(value: str) -> bool:
    """
    Cheap shape check before hashing, so obviously wrong tokens are rejected without a lookup.

    :param value: the bearer token
    :return: whether it has the shape of a key
    """
    return bool(_KEY_SHAPE.match(value))
