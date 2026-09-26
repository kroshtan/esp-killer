from pathlib import Path

import pytest

from server.keys import generate_key, hash_key, looks_like_key
from server.orgconfig import ConfigStore, OrgConfig, OrgEntry, ServerEntry, ServerIdentity, load_config, save_config
from server.ratelimit import TokenBucketLimiter


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def test_keys() -> None:
    key = generate_key()
    assert looks_like_key(key)
    assert key != generate_key()
    assert hash_key(key).startswith("sha256:")
    assert len(hash_key(key)) == len("sha256:") + 64
    assert not looks_like_key("espk_short")
    assert not looks_like_key("Bearer " + key)


def test_token_bucket() -> None:
    clock = FakeClock()
    limiter = TokenBucketLimiter(rate_per_s=1.0, burst=3, clock=clock)
    assert [limiter.acquire("a") for _ in range(3)] == [0.0, 0.0, 0.0]
    assert limiter.acquire("a") == pytest.approx(1.0)
    assert limiter.acquire("b") == 0.0  # buckets are per key
    clock.now = 0.5
    assert limiter.acquire("a") == pytest.approx(0.5)
    clock.now = 10.0
    assert [limiter.acquire("a") for _ in range(3)] == [0.0, 0.0, 0.0]  # refills only up to burst
    assert limiter.acquire("a") > 0
    assert TokenBucketLimiter.retry_after_header(0.2) == "1"
    assert TokenBucketLimiter.retry_after_header(2.1) == "3"


def _config(*hashes: str | None) -> OrgConfig:
    return OrgConfig(orgs={"org": OrgEntry(servers={f"s{i}": ServerEntry(key_hash=h) for i, h in enumerate(hashes)})})


def test_key_index_skips_revoked_and_rejects_duplicates() -> None:
    a, b = hash_key(generate_key()), hash_key(generate_key())
    assert _config(a, None, b).key_index() == {
        a: ServerIdentity("org", "s0"),
        b: ServerIdentity("org", "s2"),
    }
    with pytest.raises(ValueError, match="duplicate"):
        _config(a, a).key_index()


def test_save_is_private_and_round_trips(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    cfg = _config(hash_key(generate_key()), None)
    save_config(path, cfg)
    assert path.stat().st_mode & 0o777 == 0o600
    assert load_config(path) == cfg
    assert load_config(tmp_path / "missing.yaml") == OrgConfig()


def test_store_reloads_on_change_and_keeps_last_good_config(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    old, new = hash_key(generate_key()), hash_key(generate_key())
    save_config(path, _config(old))
    store = ConfigStore(path)
    assert store.identify(old) == ServerIdentity("org", "s0")

    save_config(path, _config(new))
    assert store.identify(old) is None
    assert store.identify(new) == ServerIdentity("org", "s0")

    path.write_text("orgs: [this is not valid")
    assert store.identify(new) == ServerIdentity("org", "s0")


def test_example_config_is_valid() -> None:
    load_config(Path(__file__).resolve().parents[2] / "config.example.yaml").key_index()
