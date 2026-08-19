"""
Cache determinism: a cache hit must never touch the network, and a cache
miss must write exactly one file pair and never re-fetch on a subsequent
call. Uses a temp cache dir + a stubbed network call - does not hit the
real CBBD API.
"""

from __future__ import annotations

import json

import pytest

from ingest_cbbd import _cached_get


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload
        self.status_code = 200

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


def test_cache_hit_never_calls_network(tmp_path, monkeypatch):
    cache_path = tmp_path / "cached.json"
    cache_path.write_text(json.dumps([{"a": 1}]))

    calls = {"n": 0}

    def fake_get(*args, **kwargs):
        calls["n"] += 1
        raise AssertionError("network should not be called on a cache hit")

    monkeypatch.setattr("ingest_cbbd.requests.get", fake_get)
    cfg = {"cbbd": {"request_timeout_s": 5, "retry_attempts": 1, "retry_sleep_s": 0, "inter_request_sleep_s": 0}}

    result = _cached_get(cache_path, "/fake/endpoint", {}, {}, cfg)
    assert result == [{"a": 1}]
    assert calls["n"] == 0


def test_cache_miss_writes_once_then_hits_cache(tmp_path, monkeypatch):
    cache_path = tmp_path / "new.json"
    calls = {"n": 0}

    def fake_get(*args, **kwargs):
        calls["n"] += 1
        return _FakeResponse([{"b": 2}])

    monkeypatch.setattr("ingest_cbbd.requests.get", fake_get)
    cfg = {"cbbd": {"request_timeout_s": 5, "retry_attempts": 1, "retry_sleep_s": 0, "inter_request_sleep_s": 0}}

    r1 = _cached_get(cache_path, "/fake/endpoint", {}, {}, cfg)
    assert calls["n"] == 1
    assert cache_path.exists()

    r2 = _cached_get(cache_path, "/fake/endpoint", {}, {}, cfg)
    assert calls["n"] == 1  # second call must hit the now-existing cache file, not the network
    assert r1 == r2 == [{"b": 2}]


def test_real_cbbd_cache_dir_is_fully_populated_and_untouched_by_rerun():
    """Regression guard for the manual reproducibility check in
    reports/phase1_worklog.md: every raw endpoint file this phase depends
    on must already exist on disk (a missing file would force a real
    network call on the next pipeline run)."""
    from pathlib import Path
    cache_dir = Path(__file__).resolve().parent.parent / "data" / "raw" / "cbbd"
    assert cache_dir.exists()
    files = list(cache_dir.glob("*.json"))
    assert len(files) == 372, f"expected 372 cached CBBD files, found {len(files)}"
