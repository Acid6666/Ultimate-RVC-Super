"""Unit tests for the GPU management service client."""

from __future__ import annotations

import time

import pytest

import requests

from gpu_client import (
    GPULease,
    GPUManager,
    GPURequestError,
    _parse_gpu_ids,
)

ACQUIRE_URL = "http://127.0.0.1:7050/v1/gpu/acquire"
RELEASE_URL = "http://127.0.0.1:7050/v1/gpu/release"
HEARTBEAT_URL = "http://127.0.0.1:7050/v1/gpu/heartbeat"


class FakeResponse:
    """Minimal stand-in for ``requests.Response``."""

    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def json(self) -> dict:
        return self._payload


def _ok_acquire() -> FakeResponse:
    return FakeResponse({"code": 0, "data": [1, 3], "msg": "成功"})


@pytest.fixture
def manager(monkeypatch: pytest.MonkeyPatch) -> GPUManager:
    """Return a manager whose HTTP calls are captured."""
    calls: list[tuple[str, dict]] = []

    def fake_post(url: str, json: dict, timeout: float) -> FakeResponse:
        calls.append((url, json))
        if url.endswith("/acquire"):
            return _ok_acquire()
        return FakeResponse({"code": 0, "data": {}, "msg": "操作成功"})

    monkeypatch.setattr(requests, "post", fake_post)
    mgr = GPUManager(device_count=2, heartbeat_interval=0.05)
    mgr.calls = calls  # type: ignore[attr-defined]
    return mgr


def test_parse_gpu_ids_supports_lists_and_dicts() -> None:
    assert _parse_gpu_ids([1, 3]) == [1, 3]
    assert _parse_gpu_ids(["2", "4"]) == [2, 4]
    assert _parse_gpu_ids([{"gpu_id": 5}]) == [5]
    assert _parse_gpu_ids(None) == []
    assert _parse_gpu_ids({}) == []


def test_acquire_returns_lease_and_sends_count(manager: GPUManager) -> None:
    lease = manager.acquire()
    assert lease is not None
    assert lease.gpu_ids == [1, 3]
    assert manager.calls[0] == (ACQUIRE_URL, {"count": 2})  # type: ignore[attr-defined]
    lease.release()


def test_acquire_returns_none_when_no_resource(
    manager: GPUManager,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        requests,
        "post",
        lambda url, json, timeout: FakeResponse(
            {"code": 7, "data": {}, "msg": "没有可用GPU资源"},
        ),
    )
    assert manager.acquire() is None


def test_acquire_raises_on_unexpected_error(
    manager: GPUManager,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        requests,
        "post",
        lambda url, json, timeout: FakeResponse(
            {"code": 9, "data": {}, "msg": "参数错误"},
        ),
    )
    with pytest.raises(GPURequestError):
        manager.acquire()


def test_acquire_raises_when_service_unreachable(
    manager: GPUManager,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def boom(url: str, json: dict, timeout: float) -> FakeResponse:
        raise requests.ConnectionError("connection refused")

    monkeypatch.setattr(requests, "post", boom)
    with pytest.raises(GPURequestError):
        manager.acquire()


def test_release_sends_gpu_ids(manager: GPUManager) -> None:
    assert manager.release([1, 3]) is True
    assert manager.calls[-1] == (RELEASE_URL, {"gpu_ids": [1, 3]})  # type: ignore[attr-defined]


def test_heartbeat_sends_gpu_ids(manager: GPUManager) -> None:
    assert manager.heartbeat([1, 3]) is True
    assert manager.calls[-1] == (HEARTBEAT_URL, {"gpu_ids": [1, 3]})  # type: ignore[attr-defined]


def test_lease_heartbeat_thread_stops_after_release(manager: GPUManager) -> None:
    # NOTE 心跳间隔最小为 1 秒，避免把管理服务打爆
    lease = GPULease(manager, [7], heartbeat_interval=1.0)
    lease.start_heartbeat()
    time.sleep(1.3)
    beats = [c for c in manager.calls if c[0].endswith("/heartbeat")]  # type: ignore[attr-defined]
    assert len(beats) >= 1
    lease.release()
    time.sleep(1.3)
    beats_after = [c for c in manager.calls if c[0].endswith("/heartbeat")]  # type: ignore[attr-defined]
    assert len(beats_after) == len(beats)
    assert manager.calls[-1] == (RELEASE_URL, {"gpu_ids": [7]})  # type: ignore[attr-defined]
