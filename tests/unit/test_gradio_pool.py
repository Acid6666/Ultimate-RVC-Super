"""Unit tests for the on-demand Gradio/GPU lifecycle pool."""

from __future__ import annotations

import os
import time

import pytest

import gradio_pool
from gpu_client import GPUUnavailableError


class FakeLease:
    """Stand-in for a GPU lease."""

    def __init__(self, gpu_ids: list[int], manager: FakeManager) -> None:
        self.gpu_ids = gpu_ids
        self.manager = manager
        self.heartbeat_started = False
        self.released = False

    def start_heartbeat(self) -> None:
        self.heartbeat_started = True

    def release(self) -> None:
        self.released = True
        self.manager.released = list(self.gpu_ids)


class FakeManager:
    """Stand-in for the GPU management client."""

    def __init__(self, gpu_ids: list[int] | None) -> None:
        self.gpu_ids = gpu_ids
        self.lease: FakeLease | None = None
        self.released: list[int] = []

    def acquire(self) -> FakeLease | None:
        if self.gpu_ids is None:
            return None
        self.lease = FakeLease(self.gpu_ids, self)
        return self.lease

    def release(self, gpu_ids: list[int]) -> bool:
        self.released = list(gpu_ids)
        return True

    def heartbeat(self, gpu_ids: list[int]) -> bool:
        return True


class FakeProc:
    """Stand-in for a subprocess handle."""

    def __init__(self, cmd: list[str], **kwargs: object) -> None:
        self.cmd = cmd
        self.kwargs = kwargs
        self.pid = 4242
        self.returncode: int | None = None
        self.waited = False

    def poll(self) -> int | None:
        return None

    def wait(self, timeout: float | None = None) -> int:
        self.waited = True
        return 0


@pytest.fixture
def pool(monkeypatch: pytest.MonkeyPatch, tmp_path) -> gradio_pool.GradioPool:
    """Return a pool with all external effects faked out."""
    spawned: list[FakeProc] = []

    def fake_popen(cmd: list[str], **kwargs: object) -> FakeProc:
        proc = FakeProc(cmd, **kwargs)
        spawned.append(proc)
        return proc

    monkeypatch.setattr(gradio_pool.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(gradio_pool.os, "killpg", lambda pgid, sig: None)
    monkeypatch.setattr(gradio_pool.os, "getpgid", lambda pid: pid)
    monkeypatch.setattr(gradio_pool, "LOG_DIR", tmp_path)
    monkeypatch.setattr(
        gradio_pool.requests,
        "get",
        lambda url, timeout: type("R", (), {"ok": True})(),
    )

    instance = gradio_pool.GradioPool()
    instance.persistent = False
    instance.idle_timeout = 0.1
    instance._new_client = lambda: "CLIENT"  # type: ignore[method-assign]
    instance.spawned = spawned  # type: ignore[attr-defined]
    return instance


def test_ensure_leases_gpu_and_starts_gradio(pool) -> None:
    pool._gpu = FakeManager([3])  # type: ignore[assignment]
    client = pool.ensure()
    assert client == "CLIENT"
    assert pool.gpu_ids == [3]
    proc = pool.spawned[0]  # type: ignore[attr-defined]
    assert proc.kwargs["env"]["CUDA_VISIBLE_DEVICES"] == "3"
    assert pool._lease.heartbeat_started is True  # type: ignore[union-attr]


def test_ensure_raises_when_no_gpu_available(pool) -> None:
    pool._gpu = FakeManager(None)  # type: ignore[assignment]
    with pytest.raises(GPUUnavailableError):
        pool.ensure()
    assert pool.spawned == []  # type: ignore[attr-defined]


def test_idle_reaper_releases_gpu_and_stops_gradio(pool) -> None:
    manager = FakeManager([2])
    pool._gpu = manager  # type: ignore[assignment]
    pool.ensure()
    proc = pool.spawned[0]  # type: ignore[attr-defined]

    pool._last_active = time.time() - pool.idle_timeout - 1
    pool._reap_once()

    assert proc.waited is True
    assert manager.released == [2]
    assert pool.gpu_ids == []
    assert pool._proc is None


def test_reaper_keeps_gpu_while_task_is_active(pool) -> None:
    manager = FakeManager([2])
    pool._gpu = manager  # type: ignore[assignment]
    with pool.session() as client:
        assert client == "CLIENT"
        pool._last_active = time.time() - pool.idle_timeout - 1
        pool._reap_once()
        assert manager.released == []
        assert pool._proc is not None
    assert pool._active == 0
