"""
Client for the GPU resource management service.

The service exposes three endpoints:

* ``POST /v1/gpu/acquire``   - lease ``count`` GPUs
* ``POST /v1/gpu/release``   - return previously leased GPUs
* ``POST /v1/gpu/heartbeat`` - keep a lease alive (建议 60 秒一次)

A lease that stops receiving heartbeats is reclaimed by the service
after 10 minutes, therefore every active lease runs a background
heartbeat thread.
"""

from __future__ import annotations

from typing import Any

import logging
import threading

import requests

logger = logging.getLogger("urvc.gpu")

DEFAULT_BASE_URL = "http://127.0.0.1:7050"
DEFAULT_TIMEOUT = 10.0
DEFAULT_HEARTBEAT_INTERVAL = 60.0

CODE_SUCCESS = 0
CODE_NO_RESOURCE = 7


class GPUUnavailableError(RuntimeError):
    """No GPU resource could be leased from the management service."""


class GPURequestError(RuntimeError):
    """The GPU management service answered with an unexpected error."""


def _parse_gpu_ids(data: Any) -> list[int]:
    """
    Normalize the ``data`` field of a response into a list of GPU ids.

    Parameters
    ----------
    data : Any
        The ``data`` field returned by the GPU management service. It is
        expected to be a list of integers, but dicts and numeric strings
        are tolerated as well.

    Returns
    -------
    list[int]
        The parsed GPU ids. An empty list is returned when nothing could
        be parsed.

    """
    if data is None or isinstance(data, bool):
        return []
    if isinstance(data, (int, float)):
        return [int(data)]
    items: list[Any] = [data] if isinstance(data, str) else list(data)
    gpu_ids: list[int] = []
    for item in items:
        if isinstance(item, bool):
            continue
        if isinstance(item, (int, float)):
            gpu_ids.append(int(item))
        elif isinstance(item, str) and item.strip().lstrip("-").isdigit():
            gpu_ids.append(int(item.strip()))
        elif isinstance(item, dict):
            for key in ("gpu_id", "gpuId", "GpuId", "id", "ID"):
                value = item.get(key)
                if isinstance(value, (int, float)) and not isinstance(
                    value,
                    bool,
                ):
                    gpu_ids.append(int(value))
                    break
    return gpu_ids


class GPULease:
    """A set of GPU ids leased from the GPU management service."""

    def __init__(
        self,
        manager: GPUManager,
        gpu_ids: list[int],
        heartbeat_interval: float = DEFAULT_HEARTBEAT_INTERVAL,
    ) -> None:
        self._manager = manager
        self._gpu_ids = list(gpu_ids)
        self._heartbeat_interval = max(1.0, float(heartbeat_interval))
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._released = False

    @property
    def gpu_ids(self) -> list[int]:
        """The GPU ids covered by this lease."""
        return list(self._gpu_ids)

    @property
    def released(self) -> bool:
        """Whether the lease has been released."""
        return self._released

    def start_heartbeat(self) -> None:
        """Start sending heartbeats in a background thread."""
        if self._thread is not None or self._released:
            return
        self._thread = threading.Thread(
            target=self._heartbeat_loop,
            name="gpu-heartbeat",
            daemon=True,
        )
        self._thread.start()

    def _heartbeat_loop(self) -> None:
        while not self._stop_event.wait(0.0):
            self.send_heartbeat()
            if self._stop_event.wait(self._heartbeat_interval):
                break

    def send_heartbeat(self) -> bool:
        """
        Send a single heartbeat for the leased GPUs.

        Returns
        -------
        bool
            True if the management service acknowledged the heartbeat.

        """
        if self._released:
            return False
        try:
            return self._manager.heartbeat(self._gpu_ids)
        except Exception:
            logger.exception("GPU %s 心跳失败", self._gpu_ids)
            return False

    def release(self) -> None:
        """Stop the heartbeat thread and return the GPUs to the pool."""
        if self._released:
            return
        self._released = True
        self._stop_event.set()
        thread, self._thread = self._thread, None
        if (
            thread is not None
            and thread.is_alive()
            and thread is not threading.current_thread()
        ):
            thread.join(timeout=5)
        try:
            self._manager.release(self._gpu_ids)
        except Exception:
            logger.exception("释放 GPU %s 失败", self._gpu_ids)


class GPUManager:
    """Thin client for the GPU resource management service."""

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        device_count: int = 1,
        heartbeat_interval: float = DEFAULT_HEARTBEAT_INTERVAL,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.device_count = max(1, int(device_count))
        self.heartbeat_interval = float(heartbeat_interval)
        self.timeout = float(timeout)

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        """
        POST a JSON payload to the management service.

        Parameters
        ----------
        path : str
            The endpoint path, e.g. ``/v1/gpu/acquire``.
        payload : dict[str, Any]
            The JSON body to send.

        Returns
        -------
        dict[str, Any]
            The decoded JSON response.

        Raises
        ------
        GPURequestError
            If the service is unreachable or answers with invalid JSON.

        """
        url = f"{self.base_url}{path}"
        try:
            response = requests.post(url, json=payload, timeout=self.timeout)
        except requests.RequestException as exc:
            raise GPURequestError(f"请求 GPU 管理服务失败: {exc}") from exc
        try:
            body = response.json()
        except ValueError as exc:
            raise GPURequestError(
                f"GPU 管理服务返回非 JSON 响应 (HTTP {response.status_code})",
            ) from exc
        if not isinstance(body, dict):
            raise GPURequestError(f"GPU 管理服务返回异常数据: {body!r}")
        return body

    def acquire(self) -> GPULease | None:
        """
        Lease GPUs from the management service.

        Returns
        -------
        GPULease | None
            The lease, or None when no GPU resource is currently free.

        Raises
        ------
        GPURequestError
            If the service is unreachable or returns an unexpected error.

        """
        body = self._post("/v1/gpu/acquire", {"count": self.device_count})
        code = body.get("code")
        msg = body.get("msg", "")
        if code == CODE_NO_RESOURCE:
            logger.warning("没有可用 GPU 资源: %s", msg)
            return None
        if code != CODE_SUCCESS:
            raise GPURequestError(f"申请 GPU 失败 (code={code}): {msg}")
        gpu_ids = _parse_gpu_ids(body.get("data"))
        if not gpu_ids:
            logger.warning("申请 GPU 返回空的资源列表: %s", body)
            return None
        logger.info("已申请 GPU: %s", gpu_ids)
        return GPULease(self, gpu_ids, self.heartbeat_interval)

    def release(self, gpu_ids: list[int]) -> bool:
        """
        Return GPUs to the management service.

        Parameters
        ----------
        gpu_ids : list[int]
            The GPU ids to release.

        Returns
        -------
        bool
            True if the service acknowledged the release.

        """
        if not gpu_ids:
            return True
        body = self._post("/v1/gpu/release", {"gpu_ids": list(gpu_ids)})
        ok = body.get("code") == CODE_SUCCESS
        if ok:
            logger.info("已释放 GPU: %s", gpu_ids)
        else:
            logger.error("释放 GPU %s 失败: %s", gpu_ids, body)
        return ok

    def heartbeat(self, gpu_ids: list[int]) -> bool:
        """
        Send a heartbeat for GPUs that are still in use.

        Parameters
        ----------
        gpu_ids : list[int]
            The GPU ids to keep alive.

        Returns
        -------
        bool
            True if the service acknowledged the heartbeat.

        """
        if not gpu_ids:
            return True
        body = self._post("/v1/gpu/heartbeat", {"gpu_ids": list(gpu_ids)})
        ok = body.get("code") == CODE_SUCCESS
        if not ok:
            logger.error("GPU %s 心跳被拒绝: %s", gpu_ids, body)
        return ok
