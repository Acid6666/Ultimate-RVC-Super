"""
On-demand lifecycle management for the Gradio backend and its GPU lease.

The FastAPI gateway is the only permanently running service. The Gradio
app (which is what actually allocates VRAM) is started lazily when the
first request arrives and is stopped again once the service has been
idle for ``URVC_IDLE_TIMEOUT`` seconds.

Lifecycle:

1. request arrives -> lease GPU(s) from the GPU management service
2. start the Gradio subprocess with ``CUDA_VISIBLE_DEVICES=<lease>``
3. proxy the request, refreshing the activity timestamp
4. idle for ``URVC_IDLE_TIMEOUT`` -> kill the subprocess (VRAM returns
   to zero) and release the GPU lease
"""

from __future__ import annotations

from typing import Any

import atexit
import logging
import os
import shlex
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import requests

from gpu_client import (
    DEFAULT_BASE_URL,
    DEFAULT_HEARTBEAT_INTERVAL,
    GPULease,
    GPUManager,
    GPURequestError,
    GPUUnavailableError,
)

logger = logging.getLogger("urvc.pool")

PROJECT_ROOT = Path(__file__).resolve().parent
LOG_DIR = PROJECT_ROOT / "logs"

DEFAULT_GRADIO_HOST = "127.0.0.1"
DEFAULT_GRADIO_PORT = 7865
DEFAULT_IDLE_TIMEOUT = 600.0
DEFAULT_STARTUP_TIMEOUT = 300.0
DEFAULT_CONTROL_PORT = 7867
REAP_INTERVAL = 10.0


class ServiceUnavailableError(RuntimeError):
    """The Gradio backend could not be started or reached."""


def _env_flag(name: str, default: bool) -> bool:
    """Read a boolean environment variable."""
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def _env_float(name: str, default: float) -> float:
    """Read a float environment variable."""
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError:
        logger.warning("环境变量 %s=%r 无效，使用默认值 %s", name, raw, default)
        return default


def _env_int(name: str, default: int) -> int:
    """Read an integer environment variable."""
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        logger.warning("环境变量 %s=%r 无效，使用默认值 %s", name, raw, default)
        return default


class GradioPool:
    """Owns the GPU lease and the Gradio subprocess behind it."""

    def __init__(self) -> None:
        self.managed = _env_flag("URVC_GRADIO_MANAGED", True)
        # 常驻模式：Gradio 进程一直在线（页面随时可访问），空闲时只卸载模型
        # 并把 GPU 名额归还给管理服务；按需模式：空闲时直接停掉进程。
        self.persistent = _env_flag("URVC_GRADIO_PERSISTENT", True)
        self.host = os.environ.get("URVC_GRADIO_HOST", DEFAULT_GRADIO_HOST)
        self.port = _env_int("URVC_GRADIO_PORT", DEFAULT_GRADIO_PORT)
        self.idle_timeout = _env_float("URVC_IDLE_TIMEOUT", DEFAULT_IDLE_TIMEOUT)
        # Gradio 进程内的本地控制服务端口，用于触发模型卸载
        self.control_port = _env_int("URVC_CONTROL_PORT", DEFAULT_CONTROL_PORT)
        self.startup_timeout = _env_float(
            "URVC_GRADIO_STARTUP_TIMEOUT",
            DEFAULT_STARTUP_TIMEOUT,
        )
        self.base_url = f"http://{self.host}:{self.port}/"
        self._gpu = GPUManager(
            base_url=os.environ.get("URVC_GPU_MANAGER_URL", DEFAULT_BASE_URL),
            device_count=_env_int("URVC_GPU_COUNT", 1),
            heartbeat_interval=_env_float(
                "URVC_GPU_HEARTBEAT_INTERVAL",
                DEFAULT_HEARTBEAT_INTERVAL,
            ),
        )
        self._lock = threading.RLock()
        self._lease: GPULease | None = None
        self._proc: subprocess.Popen[bytes] | None = None
        self._log_file: Any = None
        self._client: Any = None
        self._active = 0
        self._last_active = 0.0
        self._stop_event = threading.Event()
        self._reaper: threading.Thread | None = None
        self._started = False
        # 常驻进程启动时绑定的显卡（CUDA_VISIBLE_DEVICES），进程存活期间不可变
        self._bound_gpu_ids: list[int] | None = None
        # 该进程自启动以来是否真正跑过推理（决定空闲时是否需要卸载模型）。
        # 未跑过推理时进程尚未初始化 CUDA，显存本就是 0，调用清理反而会
        # 触发 torch 导入并创建 CUDA 上下文，白白占用显存。
        self._models_loaded = False

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    @property
    def gpu_ids(self) -> list[int]:
        """The GPU ids currently leased by this pool."""
        with self._lock:
            return list(self._lease.gpu_ids) if self._lease else []

    def start(self) -> None:
        """Start the idle reaper thread (idempotent)."""
        with self._lock:
            if self._started:
                return
            self._started = True
            self._last_active = time.time()
        if not self.managed:
            logger.info("Gradio 生命周期管理已关闭，将直接连接 %s", self.base_url)
            return
        self._reaper = threading.Thread(
            target=self._reap_loop,
            name="gradio-idle-reaper",
            daemon=True,
        )
        self._reaper.start()
        if self.persistent:
            logger.info(
                "Gradio 常驻模式已就绪：空闲 %.0f 秒后卸载模型并归还 GPU（进程保持在线）",
                self.idle_timeout,
            )
        else:
            logger.info(
                "Gradio 按需启动已就绪：空闲 %.0f 秒后释放 GPU 并停止服务",
                self.idle_timeout,
            )

    def shutdown(self) -> None:
        """Stop the Gradio subprocess and release the GPU lease."""
        self._stop_event.set()
        with self._lock:
            self._teardown_locked(release_gpu=True)

    def ensure(self) -> Any:
        """
        Return a ready-to-use Gradio client, starting the backend first
        if it is not already running.

        Returns
        -------
        Any
            A connected ``gradio_client.Client`` instance.

        Raises
        ------
        ServiceUnavailableError
            If no GPU could be leased or the backend failed to start.

        """
        with self._lock:
            self._last_active = time.time()
            if not self.managed:
                if self._client is None:
                    self._client = self._new_client()
                return self._client
            if self._proc is not None and self._proc.poll() is not None:
                logger.warning(
                    "Gradio 进程已退出 (返回码 %s)，回收资源后重新启动",
                    self._proc.returncode,
                )
                self._teardown_locked(release_gpu=True)
            if self._proc is None:
                self._start_locked()
            elif self._lease is None:
                # 常驻模式：进程仍在线，但 GPU 名额已在空闲时归还，需要重新申请
                self._reacquire_locked()
            if self._client is None:
                self._client = self._new_client()
            return self._client

    @contextmanager
    def session(self, loads_models: bool = True) -> Iterator[Any]:
        """
        Context manager wrapping one unit of GPU-backed work.

        Parameters
        ----------
        loads_models : bool
            Whether this unit of work loads models onto the GPU. Only such
            work makes it necessary to unload models when going idle.

        Yields
        ------
        Any
            A connected ``gradio_client.Client`` instance. The backend is
            kept alive for as long as the session is open.

        """
        client = self.ensure()
        with self._lock:
            self._active += 1
            if loads_models:
                self._models_loaded = True
        try:
            yield client
        finally:
            with self._lock:
                self._active = max(0, self._active - 1)
                self._last_active = time.time()

    def status(self) -> dict[str, Any]:
        """Return the current state of the pool for observability."""
        with self._lock:
            idle = time.time() - self._last_active if self._last_active else 0.0
            return {
                "managed": self.managed,
                "persistent": self.persistent,
                "gradio_running": self._proc is not None
                and self._proc.poll() is None,
                "gradio_url": self.base_url,
                "gpu_ids": list(self._lease.gpu_ids) if self._lease else [],
                "bound_gpu_ids": list(self._bound_gpu_ids or []),
                "gpu_leased": self._lease is not None,
                "active_tasks": self._active,
                "idle_seconds": round(idle, 1),
                "idle_timeout": self.idle_timeout,
            }

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------
    def _new_client(self) -> Any:
        """Create a Gradio client for the running backend."""
        from gradio_client import Client  # noqa: PLC0415

        try:
            return Client(self.base_url)
        except Exception as exc:
            raise ServiceUnavailableError(
                f"无法连接 Gradio 服务 {self.base_url}: {exc}",
            ) from exc

    def _start_locked(self) -> None:
        """Lease a GPU and start the Gradio subprocess."""
        try:
            lease = self._gpu.acquire()
        except GPURequestError as exc:
            raise ServiceUnavailableError(str(exc)) from exc
        if lease is None:
            raise GPUUnavailableError("没有可用 GPU 资源")
        try:
            lease.start_heartbeat()
            self._lease = lease
            self._bound_gpu_ids = list(lease.gpu_ids)
            self._proc = self._spawn_locked(lease.gpu_ids)
            self._wait_ready_locked()
            logger.info(
                "Gradio 已就绪 (GPU %s, pid=%s, url=%s)",
                lease.gpu_ids,
                self._proc.pid,
                self.base_url,
            )
        except Exception:
            self._teardown_locked(release_gpu=True)
            raise

    def _reacquire_locked(self) -> None:
        """
        Re-lease GPUs for an already running (persistent) Gradio process.

        The devices a process may use are fixed at startup through
        ``CUDA_VISIBLE_DEVICES``. If the management service hands back a
        different card the process has to be restarted.
        """
        try:
            lease = self._gpu.acquire()
        except GPURequestError as exc:
            raise ServiceUnavailableError(str(exc)) from exc
        if lease is None:
            raise GPUUnavailableError("没有可用 GPU 资源")
        bound = self._bound_gpu_ids
        if bound and lease.gpu_ids != bound:
            logger.info(
                "重新申请到 GPU %s，与进程绑定的 %s 不一致，重启 Gradio 以切换显卡",
                lease.gpu_ids,
                bound,
            )
            lease.release()
            self._teardown_locked(release_gpu=False)
            self._start_locked()
            return
        lease.start_heartbeat()
        self._lease = lease
        logger.info("已重新申请 GPU: %s", lease.gpu_ids)

    def _build_command(self) -> list[str]:
        """Build the command used to start the Gradio app."""
        override = os.environ.get("URVC_GRADIO_CMD")
        if override:
            return shlex.split(override)
        python = os.environ.get("URVC_GRADIO_PYTHON") or sys.executable
        return [
            python,
            "src/ultimate_rvc/web/main.py",
            "--listen",
            "--listen-host",
            self.host,
            "--listen-port",
            str(self.port),
        ]

    def _spawn_locked(self, gpu_ids: list[int]) -> subprocess.Popen[bytes]:
        """Spawn the Gradio subprocess bound to the leased GPUs."""
        LOG_DIR.mkdir(exist_ok=True)
        log_file = Path(LOG_DIR / "web.log").open("ab")  # noqa: SIM115
        self._log_file = log_file
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = ",".join(str(gpu_id) for gpu_id in gpu_ids)
        # 空闲清理统一由网关控制：默认关闭 Gradio 自带的监控线程，否则长
        # 任务进行中可能被自动卸载模型而中断
        env["URVC_GRADIO_IDLE_TIMEOUT"] = os.environ.get(
            "URVC_GRADIO_IDLE_TIMEOUT",
            "0",
        )
        env["URVC_CONTROL_PORT"] = str(self.control_port)
        cmd = self._build_command()
        logger.info(
            "启动 Gradio: CUDA_VISIBLE_DEVICES=%s %s",
            env["CUDA_VISIBLE_DEVICES"],
            " ".join(cmd),
        )
        return subprocess.Popen(
            cmd,
            cwd=str(PROJECT_ROOT),
            env=env,
            stdout=log_file,
            stderr=log_file,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )

    def _wait_ready_locked(self) -> None:
        """Block until the Gradio backend answers its startup probe."""
        url = f"http://{self.host}:{self.port}/gradio_api/startup-events"
        deadline = time.time() + self.startup_timeout
        while time.time() < deadline:
            if self._proc is None or self._proc.poll() is not None:
                raise ServiceUnavailableError("Gradio 启动失败，请查看 logs/web.log")
            try:
                response = requests.get(url, timeout=5)
            except requests.RequestException:
                response = None
            if response is not None and response.ok:
                return
            time.sleep(2)
        raise ServiceUnavailableError(
            f"Gradio 启动超时 ({self.startup_timeout:.0f} 秒)",
        )

    def _close_log(self) -> None:
        """Close the log file used by the Gradio subprocess."""
        log_file, self._log_file = self._log_file, None
        if log_file is not None:
            try:
                log_file.close()
            except OSError:
                logger.exception("关闭 Gradio 日志文件失败")

    def _teardown_locked(self, release_gpu: bool = True) -> None:
        """Stop the Gradio subprocess and optionally free the GPU lease."""
        proc, self._proc = self._proc, None
        self._client = None
        if proc is not None:
            self._kill(proc)
            self._bound_gpu_ids = None
            self._models_loaded = False
        self._close_log()
        if release_gpu:
            lease, self._lease = self._lease, None
            if lease is not None:
                lease.release()

    def _kill(self, proc: subprocess.Popen[bytes]) -> None:
        """Terminate a subprocess and all of its children."""
        if proc.poll() is not None:
            return
        logger.info("停止 Gradio 进程 (pid=%s)", proc.pid)
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except OSError:
            proc.terminate()
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            logger.warning("Gradio 进程未响应 SIGTERM，强制结束 (pid=%s)", proc.pid)
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except OSError:
                proc.kill()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                logger.error("无法结束 Gradio 进程 (pid=%s)", proc.pid)

    def _reap_loop(self) -> None:
        """Periodically release resources once the service goes idle."""
        while not self._stop_event.wait(REAP_INTERVAL):
            try:
                self._reap_once()
            except Exception:
                logger.exception("空闲回收任务异常")
        logger.info("空闲回收线程已停止")

    def _reap_once(self) -> None:
        """Release GPU resources once the service has been idle too long."""
        with self._lock:
            if self._proc is None or self._lease is None:
                return
            if self._active > 0:
                return
            idle = time.time() - self._last_active
            if idle < self.idle_timeout:
                return
            if self.persistent:
                logger.info(
                    "已空闲 %.0f 秒 (阈值 %.0f 秒)，卸载模型并归还 GPU %s（Gradio 保持在线）",
                    idle,
                    self.idle_timeout,
                    self._lease.gpu_ids,
                )
                if self._models_loaded:
                    self._unload_models_locked()
                    self._models_loaded = False
                lease, self._lease = self._lease, None
                lease.release()
                return
            logger.info(
                "已空闲 %.0f 秒 (阈值 %.0f 秒)，停止 Gradio 并释放 GPU %s",
                idle,
                self.idle_timeout,
                self._lease.gpu_ids,
            )
            self._teardown_locked(release_gpu=True)

    def _unload_models_locked(self) -> None:
        """Ask the persistent Gradio process to drop its loaded models."""
        url = f"http://127.0.0.1:{self.control_port}/cleanup"
        try:
            response = requests.post(url, timeout=60)
        except requests.RequestException as exc:
            logger.warning("通知 Gradio 卸载模型失败: %s", exc)
            return
        if response.ok:
            logger.info("已通知 Gradio 卸载模型并释放显存")
        else:
            logger.warning(
                "Gradio 卸载模型返回异常: HTTP %s %s",
                response.status_code,
                response.text[:200],
            )


_pool: GradioPool | None = None
_pool_lock = threading.Lock()


def get_pool() -> GradioPool:
    """Return the process-wide Gradio pool."""
    global _pool
    if _pool is None:
        with _pool_lock:
            if _pool is None:
                _pool = GradioPool()
    return _pool


def shutdown_pool() -> None:
    """Release the GPU lease held by the process-wide pool, if any."""
    if _pool is not None:
        _pool.shutdown()


atexit.register(shutdown_pool)
