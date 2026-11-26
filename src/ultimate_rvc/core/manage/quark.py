"""Quark cloud drive integration for importing RVC voice models.

Quark does not offer a public API, so this module uses the same private
endpoints that the web client itself calls. Authentication is done with the
user's own cookie, which is stored locally with restrictive permissions.
"""

from __future__ import annotations

import json
import os
import random
import re
import tempfile
import time
from pathlib import Path
from typing import Any, Callable

import requests

from ultimate_rvc.common import CONFIG_DIR
from ultimate_rvc.core.manage.models import upload_voice_model

BASE_URL = "https://drive-pc.quark.cn"
COMMON_PARAMS = {"pr": "ucpro", "fr": "pc", "uc_param_str": ""}
COOKIE_FILE = CONFIG_DIR / "quark_cookie.json"
# Quark returns code 23018 ("download file size limit") for large files when
# the request looks like a browser. The PC client UA is required.
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    " (KHTML, like Gecko) quark-cloud-drive/2.5.56 Chrome/100.0.4896.160"
    " Electron/18.3.5.12 Safari/537.36 Channel/pckk_other_ch"
)
MODEL_SUFFIXES = {".pth", ".index", ".zip"}
MODE_DRIVE = "我的网盘"
MODE_SHARE = "分享链接"
MODE_SEARCH = "关键词搜索"
TRANSFER_FOLDER = "URVC-Models"

ProgressCallback = Callable[[float, str], None] | None


class QuarkError(RuntimeError):
    """Raised when a Quark Drive operation fails."""


def normalize_cookie(raw: str) -> str:
    """Convert pasted cookie data into a ``name=value; ...`` header string.

    Accepts a raw header string as well as the JSON exported by browser
    extensions such as Cookie-Editor (a list of cookie objects).
    """
    raw = raw.strip()
    if not raw:
        return ""
    if raw.startswith("[") or raw.startswith("{"):
        try:
            data = json.loads(raw)
        except ValueError as exc:
            raise QuarkError("Cookie 看起来是 JSON，但解析失败，请重新复制。") from exc
        if isinstance(data, dict):
            data = [data]
        if not isinstance(data, list):
            raise QuarkError("Cookie JSON 格式不正确。")
        pairs = [
            f"{item['name']}={item['value']}"
            for item in data
            if isinstance(item, dict) and item.get("name")
        ]
        if not pairs:
            raise QuarkError("Cookie JSON 中没有有效的 name/value 项。")
        return "; ".join(pairs)
    return raw


def save_quark_cookie(cookie: str) -> str:
    """Persist the user's Quark cookie with 0600 permissions."""
    cookie = normalize_cookie(cookie)
    if not cookie:
        raise QuarkError("Cookie 不能为空。")
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    COOKIE_FILE.write_text(
        json.dumps({"cookie": cookie}, ensure_ascii=False),
        encoding="utf-8",
    )
    os.chmod(COOKIE_FILE, 0o600)
    return f"Cookie 已保存到 {COOKIE_FILE}（权限 600）。"


def load_quark_cookie() -> str:
    """Load the stored Quark cookie."""
    if not COOKIE_FILE.exists():
        raise QuarkError(
            "尚未配置夸克 Cookie，请先在「Settings → Quark Drive」中保存。",
        )
    try:
        data = json.loads(COOKIE_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise QuarkError(f"读取 Cookie 文件失败: {exc}") from exc
    cookie = normalize_cookie(str(data.get("cookie") or ""))
    if not cookie:
        raise QuarkError("Cookie 文件为空，请重新保存。")
    return cookie


def clear_quark_cookie() -> str:
    """Remove the stored Quark cookie."""
    if COOKIE_FILE.exists():
        COOKIE_FILE.unlink()
    return "已清除本地保存的夸克 Cookie。"


def has_quark_cookie() -> bool:
    """Return whether a non-empty cookie has been configured."""
    try:
        load_quark_cookie()
    except QuarkError:
        return False
    return True


def _parse_share_url(url: str) -> tuple[str, str]:
    match = re.search(r"/s/([A-Za-z0-9]+)", url)
    if not match:
        raise QuarkError("分享链接格式不正确，应为 https://pan.quark.cn/s/xxxxx")
    passcode_match = re.search(r"[?&]pwd=([A-Za-z0-9]+)", url)
    return match.group(1), passcode_match.group(1) if passcode_match else ""


def _find_download_url(obj: Any) -> str | None:
    url_keys = {"download_url", "downloadurl", "download_link", "url", "link"}
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key.lower() in url_keys and isinstance(value, str) and value.startswith("http"):
                return value
        for value in obj.values():
            found = _find_download_url(value)
            if found:
                return found
    elif isinstance(obj, list):
        for item in obj:
            found = _find_download_url(item)
            if found:
                return found
    return None


class QuarkClient:
    """Small client for the private Quark Drive web API."""

    def __init__(self, cookie: str | None = None) -> None:
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": USER_AGENT,
                "Cookie": cookie or load_quark_cookie(),
                "Referer": "https://pan.quark.cn/",
            },
        )

    def _request(
        self,
        method: str,
        path: str,
        params: dict[str, Any] | None = None,
        body: dict[str, Any] | None = None,
        timeout: int = 30,
    ) -> tuple[dict[str, Any], requests.Response]:
        try:
            response = self.session.request(
                method,
                f"{BASE_URL}{path}",
                params={**COMMON_PARAMS, **(params or {})},
                json=body,
                timeout=timeout,
            )
        except requests.RequestException as exc:
            raise QuarkError(f"请求夸克网盘失败: {exc}") from exc
        try:
            payload = response.json()
        except ValueError as exc:
            raise QuarkError(
                f"夸克网盘返回了非 JSON 响应 (HTTP {response.status_code})",
            ) from exc
        code = payload.get("code", payload.get("status"))
        if str(code) not in ("0", "200"):
            raise QuarkError(
                str(
                    payload.get("message")
                    or payload.get("msg")
                    or f"夸克接口错误 (code={code})",
                ),
            )
        return payload, response

    def _get(self, path: str, params: dict[str, Any] | None = None, **kwargs: Any):
        return self._request("GET", path, params=params, **kwargs)

    def _post(
        self,
        path: str,
        params: dict[str, Any] | None = None,
        body: dict[str, Any] | None = None,
        **kwargs: Any,
    ):
        return self._request("POST", path, params=params, body=body, **kwargs)

    @staticmethod
    def _items(payload: dict[str, Any]) -> list[dict[str, Any]]:
        return list((payload.get("data") or {}).get("list") or [])

    def list_directory(self, fid: str = "0") -> list[dict[str, Any]]:
        """List all entries of a directory in the user's own drive."""
        items: list[dict[str, Any]] = []
        page = 1
        while True:
            payload, _ = self._get(
                "/1/clouddrive/file/sort",
                {
                    "pdir_fid": fid,
                    "_page": page,
                    "_size": 50,
                    "_fetch_total": 1,
                    "_fetch_sub_dirs": 0,
                    "_sort": "file_type:asc,updated_at:desc",
                    "_fetch_full_path": 0,
                    "fetch_all_file": 1,
                    "fetch_risk_file_name": 1,
                },
            )
            batch = self._items(payload)
            items.extend(batch)
            total = int((payload.get("metadata") or {}).get("_total") or len(items))
            if len(batch) < 50 or len(items) >= total:
                break
            page += 1
        return items

    def get_file_info(self, fid: str) -> dict[str, Any]:
        """Get metadata of one file or folder, including its parent fid."""
        payload, _ = self._get("/1/clouddrive/file/info", {"fid": fid})
        return dict(payload.get("data") or {})

    def _path_for_item(
        self,
        item: dict[str, Any],
        cache: dict[str, dict[str, Any]],
    ) -> str:
        parts = [str(item.get("file_name") or "")]
        fid = str(item.get("pdir_fid") or "0")
        for _ in range(20):
            if not fid or fid == "0":
                break
            info = cache.get(fid)
            if info is None:
                info = self.get_file_info(fid)
                cache[fid] = info
            name = info.get("file_name")
            if not name:
                break
            parts.append(str(name))
            parent = str(info.get("pdir_fid") or "0")
            if parent == fid:
                break
            fid = parent
        return "/" + "/".join(reversed(parts))

    def add_full_paths(self, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Attach a ``_path`` field containing the full drive path."""
        cache: dict[str, dict[str, Any]] = {"0": {"file_name": ""}}
        for item in items:
            try:
                item["_path"] = self._path_for_item(item, cache)
            except QuarkError:
                item["_path"] = "/" + str(item.get("file_name") or "")
        return items

    def search_files(self, keyword: str) -> list[dict[str, Any]]:
        """Search the whole drive for files matching a keyword."""
        keyword = keyword.strip()
        if not keyword:
            raise QuarkError("请输入搜索关键词。")
        items: list[dict[str, Any]] = []
        page = 1
        while page <= 20:
            payload, _ = self._get(
                "/1/clouddrive/file/search",
                {"q": keyword, "_page": page, "_size": 50},
            )
            batch = self._items(payload)
            items.extend(batch)
            if len(batch) < 50:
                break
            page += 1
        return items

    def get_stoken(self, share_url: str, passcode: str = "") -> tuple[str, str]:
        """Resolve a share URL into (pwd_id, stoken)."""
        pwd_id, url_passcode = _parse_share_url(share_url)
        try:
            payload, _ = self._post(
                "/1/clouddrive/share/sharepage/token",
                {"__t": time.time(), "__dt": 1},
                {"pwd_id": pwd_id, "passcode": passcode.strip() or url_passcode},
            )
        except QuarkError as exc:
            message = str(exc)
            if "41006" in message or "分享不存在" in message:
                raise QuarkError(
                    "分享链接无效或已失效，请检查链接是否正确；"
                    "若链接带提取码，请填到「提取码」输入框。",
                ) from exc
            if "41008" in message or "提取码" in message:
                raise QuarkError("该分享需要提取码，请填写后再试。") from exc
            raise
        stoken = str((payload.get("data") or {}).get("stoken") or "")
        if not stoken:
            raise QuarkError("获取分享 token 失败，请检查分享链接和提取码。")
        return pwd_id, stoken

    def list_share(
        self,
        share_url: str,
        passcode: str = "",
        fid: str = "0",
    ) -> list[dict[str, Any]]:
        """List entries inside a share (or one of its folders)."""
        pwd_id, stoken = self.get_stoken(share_url, passcode)
        items: list[dict[str, Any]] = []
        page = 1
        while True:
            payload, _ = self._get(
                "/1/clouddrive/share/sharepage/detail",
                {
                    "pwd_id": pwd_id,
                    "stoken": stoken,
                    "pdir_fid": fid,
                    "_page": page,
                    "_size": 50,
                    "_fetch_total": 1,
                    "_fetch_sub_dirs": 0,
                    "_sort": "file_type:asc,updated_at:desc",
                    "_fetch_full_path": 0,
                    "fetch_all_file": 1,
                    "fetch_risk_file_name": 1,
                },
            )
            batch = self._items(payload)
            items.extend(batch)
            total = int((payload.get("metadata") or {}).get("_total") or len(items))
            if len(batch) < 50 or len(items) >= total:
                break
            page += 1
        return items

    def save_share_items(
        self,
        pwd_id: str,
        stoken: str,
        items: list[dict[str, Any]],
        to_fid: str = "0",
    ) -> list[str]:
        """Transfer selected share items into the user's own drive."""
        payload, _ = self._post(
            "/1/clouddrive/share/sharepage/save",
            {
                "app": "clouddrive",
                "__dt": int(random.uniform(1, 5) * 60 * 1000),
                "__t": time.time(),
            },
            {
                "fid_list": [str(item["fid"]) for item in items],
                "fid_token_list": [str(item.get("fid_token") or "") for item in items],
                "to_pdir_fid": to_fid,
                "pwd_id": pwd_id,
                "stoken": stoken,
                "pdir_fid": "0",
                "scene": "link",
            },
        )
        task_id = str((payload.get("data") or {}).get("task_id") or "")
        if not task_id:
            raise QuarkError("转存失败：未获得任务 ID。")
        for _ in range(60):
            task_payload, _ = self._get(
                "/1/clouddrive/task",
                {"task_id": task_id, "retry_index": 0},
            )
            data = task_payload.get("data") or {}
            if data.get("status") == 2:
                return [str(fid) for fid in (data.get("save_as") or {}).get("save_as_top_fids") or []]
            if data.get("status") not in (0, 1):
                raise QuarkError(f"转存任务失败: {data.get('message') or data}")
            time.sleep(1)
        raise QuarkError("转存任务超时。")

    def get_download_url(self, fid: str) -> str:
        """Request a direct download URL for one file."""
        payload, _ = self._post(
            "/1/clouddrive/file/download",
            body={"fids": [str(fid)]},
        )
        url = _find_download_url(payload)
        if not url:
            raise QuarkError(f"未能从夸克响应中获取下载直链: {payload}")
        return url

    def download_file(
        self,
        fid: str,
        destination: Path,
        progress: ProgressCallback = None,
        label: str = "",
    ) -> Path:
        """Stream one file to ``destination``."""
        url = self.get_download_url(fid)
        try:
            with self.session.get(url, stream=True, timeout=120) as response:
                if response.status_code not in (200, 206):
                    raise QuarkError(f"下载失败，HTTP {response.status_code}")
                total = int(response.headers.get("Content-Length") or 0)
                downloaded = 0
                with destination.open("wb") as output:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if not chunk:
                            continue
                        output.write(chunk)
                        downloaded += len(chunk)
                        if progress and total:
                            progress(
                                min(0.99, downloaded / total),
                                desc=f"下载 {label or destination.name}",
                            )
        except requests.RequestException as exc:
            raise QuarkError(f"下载文件失败: {exc}") from exc
        return destination


def _ensure_transfer_folder(client: QuarkClient) -> str:
    """Return the fid of a dedicated folder for transferred share items."""
    for item in client.list_directory("0"):
        if item.get("dir") and item.get("file_name") == TRANSFER_FOLDER:
            return str(item["fid"])
    payload, _ = client._post(
        "/1/clouddrive/file",
        body={
            "pdir_fid": "0",
            "file_name": "",
            "dir_path": TRANSFER_FOLDER,
            "dir_init_lock": False,
        },
    )
    fid = str((payload.get("data") or {}).get("fid") or "")
    if not fid:
        raise QuarkError(f"创建转存文件夹「{TRANSFER_FOLDER}」失败。")
    return fid


def list_quark_items(
    mode: str,
    location: str,
    passcode: str,
    fid: str = "0",
) -> list[dict[str, Any]]:
    """List entries either in the user's drive or inside a share."""
    client = QuarkClient()
    if mode == MODE_SHARE:
        if not location.strip():
            raise QuarkError("请输入夸克分享链接。")
        items = client.list_share(location.strip(), passcode.strip(), fid)
    elif mode == MODE_SEARCH and (not fid or fid == "0"):
        items = client.search_files(location)
    else:
        items = client.list_directory(fid or "0")
    return client.add_full_paths(items)


def _collect_model_files(
    client: QuarkClient,
    fids: list[str],
    source_items: list[dict[str, Any]] | None = None,
    depth: int = 0,
) -> list[tuple[str, str]]:
    """Resolve folders and keep only model files, up to a limited depth."""
    if depth > 3:
        return []
    result: list[tuple[str, str]] = []
    for index, fid in enumerate(fids):
        source = source_items[index] if source_items and index < len(source_items) else {}
        if source.get("dir"):
            children = client.list_directory(str(fid))
            result.extend(
                _collect_model_files(
                    client,
                    [str(child["fid"]) for child in children],
                    children,
                    depth + 1,
                ),
            )
            continue
        name = str(source.get("file_name") or "")
        if Path(name).suffix.lower() in MODEL_SUFFIXES:
            result.append((str(fid), name))
    return result


def import_quark_models(
    mode: str,
    location: str,
    passcode: str,
    selected_items: list[dict[str, Any]],
    model_name: str,
    progress: ProgressCallback = None,
) -> str:
    """Download selected Quark files and install them as a voice model."""
    model_name = model_name.strip()
    if not model_name:
        raise QuarkError("请输入模型名称。")
    if not selected_items:
        raise QuarkError("请先在文件列表中选择文件。")

    client = QuarkClient()
    if mode == MODE_SHARE:
        pwd_id, stoken = client.get_stoken(location.strip(), passcode.strip())
        to_fid = _ensure_transfer_folder(client)
        saved_fids = client.save_share_items(pwd_id, stoken, selected_items, to_fid)
        candidates = _collect_model_files(client, saved_fids, selected_items)
    else:
        candidates = [
            (str(item["fid"]), str(item.get("file_name") or ""))
            for item in selected_items
            if Path(str(item.get("file_name") or "")).suffix.lower() in MODEL_SUFFIXES
        ]

    if not candidates:
        raise QuarkError("所选内容中没有找到 .pth / .index / .zip 模型文件。")

    with tempfile.TemporaryDirectory(prefix="urvc_quark_") as temp_dir:
        files: list[Path] = []
        for index, (fid, name) in enumerate(candidates):
            destination = Path(temp_dir) / (name or f"model_{index}.pth")
            client.download_file(fid, destination, progress, name)
            files.append(destination)
        upload_voice_model(files, model_name)

    return f"已从夸克网盘导入模型「{model_name}」。"
