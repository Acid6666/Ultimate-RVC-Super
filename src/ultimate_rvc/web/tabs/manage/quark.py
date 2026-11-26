"""Gradio tab for importing voice models from Quark Drive."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from functools import partial

import gradio as gr

from ultimate_rvc.core.manage.models import get_voice_model_names
from ultimate_rvc.core.manage.quark import (
    MODE_DRIVE,
    MODE_SEARCH,
    MODE_SHARE,
    QuarkError,
    has_quark_cookie,
    import_quark_models,
    list_quark_items,
)
from ultimate_rvc.web.common import exception_harness, update_dropdowns

if TYPE_CHECKING:
    from ultimate_rvc.web.config.main import TotalConfig


def _item_label(item: dict[str, Any]) -> str:
    path = item.get("_path") or "/" + str(item.get("file_name") or "")
    if item.get("dir"):
        return f"📁 {path}"
    size_mb = float(item.get("size") or 0) / 1024 / 1024
    return f"📄 {path} ({size_mb:.1f} MB)"


def _list_items(
    mode: str,
    location: str,
    passcode: str,
    fid: str,
    history: list[str],
) -> tuple[Any, dict[str, Any], str, list[str], str]:
    items = list_quark_items(mode, location, passcode, fid)
    labels: list[str] = []
    seen: dict[str, int] = {}
    for item in items:
        base = _item_label(item)
        seen[base] = seen.get(base, 0) + 1
        labels.append(base if seen[base] == 1 else f"{base} #{seen[base]}")
    mapping = {label: item for label, item in zip(labels, items)}
    message = f"找到 {len(items)} 项。"
    if mode == MODE_SEARCH and len(items) >= 1000:
        message += " 结果较多仅显示前 1000 项，请细化关键词。"
    return (
        gr.Dropdown(choices=labels, value=[]),
        mapping,
        fid,
        history,
        message,
    )


def _enter_folder(
    mode: str,
    location: str,
    passcode: str,
    selected: list[str],
    mapping: dict[str, Any],
    fid: str,
    history: list[str],
) -> tuple[Any, dict[str, Any], str, list[str], str]:
    item = (mapping or {}).get(selected[0]) if selected else None
    if not item or not item.get("dir"):
        raise gr.Error("请先选中一个文件夹。")
    return _list_items(
        mode,
        location,
        passcode,
        str(item["fid"]),
        [*history, fid],
    )


def _go_up(
    mode: str,
    location: str,
    passcode: str,
    history: list[str],
) -> tuple[Any, dict[str, Any], str, list[str], str]:
    if not history:
        raise gr.Error("已经在根目录了。")
    return _list_items(mode, location, passcode, history[-1], history[:-1])


def _download(
    mode: str,
    location: str,
    passcode: str,
    selected: list[str],
    mapping: dict[str, Any],
    model_name: str,
    progress: gr.Progress = gr.Progress(),
) -> str:
    items = [(mapping or {}).get(label) for label in selected]
    items = [item for item in items if item is not None]
    return import_quark_models(mode, location, passcode, items, model_name, progress)


def _refresh_status() -> str:
    return "已配置" if has_quark_cookie() else "未配置，请前往 Settings 保存"


def render(total_config: TotalConfig) -> gr.Textbox:
    """Render the Quark Drive sub-tab and return its cookie status box."""
    tab_config = total_config.management.model
    configured = has_quark_cookie()
    with gr.Tab("Quark Drive"):
        gr.Markdown(
            "从夸克网盘直接导入 RVC 模型。请先在 **Settings → Quark Drive** "
            "中保存你的 Cookie。默认按**关键词全盘搜索**，勾选结果中的文件即可下载为模型；"
            "也支持浏览自己的网盘或粘贴分享链接。",
        )
        gr.Markdown(
            "Cookie 仅保存在本地 `config/quark_cookie.json`（权限 600），"
            "不会上传到任何第三方服务。",
        )
        with gr.Row():
            cookie_status = gr.Textbox(
                label="Cookie 状态",
                value="已配置" if configured else "未配置，请前往 Settings 保存",
                interactive=False,
                scale=3,
            )
            refresh_btn = gr.Button("刷新状态", scale=1)
        with gr.Row():
            mode = gr.Radio(
                [MODE_SEARCH, MODE_SHARE, MODE_DRIVE],
                value=MODE_SEARCH,
                label="来源",
            )
            passcode = gr.Textbox(
                label="提取码（分享链接可选）",
                placeholder="如 4g6h",
                max_lines=1,
            )
        location = gr.Textbox(
            label="搜索关键词 / 分享链接",
            placeholder="例如：模型、RVC、xxx（搜索模式下填关键词）",
        )
        with gr.Row():
            current_fid = gr.Textbox(
                label="当前目录 ID",
                value="0",
                interactive=False,
            )
            list_btn = gr.Button("列出文件")
            enter_btn = gr.Button("进入文件夹")
            up_btn = gr.Button("返回上级")
        file_dropdown = gr.Dropdown(
            label="选择文件或文件夹（可多选文件）",
            choices=[],
            multiselect=True,
            interactive=True,
        )
        items_state = gr.State({})
        history_state = gr.State([])
        msg = gr.Textbox(label="结果", interactive=False)
        with gr.Row():
            model_name = gr.Textbox(
                label="模型名称",
                placeholder="例如：我的模型",
                max_lines=1,
                scale=3,
            )
            download_btn = gr.Button("下载为模型", variant="primary", scale=1)

        refresh_btn.click(
            _refresh_status,
            outputs=cookie_status,
            api_name="quark_refresh_status",
        )

        list_outputs = [
            file_dropdown,
            items_state,
            current_fid,
            history_state,
            msg,
        ]
        list_inputs = [mode, location, passcode, current_fid, history_state]

        list_btn.click(
            exception_harness(_list_items),
            inputs=list_inputs,
            outputs=list_outputs,
            api_name="quark_list_items",
        )
        enter_btn.click(
            exception_harness(_enter_folder),
            inputs=[mode, location, passcode, file_dropdown, items_state, current_fid, history_state],
            outputs=list_outputs,
            api_name="quark_enter_folder",
        )
        up_btn.click(
            exception_harness(_go_up),
            inputs=[mode, location, passcode, history_state],
            outputs=list_outputs,
            api_name="quark_go_up",
        )
        download_click = download_btn.click(
            exception_harness(_download),
            inputs=[mode, location, passcode, file_dropdown, items_state, model_name],
            outputs=msg,
            api_name="quark_download_model",
        )
        download_click.success(
            partial(update_dropdowns, get_voice_model_names, 5, [], [4]),
            outputs=[
                total_config.song.one_click.voice_model.instance,
                total_config.song.multi_step.voice_model.instance,
                total_config.speech.one_click.voice_model.instance,
                total_config.speech.multi_step.voice_model.instance,
                tab_config.voices.instance,
            ],
            api_name="quark_refresh_models",
            show_progress="hidden",
        )

    return cookie_status
