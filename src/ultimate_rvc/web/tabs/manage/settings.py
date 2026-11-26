"""Module which defines the code for the "Settings" tab."""

from __future__ import annotations

from typing import TYPE_CHECKING

from functools import partial

import gradio as gr

from ultimate_rvc.core.manage.config import (
    delete_all_configs,
    delete_configs,
    get_config_names,
)
from ultimate_rvc.core.manage.quark import (
    clear_quark_cookie,
    has_quark_cookie,
    save_quark_cookie,
)
from ultimate_rvc.core.manage.settings import delete_temp_files
from ultimate_rvc.web.common import (
    confirm_box_js,
    confirmation_harness,
    exception_harness,
    load_total_config_values,
    render_msg,
    save_total_config_values,
    setup_delete_event,
    update_dropdowns,
)

if TYPE_CHECKING:
    from ultimate_rvc.web.config.main import SettingsManagementConfig, TotalConfig


def render(total_config: TotalConfig, quark_status: gr.Textbox | None = None) -> None:
    """
    Render "Settings" tab.

    Parameters
    ----------
    total_config : TotalConfig
        The total configuration object containing all the settings.

    """
    tab_config = total_config.management.settings
    tab_config.dummy_checkbox.instantiate()
    _render_config_files_tab(tab_config, total_config)
    _render_temp_files_tab(tab_config)
    _render_quark_cookie_tab(quark_status)


def _render_config_files_tab(
    tab_config: SettingsManagementConfig,
    total_config: TotalConfig,
) -> None:
    components = [config.instance for config in total_config.all]
    with gr.Tab("Configuration files"):
        with gr.Accordion("Save configuration"), gr.Row():
            with gr.Column():
                save_config_name = gr.Textbox(
                    label="Configuration name",
                    info=(
                        "The name of the configuration to save the current UI"
                        " settings to."
                    ),
                    placeholder="Enter a name for the configuration",
                    max_lines=1,
                    value="Default Configuration",
                )
                save_config_btn = gr.Button("Save", variant="primary")
            with gr.Column():
                save_config_msg = gr.Textbox(label="Output message", interactive=False)

        with gr.Accordion("Load configuration", open=False), gr.Row():
            with gr.Column():
                tab_config.load_config_name.instance.render()
                load_config_btn = gr.Button("Load", variant="primary")
            with gr.Column():
                load_config_msg = gr.Textbox(label="Output message", interactive=False)

        save_config_btn.click(
            exception_harness(save_total_config_values),
            inputs=[save_config_name, *components],
            outputs=save_config_msg,
        ).success(
            partial(render_msg, "[-] Successfully saved configuration!"),
            outputs=save_config_msg,
            show_progress="hidden",
        ).then(
            partial(update_dropdowns, get_config_names, 1, value_indices=[0]),
            inputs=save_config_name,
            outputs=tab_config.load_config_name.instance,
            show_progress="hidden",
        ).then(
            partial(update_dropdowns, get_config_names, 1, [], [0]),
            outputs=tab_config.delete_config_names.instance,
            show_progress="hidden",
        )

        load_config_btn.click(
            exception_harness(load_total_config_values),
            inputs=tab_config.load_config_name.instance,
            outputs=components,
            show_progress_on=load_config_msg,
        ).success(
            partial(render_msg, "[-] Successfully loaded configuration!"),
            outputs=load_config_msg,
            show_progress="hidden",
        )

        with gr.Accordion("Delete configuration(s)", open=False), gr.Row():
            with gr.Column():
                tab_config.delete_config_names.instance.render()
                delete_config_btn = gr.Button("Delete selected", variant="secondary")
                delete_all_config_btn = gr.Button("Delete all", variant="primary")
            with gr.Column():
                delete_config_msg = gr.Textbox(
                    label="Output message",
                    interactive=False,
                )
        delete_config_click = setup_delete_event(
            delete_config_btn,
            delete_configs,
            [
                tab_config.dummy_checkbox.instance,
                tab_config.delete_config_names.instance,
            ],
            delete_config_msg,
            "Are you sure you want to delete the configurations with the selected"
            " names?",
            "[-] Successfully deleted the configurations with the selected names!",
        )
        delete_all_config_click = setup_delete_event(
            delete_all_config_btn,
            delete_all_configs,
            [tab_config.dummy_checkbox.instance],
            delete_config_msg,
            "Are you sure you want to delete all configuration?",
            "[-] Successfully deleted all configuration!",
        )
        for event in [delete_config_click, delete_all_config_click]:
            event.success(
                partial(update_dropdowns, get_config_names, 2, [], [1]),
                outputs=[
                    tab_config.load_config_name.instance,
                    tab_config.delete_config_names.instance,
                ],
                show_progress="hidden",
            )


def _render_temp_files_tab(tab_config: SettingsManagementConfig) -> None:
    with gr.Tab("Temporary files"):
        gr.Markdown("")
        with gr.Row(equal_height=True):
            temporary_files_btn = gr.Button("Delete all", variant="primary")
            temporary_files_msg = gr.Textbox(label="Output message", interactive=False)

        temporary_files_btn.click(
            confirmation_harness(delete_temp_files),
            inputs=tab_config.dummy_checkbox.instance,
            outputs=temporary_files_msg,
            js=confirm_box_js(
                "Are you sure you want to delete all temporary files? Any files"
                " uploaded directly via the UI will not be available for further"
                " processing until they are re-uploaded.",
            ),
        ).success(
            partial(
                render_msg,
                "[-] Successfully deleted all temporary files!",
            ),
            outputs=temporary_files_msg,
            show_progress="hidden",
        )


def _render_quark_cookie_tab(quark_status: gr.Textbox | None = None) -> None:
    """Render the Quark Drive cookie configuration section."""
    configured = has_quark_cookie()
    with gr.Accordion("Quark Drive", open=False):
        gr.Markdown(
            "获取方式：浏览器登录 `pan.quark.cn` → F12 打开开发者工具 → "
            "Network → 刷新页面 → 点任意发往 `drive.quark.cn` 的请求 → "
            "复制请求头中的完整 `Cookie` 并粘贴到下方。\n\n"
            "Cookie 仅保存在本地 `config/quark_cookie.json`（权限 600），"
            "过期后重新粘贴即可。",
        )
        status = gr.Textbox(
            label="状态",
            value="已配置" if configured else "未配置",
            interactive=False,
        )
        cookie_box = gr.Textbox(
            label="Cookie",
            type="password",
            placeholder="已配置时留空表示不修改" if configured else "粘贴完整 Cookie",
        )
        with gr.Row():
            save_btn = gr.Button("保存 Cookie", variant="primary")
            clear_btn = gr.Button("清除 Cookie", variant="secondary")
        msg = gr.Textbox(label="输出信息", interactive=False)

        save_outputs = [msg, status, cookie_box]
        clear_outputs = [msg, status, cookie_box]
        if quark_status is not None:
            save_outputs.append(quark_status)
            clear_outputs.append(quark_status)

        def _save(cookie: str) -> tuple[str, ...]:
            message = save_quark_cookie(cookie)
            result = (message, "已配置", "")
            return (*result, "已配置") if quark_status is not None else result

        def _clear() -> tuple[str, ...]:
            message = clear_quark_cookie()
            result = (message, "未配置", "")
            return (*result, "未配置") if quark_status is not None else result

        save_btn.click(
            exception_harness(_save),
            inputs=cookie_box,
            outputs=save_outputs,
            api_name="quark_save_cookie",
        )
        clear_btn.click(
            exception_harness(_clear),
            outputs=clear_outputs,
            api_name="quark_clear_cookie",
        )
