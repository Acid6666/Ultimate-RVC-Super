from typing import List

import logging
import os
import shutil
import subprocess
import tempfile
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.docs import (
    get_redoc_html,
    get_swagger_ui_html,
)
from fastapi.responses import JSONResponse, StreamingResponse
from gradio_client import handle_file

import requests

from pydantic import BaseModel, Field

from gpu_client import GPUUnavailableError
from gradio_pool import ServiceUnavailableError, get_pool, shutdown_pool

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("urvc.api")

# Gradio 后端按需拉起：有请求时向 GPU 管理服务申请显卡并启动，
# 空闲 URVC_IDLE_TIMEOUT 秒后停止进程并归还显卡。
pool = get_pool()


@asynccontextmanager
async def lifespan(_: FastAPI):
    pool.start()
    try:
        yield
    finally:
        shutdown_pool()


app = FastAPI(title="Voice Conversion Client API", docs_url=None, redoc_url=None, convert_url=None, lifespan=lifespan)
origins = ["*"]

# 创建下载文件存储目录
DOWNLOAD_DIR = Path("/tmp/voice_conversion_downloads")
DOWNLOAD_DIR.mkdir(exist_ok=True)


# 添加 CORS 中间件
app.add_middleware(
CORSMiddleware,
allow_origins=origins,
allow_credentials=True,
allow_methods=["*"],
allow_headers=["*"],
)


def get_valid_model_name(client, input_name: str) -> str:
    """
    解析复杂的 Gradio 返回结构，并智能匹配模型名称
    """
    try:
        # 1. 获取原始数据
        result = client.predict(api_name="/_init_dropdowns")
        raw_data = result[6] if len(result) > 6 else []

        # 2. 【关键修复】数据清洗：从复杂结构中提取纯净的模型名称列表
        # 目标是将 raw_data 清洗为 simple_choices = ['ldh_500', 'Eminem', ...]
        simple_choices = []

        if isinstance(raw_data, dict) and "choices" in raw_data:
            # 情况 A: 返回的是 {'choices': [['val', 'label'], ...]}
            # 我们只需要取每个子列表的第一个元素（value）
            for item in raw_data["choices"]:
                if isinstance(item, list) and len(item) > 0:
                    simple_choices.append(str(item[0]))
                else:
                    simple_choices.append(str(item))
        elif isinstance(raw_data, list):
            # 情况 B: 返回的是 [['val', 'label'], ...] 或 ['val', 'val']
            for item in raw_data:
                if isinstance(item, list) and len(item) > 0:
                    simple_choices.append(str(item[0]))
                else:
                    simple_choices.append(str(item))
        else:
            # 情况 C: 可能是其他未知结构，尝试强转
            print(f"⚠️ 未知模型列表结构: {type(raw_data)}")
            simple_choices = []

        # 3. 如果列表为空，报错
        if not simple_choices:
            raise ValueError(f"解析后的模型列表为空。原始数据类型: {type(raw_data)}")

        # print(f"🔍 解析到的可用模型: {simple_choices}") # 调试用

        # 4. 智能匹配逻辑
        input_name = str(input_name).strip()

        # (A) 精确匹配 (如 'ldh_500' == 'ldh_500')
        if input_name in simple_choices:
            return input_name

        # (B) 尝试去掉 .pth (防止用户传 'ldh_500.pth' 但列表里是 'ldh_500')
        if input_name.endswith(".pth"):
            name_no_pth = input_name[:-4]
            if name_no_pth in simple_choices:
                return name_no_pth

        # (C) 尝试加上 .pth (防止用户传 'ldh_500' 但列表里是 'ldh_500.pth')
        name_with_pth = f"{input_name}.pth"
        if name_with_pth in simple_choices:
            return name_with_pth

        # 匹配失败，抛出包含清洗后列表的异常
        raise ValueError(f"找不到模型 '{input_name}'。当前可用模型: {simple_choices}")

    except Exception as e:
        raise e


def download_file_from_url(url: str) -> str:
    """从URL下载文件并返回本地路径"""
    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()

        # 生成唯一文件名
        filename = f"{uuid.uuid4()}.wav"
        file_path = DOWNLOAD_DIR / filename

        with Path(file_path).open("wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

        return str(file_path)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"无法从URL下载文件: {e!s}")


def convert_to_mp3(input_path: str, output_path: str) -> str:
    """使用FFmpeg将音频文件转换为MP3格式"""
    try:
        cmd = [
            "ffmpeg", "-i", input_path,
            "-codec:a", "libmp3lame",
            "-q:a", "2",  # 高质量
            "-y",  # 覆盖输出文件
            output_path
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        return output_path
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"音频格式转换失败: {e.stderr.decode()}")


@app.get("/gpu/status", summary="查看 GPU 租约与 Gradio 后端状态")
def gpu_status():
    return pool.status()


@app.get("/models", summary="获取可用模型列表")
def get_models():
    try:
        # 仅刷新下拉框，不加载模型，空闲时无需触发模型卸载
        with pool.session(loads_models=False) as client:
            result = client.predict(api_name="/_init_dropdowns")
            return {"voice_models": result[6] if len(result) > 6 else []}
    except (GPUUnavailableError, ServiceUnavailableError) as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate")
async def convert_voice(
    source_file: UploadFile = File(None, description="待转换音频文件"),
    source_url: str = Form(None, description="待转换音频URL"),
    voice_model: str = Form(..., description="声音模型，必须与RVC下拉列表一致"),
    vocal_pitch_shift: float = Form(0, description="用于控制转换后人声的音高八度偏移量，男声转女声时设为1，女声转男声设为-1"),
    overall_pitch_shift: float = Form(0, description="半音数的音高变换（男变女[0,-12]、女变男[0,12]、性别不变0)"),
    index_rate: float = Form(0.7, description="增大该值可使转换结果更偏向目标音色模型的特征"),
    format: str = Form("mp3", description="音频格式，mp3, wav, flac 等"),
    instrumental_pitch_shift: float = Form(None, description="独立伴奏半音偏移，None=跟随 overall_pitch_shift，0=伴奏不变调"),
    backup_vocal_pitch_shift: float = Form(None, description="独立和声半音偏移，None=跟随 overall_pitch_shift，0=和声不变调"),
):
    """
    使用gradio_client调用已运行的Gradio服务进行语音转换，并立即返回音频文件
    支持文件上传和URL两种输入方式
    """
    source_path = None
    mp3_temp_path = None

    if not source_file and not source_url:
        raise HTTPException(status_code=400, detail="必须提供 source_file 或 source_url")

    try:
        # 有请求时才会申请 GPU 并拉起 Gradio 后端
        with pool.session() as client:
            # 1. 准备本地临时文件
            if source_file:
                suffix = os.path.splitext(source_file.filename)[1] or ".mp3"
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    shutil.copyfileobj(source_file.file, tmp)
                    source_path = tmp.name
            else:
                source_path = download_file_from_url(source_url)

            # 2. 获取真实模型名
            real_model_name = get_valid_model_name(client, voice_model)

            # 3. 【关键修复】分两步走：先上传，再推理
            # 第一步：上传文件到 RVC 音频组件 (Audio Component)
            # 这会返回服务器上的路径字符串
            print("正在上传文件到 RVC 服务器...")
            upload_result = client.predict(
                x=handle_file(source_path),  # 这里必须用 handle_file，因为 update_value_2 接收文件
                api_name="/update_value_2"
            )

            # 解析上传结果，获取纯净的路径字符串
            server_path = None
            if isinstance(upload_result, dict) and "value" in upload_result:
                server_path = upload_result["value"]
            else:
                server_path = upload_result

            print(f"RVC 服务器内部路径: {server_path}")

            if not server_path:
                raise ValueError("文件上传到 RVC 失败，返回路径为空")

            # 使用gradio_client调用服务
            result = client.predict(
            param_0=server_path,       # [输入] 源音频路径 (Source) - 必须是服务器本地的绝对路径
            param_1=real_model_name,            # [核心] 模型名称 (Voice Model) - 必填，例如 'mi-test'
            param_2=vocal_pitch_shift,               # [核心] 变调 (Pitch Shift) - 半音数，整数。男转女+12，女转男-12
            param_3=overall_pitch_shift,               # [保留] 整体变调 (Overall Pitch Shift) - 通常保持 0
            param_4="rmvpe",         # [核心] F0预测算法 (Algorithm) - 推荐 rmvpe 或 fcpe
            param_5=index_rate,             # [核心] 特征检索率 (Index Rate) - 0到1，越大越像目标音色但可能由于底噪
            param_6=1,               # [调节] 响度混合率 (RMS Mix) - 0到1，控制输出音量包络的混合程度
            param_7=0.33,            # [调节] 保护率 (Protect Rate) - 0到0.5，保护无声母音/呼吸声避免电音
            param_8=False,           # [处理] 分割输入音频 (Split Input) - 是否自动切片
            param_9=False,           # [效果] 自动调音 (Autotune) - 是否开启自动修音
            param_10=1,              # [效果] 自动调音强度 (Autotune Strength)
            param_11=False,          # [效果] 启用预测音高 (Proposed Pitch)
            param_12=155,            # [效果] 预测音高阈值 (Threshold)
            param_13=False,          # [效果] 清理转换后的音频 (Clean Output) - AI降噪
            param_14=0.7,            # [效果] 清理强度 (Clean Strength)
            param_15="chinese-hubert-base",   # [核心] 编码器 (Embedder) - 默认为 contentvec，中文模型选 chinese-hubert-base
            param_16=None,           # [核心] 自定义编码器 (Custom Embedder) - 通常为空
            param_17=0,              # [参数] 说话人ID (Speaker ID) - 仅在模型包含多个角色时使用
            param_18=0.15,           # [混响] 房间大小 (Room Size)
            param_19=0.2,            # [混响] 湿润度 (Wetness Level)
            param_20=0.8,            # [混响] 干燥度 (Dryness Level)
            param_21=0.7,            # [混响] 阻尼 (Damping Level)
            param_22=0,              # [音量] 主人声增益 (Main Gain) - 转换后的人声音量
            param_23=0,              # [音量] 伴奏增益 (Instrumentals Gain) - 伴奏音量
            param_24=0,              # [音量] 备份人声增益 (Backup Gain)
            param_25=44100,        # [输出] 采样率 (Sample Rate) - 注意：某些版本此处需要传数字类型 44100
            param_26=format,          # [输出] 音频格式 (Format) - mp3, wav, flac 等
            param_27="output_format",      # [输出] 输出文件名 (Output Name) - 仅文件名，不带后缀
            param_28=instrumental_pitch_shift,  # 伴奏独立半音偏移 (None=跟随)
            param_29=backup_vocal_pitch_shift,  # 和声独立半音偏移 (None=跟随)
            api_name="/partial_6"
            )

            # 创建MP3临时文件
            mp3_temp_path = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4()}.mp3")
            convert_to_mp3(result[0], mp3_temp_path)

            # 读取MP3文件内容
            audio_content = Path(mp3_temp_path).read_bytes()

            # 清理临时文件
            if Path(source_path).exists():
                Path(source_path).unlink()
            if Path(mp3_temp_path).exists():
                Path(mp3_temp_path).unlink()

            # 返回MP3音频文件
            return StreamingResponse(
                iter([audio_content]),
                media_type="audio/mpeg",
                headers={
                    "Content-Disposition": "attachment; filename=converted_audio.mp3"
                }
            )

    except (GPUUnavailableError, ServiceUnavailableError) as e:
        # 清理临时文件
        if source_path and Path(source_path).exists():
            Path(source_path).unlink()
        return JSONResponse(
            status_code=503,
            content={"error": f"GPU 资源不可用: {e!s}"}
        )
    except Exception as e:
        # 清理临时文件
        if source_path and Path(source_path).exists():
            Path(source_path).unlink()

        # 尝试读取可能存在的错误信息
        error_detail = str(e)
        try:
            # 如果是Gradio的错误，可能包含更多细节
            if hasattr(e, "message"):
                error_detail = e.message
        except:
            pass

        return JSONResponse(
            status_code=500,
            content={"error": f"转换失败: {error_detail}"}
        )


# =======================
# Step 1: 上传数据集 & 预处理
# =======================
@app.post("/model/train/preprocess_dataset", summary="Step 1: 上传数据集并预处理")
async def train_model(
    background_tasks: BackgroundTasks,
    dataset: list[UploadFile] = File(..., description="请直接多选上传 wav/mp3 干声文件"),
    model_name: str = Form(..., description="模型名称 (文件夹名)，必须已存在于 logs/ 下"),

):
    # 验证模型是否存在 (复用之前的逻辑)
    try:
        with pool.session() as client:
            real_model_name = get_valid_model_name(client, model_name)
    except (GPUUnavailableError, ServiceUnavailableError) as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception:
        real_model_name = model_name

    model_dataset_dir = os.path.join("ultimate-rvc/audio/training/", real_model_name)
    Path(model_dataset_dir).mkdir(parents=True)
    saved_count = 0
    try:
        print(f"📦 正在接收 {len(dataset)} 个文件，存入: {model_dataset_dir}")
        for file in dataset:
            # 跳过空文件或非音频文件(简单判断)
            if not file.filename:
                continue

            file_path = os.path.join(model_dataset_dir, file.filename)

            # 保存文件到磁盘
            with Path(file_path).open("wb") as f:
                shutil.copyfileobj(file.file, f)
            saved_count += 1

        print(f"✅ 成功保存 {saved_count} 个文件")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"文件保存失败: {e!s}")

    abs_dataset_path = os.path.abspath(model_dataset_dir)
    # 定义预处理后台任务

    def run_preprocess():
        print(f"🔄 [后台任务] 开始预处理数据: {model_name} ...")
        try:
            # 后台任务期间保持 GPU 租约与心跳
            with pool.session() as client:
                result = client.predict(
                real_model_name,         # param_0: Model name
                abs_dataset_path,                   # param_1: Dataset path (绝对路径)
                40000,              # param_2: Sample rate (String "40000")
                "post",             # param_3: Normalization mode (none/pre/post)
                True,               # param_4: Filter audio
                False,              # param_5: Clean audio (建议False，太慢)
                0.7,                # param_6: Cleaning intensity
                "Automatic",        # param_7: Audio splitting method
                3,                # param_8: Chunk length
                0.3,                # param_9: Overlap length
                96,           # param_10: CPU cores
                api_name="/_wrapped_fn_12"
                )
            print(f"✅ [后台任务] 预处理完成: {result}")
        except (GPUUnavailableError, ServiceUnavailableError) as e:
            print(f"❌ [后台任务] 预处理失败，GPU 资源不可用: {e}")
        except Exception as e:
            print(f"❌ [后台任务] 预处理失败: {e}")
            import traceback
            traceback.print_exc()

    background_tasks.add_task(run_preprocess)

    return {
        "status": "queued",
        "message": "数据集已上传，预处理任务已后台启动。",
        "dataset_path": abs_dataset_path
    }


# =======================
# Step 2: 特征提取
# =======================
@app.post("/model/extract_features", summary="Step 2: 特征提取")
async def extract_features(
    background_tasks: BackgroundTasks,
    model_name: str = Form(..., description="模型名称"),
    # gpu_id: int = Form(0, description="GPU ID"),
):
    """
    RVC 训练第二步：
    提取音高(f0)和特征(embedding)，生成 .npy 文件。
    必须在 Step 1 完成后调用。
    """
    # 验证模型是否存在 (复用之前的逻辑)
    try:
        with pool.session() as client:
            real_model_name = get_valid_model_name(client, model_name)
    except (GPUUnavailableError, ServiceUnavailableError) as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception:
        real_model_name = model_name

    def run_extraction():
        print(f"🔄 [后台任务] 开始特征提取: {model_name} (Embedder: chinese-hubert-base)...")
        try:
            # 后台任务期间保持 GPU 租约与心跳
            with pool.session() as client:
                result = client.predict(
                real_model_name,     # param_0: Model name
                "rmvpe",             # param_1: F0 method
                "chinese-hubert-base",       # param_2: Embedder model (chinese-hubert-base)
                None,               # param_3: Custom embedder
                2,                  # param_4: Include mutes (2 = 默认)
                96,                 # param_5: CPU cores (提取特征主要靠GPU，CPU参数影响不大)
                "Automatic",        # param_6: Hardware acceleration
                [0],       # param_7: GPUs (List，CUDA_VISIBLE_DEVICES 下的序号)
                api_name="/_wrapped_fn_13"
                )
            print(f"✅ [后台任务] 特征提取完成: {result}")

        except (GPUUnavailableError, ServiceUnavailableError) as e:
            print(f"❌ [后台任务] 特征提取失败，GPU 资源不可用: {e}")
        except Exception as e:
            print(f"❌ [后台任务] 特征提取失败: {e}")
            import traceback
            traceback.print_exc()

    background_tasks.add_task(run_extraction)

    return {
        "status": "queued",
        "message": f"特征提取任务已后台启动 (Model: {model_name})",
        "info": "请等待控制台显示 '特征提取完成' 后再执行 Step 3 (Train)"
    }


# =======================
# Step 3: 模型训练
# =======================
@app.post("/model/train", summary="Step 3: 启动模型训练")
async def train_model(
    background_tasks: BackgroundTasks,
    model_name: str = Form(..., description="模型名称 (文件夹名)，必须已存在于 logs/ 下"),
    epochs: int = Form(500, description="训练总轮数 (Epochs)"),
    batch_size: int = Form(8, description="批次大小 (Batch Size)，显存小请填 4 或 2"),
    overtraining_detector: bool = Form(True, description="是否启用过训练检测 (Overtraining Detector)"),
    Overtraining_threshold: int = Form(50, description="过训练检测阈值 (Overtraining Threshold)"),
    save_interval: int = Form(100, description="保存频率 (每隔多少轮保存一次模型)"),
):
    """
    启动 RVC 模型的训练流程 (对应 WebUI 的 Step 3)。
    
    注意：
    1. 这是一个耗时操作！
    2. 此接口假设你已经完成了 '数据预处理' 和 '特征提取'。
    3. 为了不阻塞 API，这里演示使用后台任务触发训练。
    """
    # 验证模型是否存在 (复用之前的逻辑)
    try:
        real_model_name = get_valid_model_name(model_name)
    except Exception:
        # 如果是新模型，可能还没在下拉列表里，这里允许不做强校验，或者根据你的业务逻辑决定
        # RVC 的逻辑是：只要 logs/{model_name} 文件夹存在且有处理好的数据就能训练
        # 这里我们假设用户传入的是正确的
        real_model_name = model_name

    print(f"🚀 接收到训练请求: {real_model_name}, Epochs: {epochs}")

    # 定义训练逻辑函数
    def run_training_task():
        print(f"🔄 [后台任务] 开始训练模型: {real_model_name} ...")
        try:
            # 训练全程保持 GPU 租约与心跳，避免被管理服务回收
            with pool.session() as client:
                result = client.predict(
                    real_model_name,            # param_0: Model name
                    epochs,          # param_1: Epochs
                    batch_size,      # param_2: Batch size
                    overtraining_detector,  # param_3: Detect overtraining
                    Overtraining_threshold,                  # param_4: Overtraining threshold (默认50)
                    "HiFi-GAN",                 # param_5: Vocoder
                    "Auto",                     # param_6: Index algorithm
                    "Default",                  # param_7: Pretrained model type
                    None,                       # param_8: Custom pretrained model
                    save_interval,   # param_9: Save interval
                    False,                      # param_10: Save all checkpoints
                    False,       # param_11: Save all weights
                    False,                      # param_12: Clear saved data
                    True,                      # param_13: Upload voice model (训练完自动上传?)
                    real_model_name,        # param_14: Upload name (必填的占位符)
                    "Automatic",                # param_15: Hardware acceleration
                    [0],               # param_16: GPUs (List，CUDA_VISIBLE_DEVICES 下的序号)
                    "fp32",                     # param_17: Precision (fp32/fp16)
                    False,                      # param_18: Preload dataset
                    False,                      # param_19: Reduce memory usage
                    api_name="/_wrapped_fn_14"
                    )
            print(f"✅ [后台任务] 训练完成! 结果: {result}")

        except (GPUUnavailableError, ServiceUnavailableError) as e:
            print(f"❌ [后台任务] 训练失败，GPU 资源不可用: {e}")
        except Exception as e:
            print(f"❌ [后台任务] 训练失败: {e}")
            import traceback
            traceback.print_exc()

    # 将训练任务加入后台队列，立即返回响应给用户
    background_tasks.add_task(run_training_task)

    return {
        "status": "queued",
        "message": f"模型 '{real_model_name}' 的训练任务已在后台启动。",
        "config": {
            "epochs": epochs,
            "batch_size": batch_size,
            "gpu": pool.gpu_ids
        },
        "note": "训练可能需要很长时间，请留意服务器控制台日志查看进度。"
    }


@app.post("/models/upload", summary="上传新的语音模型 (.pth/.index)")
async def upload_model(
    model_name: str = Form(..., description="模型名称 (不带后缀，例如: my_custom_voice)"),
    files: list[UploadFile] = File(..., description="请上传 .pth 文件和可选的 .index 文件")
):
    """
    上传 RVC 模型文件到服务器。
    支持同时上传 .pth 和 .index 文件。
    """
    temp_files = []
    temp_paths = []

    try:
        with pool.session() as client:
            # 1. 将上传的文件保存到临时目录
            # Gradio Client 的 handle_file 需要本地真实路径
            for file in files:
                suffix = os.path.splitext(file.filename)[1]
                # 创建临时文件
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
                shutil.copyfileobj(file.file, tmp)
                tmp.close()  # 关闭句柄让 gradio 能读取

                temp_files.append(tmp.name)
                temp_paths.append(handle_file(tmp.name))  # 包装成 Gradio 文件句柄

            print(f"📦 正在上传 {len(temp_files)} 个文件用于模型: {model_name}...")

            # 2. 调用 Gradio 上传接口 (/_wrapped_fn_9)
            # 注意：param_0 接受的是一个 list
            result = client.predict(
                param_0=temp_paths,   # [handle_file(pth), handle_file(index)]
                param_1=model_name,   # 模型名称
                api_name="/_wrapped_fn_9"
            )

            print(f"✅ 上传响应: {result}")

            # 3. 强制刷新模型列表 (/_init_dropdowns)
            # 这一步很重要，否则上传后去调用 generate 可能会报找不到模型
            print("🔄 刷新模型列表...")
            client.predict(api_name="/_init_dropdowns")

            return {
                "status": "success",
                "message": f"模型 '{model_name}' 上传成功",
                "details": result
            }

    except (GPUUnavailableError, ServiceUnavailableError) as e:
        raise HTTPException(status_code=503, detail=f"GPU 资源不可用: {e!s}")
    except Exception as e:
        print(f"❌ 上传失败: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"模型上传失败: {e!s}")

    finally:
        # 4. 清理临时文件
        for path in temp_files:
            if path and Path(path).exists():
                try: Path(path).unlink()
                except: pass


# 添加自定义Swagger UI路由，使用国内CDN
@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui_html():
    return get_swagger_ui_html(
    openapi_url=app.openapi_url,
    title=app.title + " - Swagger UI",
    oauth2_redirect_url=app.swagger_ui_oauth2_redirect_url,
    swagger_js_url="https://cdn.bootcdn.net/ajax/libs/swagger-ui/5.17.14/swagger-ui-bundle.js",
    swagger_css_url="https://cdn.bootcdn.net/ajax/libs/swagger-ui/5.17.14/swagger-ui.css",
)


# 添加自定义ReDoc路由，使用国内CDN
@app.get("/redoc", include_in_schema=False)
async def custom_redoc_html():
    return get_redoc_html(
    openapi_url=app.openapi_url,
    title=app.title + " - ReDoc",
    redoc_js_url="https://cdn.bootcdn.net/ajax/libs/redoc/2.0.0/redoc.standalone.js",
)


@app.get("/")
async def root():
    return {"message": "Voice Conversion Client API is running. Use POST /convert to convert voice."}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host=os.environ.get("URVC_GATEWAY_HOST", "127.0.0.1"),
        port=int(os.environ.get("URVC_GATEWAY_PORT", "7866")),
    )
