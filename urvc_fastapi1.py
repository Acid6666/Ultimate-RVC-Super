import shutil
import os
import tempfile
import uuid
import traceback
from fastapi import FastAPI, HTTPException, File, UploadFile, Form, Request
from fastapi.staticfiles import StaticFiles
from enum import Enum
from gradio_client import Client

# =======================
# 配置部分
# =======================

OUTPUT_DIR = "outputs"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

app = FastAPI(title="RVC API Service", version="2.0.0")
app.mount("/download", StaticFiles(directory=OUTPUT_DIR), name="download")

GRADIO_URL = "http://127.0.0.1:7865/"

print(f"🔗 正在连接 Gradio 服务: {GRADIO_URL} ...")
try:
    client = Client(GRADIO_URL)
    print(f"✅ 连接成功!")
except Exception as e:
    print(f"❌ 连接失败: {e}")

class PitchAlgoEnum(str, Enum):
    rmvpe = "rmvpe"
    crepe = "crepe"
    crepe_tiny = "crepe-tiny"
    fcpe = "fcpe"

@app.get("/models", summary="获取可用模型列表")
def get_models():
    try:
        result = client.predict(api_name="/_init_dropdowns")
        return {"voice_models": result[6] if len(result) > 6 else []}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate", summary="RVC 推理 (直接路径模式)")
def generate_audio(
    request: Request,
    source: UploadFile = File(..., description="源音频文件"),
    voice_model_name: str = Form(..., description="模型名称 (必须与RVC下拉列表一致)"),
    pitch_shift: float = Form(0, description="变调 (半音)"),
    f0_method: PitchAlgoEnum = Form(PitchAlgoEnum.rmvpe, description="F0算法"),
    index_rate: float = Form(0, description="索引率 (建议先设为0测试)"), # 改默认为0，先跑通
    output_format: str = Form("mp3", description="输出格式")
):
    temp_file_path = None
    try:
        # --- 1. 保存文件 (确保文件名安全，防止中文路径问题) ---
        # 即使上传的是中文名，我们在本地存为 uuid.mp3 这种纯 ASCII 名
        # 这样传递给 RVC 时，它是安全的绝对路径
        ext = os.path.splitext(source.filename)[1] if source.filename else ".mp3"
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            shutil.copyfileobj(source.file, tmp)
            temp_file_path = tmp.name
        
        # 获取绝对路径
        abs_input_path = os.path.abspath(temp_file_path)
        print(f"📂 输入文件已就绪 (绝对路径): {abs_input_path}")

        # --- 2. 准备参数 ---
        # 这里的 embedder 依然尝试用 chinese-hubert-base
        embedder = "chinese-hubert-base"
        
        args = [
            abs_input_path,             # param_0: 直接传本地绝对路径！
            voice_model_name,           # param_1
            float(pitch_shift),         # param_2: 强转 float
            float(0),                   # param_3
            f0_method.value,            # param_4
            float(index_rate),          # param_5
            float(1),                   # param_6: RMS mix
            float(0.33),                # param_7: Protect
            False,                      # param_8: Split voice
            False,                      # param_9
            float(1),                   # param_10
            False,                      # param_11
            float(155),                 # param_12
            False,                      # param_13
            float(0.7),                 # param_14
            
            embedder,                   # param_15: 编码器
            
            None, 0, 0.15, 0.2, 0.8, 0.7, 0, 0, 0,
            44100,                      # param_25: int
            output_format,              # param_26
            "fastapi_out"               # param_27
        ]

        # --- 3. 调用推理 ---
        print(f"🔄 开始推理: 模型={voice_model_name}, 变调={pitch_shift}, 编码器={embedder}")
        
        try:
            result = client.predict(*args, api_name="/partial_6")
        except Exception as e_infer:
            # 自动降级重试逻辑
            if "is not in the list of choices" in str(e_infer) and embedder in str(e_infer):
                print(f"⚠️ 警告: 不支持 {embedder}，降级为 contentvec 重试...")
                args[15] = "contentvec"
                result = client.predict(*args, api_name="/partial_6")
            else:
                raise e_infer

        # --- 4. 处理结果 ---
        # result[1] = 混合音频 (Song)
        # result[8] = 纯人声 (Converted Vocals)
        # 既然你说输出的是原音频，我们这次同时把 [1] 和 [8] 都保存下来供你对比
        
        output_mix_path = result[1]
        output_vocal_path = result[8]
        
        if not output_mix_path:
             raise Exception("RVC 未返回有效音频")

        # 保存混合音频
        filename_mix = f"mix_{uuid.uuid4()}.{output_format}"
        target_path_mix = os.path.join(OUTPUT_DIR, filename_mix)
        shutil.copy(output_mix_path, target_path_mix)

        # 保存纯人声 (如果有)
        url_vocal = None
        if output_vocal_path and os.path.exists(output_vocal_path):
            filename_vocal = f"vocal_{uuid.uuid4()}.{output_format}"
            target_path_vocal = os.path.join(OUTPUT_DIR, filename_vocal)
            shutil.copy(output_vocal_path, target_path_vocal)
            url_vocal = str(request.base_url) + f"download/{filename_vocal}"

        url_mix = str(request.base_url) + f"download/{filename_mix}"

        return {
            "status": "success",
            "download_url": url_mix,         # 默认下载链接 (混合)
            "download_url_vocal": url_vocal, # 纯人声链接 (用于检查是否转换成功)
            "info": f"Used model: {voice_model_name}"
        }

    except Exception as e:
        print(f"❌ 错误: {e}")
        traceback.print_exc() # 打印完整堆栈
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        # 清理输入临时文件
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
            except:
                pass

if __name__ == "__main__":
    import uvicorn
    # 请确保端口是 7866
    uvicorn.run(app, host="172.22.14.2", port=7866)