cd "$(dirname "$0")"

# 本脚本仅用于「未接入 systemd」时的手动后台启动。
# 已执行 systemd 托管后（推荐），请改用：
#   systemctl start|stop|restart|status urvc
#   journalctl -u urvc -f        或     tail -f logs/urvc.log
if systemctl is-active --quiet urvc.service 2>/dev/null; then
    echo "urvc.service 已由 systemd 托管并正在运行，请先执行: systemctl stop urvc"
    exit 1
fi

# 先杀掉旧的进程（如果存在）
pkill -f "urvc_fastapi.py" 2>/dev/null
pkill -f "web/main.py" 2>/dev/null
sleep 1

# 确保端口释放
echo "Cleaning up old processes and ports..."

# ---------------------------------------------------------------
# GPU 资源由 127.0.0.1:7050 统一管理：
#   - 有请求时  -> acquire GPU，并以 CUDA_VISIBLE_DEVICES=<gpu_id>
#                  拉起 Gradio（真正占用显存的进程）
#   - 空闲超时  -> 停掉 Gradio（显存归零）+ release GPU
#   - 占用期间  -> 每 60 秒发一次 heartbeat
# 因此这里只需要常驻 FastAPI 网关（7866），它本身不占用显存。
# ---------------------------------------------------------------
export URVC_GPU_MANAGER_URL=${URVC_GPU_MANAGER_URL:-http://127.0.0.1:7050}
export URVC_GPU_COUNT=${URVC_GPU_COUNT:-1}
export URVC_GPU_HEARTBEAT_INTERVAL=${URVC_GPU_HEARTBEAT_INTERVAL:-60}
export URVC_IDLE_TIMEOUT=${URVC_IDLE_TIMEOUT:-600}
export URVC_GRADIO_HOST=${URVC_GRADIO_HOST:-127.0.0.1}
export URVC_GRADIO_PORT=${URVC_GRADIO_PORT:-7865}
export URVC_GRADIO_STARTUP_TIMEOUT=${URVC_GRADIO_STARTUP_TIMEOUT:-300}
export URVC_GRADIO_MANAGED=${URVC_GRADIO_MANAGED:-1}

mkdir -p logs

echo "Starting FastAPI gateway on port 7866 (Gradio will start on demand)..."
nohup uv run python urvc_fastapi.py >> logs/api.log 2>&1 &

# 等待网关就绪
echo "Waiting for FastAPI gateway to start..."
for i in $(seq 1 30); do
    if curl -s "http://${URVC_GRADIO_HOST}:7866/gpu/status" > /dev/null 2>&1; then
        echo "FastAPI gateway is ready!"
        curl -s "http://${URVC_GRADIO_HOST}:7866/gpu/status"
        echo ""
        break
    fi
    if [ $i -eq 30 ]; then
        echo "ERROR: FastAPI gateway failed to start within 60 seconds"
        exit 1
    fi
    sleep 2
done

echo "All services started!"
echo "说明：Gradio 会在首个 /generate 等请求到达时自动申请 GPU 并启动，"
echo "      空闲 ${URVC_IDLE_TIMEOUT} 秒后自动停止并归还 GPU。"
