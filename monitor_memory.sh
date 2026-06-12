#!/bin/bash

# 显存监控脚本

echo "=========================================="
echo "RVC 服务显存监控"
echo "=========================================="
echo ""

# 检查 nvidia-smi 是否可用
if ! command -v nvidia-smi &> /dev/null; then
    echo "错误: nvidia-smi 命令不可用"
    exit 1
fi

# 显示 GPU 信息
nvidia-smi --query-gpu=index,name,driver_version,memory.total,memory.free,memory.used --format=csv,noheader

echo ""
echo "开始监控显存使用情况（按 Ctrl+C 停止）..."
echo ""

# 持续监控显存使用
while true; do
    clear
    echo "=========================================="
    date
    echo "=========================================="
    echo ""
    nvidia-smi --query-gpu=index,name,memory.total,memory.free,memory.used,utilization.gpu,utilization.memory --format=csv,noheader,nounits | awk -F', ' '{
        printf "GPU %s: %s\n", $1, $2
        printf "  显存: %s MB / %s MB (已用: %s MB)\n", $5, $3, $4
        printf "  GPU 利用率: %s%%, 显存利用率: %s%%\n", $6, $7
        printf "  空闲显存: %s MB\n\n", $5
    }'
    echo ""
    echo "RVC 进程:"
    ps aux | grep -E "ultimate_rvc|python.*main" | grep -v grep || echo "  (未找到 RVC 进程)"
    echo ""
    echo "下次刷新: 5秒后 (按 Ctrl+C 退出)"
    sleep 5
done
