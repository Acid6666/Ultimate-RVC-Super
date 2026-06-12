#!/bin/bash

# RVC 服务启动脚本
#
# 使用方法:
#   CUDA_VISIBLE_DEVICES=0 ./start_service.sh
#   CUDA_VISIBLE_DEVICES=1 nohup ./start_service.sh >> logs/web.log 2>&1 &
#   GPU_ID=2 ./start_service.sh

echo "=========================================="
echo "RVC 服务启动脚本"
echo "=========================================="
echo ""

# 检查并设置 GPU
if [ -n "$GPU_ID" ]; then
    export CUDA_VISIBLE_DEVICES=$GPU_ID
    echo "使用 GPU: $GPU_ID"
elif [ -n "$CUDA_VISIBLE_DEVICES" ]; then
    echo "使用 GPU: $CUDA_VISIBLE_DEVICES"
else
    echo "警告: 未指定 GPU，使用默认设置"
fi

echo ""
echo "显存优化特性："
echo "- 启动时立即清理显存"
echo "- 空闲 5 分钟后自动清理"
echo "- 使用优化的清理流程"
echo ""

# 检查 CUDA 是否可用
echo "检查 CUDA 环境..."
python3 -c "import torch; print(f'CUDA 可用: {torch.cuda.is_available()}'); print(f'CUDA 设备数量: {torch.cuda.device_count() if torch.cuda.is_available() else 0}')"

if [ $? -ne 0 ]; then
    echo "错误: CUDA 环境检查失败"
    exit 1
fi

echo ""
echo "启动 RVC 服务..."
echo ""

# 启动服务
cd /data/lizilong/ultimate-rvc

# 可以通过环境变量配置
# URVC_CONFIG: 配置文件名称
# YT_COOKIEFILE: YouTube cookie 文件路径
# LISTEN_PORT: 监听端口
export URVC_CONFIG=${URVC_CONFIG:-}
export YT_COOKIEFILE=${YT_COOKIEFILE:-}
export LISTEN_PORT=${LISTEN_PORT:-7860}

# 启动服务
python3 -m ultimate_rvc.web.main \
    --listen \
    --listen-host 172.22.14.2 \
    --listen-port $LISTEN_PORT

echo ""
echo "RVC 服务已停止"
