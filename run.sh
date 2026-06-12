cd /data/lizilong/ultimate-rvc
CUDA_VISIBLE_DEVICES=2 uv run python /data/lizilong/ultimate-rvc/urvc_fastapi.py >> logs/api.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 uv run python src/ultimate_rvc/web/main.py --listen --listen-host 172.22.14.2 --listen-port 7865 >> logs/web.log 2>&1 &