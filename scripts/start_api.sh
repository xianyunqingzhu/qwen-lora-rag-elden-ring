#!/bin/bash

PROJECT_DIR="/root/autodl-fs/qwen_api"
LOG_DIR="$PROJECT_DIR/logs"
LOG_FILE="$LOG_DIR/api.log"

PORT=9000

mkdir -p "$LOG_DIR"

echo "========================================"
echo "[INFO] 启动 FastAPI 服务"
echo "[INFO] 项目目录: $PROJECT_DIR"
echo "[INFO] 端口: $PORT"
echo "[INFO] 日志文件: $LOG_FILE"
echo "========================================"

if lsof -i:$PORT >/dev/null 2>&1; then
    echo "[WARN] 端口 $PORT 已被占用，API 可能已经在运行"
    lsof -i:$PORT
    exit 0
fi

nohup bash -c "
source /root/miniconda3/etc/profile.d/conda.sh
conda activate qwen

cd $PROJECT_DIR

uvicorn main:app \
  --host 0.0.0.0 \
  --port $PORT
" > "$LOG_FILE" 2>&1 &

echo "[SUCCESS] FastAPI 已在后台启动"
echo "[INFO] PID: $!"
echo "[INFO] 查看日志：tail -f $LOG_FILE"
echo "[INFO] 测试服务：curl http://127.0.0.1:$PORT/"
