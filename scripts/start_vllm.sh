#!/bin/bash

# =========================
# vLLM 启动脚本
# =========================

PROJECT_DIR="/root/autodl-fs/qwen_api"
LOG_DIR="$PROJECT_DIR/logs"
LOG_FILE="$LOG_DIR/vllm.log"

MODEL_PATH="/root/autodl-fs/models/qwen2.5-7b-elden-ring-merged"
PORT=8000

mkdir -p "$LOG_DIR"

echo "========================================"
echo "[INFO] 启动 vLLM 服务"
echo "[INFO] 模型路径: $MODEL_PATH"
echo "[INFO] 端口: $PORT"
echo "[INFO] 日志文件: $LOG_FILE"
echo "========================================"

# 如果端口已被占用，提示并退出
if curl -s http://127.0.0.1:$PORT/v1/models >/dev/null 2>&1; then
    echo "[WARN] vLLM 已经在运行"
    exit 0
fi

# 启动 vLLM
nohup bash -c "
source /root/miniconda3/etc/profile.d/conda.sh
conda activate qwen

vllm serve $MODEL_PATH \
  --host 0.0.0.0 \
  --port $PORT \
  --dtype bfloat16
" > "$LOG_FILE" 2>&1 &

echo "[SUCCESS] vLLM 已在后台启动"
echo "[INFO] PID: $!"
echo "[INFO] 查看日志：tail -f $LOG_FILE"
echo "[INFO] 测试服务：curl http://127.0.0.1:$PORT/v1/models"
