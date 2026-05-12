#!/bin/bash

SCRIPT_DIR="/root/autodl-fs/qwen_api/scripts"

echo "========================================"
echo "[INFO] 重启 AI 服务"
echo "========================================"

echo "[STEP 1] 停止旧服务..."
bash "$SCRIPT_DIR/stop_all.sh"

echo ""
echo "[STEP 2] 启动 vLLM..."
bash "$SCRIPT_DIR/start_vllm.sh"

echo ""
echo "[STEP 3] 等待 vLLM 就绪..."

VLLM_READY=0

for i in {1..60}; do
    if curl -s http://127.0.0.1:8000/v1/models >/dev/null 2>&1; then
        echo "[SUCCESS] vLLM 启动成功"
        VLLM_READY=1
        break
    fi

    echo "[INFO] vLLM 初始化中... $i/60"
    sleep 5
done

if [ "$VLLM_READY" -ne 1 ]; then
    echo "[ERROR] vLLM 启动超时"
    echo "[INFO] 查看日志：tail -f /root/autodl-fs/qwen_api/logs/vllm.log"
    exit 1
fi

echo ""
echo "[STEP 4] 启动 FastAPI..."
bash "$SCRIPT_DIR/start_api.sh"

echo ""
echo "[STEP 5] 等待 FastAPI 就绪..."

API_READY=0

for i in {1..90}; do
    if curl -s http://127.0.0.1:9000/ >/dev/null 2>&1; then
        echo "[SUCCESS] FastAPI 启动成功"
        API_READY=1
        break
    fi

    echo "[INFO] FastAPI 初始化中... $i/30"
    sleep 3
done

if [ "$API_READY" -ne 1 ]; then
    echo "[ERROR] FastAPI 启动超时"
    echo "[INFO] 查看日志：tail -f /root/autodl-fs/qwen_api/logs/api.log"
    exit 1
fi

echo ""
echo "========================================"
echo "[SUCCESS] 所有服务启动完成"
echo "========================================"

echo "vLLM:    http://127.0.0.1:8000/v1/models"
echo "FastAPI: http://127.0.0.1:9000/"
echo "RAG:     POST http://127.0.0.1:9000/rag/chat"
