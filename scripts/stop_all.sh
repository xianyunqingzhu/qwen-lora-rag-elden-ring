#!/bin/bash

echo "========================================"
echo "[INFO] 停止 FastAPI 和 vLLM 服务"
echo "========================================"

echo "[INFO] 停止 FastAPI..."
pkill -f "uvicorn main:app"

echo "[INFO] 停止 vLLM..."
pkill -f "vllm serve"

sleep 5

echo "[INFO] 检查 vLLM..."
if curl -s http://127.0.0.1:8000/v1/models >/dev/null 2>&1; then
    echo "[WARN] vLLM 可能仍在运行"
else
    echo "[OK] vLLM 已停止"
fi

echo "[INFO] 检查 FastAPI..."
if curl -s http://127.0.0.1:9000/ >/dev/null 2>&1; then
    echo "[WARN] FastAPI 可能仍在运行"
else
    echo "[OK] FastAPI 已停止"
fi

echo "[SUCCESS] 停止完成"
