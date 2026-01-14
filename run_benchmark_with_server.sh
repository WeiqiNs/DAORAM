#!/bin/bash
# 启动 DAORAM socket server 和 benchmark client
# 用法：bash run_benchmark_with_server.sh

# 项目根目录
ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"

echo "[INFO] Cleaning up ports..."
lsof -ti:10000 | xargs kill -9 2>/dev/null

run_protocol() {
    PROTO=$1
    echo "========================================"
    echo "[INFO] Starting benchmark for: $PROTO"
    echo "========================================"
    
    # Start Server in background
    PYTHONPATH=. python demo/server.py > /dev/null 2>&1 &
    SERVER_PID=$!
    echo "[INFO] Server started (PID: $SERVER_PID). Waiting for ready..."
    sleep 2
    
    # Run Benchmark
    echo "[INFO] Running client..."
    PYTHONPATH=. python scripts/benchmark_somap_wan.py --ops 20 --latency 0.02 --protocol $PROTO
    
    # Kill Server
    echo "[INFO] Killing server..."
    kill $SERVER_PID 2>/dev/null
    wait $SERVER_PID 2>/dev/null
    # Force kill just in case
    lsof -ti:10000 | xargs kill -9 2>/dev/null
    sleep 1
}

run_protocol "bottom_up"
run_protocol "top_down"
run_protocol "baseline"

echo "========================================"
echo "[INFO] All benchmarks completed."
echo "========================================"
