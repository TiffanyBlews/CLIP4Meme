#!/bin/bash
# 单卡版本的 Jina 特征预提取脚本

set -e

# 切换到项目根目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

echo "[Info] Working directory: $(pwd)"

# 检查 GPU
if command -v nvidia-smi &> /dev/null; then
    NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
    echo "[Info] Detected $NUM_GPUS GPU(s)"
    
    if [ $NUM_GPUS -gt 0 ]; then
        # 使用第一张 GPU
        DEVICE="cuda:0"
        echo "[Info] Using GPU: $DEVICE"
    else
        DEVICE="cpu"
        echo "[Info] No GPU detected, using CPU"
    fi
else
    DEVICE="cpu"
    echo "[Info] nvidia-smi not found, using CPU"
fi

# 环境变量
export TOKENIZERS_PARALLELISM=false
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

echo "[Info] Starting Jina features pre-extraction (single GPU)..."
echo "[Info] This will extract features for all training queries (one-time process)"
echo ""

# 启动预提取（单卡，不需要 accelerate）    # --eval_cache \

python models/rerank/precompute_jina_features.py \
    --data_path "./imgflip_data/msrvtt" \
    --image_path "./imgflip_data/images" \
    --save_dir "./checkpoints_rerank_score_only" \
    --max_queries 10000 \
    --resume
    --device "$DEVICE" \
    --topk_base 50 \
    --jina_micro_batch 50 \
    --image_max_side 256 \
    --label_mode inv_rank \
    --seed 42

echo ""
echo "[Info] Pre-extraction completed!"
echo "[Info] Cache saved to: $CACHE_PATH"
echo "[Info] You can now run training with: bash models/rerank/run_score_only_train.sh"

