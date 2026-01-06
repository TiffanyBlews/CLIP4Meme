#!/bin/bash
# 仅训练 score 层的优化脚本（无需 FSDP/DeepSpeed，简单 DDP 即可）

set -e

# 切换到项目根目录（而不是脚本所在目录）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

echo "[Info] Working directory: $(pwd)"

# 检查 GPU 数量
NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo "[Info] Detected $NUM_GPUS GPUs"

# 环境变量
export TOKENIZERS_PARALLELISM=false
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

echo "[Info] Starting score-only distributed training..."
echo "[Info] Optimizations enabled:"
echo "  - Pre-extracted Jina features (no repeated forward pass)"
echo "  - Frozen backbone (no gradient computation)"
echo "  - Large batch size (only training small score head)"
echo "  - Simple DDP (score head is tiny, no need for FSDP)"
echo ""

# 启动训练（使用相对于项目根目录的路径）
accelerate launch --multi_gpu --num_processes=$NUM_GPUS \
    models/rerank/jina_m0_score_only_train_ddp.py \
    --data_path "./imgflip_data/msrvtt" \
    --image_path "./imgflip_data/images" \
    --save_dir "./checkpoints_rerank_score_only" \
    --epochs 5 \
    --lr 1e-5 \
    --batch_size 512 \
    --gradient_accumulation_steps 1 \
    --loss_type lambdarank \
    --label_mode inv_rank \
    --topk_base 50 \
    --jina_micro_batch 50 \
    --image_max_side 256 \
    --seed 42

echo ""
echo "[Info] Training completed!"

