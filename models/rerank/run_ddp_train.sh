#!/bin/bash
# 多卡分布式训练启动脚本
# 使用方式: bash run_ddp_train.sh [CONFIG_TYPE]
# CONFIG_TYPE 可选: ddp (默认), fsdp, deepspeed

set -e

# 切换到项目根目录（而不是脚本所在目录）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

echo "[Info] Working directory: $(pwd)"

# 配置类型（默认 ddp）
CONFIG_TYPE="${1:-ddp}"

# 根据配置类型选择配置文件
case "$CONFIG_TYPE" in
    ddp)
        CONFIG_FILE="accelerate_config_ddp.yaml"
        echo "[Info] Using DDP (Data Distributed Parallel) mode"
        ;;
    fsdp)
        CONFIG_FILE="accelerate_config_fsdp.yaml"
        echo "[Info] Using FSDP (Fully Sharded Data Parallel) mode"
        ;;
    deepspeed)
        CONFIG_FILE="accelerate_config_deepspeed.yaml"
        echo "[Info] Using DeepSpeed ZeRO mode"
        ;;
    *)
        echo "[Error] Unknown config type: $CONFIG_TYPE"
        echo "  Available options: ddp, fsdp, deepspeed"
        exit 1
        ;;
esac

# 检查配置文件是否存在（使用完整路径）
CONFIG_FILE_FULL="$SCRIPT_DIR/$CONFIG_FILE"
if [ ! -f "$CONFIG_FILE_FULL" ]; then
    echo "[Error] Config file not found: $CONFIG_FILE_FULL"
    exit 1
fi

# 检查 GPU 数量
NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo "[Info] Detected $NUM_GPUS GPUs"

# 设置环境变量
export TOKENIZERS_PARALLELISM=false
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

# 启动训练
echo "[Info] Starting distributed training..."
echo "[Info] Config file: $CONFIG_FILE_FULL"
echo ""

accelerate launch --config_file "$CONFIG_FILE_FULL" \
    models/rerank/jina_m0_lora_train_ddp.py \
    --data_path "./imgflip_data/msrvtt" \
    --image_path "./imgflip_data/images" \
    --save_dir "./checkpoints_rerank_ddp" \
    --epochs 5 \
    --lr 1e-5 \
    --batch_size 128 \
    --gradient_accumulation_steps 4 \
    --mixed_precision fp16 \
    --train_lora false \
    --lora_r 8 \
    --lora_alpha 16 \
    --loss_type lambdarank \
    --label_mode inv_rank \
    --topk_base 50 \
    --scheduler cosine_warmup \
    --warmup_steps 100 \
    --jina_micro_batch 32 \
    --image_max_side 256 \
    --eval_sample_limit 200 \
    --seed 42

echo ""
echo "[Info] Training completed!"

