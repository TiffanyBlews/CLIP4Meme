docker exec -it cuda_1111 bash
conda activate neo_meme
cd ljj

DATA_PATH="./imgflip_data/msrvtt"
IMAGE_PATH="./imgflip_data/images"
JINA_API_KEY=jina_8c082cdb7f8e4608ba8d15d5fc72f1deJal4bOxisAOBOtelYXXQ8k-A2O1v
# DATA_PATH="./data/input_file"
# IMAGE_PATH="./data/image"
PYTORCH_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=0,1 \
nohup accelerate launch \
  --config_file models/rerank/accelerate_config_ddp.yaml \
  models/rerank/jina_m0_lora_train_ddp.py \
  --data_path "$DATA_PATH" \
  --image_path "$IMAGE_PATH" \
  --save_dir ./checkpoints_rerank_ddp \
  --epochs 2 \
  --batch_size 32 \
  --topk_base 50 \
  --lr 1e-5 \
  --loss_type lambdarank \
  --label_mode inv_rank \
  --image_max_side 256 \
  --qwen_min_side 32 \
  --jina_micro_batch 10 \
  --train_lora true \
  --train_sample_limit 5000 \
  --eval_sample_limit 1000 \
  --wandb_project jina_rerank_ddp \
  --wandb_entity wangtn \
  --wandb_mode online \
  > 0106_jina_m0_ddp_train.log 2>&1 &


CUDA_VISIBLE_DEVICES=1 nohup python models/IEF/train.py --train qi --data_path "$DATA_PATH" --image_path "$IMAGE_PATH" --save_dir ief_imgflip  --wandb_project clip4meme --wandb_entity wangtn > 1211_qi_imgflip.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup python models/IEF/train.py --train qc --data_path "$DATA_PATH" --image_path "$IMAGE_PATH" --save_dir ief_imgflip  --checkpoint_path  checkpoints_imgflip/new_qc_qi.pth --wandb_project clip4meme --wandb_entity wangtn > 1211_qc_imgflip.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup python models/IEF/inference_fusion.py --multi_alpha --checkpoint checkpoints_imgflip/new_qc.pth --data_path "$DATA_PATH" --image_path "$IMAGE_PATH" > 1211_fusion_imgflip.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup python models/clip/clip_train.py --data_path "$DATA_PATH" --image_path "$IMAGE_PATH" --save_dir clip_imgflip --model_save_name clip_imgflip.pt > 1211_clip_train_imgflip.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup python models/clip/clip_infer.py --data_path "$DATA_PATH" --image_path "$IMAGE_PATH" --checkpoint checkpoints_clip/clip_imgflip.pt   > 111_infer_imgflip_vanilla_clip.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup python models/clip/clip_infer.py --data_path "$DATA_PATH" --image_path "$IMAGE_PATH" --checkpoint ./checkpoints_clip/clip_imgflip_image_emotion_fusion.pt --use_image_emotion_fusion  > 1223_infer_imgflip.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup python models/clip/clip_infer.py --data_path "$DATA_PATH" --image_path "$IMAGE_PATH"  --zero_shot --save_dir ./checkpoints_rerank_imgflip > 1211_clip_zero_shot_imgflip.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup python models/clip/clip_infer.py --data_path "$DATA_PATH" --image_path "$IMAGE_PATH"  --checkpoint ./checkpoints_clip/clip_imgflip_image_emotion_fusion.pt --use_image_emotion_fusion --topk_base 100   --num_samples 1000 --sample_mode random --sample_seed 114  > 1223_clip_imgflip.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup python models/clip/clip_infer.py --data_path "$DATA_PATH" --image_path "$IMAGE_PATH"  --checkpoint ./checkpoints_clip/clip_imgflip_image_emotion_fusion.pt --use_image_emotion_fusion --topk_base 100   --num_samples 300 --sample_mode random --sample_seed 114514 --use_rerank_api   --jina_model_name jina-reranker-m0  --jina_api_key "$JINA_API_KEY" --subset_image_pool  > 1225_clip_jina_imgflip_infer.log 2>&1 &
# CUDA_VISIBLE_DEVICES=1 nohup python models/clip/clip_infer.py --data_path "$DATA_PATH" --image_path "$IMAGE_PATH" --checkpoint ./checkpoints_clip/clip_imgflip_no_fusion.pt > 1210_imgflip_clip.log 2>&1 &
# CUDA_VISIBLE_DEVICES=1 nohup python models/clip/clip_infer.py --rerank_model_path checkpoints_rerank/rerank_best_R@1_epoch_1_linear.pt > 1207_rerank_linear_infer.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup python models/rerank/rerank_train.py --use_clip_score  --lr 1e-5 --keep_all_best --epochs 10 --batch_size 128 --model_type mlp --loss_type lambdarank --label_mode inv_rank --clip_checkpoint ./checkpoints_clip/clip_imgflip.pt --topk_base 100 --extra_negatives 50 --wandb_project meme_rerank --wandb_entity wangtn --wandb_mode online > 1219_rerank_mlp_clipsim.log 2>&1 &


PYTORCH_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=1 nohup python models/rerank/jina_m0_lora_train.py   --device cuda:0   --data_path "$DATA_PATH"   --image_path "$IMAGE_PATH"   --save_dir ./checkpoints_rerank_score_only   --epochs 10   --batch_size 32   --topk_base 50   --lr 1e-6   --loss_type lambdarank   --label_mode inv_rank   --image_max_side 256   --qwen_min_side 32   --jina_micro_batch 25   --grad_checkpointing --use_jina_cache  --eval_before_train   --eval_sample_limit 1000   --wandb_project jina_rerank   --wandb_entity wangtn   --wandb_mode online   > 0105_jina_m0_score_train.log 2>&1 &

PYTORCH_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=1 nohup python models/rerank/jina_m0_lora_train.py   --device cuda:0   --data_path "$DATA_PATH"   --image_path "$IMAGE_PATH"   --save_dir ./checkpoints_rerank   --epochs 2   --batch_size 32   --topk_base 50   --lr 1e-5   --loss_type lambdarank   --label_mode inv_rank   --image_max_side 256   --qwen_min_side 32 --train_sample_limit 5000  --jina_micro_batch 25   --grad_checkpointing   --eval_sample_limit 1000   --wandb_project jina_rerank   --wandb_entity wangtn   --wandb_mode online   > 0105_jina_m0_lora_train.log 2>&1 &


PYTORCH_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0 nohup python models/clip/clip_infer.py   --data_path "$DATA_PATH"   --image_path "$IMAGE_PATH"   --checkpoint ./checkpoints_clip/clip_imgflip_image_emotion_fusion.pt   --use_image_emotion_fusion   --topk_base 50   --use_jina_lora   --jina_lora_path ./checkpoints_rerank/jina_m0_lora_epoch_2_pre_eval_jina_m0_lora_lambdarank_inv_rank.pt   --num_samples 500   --sample_mode random   --sample_seed 114 > 1230_jina_m0_lora_infer2.log 2>&1 &

# 注意：使用 --use_jina_cache 时，必须指定 --jina_lora_path 来加载训练好的 checkpoint
# 否则会使用未训练的 base model，导致性能下降
# 确保缓存是在训练好的模型上提取的（使用相同的 checkpoint）
PYTORCH_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0 nohup python models/clip/clip_infer.py   --data_path "$DATA_PATH"   --image_path "$IMAGE_PATH"   --checkpoint ./checkpoints_clip/clip_imgflip_image_emotion_fusion.pt   --use_image_emotion_fusion --use_jina_cache  --topk_base 50   --use_jina_lora   --jina_lora_path ./checkpoints_rerank_ddp/jina_m0_lora_epoch_2_pre_eval_jina_m0_lora_lambdarank_inv_rank.pt   --num_samples 1000   --sample_mode random   --sample_seed 114 > 0105_jina_m0_infer.log 2>&1 &

HF_ENDPOINT=https://hf-mirror.com CUDA_VISIBLE_DEVICES=1 nohup python models/clip/clip_infer.py --data_path "$DATA_PATH" --image_path "$IMAGE_PATH" --checkpoint ./checkpoints_clip/clip_imgflip_image_emotion_fusion.pt --use_image_emotion_fusion --topk_base 100 --use_jina_lora --jina_lora_path ./checkpoints_rerank/jina_m0_lora_best_R@1_jina_m0_lora_lambdarank_inv_rank.pt --num_samples 1000 --sample_mode random --sample_seed 114 > 1223_clip_jina_lora_infer.log 2>&1 &


CUDA_VISIBLE_DEVICES=1 nohup python models/lavis/train_blip.py --model_name blip_feature_extractor --model_type base --data_path ./imgflip_data/msrvtt --image_path ./imgflip_data/images --epochs 3 --lr 1e-5 --wandb_project clip4meme --wandb_entity wangtn > 1210_train_blip.log 2>&1 &
HF_ENDPOINT=https://hf-mirror.com WANDB_PROJECT=clip4meme WANDB_ENTITY=wangtn CUDA_VISIBLE_DEVICES=1 nohup python3 models/vsepp/train_vsepp.py --mode msrvtt --repo_root "$(pwd)/external/vsepp" --train_csv "$(pwd)/imgflip_data/new/train.csv" --train_json "$(pwd)/imgflip_data/msrvtt/train_data.json" --train_emotion "$(pwd)/imgflip_data/msrvtt/train_emotion.json" --val_json "$(pwd)/imgflip_data/msrvtt/test_data.json" --val_emotion "$(pwd)/imgflip_data/msrvtt/test_emotion.json" --image_dir "$(pwd)/imgflip_data/images" --max_violation --finetune --num_epochs 5 --val_step 500 > 1210_train2_vsepp.log 2>&1 &

