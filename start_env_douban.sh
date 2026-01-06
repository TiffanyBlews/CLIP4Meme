docker exec -it cuda_1111 bash
conda activate neo_meme
cd ljj

# DATA_PATH="./douban_data/msrvtt"
# IMAGE_PATH="./douban_data/images"
DATA_PATH="./data/input_file"
IMAGE_PATH="./data/image"
CUDA_VISIBLE_DEVICES=1 nohup python models/IEF/train.py --train qi --data_path "$DATA_PATH" --image_path "$IMAGE_PATH" --save_dir ief_douban  --wandb_project clip4meme --wandb_entity wangtn > 1211_qi_douban.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup python models/IEF/train.py --train qc --data_path "$DATA_PATH" --image_path "$IMAGE_PATH" --save_dir ief_douban  --checkpoint_path  checkpoints_douban/new_qc_qi.pth --wandb_project clip4meme --wandb_entity wangtn > 1211_qc_douban.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup python models/IEF/inference_fusion.py --multi_alpha --checkpoint checkpoints_douban/stage2_qc_best.pth --data_path "$DATA_PATH" --image_path "$IMAGE_PATH" > 1211_fusion_douban.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup python models/clip/clip_train.py --data_path "$DATA_PATH" --image_path "$IMAGE_PATH" --save_dir clip_douban --model_save_name clip_douban.pt > 1211_clip_train_douban.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup python models/clip/clip_infer.py --data_path "$DATA_PATH" --image_path "$IMAGE_PATH"  --save_dir checkpoints_douban --checkpoint ./checkpoints_clip/clip_douban_image_emotion_fusion.pt --use_image_emotion_fusion  > 111_infer_douban.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup python models/clip/clip_infer.py --data_path "$DATA_PATH" --image_path "$IMAGE_PATH"  --save_dir clip_douban --checkpoint clip_douban/clip_douban.pt   > 111_infer_douban_vanilla_clip.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup python models/clip/clip_infer.py  --data_path "$DATA_PATH" --image_path "$IMAGE_PATH"  --zero_shot --save_dir ./checkpoints_rerank_douban > 1209_clip_zero_shot_douban.log 2>&1 &
# CUDA_VISIBLE_DEVICES=1 nohup python models/clip/clip_infer.py --data_path "$DATA_PATH" --image_path "$IMAGE_PATH" --checkpoint ./checkpoints_clip/clip_douban_no_fusion.pt > 1210_douban_clip.log 2>&1 &
# CUDA_VISIBLE_DEVICES=1 nohup python models/clip/clip_infer.py --rerank_model_path checkpoints_rerank/rerank_best_R@1_epoch_1_linear.pt > 1207_rerank_linear_infer.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup python models/lavis/train_blip.py --model_name blip_feature_extractor --model_type base --data_path ./imgflip_data/msrvtt --image_path ./imgflip_data/images --epochs 3 --lr 1e-5 --wandb_project clip4meme --wandb_entity wangtn > 1210_train_blip.log 2>&1 &
HF_ENDPOINT=https://hf-mirror.com WANDB_PROJECT=clip4meme WANDB_ENTITY=wangtn CUDA_VISIBLE_DEVICES=1 nohup python3 models/vsepp/train_vsepp.py --mode msrvtt --repo_root "$(pwd)/external/vsepp" --train_csv "$(pwd)/imgflip_data/new/train.csv" --train_json "$(pwd)/imgflip_data/msrvtt/train_data.json" --train_emotion "$(pwd)/imgflip_data/msrvtt/train_emotion.json" --val_json "$(pwd)/imgflip_data/msrvtt/test_data.json" --val_emotion "$(pwd)/imgflip_data/msrvtt/test_emotion.json" --image_dir "$(pwd)/imgflip_data/images" --max_violation --finetune --num_epochs 5 --val_step 500 > 1210_train2_vsepp.log 2>&1 &

docker exec -it cuda_1111 bash
conda activate lavis
cd ljj
HF_ENDPOINT=https://hf-mirror.com CUDA_VISIBLE_DEVICES=1 nohup python models/lavis/train_blip.py --model_name blip_retrieval --data_path ./imgflip_data/msrvtt --image_path ./imgflip_data/images --epochs 3 --lr 1e-5 --wandb_project clip4meme --wandb_entity wangtn > 1210_train_blip.log 2>&1 &

HF_ENDPOINT=https://hf-mirror.com CUDA_VISIBLE_DEVICES=1 nohup python models/lavis/train_albef.py --model_name albef_retrieval  --data_path ./imgflip_data/msrvtt --image_path ./imgflip_data/images --epochs 3 --lr 1e-5 --wandb_project clip4meme  > 1210_train_albef.log 2>&1 &
HF_ENDPOINT=https://hf-mirror.com WANDB_PROJECT=visualbert4meme WANDB_ENTITY=wangtn CUDA_VISIBLE_DEVICES=1 nohup python models/visualbert/visualbert_train.py --data_path ./imgflip_data/msrvtt --image_path ./imgflip_data/images --epochs 5 --batch_size 64 --lr 5e-6 --save_dir ./checkpoints_visualbert > 1210_train_visualbert.log 2>&1 &
HF_ENDPOINT=https://hf-mirror.com CUDA_VISIBLE_DEVICES=1 nohup python models/visualbert/visualbert_infer.py --checkpoint_path ./checkpoints_visualbert/visualbert_imgflip.pt --data_path ./imgflip_data/msrvtt --image_path ./imgflip_data/images --batch_size 64 --visual_seq_len 10 > 1210_infer_visualbert.log 2>&1 &
