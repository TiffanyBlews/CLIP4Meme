# train.py

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
from tqdm import tqdm
import os
import argparse
from collections import defaultdict

from model import CLIP4Meme
from dataset import MSRVTT_Dataset

# 配置信息保持不变
CONFIG = {
    "data_path": "/root/zt/data_topics/input_file",
    "image_path": "/root/zt/data_topics/image",
    "train_csv": "train_video_id_9k.csv",
    "train_json": "train_9k.json",
    "train_emotion_json": "train_emotion.json",
    "test_json": "test_4k.json",
    "test_emotion_json": "test_emotion.json",
    
    "batch_size": 64, "epochs": 6, "lr": 1e-4, "coef_lr": 1e-3,
    "weight_decay": 0.01, "min_lr": 1e-8,
    
    "pretrained_clip_name": "ViT-B/32", "use_emotion_fusion": True, "seed": 42,

    "save_dir": "./checkpoints_final",
    # *** 修改点: 更新了阶段一 checkpoint 的默认路径 ***
    "stage1_checkpoint_path": "./checkpoints_final/stage1_qi_best.pth"
}

def symmetric_contrastive_loss(sim_matrix):
    labels = torch.arange(sim_matrix.size(0), device=sim_matrix.device)
    loss = (F.cross_entropy(sim_matrix, labels) + F.cross_entropy(sim_matrix.t(), labels)) / 2
    return loss

def eval_epoch(model, test_dataset, test_dataloader, device, stage, print_n_samples=5):
    """
    评估函数已更新，为阶段一（QI）实现完整、精确但较慢的评估逻辑。
    """
    model.eval()
    print(f"\n--- Evaluating Stage {stage} ({'QI-Accurate' if stage == 1 else 'QC'}) ---")
    
    # 1. 收集元数据 (与之前相同)
    all_video_ids_meta = [item['video_id'] for item in test_dataset.data]
    all_captions_meta = [item['query'] for item in test_dataset.data]
    all_titles_meta = [item['candidate_texts'][0] if item['candidate_texts'] else "" for item in test_dataset.data]
    caption_to_indices = defaultdict(list)
    for idx, cap in enumerate(all_captions_meta):
        caption_to_indices[cap].append(idx)
    
    # 2. 缓存所有查询和目标的特征
    all_q_feats, all_target_feats = [], []
    with torch.no_grad():
        # 步骤 2.1: 高效地一次性缓存所有查询文本的特征
        print("Caching all query features...")
        for batch in tqdm(test_dataloader, desc="Caching Queries"):
            query_ids = batch['query_ids'].to(device)
            q_feat = model.encode_text_pooled(query_ids)
            all_q_feats.append(q_feat.cpu())
        all_q_feats = torch.cat(all_q_feats, dim=0)

        # 步骤 2.2: 根据不同阶段计算目标特征
        print(f"Calculating all target features for Stage {stage}...")
        if stage == 1: # 评估 QI 分支 (精确但缓慢的路径)
            # 我们需要为每个图文对计算其完整的融合特征
            for batch in tqdm(test_dataloader, desc="Calculating Fused QI Features"):
                # *** 这是核心修改：我们直接调用模型 stage1_qi 的前向传播 ***
                # *** 以获取与训练时完全一致的、经过Co-attention的特征 ***
                sim_matrix_batch = model(
                    query_ids=batch['query_ids'].to(device), # 传入以满足函数签名，但内部不会用于最终输出
                    candidate_ids=batch['candidate_ids'].to(device),
                    image=batch['image'].to(device),
                    emotion_ids=batch['emotion_input_ids'].to(device),
                    emotion_mask=batch['emotion_attention_mask'].to(device),
                    training_stage='stage1_qi'
                )
                # 这里的 sim_matrix_batch 是 Query vs Fused(Image,Context)
                # 为了得到独立的 Fused(Image,Context) 特征，我们需要分解它
                # 一个简化的做法是假设logit_scale和归一化效果可以近似抵消
                # 或者，更严谨地，我们需要修改模型返回融合后的特征
                # 为简单起见，我们直接在评估时重新计算融合特征
                
                # --- 重复模型内部逻辑来获取融合特征 ---
                image_feat = model.encode_image(batch['image'].to(device))
                candidate_feat_tokens, candidate_mask = model.encode_text_tokens(batch['candidate_ids'].to(device))
                if batch['emotion_input_ids'] is not None:
                    emotion_feat_bert = model.encode_emotion(batch['emotion_input_ids'].to(device), batch['emotion_attention_mask'].to(device))
                    projected_emotion_feat = model.emotion_projection(emotion_feat_bert.type(model.emotion_projection.weight.dtype))
                    image_feat = image_feat + projected_emotion_feat
                image_feat_seq = F.normalize(image_feat, dim=-1).unsqueeze(1)
                image_mask_for_attention = torch.ones(image_feat_seq.size(0), 1, 1, 1).to(device)
                
                fused_image_feat, _ = model.co_attention(
                    image_feat_seq, image_mask_for_attention,
                    candidate_feat_tokens, candidate_mask
                )
                final_fused_feat = fused_image_feat.squeeze(1)
                all_target_feats.append(final_fused_feat.cpu())

        elif stage == 2: # 评估 QC 分支 (逻辑不变，本身就是高效的)
            for batch in tqdm(test_dataloader, desc="Caching QC Features"):
                candidate_ids = batch['candidate_ids'].to(device)
                c_feat = model.encode_text_pooled(candidate_ids)
                all_target_feats.append(c_feat.cpu())
        
        all_target_feats = torch.cat(all_target_feats, dim=0)

    # 3. 计算总相似度矩阵并评估
    all_q_feats = F.normalize(all_q_feats, dim=-1)
    all_target_feats = F.normalize(all_target_feats, dim=-1)
    sim_matrix = torch.matmul(all_q_feats, all_target_feats.t())
    
    # 4. R@k 计算和详细打印逻辑 (与之前相同)
    # ... (省略与上一版完全相同的 R@k 计算和打印代码) ...
    print("\n--- Computing R@k & Detailed Predictions ---")
    num_queries = sim_matrix.shape[0]
    recalls = {}
    for k in [1, 5, 10]:
        hits = 0
        for q_idx in range(num_queries):
            top_preds = torch.topk(sim_matrix[q_idx], k=k).indices.tolist()
            gt_indices = caption_to_indices[all_captions_meta[q_idx]]
            if any(p in gt_indices for p in top_preds):
                hits += 1
            if q_idx < print_n_samples and k == 10:
                print(f"\n{'='*22} Sample {q_idx+1}/{print_n_samples} {'='*22}")
                print(f"  [Query]      VID: {all_video_ids_meta[q_idx]}, Caption: {all_captions_meta[q_idx]}")
                print(f"  [GT Indices] {gt_indices}")
                print("  [Top-5 Predictions]")
                top5 = torch.topk(sim_matrix[q_idx], k=5)
                for rank, pred_idx in enumerate(top5.indices.tolist()):
                    mark, score = "✔️" if pred_idx in gt_indices else "❌", top5.values[rank]
                    print(f"    {rank+1}. {mark} score={score:.4f} | VID={all_video_ids_meta[pred_idx]} | Cap: {all_captions_meta[pred_idx]}")
        recalls[f'R@{k}'] = (hits / num_queries) * 100
    print("\n--- Final Evaluation Results ---")
    for k, v in recalls.items(): print(f"  {k}: {v:.2f}%")
    return recalls
    
def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.save_dir, exist_ok=True)
    
    stage_name = 'QI' if args.stage == 1 else 'QC'
    print(f"--- Starting Training for Stage {args.stage}: {stage_name} ---")

    # 1. 初始化模型和数据
    model = CLIP4Meme(pretrained_clip_name=args.pretrained_clip_name)
    model.to(device)
    
    train_dataset = MSRVTT_Dataset(
        # ... (数据加载参数与之前相同) ...
        csv_path=os.path.join(args.data_path, args.train_csv),
        json_path=os.path.join(args.data_path, args.train_json),
        features_path=args.image_path,
        emotion_json_path=os.path.join(args.data_path, args.train_emotion_json),
        bert_tokenizer=model.get_bert_tokenizer(),
        clip_preprocess=model.clip_preprocess,
        is_train=True,
        load_image=(args.stage == 1) # 阶段一加载图片，阶段二不加载
    )
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    
    test_dataset = MSRVTT_Dataset(
        # ... (测试数据加载参数与之前相同) ...
        csv_path=os.path.join(args.data_path, args.test_json),
        json_path=os.path.join(args.data_path, args.test_json),
        features_path=args.image_path,
        emotion_json_path=os.path.join(args.data_path, args.test_emotion_json),
        bert_tokenizer=model.get_bert_tokenizer(),
        clip_preprocess=model.clip_preprocess,
        is_train=False,
        load_image=True # 评估时总是加载图片，以备阶段一评估
    )
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # 2. 优化器 (与之前相同)
    # ... (省略与之前相同的 optimizer 定义代码) ...
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.clip.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay, 'lr': args.lr * args.coef_lr},
        {'params': [p for n, p in model.clip.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0, 'lr': args.lr * args.coef_lr},
        {'params': [p for n, p in model.bert_model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay, 'lr': 0 if args.stage == 2 else args.lr * args.coef_lr},
        {'params': [p for n, p in model.bert_model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0, 'lr': 0 if args.stage == 2 else args.lr * args.coef_lr},
        {'params': [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith(('clip.', 'bert_model.')) and not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay, 'lr': args.lr},
        {'params': [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith(('clip.', 'bert_model.')) and any(nd in n for nd in no_decay)], 'weight_decay': 0.0, 'lr': args.lr}
    ]
    optimizer = optim.AdamW(optimizer_grouped_parameters, lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.min_lr)

    # 3. 加载阶段一模型 (仅在阶段二)
    if args.stage == 2:
        if not os.path.exists(args.stage1_checkpoint_path):
            raise FileNotFoundError(f"Stage 2 requires Stage 1 checkpoint: {args.stage1_checkpoint_path}")
        print(f"--- Loading Stage 1 (QI) checkpoint for Stage 2 (QC) training ---")
        checkpoint = torch.load(args.stage1_checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])

    # 4. 训练循环
    best_r1 = 0.0
    for epoch in range(args.epochs):
        model.train()
        pbar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{args.epochs} [{stage_name}]")
        for batch in pbar:
            optimizer.zero_grad()
            
            # *** 修改点: 调用模型时传入正确的参数和 stage name ***
            sim_matrix = model(
                query_ids=batch['query_ids'].to(device),
                candidate_ids=batch['candidate_ids'].to(device),
                image=batch['image'].to(device),
                emotion_ids=batch['emotion_input_ids'].to(device),
                emotion_mask=batch['emotion_attention_mask'].to(device),
                training_stage=f'stage{args.stage}_{stage_name.lower()}'
            )
            
            loss = symmetric_contrastive_loss(sim_matrix)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            pbar.set_postfix({"Loss": f"{loss.item():.4f}"})
        
        scheduler.step()
        
        # 评估和保存
        eval_results = eval_epoch(model, test_dataset, test_dataloader, device, args.stage)
        if eval_results.get('R@1', 0) > best_r1:
            best_r1 = eval_results.get('R@1', 0)
            # *** 修改点: 更新保存路径 ***
            save_path = os.path.join(args.save_dir, f'stage{args.stage}_{stage_name.lower()}_best.pth')
            print(f"*** New best R@1: {best_r1:.2f}%. Saving model to {save_path} ***")
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict()}, save_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train Meme-Retrieval model based on paper's QI/QC branches.")
    parser.add_argument('--stage', type=int, required=True, choices=[1, 2], help="1 for QI-branch, 2 for QC-branch.")
    for key, value in CONFIG.items():
        parser.add_argument(f'--{key}', type=type(value) if value is not None else str, default=value)
    parsed_args = parser.parse_args()
    main(parsed_args)