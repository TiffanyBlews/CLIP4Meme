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

CONFIG = {
    "data_path": "/root/zt/imgflip_results/msrvtt",
    "image_path": "/root/zt/imgflip_results/images",
    "train_csv": "train_ids.csv",
    "train_json": "train_data.json",
    "train_emotion_json": "train_emotion.json",
    "test_json": "test_data.json",
    "test_emotion_json": "test_emotion.json",
    
    "batch_size": 64, "epochs": 20, "lr": 5e-4, "coef_lr": 1e-3,
    "weight_decay": 0.01, "min_lr": 0,
    
    "pretrained_clip_name": "ViT-B/32", "use_emotion_fusion": True, "seed": 42,

    "save_dir": "./checkpoints_imgflip",
    "checkpoint_path": "./checkpoints_imgflip/stage2_qc_best.pth"
}

def symmetric_contrastive_loss(sim_matrix):
    labels = torch.arange(sim_matrix.size(0), device=sim_matrix.device)
    loss = (F.cross_entropy(sim_matrix, labels) + F.cross_entropy(sim_matrix.t(), labels)) / 2
    return loss

def eval_epoch(model, test_dataset, test_dataloader, device, branch, print_n_samples=5):
    model.eval()
    print(f"\n--- Evaluating {branch} branch ---")
    
    # 1. 收集元数据
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

        # 步骤 2.2: 根据不同分支计算目标特征
        print(f"Calculating all target features for {branch} branch...")
        if branch == 'QI': # 评估 QI 分支 (使用return_features避免重复计算)
            # 使用模型的return_features功能，避免重复计算
            for batch in tqdm(test_dataloader, desc="Calculating Fused QI Features"):
                query_feat, fused_feat = model(
                    query_ids=batch['query_ids'].to(device),
                    candidate_ids=batch['candidate_ids'].to(device),
                    image=batch['image'].to(device),
                    emotion_ids=batch['emotion_input_ids'].to(device),
                    emotion_mask=batch['emotion_attention_mask'].to(device),
                    branch='qi',
                    return_features=True
                )
                # 直接使用模型返回的融合特征，无需重复计算
                all_target_feats.append(fused_feat.cpu())

        elif branch == 'QC': # 评估 QC 分支 (使用return_features保持一致性)
            for batch in tqdm(test_dataloader, desc="Caching QC Features"):
                query_feat, candidate_feat = model(
                    query_ids=batch['query_ids'].to(device),
                    candidate_ids=batch['candidate_ids'].to(device),
                    image=batch['image'].to(device),  # QC不需要图片，但为了保持接口一致
                    emotion_ids=batch['emotion_input_ids'].to(device),
                    emotion_mask=batch['emotion_attention_mask'].to(device),
                    branch='qc',
                    return_features=True
                )
                # 直接使用模型返回的特征，保持与QI分支的一致性
                all_target_feats.append(candidate_feat.cpu())
        
        all_target_feats = torch.cat(all_target_feats, dim=0)

    # 3. 计算总相似度矩阵并评估
    all_q_feats = F.normalize(all_q_feats, dim=-1)
    all_target_feats = F.normalize(all_target_feats, dim=-1)
    sim_matrix = torch.matmul(all_q_feats, all_target_feats.t())
    
    # 4. R@k 计算和详细打印逻辑
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
    print(args)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.save_dir, exist_ok=True)
    
    branch_name = args.train.upper()
    print(f"--- Starting Training for {branch_name} branch ---")

    # 1. 初始化模型和数据
    model = CLIP4Meme(pretrained_clip_name=args.pretrained_clip_name)
    model.to(device)
    
    train_dataset = MSRVTT_Dataset(
        csv_path=os.path.join(args.data_path, args.train_csv),
        json_path=os.path.join(args.data_path, args.train_json),
        features_path=args.image_path,
        emotion_json_path=os.path.join(args.data_path, args.train_emotion_json),
        bert_tokenizer=model.get_bert_tokenizer(),
        clip_preprocess=model.clip_preprocess,
        is_train=True,
        load_image=(args.train.lower() == 'qi') # QI分支加载图片，QC分支不加载
    )
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    
    test_dataset = MSRVTT_Dataset(
        csv_path=os.path.join(args.data_path, args.test_json),
        json_path=os.path.join(args.data_path, args.test_json),
        features_path=args.image_path,
        emotion_json_path=os.path.join(args.data_path, args.test_emotion_json),
        bert_tokenizer=model.get_bert_tokenizer(),
        clip_preprocess=model.clip_preprocess,
        is_train=False,
        load_image=True # 评估时总是加载图片，以备QI分支评估
    )
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # 2. 优化器 
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.clip.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay, 'lr': args.lr * args.coef_lr},
        {'params': [p for n, p in model.clip.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0, 'lr': args.lr * args.coef_lr},
        {'params': [p for n, p in model.bert_model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay, 'lr': 0 if args.train.lower() == 'qc' else args.lr * args.coef_lr},
        {'params': [p for n, p in model.bert_model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0, 'lr': 0 if args.train.lower() == 'qc' else args.lr * args.coef_lr},
        {'params': [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith(('clip.', 'bert_model.')) and not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay, 'lr': args.lr},
        {'params': [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith(('clip.', 'bert_model.')) and any(nd in n for nd in no_decay)], 'weight_decay': 0.0, 'lr': args.lr}
    ]
    optimizer = optim.AdamW(optimizer_grouped_parameters, lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.min_lr)

    # 3. 加载检查点 (QI和QC分支共享相同的检查点)
    if os.path.exists(args.checkpoint_path):
        print(f"--- Loading checkpoint for {branch_name} branch training ---")
        checkpoint = torch.load(args.checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        print("--- No checkpoint found, starting from scratch ---")

    # 4. 训练循环
    best_r1 = 0.0
    for epoch in range(args.epochs):
        model.train()
        pbar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{args.epochs} [{branch_name}]")
        for batch in pbar:
            optimizer.zero_grad()
            
            # 调用模型时传入正确的分支名称
            sim_matrix = model(
                query_ids=batch['query_ids'].to(device),
                candidate_ids=batch['candidate_ids'].to(device),
                image=batch['image'].to(device),
                emotion_ids=batch['emotion_input_ids'].to(device),
                emotion_mask=batch['emotion_attention_mask'].to(device),
                branch=branch_name.lower()
            )
            
            loss = symmetric_contrastive_loss(sim_matrix)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            pbar.set_postfix({"Loss": f"{loss.item():.4f}"})
        
        scheduler.step()
        
        # 评估和保存
        eval_results = eval_epoch(model, test_dataset, test_dataloader, device, branch_name)
        if eval_results.get('R@1', 0) > best_r1:
            best_r1 = eval_results.get('R@1', 0)
            
            # 构建新的检查点文件名：原文件名 + 当前分支名
            base_name = os.path.splitext(os.path.basename(args.checkpoint_path))[0]
            if base_name == "checkpoint":  # 如果是默认名称，直接使用分支名
                new_checkpoint_name = f"{branch_name.lower()}.pth"
            else:  # 否则在原文件名后添加当前分支名
                new_checkpoint_name = f"{base_name}_{branch_name.lower()}.pth"
            
            save_path = os.path.join(args.save_dir, new_checkpoint_name)
            print(f"*** New best R@1: {best_r1:.2f}%. Saving model to {save_path} ***")
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict()}, save_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train Meme-Retrieval model based on QI/QC branches.")
    parser.add_argument('--train', type=str, required=True, choices=['qi', 'qc'], help="Train QI or QC branch")
    for key, value in CONFIG.items():
        parser.add_argument(f'--{key}', type=type(value) if value is not None else str, default=value)
    parsed_args = parser.parse_args()
    main(parsed_args)