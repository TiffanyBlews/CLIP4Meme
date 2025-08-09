# train2.py (已适配 EFB-V2)

from collections import defaultdict
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import os
from torch.optim.lr_scheduler import CosineAnnealingLR

# <--- 适配 EFB-V2: 导入新的模型
from model_efb_v2 import EFB_V3
from dataset import MSRVTT_Dataset

CONFIG = {
    # Path settings
    "data_path": "/root/zt/data_topics/input_file",

    # Training hyperparameters
    "batch_size": 64,
    "batch_size_val": 128,
    "lr": 1e-4,
    "coef_lr": 1e-3,
    "epochs": 6,
    "max_grad_norm": 1.0,
    "min_lr": 1e-8,

    # Data parameters
    "max_words": 32,
    "max_frames": 1,
    "image_resolution": 224,
    "num_workers": 4,

    # Model parameters
    "pretrained_clip_name": "ViT-B/32",
    # <--- 适配 EFB-V2: 新增模型行为开关，False 对应 clip4clip 等价模式
    "use_emotion_fusion": True,
    # Seed
    "seed": 42,

    # Checkpoint / Eval settings
    "checkpoint_path": None,
    "eval_only": False,
    "save_dir": "./checkpoints"
}


# ------------------------------------------------------------------
# 工具函数
# ------------------------------------------------------------------
def symmetric_contrastive_loss(sim_matrix):
    """Symmetric Cross-Entropy loss (InfoNCE)"""
    batch_size = sim_matrix.size(0)
    labels = torch.arange(batch_size, device=sim_matrix.device)
    loss_rows = F.cross_entropy(sim_matrix, labels)
    loss_cols = F.cross_entropy(sim_matrix.t(), labels)
    return (loss_rows + loss_cols) / 2.0


def eval_epoch(model, test_dataset, test_dataloader, device, use_emotion_fusion, print_n_samples=5):
    # <--- 适配 EFB-V2: 整个函数重构以支持双模态，并保持全局R@k评估逻辑
    model.eval()

    # 1. 收集元数据 (不变)
    print("--- Collecting metadata ---")
    all_video_ids_meta = [item['video_id'] for item in test_dataset.data]
    all_captions_meta = [item['query'] for item in test_dataset.data]
    all_titles_meta = [item['context'] for item in test_dataset.data]

    caption_to_indices = defaultdict(list)
    for idx, cap in enumerate(all_captions_meta):
        caption_to_indices[cap].append(idx)
    print(f"Found {len(caption_to_indices)} unique captions.")

    # 2. 缓存基础特征
    print("--- Caching base features ---")
    all_query_feats_final = []
    all_image_feats = []
    
    # 根据模式决定是否缓存额外特征
    cached_context_tokens = [] if not use_emotion_fusion else None
    cached_emotion_feats = [] if use_emotion_fusion else None

    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Caching"):
            image = batch['image'].to(device)
            query = batch['query'].to(device)
            
            all_query_feats_final.append(model.encode_text_final_feature(query).cpu())
            all_image_feats.append(model.encode_image(image).cpu())

            if use_emotion_fusion:
                e_ids = batch['emotion_input_ids'].to(device)
                e_mask = batch['emotion_attention_mask'].to(device)
                e_bert = model.encode_emotion(e_ids, e_mask)
                cached_emotion_feats.append(model.emotion_projection(e_bert).cpu())
            else:
                context_tokens, _ = model.encode_text_all_tokens(batch['context'].to(device))
                cached_context_tokens.append(context_tokens.cpu())

    all_query_feats_final = torch.cat(all_query_feats_final, dim=0)
    all_image_feats = torch.cat(all_image_feats, dim=0)
    all_query_feats_final = F.normalize(all_query_feats_final, dim=-1)
    all_image_feats = F.normalize(all_image_feats, dim=-1)

    # 3. 计算最终的相似度矩阵
    print("--- Computing final similarity matrix ---")
    if use_emotion_fusion:
        # 模式一: 情感融合逻辑
        print("Using Emotion-Fusion mode for evaluation.")
        all_emotion_feats = torch.cat(cached_emotion_feats, dim=0)
        all_emotion_feats = F.normalize(all_emotion_feats, dim=-1)
        final_image_features = F.normalize(all_image_feats + all_emotion_feats, dim=-1)
        final_sim = torch.matmul(all_query_feats_final, final_image_features.t())
    else:
        # 模式二: clip4clip 等价逻辑
        print("Using clip4clip-equivalent (Co-attention) mode for evaluation.")
        all_context_tokens = torch.cat(cached_context_tokens, dim=0)
        
        # 准备 co-attention 输入
        img_seq = all_image_feats.unsqueeze(1).to(device)
        ctx_seq = all_context_tokens.to(device)
        img_mask = torch.ones(img_seq.size(0), 1, 1, 1).to(device)
        ctx_mask_raw = (ctx_seq != 0).any(dim=-1).float() # check for non-zero vectors
        ctx_mask = (1.0 - ctx_mask_raw.unsqueeze(1).unsqueeze(2)) * -10000.0

        # 在 GPU 上执行 Co-attention (可能需要分批处理以防OOM)
        with torch.no_grad():
             fused_image_feat, _ = model.co_attention(img_seq, img_mask, ctx_seq, ctx_mask)

        final_image_features = fused_image_feat.squeeze(1).cpu()
        final_image_features = F.normalize(final_image_features, dim=-1)
        final_sim = torch.matmul(all_query_feats_final, final_image_features.t())

    # 4. 计算 R@k (逻辑不变)
    print("\n--- Computing R@k & Detailed Predictions ---")
    num_queries = final_sim.shape[0]
    recalls = {}

    for k in [1, 5, 10, 50]:
        hits = 0
        for q_idx in range(num_queries):
            top_preds = torch.topk(final_sim[q_idx], k=k).indices.tolist()
            gt_indices = caption_to_indices[all_captions_meta[q_idx]]
            hits += any(p in gt_indices for p in top_preds)
            
            # ---- 详细打印 ----
            if q_idx < print_n_samples:
                print(f"\n{'='*22} Sample {q_idx+1}/{print_n_samples} {'='*22}")
                print(f"[Query]      VID: {all_video_ids_meta[q_idx]}, Cap: {all_captions_meta[q_idx]}, Title: {all_titles_meta[q_idx]}")
                print(f"[GT indices] {gt_indices}")
                print("[Top-5 Predictions]")
                top5 = torch.topk(final_sim[q_idx], k=5)
                for rank, pred_idx in enumerate(top5.indices.tolist()):
                    mark = "✔️" if pred_idx in gt_indices else "❌"
                    print(f"  {rank+1}. {mark} score={top5.values[rank]:.4f}  VID={all_video_ids_meta[pred_idx]}, Cap: {all_captions_meta[pred_idx]}, Title: {all_titles_meta[pred_idx]}")
        recalls[f'R@{k}'] = (hits / num_queries) * 100

    print("\n--- Final R@k ---")
    for k, v in recalls.items():
        print(f"{k}: {v:.2f}%")
    return recalls

# ------------------------------------------------------------------
# 主函数
# ------------------------------------------------------------------
def main():
    torch.manual_seed(CONFIG['seed'])
    np.random.seed(CONFIG['seed'])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # <--- 适配 EFB-V2: 实例化新模型，移除无用参数
    print(f"Initializing EFB-V2 model. Emotion fusion mode: {CONFIG['use_emotion_fusion']}")
    model = EFB_V3(
        pretrained_clip_name=CONFIG['pretrained_clip_name'],
    )
    model.to(device)

    # 2. 数据
    clip_preprocess = model.clip_preprocess
    bert_tokenizer = model.get_bert_tokenizer()

    train_dataset = MSRVTT_Dataset(
        csv_path=os.path.join(CONFIG['data_path'], "train_video_id_9k.csv"),
        json_path=os.path.join(CONFIG['data_path'], "train_9k.json"),
        features_path=os.path.join(CONFIG['data_path'], "../image"),
        emotion_json_path=os.path.join(CONFIG['data_path'], "train_emotion.json"),
        bert_tokenizer=bert_tokenizer,
        clip_preprocess=clip_preprocess,
        max_words=CONFIG['max_words'],
        is_train=True
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=True,
        num_workers=CONFIG['num_workers']
    )

    test_dataset = MSRVTT_Dataset(
        csv_path=os.path.join(CONFIG['data_path'], "test_4k.json"),
        json_path=os.path.join(CONFIG['data_path'], "test_4k.json"),
        features_path=os.path.join(CONFIG['data_path'], "../image"),
        emotion_json_path=os.path.join(CONFIG['data_path'], "test_emotion.json"),
        bert_tokenizer=bert_tokenizer,
        clip_preprocess=clip_preprocess,
        max_words=CONFIG['max_words'],
        is_train=False
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=CONFIG['batch_size_val'],
        shuffle=False,
        num_workers=CONFIG['num_workers']
    )

    # 3. 优化器
    # clip_params = list(model.clip.parameters())
    # bert_params = list(model.bert_model.parameters())
    # other_params = [p for p in model.parameters() if id(p) not in [id(q) for q in clip_params + bert_params]]
    # optimizer = optim.AdamW([
    #     {'params': clip_params, 'lr': CONFIG['lr'] * CONFIG['coef_lr']},
    #     {'params': bert_params, 'lr': 0},
    #     {'params': other_params, 'lr': CONFIG['lr']}
    # ])
# -------------------- 这是新的、修改后的代码 (请用它替换) --------------------
    # 3. 优化器 (精细化权重衰减设置)
    print("--- Setting up optimizer with fine-grained weight decay ---")

    # 首先，在你的 CONFIG 字典中添加一个 weight_decay 值
    # 例如: "weight_decay": 0.01,
    CONFIG.setdefault("weight_decay", 0.01) # 如果未设置，则默认为 0.01

    # 定义不应进行权重衰减的参数名称关键词
    no_decay = ["bias", "LayerNorm.weight"]

    # 将模型的参数按模块和是否需要衰减进行分组
    # EFB-V3 模型包含 'clip', 'bert_model', 'emotion_projection', 'co_attention'
    clip_named_params = model.clip.named_parameters()
    bert_named_params = model.bert_model.named_parameters()
    
    # 获取 "other" 参数的名称
    other_modules_prefixes = ('emotion_projection.', 'co_attention.')
    other_named_params = [(n, p) for n, p in model.named_parameters() if n.startswith(other_modules_prefixes)]

    # 创建6个参数组
    optimizer_grouped_parameters = [
        # CLIP - 需要衰减 (应用 coef_lr 和 weight_decay)
        {
            'params': [p for n, p in clip_named_params if not any(nd in n for nd in no_decay)],
            'weight_decay': CONFIG['weight_decay'],
            'lr': CONFIG['lr'] * CONFIG['coef_lr']
        },
        # CLIP - 不需要衰减 (应用 coef_lr，但不应用 weight_decay)
        {
            'params': [p for n, p in clip_named_params if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0,
            'lr': CONFIG['lr'] * CONFIG['coef_lr']
        },
        # BERT - 需要衰减 (应用 coef_lr 和 weight_decay)
        {
            'params': [p for n, p in bert_named_params if not any(nd in n for nd in no_decay)],
            'weight_decay': CONFIG['weight_decay'],
            'lr': CONFIG['lr'] * CONFIG['coef_lr']
        },
        # BERT - 不需要衰减 (应用 coef_lr，但不应用 weight_decay)
        {
            'params': [p for n, p in bert_named_params if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0,
            'lr': CONFIG['lr'] * CONFIG['coef_lr']
        },
        # Other (新模块) - 需要衰减 (使用基础 lr 和 weight_decay)
        {
            'params': [p for n, p in other_named_params if not any(nd in n for nd in no_decay)],
            'weight_decay': CONFIG['weight_decay'],
            'lr': CONFIG['lr']
        },
        # Other (新模块) - 不需要衰减 (使用基础 lr，但不应用 weight_decay)
        {
            'params': [p for n, p in other_named_params if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0,
            'lr': CONFIG['lr']
        }
    ]

    optimizer = optim.AdamW(optimizer_grouped_parameters)
    print(f"--- Optimizer setup complete. Total groups: {len(optimizer.param_groups)} ---")

    # <--- 新增: 4. 学习率调度器
    scheduler = CosineAnnealingLR(
        optimizer, T_max=CONFIG['epochs'], eta_min=CONFIG['min_lr']
    )

    # 4. 加载 checkpoint（若指定）
    start_epoch = 0
    ckpt_path = CONFIG.get("checkpoint_path")
    if ckpt_path and os.path.isfile(ckpt_path):
        print(f"--- Loading checkpoint from: {ckpt_path} ---")
        checkpoint = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        if not CONFIG.get("eval_only", False):
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            # <--- 新增: 加载 scheduler 状态
            if 'scheduler_state_dict' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                print("--- Scheduler state loaded successfully! ---")
            start_epoch = checkpoint.get('epoch', -1) + 1
        print("--- Checkpoint loaded successfully! ---")

    # 6. 仅评估模式
    if CONFIG.get("eval_only", False):
        if not CONFIG["checkpoint_path"]:
            print("Error: eval_only=True requires checkpoint_path to be set.")
            return
        print("--- Running in Evaluation-Only Mode ---")
        # <--- 适配 EFB-V2: 更新 eval_epoch 的调用
        eval_epoch(model, test_dataset, test_dataloader, device, CONFIG['use_emotion_fusion'])
        return

    # 7. 训练循环
    best_r1 = 0.0
    os.makedirs(CONFIG['save_dir'], exist_ok=True)

    for epoch in range(start_epoch, CONFIG['epochs']):
        print(f"\n--- Epoch {epoch + 1}/{CONFIG['epochs']} ---")
        model.train()
        running_loss = 0.0

        pbar = tqdm(train_dataloader, desc=f"Training Epoch {epoch + 1}")
        for i, batch in enumerate(pbar):
            image = batch['image'].to(device)
            query = batch['query'].to(device)
            context = batch['context'].to(device)
            emotion_ids = batch['emotion_input_ids'].to(device)
            emotion_mask = batch['emotion_attention_mask'].to(device)

            optimizer.zero_grad()

            # <--- 适配 EFB-V2: 更新模型调用和损失计算
            # 现在 model 直接返回最终的相似度矩阵
            sim_matrix = model(
                image, query, context, emotion_ids, emotion_mask, 
                use_emotion_fusion=CONFIG['use_emotion_fusion']
            )

            # 损失计算被简化
            total_loss = symmetric_contrastive_loss(sim_matrix)

            if torch.isnan(total_loss):
                print(f"Warning: NaN loss at batch {i}")
                continue

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG['max_grad_norm'])
            optimizer.step()

            running_loss += total_loss.item()
            pbar.set_postfix({"Loss": f"{total_loss.item():.4f}", "LR": f"{optimizer.param_groups[0]['lr']}"})

        avg_loss = running_loss / len(train_dataloader)
        print(f"Epoch {epoch + 1} Average Training Loss: {avg_loss:.4f}")

        # 8. 评估 & 保存最佳
        # <--- 适配 EFB-V2: 更新 eval_epoch 的调用
        eval_results = eval_epoch(model, test_dataset, test_dataloader, device, CONFIG['use_emotion_fusion'])
        current_r1 = eval_results.get('R@1', 0)
        if current_r1 > best_r1:
            best_r1 = current_r1
            print(f"*** New best R@1: {best_r1:.2f}%. Saving checkpoint... ***")
            save_path = os.path.join(CONFIG['save_dir'], 'best_model.pth')
            # ... (保存逻辑不变) ...
            
        # 9. 更新学习率 (不变)
        scheduler.step()
        print(f"Scheduler step finished. New LR for 'other_params': {scheduler.get_last_lr()[-1]:.6f}")


if __name__ == '__main__':
    main()