# train.py
from collections import defaultdict
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import os
# <--- 新增: 导入 Scheduler
from torch.optim.lr_scheduler import CosineAnnealingLR

# Import our custom modules
from model import EFB
from dataset import MSRVTT_Dataset

CONFIG = {
    # Path settings
    "data_path": "/root/zt/data_topics/input_file",

    # Training hyperparameters
    "batch_size": 128,
    "batch_size_val": 128,
    "lr": 1e-4,
    "coef_lr": 1e-3,
    "epochs": 6,
    "max_grad_norm": 1.0,
    "min_lr": 1e-8,             # <--- 新增: Scheduler 的最小学习率

    # Data parameters
    "max_words": 32,
    "max_frames": 1,
    "image_resolution": 224,
    "num_workers": 4,

    # Model parameters
    "pretrained_clip_name": "ViT-B/32",
    "sim_header": "seqTransf",
    "interaction": "wti",
    "alpha": 0,

    # Seed
    "seed": 42,

    # ------------------------------------------------------------------
    # 新增：与 checkpoint / eval 相关的配置项
    # ------------------------------------------------------------------
    # "checkpoint_path": './checkpoints/best_model.pth',   # 如需加载 ckpt，把路径写在这里
    "checkpoint_path": None,
    "eval_only": False,         # 仅评估开关（True/False）
    "save_dir": "./checkpoints"
}


# ------------------------------------------------------------------
# 工具函数
# ------------------------------------------------------------------
def symmetric_contrastive_loss(sim_matrix):
    """
    Symmetric Cross-Entropy loss (InfoNCE)
    """
    batch_size = sim_matrix.size(0)
    labels = torch.arange(batch_size, device=sim_matrix.device)

    loss_rows = F.cross_entropy(sim_matrix, labels)
    loss_cols = F.cross_entropy(sim_matrix.t(), labels)
    return (loss_rows + loss_cols) / 2.0


def eval_epoch(model, test_dataset, test_dataloader, device, alpha, print_n_samples=5):
    """
    支持分组检索，并打印前 `print_n_samples` 个样本的预测细节。
    评估阶段仅用 Q-I 相似度 (sim_qi)，防止 title 泄露。
    """
    model.eval()

    # 1. 收集元数据
    print("--- Collecting metadata ---")
    all_video_ids_meta = [item['video_id'] for item in test_dataset.data]
    all_captions_meta  = [item['query']    for item in test_dataset.data]
    all_titles_meta    = [item['context']  for item in test_dataset.data]

    caption_to_indices = defaultdict(list)
    for idx, cap in enumerate(all_captions_meta):
        caption_to_indices[cap].append(idx)
    print(f"Found {len(caption_to_indices)} unique captions.")

    # 2. 缓存特征
    all_query_feats, all_context_feats, all_fused_image_feats = [], [], []
    print("--- Caching features ---")
    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Caching"):
            image  = batch['image'].to(device)
            query  = batch['query'].to(device)
            context= batch['context'].to(device)
            e_ids  = batch['emotion_input_ids'].to(device)
            e_mask = batch['emotion_attention_mask'].to(device)

            img_f   = F.normalize(model.encode_image(image), dim=-1)
            q_f     = F.normalize(model.encode_text(query), dim=-1)
            c_f     = F.normalize(model.encode_text(context), dim=-1)

            e_bert  = model.encode_emotion(e_ids, e_mask)
            e_f     = F.normalize(model.emotion_projection(e_bert), dim=-1)
            fused_f = F.normalize(img_f + e_f, dim=-1)

            all_query_feats.append(q_f.cpu())
            all_context_feats.append(c_f.cpu())
            all_fused_image_feats.append(fused_f.cpu())

    all_query_feats       = torch.cat(all_query_feats,       dim=0)
    all_context_feats     = torch.cat(all_context_feats,     dim=0)
    all_fused_image_feats = torch.cat(all_fused_image_feats, dim=0)

    # 3. 计算相似度（仅用 Q-I）
    print("--- Computing similarity (QI only) ---")
    sim_qc = torch.matmul(all_query_feats, all_context_feats.t())

    seq_in    = torch.stack([all_query_feats, all_fused_image_feats], dim=0).to(device)
    fused_out = model.cross_modal_transformer(seq_in)
    q_refined = F.normalize(fused_out[0].cpu(), dim=-1)
    i_refined = F.normalize(fused_out[1].cpu(), dim=-1)
    sim_qi    = torch.matmul(q_refined, i_refined.t())
    final_sim = alpha * sim_qi + (1 - alpha) * sim_qc

    # 4. 计算 R@k 并打印详细结果
    print("\n--- Computing R@k & Detailed Predictions ---")
    num_queries = final_sim.shape[0]
    recalls     = {}

    for k in [1, 5, 10, 50]:
        hits = 0
        for q_idx in range(num_queries):
            top_preds = torch.topk(final_sim[q_idx], k=k).indices.tolist()
            gt_indices= caption_to_indices[all_captions_meta[q_idx]]
            hits += any(p in gt_indices for p in top_preds)

            # ---- 详细打印 ----
            if q_idx < print_n_samples:
                print(f"\n{'='*22} Sample {q_idx+1}/{print_n_samples} {'='*22}")
                print(f"[Query]  VID: {all_video_ids_meta[q_idx]}")
                print(f"         Cap: {all_captions_meta[q_idx]}")
                print(f"         Tit: {all_titles_meta[q_idx]}")
                print(f"[GT indices] {gt_indices}")
                print("[Top-5 Predictions]")
                top5 = torch.topk(final_sim[q_idx], k=5)
                for rank, pred_idx in enumerate(top5.indices.tolist()):
                    mark = "✔️" if pred_idx in gt_indices else "❌"
                    print(f"  {rank+1}. {mark} score={top5.values[rank]:.4f}  "
                          f"VID={all_video_ids_meta[pred_idx]}  "
                          f"Cap={all_captions_meta[pred_idx]}")
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

    # 1. 模型
    model = EFB(
        pretrained_clip_name=CONFIG['pretrained_clip_name'],
        sim_header=CONFIG['sim_header'],
        interaction=CONFIG['interaction']
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
    clip_params = list(model.clip.parameters())
    bert_params = list(model.bert_model.parameters())
    other_params = [p for p in model.parameters() if id(p) not in [id(q) for q in clip_params + bert_params]]
    optimizer = optim.AdamW([
        {'params': clip_params, 'lr': CONFIG['lr'] * CONFIG['coef_lr']},
        {'params': bert_params, 'lr': CONFIG['lr'] * CONFIG['coef_lr']},
        {'params': other_params, 'lr': CONFIG['lr']}
    ])
    
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

    # 5. 仅评估模式
    if CONFIG.get("eval_only", False):
        if not ckpt_path:
            print("Error: eval_only=True requires checkpoint_path to be set.")
            return
        print("--- Running in Evaluation-Only Mode ---")
        eval_epoch(model, test_dataset, test_dataloader, device, CONFIG['alpha'])
        return

    # 6. 训练循环
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

            sim_matrices = model(image, query, context, emotion_ids, emotion_mask)
            sim_qi = sim_matrices['sim_qi']
            sim_qc = sim_matrices['sim_qc']

            loss_qi = symmetric_contrastive_loss(sim_qi)
            loss_qc = symmetric_contrastive_loss(sim_qc)
            total_loss = loss_qi + loss_qc

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

        # 7. 评估 & 保存最佳
        eval_results = eval_epoch(model, test_dataset, test_dataloader, device, CONFIG['alpha'])
        current_r1 = eval_results.get('R@1', 0)
        if current_r1 > best_r1:
            best_r1 = current_r1
            print(f"*** New best R@1: {best_r1:.2f}%. Saving checkpoint... ***")
            save_path = os.path.join(CONFIG['save_dir'], 'best_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(), # <--- 新增: 保存 scheduler 状态
                'eval_results': eval_results,
            }, save_path)
            print(f"*** Model saved to {save_path} ***")

        # <--- 新增: 更新学习率
        scheduler.step()
        print(f"Scheduler step finished. New LR for 'other_params': {scheduler.get_last_lr()[-1]:.6f}")


if __name__ == '__main__':
    main()