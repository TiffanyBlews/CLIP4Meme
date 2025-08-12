# inference_fusion.py
# 演示如何使用融合方法进行推理

import torch
import torch.nn.functional as F
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import argparse
import os

from model import CLIP4Meme
from dataset import MSRVTT_Dataset
from torch.utils.data import DataLoader

def evaluate_fusion(model, test_dataset, test_dataloader, device, 
                   alpha=0.6, temperature=0.05, print_n_samples=5):
    """
    评估融合方法的性能
    """
    model.eval()
    print(f"\n--- Evaluating Fusion Method (α={alpha:.1f}, QI:{alpha:.1f} + QC:{1-alpha:.1f}) ---")
    
    # 1. 收集元数据
    all_video_ids_meta = [item['video_id'] for item in test_dataset.data]
    all_captions_meta = [item['query'] for item in test_dataset.data]
    all_titles_meta = [item['candidate_texts'][0] if item['candidate_texts'] else "" for item in test_dataset.data]
    caption_to_indices = defaultdict(list)
    for idx, cap in enumerate(all_captions_meta):
        caption_to_indices[cap].append(idx)
    
    # 2. 使用融合方法进行推理 - 修复：先收集所有特征
    all_qi_features = []
    all_qc_features = []
    all_query_features = []
    
    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Collecting Features"):
            # 获取QI分支特征
            qi_query_feat, qi_target_feat = model(
                query_ids=batch['query_ids'].to(device),
                candidate_ids=batch['candidate_ids'].to(device),
                image=batch['image'].to(device),
                emotion_ids=batch['emotion_input_ids'].to(device),
                emotion_mask=batch['emotion_attention_mask'].to(device),
                training_stage='stage1_qi',
                return_features=True
            )
            
            # 获取QC分支特征
            qc_query_feat, qc_target_feat = model(
                query_ids=batch['query_ids'].to(device),
                candidate_ids=batch['candidate_ids'].to(device),
                image=batch['image'].to(device),
                emotion_ids=batch['emotion_input_ids'].to(device),
                emotion_mask=batch['emotion_attention_mask'].to(device),
                training_stage='stage2_qc',
                return_features=True
            )
            
            all_qi_features.append(qi_target_feat.cpu())
            all_qc_features.append(qc_target_feat.cpu())
            all_query_features.append(qi_query_feat.cpu())  # 使用QI的query特征
    
    # 3. 拼接所有特征
    all_qi_features = torch.cat(all_qi_features, dim=0)
    all_qc_features = torch.cat(all_qc_features, dim=0)
    all_query_features = torch.cat(all_query_features, dim=0)
    
    print(f"特征收集完成: QI={all_qi_features.shape}, QC={all_qc_features.shape}, Query={all_query_features.shape}")
    
    # 4. 计算全局相似度矩阵
    all_query_features = F.normalize(all_query_features, dim=-1)
    all_qi_features = F.normalize(all_qi_features, dim=-1)
    all_qc_features = F.normalize(all_qc_features, dim=-1)
    
    # 计算QI和QC的相似度矩阵
    qi_sim_matrix = torch.matmul(all_query_features, all_qi_features.t())
    qc_sim_matrix = torch.matmul(all_query_features, all_qc_features.t())
    
    # 应用温度参数
    qi_sim_matrix = qi_sim_matrix / temperature
    qc_sim_matrix = qc_sim_matrix / temperature
    
    # 融合相似度矩阵
    fused_sim_matrix = alpha * qi_sim_matrix + (1 - alpha) * qc_sim_matrix
    
    print(f"相似度矩阵计算完成: 融合={fused_sim_matrix.shape}, QI={qi_sim_matrix.shape}, QC={qc_sim_matrix.shape}")
    
    # 5. 计算R@k
    print("\n--- Computing R@k for Fusion Method ---")
    num_queries = fused_sim_matrix.shape[0]
    recalls = {}
    
    for k in [1, 5, 10]:
        hits = 0
        for q_idx in range(num_queries):
            top_preds = torch.topk(fused_sim_matrix[q_idx], k=k).indices.tolist()
            gt_indices = caption_to_indices[all_captions_meta[q_idx]]
            if any(p in gt_indices for p in top_preds):
                hits += 1
                
            # 打印前几个样本的详细预测
            if q_idx < print_n_samples and k == 10:
                print(f"\n{'='*22} Sample {q_idx+1}/{print_n_samples} {'='*22}")
                print(f"  [Query]      VID: {all_video_ids_meta[q_idx]}, Caption: {all_captions_meta[q_idx]}")
                print(f"  [GT Indices] {gt_indices}")
                print("  [Top-5 Predictions (Fused)]")
                
                top5_fused = torch.topk(fused_sim_matrix[q_idx], k=5)
                top5_qi = torch.topk(qi_sim_matrix[q_idx], k=5)
                top5_qc = torch.topk(qc_sim_matrix[q_idx], k=5)
                
                for rank in range(5):
                    pred_idx = top5_fused.indices[rank]
                    fused_score = top5_fused.values[rank]
                    qi_score = qi_sim_matrix[q_idx][pred_idx]
                    qc_score = qc_sim_matrix[q_idx][pred_idx]
                    
                    mark = "✔️" if pred_idx in gt_indices else "❌"
                    print(f"    {rank+1}. {mark} Fused={fused_score:.4f} | QI={qi_score:.4f} | QC={qc_score:.4f}")
                    print(f"        VID={all_video_ids_meta[pred_idx]} | Cap: {all_captions_meta[pred_idx]}")
        
        recalls[f'R@{k}'] = (hits / num_queries) * 100
    
    print("\n--- Final Fusion Results ---")
    for k, v in recalls.items():
        print(f"  {k}: {v:.2f}%")
    
    return recalls, fused_sim_matrix, qi_sim_matrix, qc_sim_matrix

def compare_branches(model, test_dataset, test_dataloader, device, alpha=0.6):
    """
    比较单独使用QI分支、QC分支和融合方法的性能
    """
    print("\n" + "="*60)
    print("COMPARING DIFFERENT BRANCHES")
    print("="*60)
    
    # 1. 评估QI分支
    print("\n--- Evaluating QI Branch Only ---")
    qi_recalls = evaluate_fusion(model, test_dataset, test_dataloader, device, 
                               alpha=1.0)
    
    # 2. 评估QC分支  
    print("\n--- Evaluating QC Branch Only ---")
    qc_recalls = evaluate_fusion(model, test_dataset, test_dataloader, device,
                               alpha=0.0)
    
    # 3. 评估融合方法
    print("\n--- Evaluating Fusion Method ---")
    fusion_recalls = evaluate_fusion(model, test_dataset, test_dataloader, device,
                                  alpha=alpha)
    
    # 4. 总结比较结果
    print("\n" + "="*60)
    print("SUMMARY COMPARISON")
    print("="*60)
    print(f"{'Metric':<10} {'QI Only':<12} {'QC Only':<12} {'Fusion':<12}")
    print("-" * 60)
    for k in [1, 5, 10]:
        # evaluate_fusion返回(recalls, fused_sim_matrix, qi_sim_matrix, qc_sim_matrix)
        # 我们需要第一个元素recalls
        qi_val = qi_recalls[0].get(f'R@{k}', 0)
        qc_val = qc_recalls[0].get(f'R@{k}', 0)
        fusion_val = fusion_recalls[0].get(f'R@{k}', 0)
        print(f"{'R@'+str(k):<10} {qi_val:<12.2f} {qc_val:<12.2f} {fusion_val:<12.2f}")

def main():
    parser = argparse.ArgumentParser(description="Evaluate fusion method for QI+QC branches")
    parser.add_argument('--checkpoint', type=str, default="checkpoints_imgflip/stage2_qc_best.pth",
                       help="Path to model checkpoint")
    parser.add_argument('--data_path', type=str, default="/root/zt/imgflip_results/msrvtt",
                       help="Path to data directory")
    parser.add_argument('--image_path', type=str, default="/root/zt/imgflip_results/images",
                       help="Path to image directory")
    parser.add_argument('--test_json', type=str, default="test_data.json",
                       help="Test JSON file")
    parser.add_argument('--test_emotion_json', type=str, default="test_emotion.json",
                       help="Test emotion JSON file")
    parser.add_argument('--batch_size', type=int, default=128,
                       help="Batch size for inference")
    parser.add_argument('--alpha', type=float, default=0.5,
                       help="Weight α for QI branch in fusion: Final = α×QI + (1-α)×QC")
    parser.add_argument('--compare', action='store_true', default=True,
                       help="Compare all branches")
    
    args = parser.parse_args()
    
    # 检查checkpoint文件
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"checkpoint not found: {args.checkpoint}")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # 1. 初始化模型
    model = CLIP4Meme()
    model.to(device)
    
    # 2. 只需要加载第二阶段的checkpoint
    print("Loading Stage 2 (QC) checkpoint (contains both QI and QC parameters)...")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 3. 准备测试数据
    from dataset import MSRVTT_Dataset
    test_dataset = MSRVTT_Dataset(
        csv_path=os.path.join(args.data_path, args.test_json),
        json_path=os.path.join(args.data_path, args.test_json),
        features_path=args.image_path,
        emotion_json_path=os.path.join(args.data_path, args.test_emotion_json),
        bert_tokenizer=model.get_bert_tokenizer(),
        clip_preprocess=model.clip_preprocess,
        is_train=False,
        load_image=True
    )
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    # 4. 执行推理
    if args.compare:
        compare_branches(model, test_dataset, test_dataloader, device, alpha=args.alpha)
    else:
        # 只评估融合方法
        evaluate_fusion(model, test_dataset, test_dataloader, device,
                      alpha=args.alpha)

if __name__ == '__main__':
    main() 