# jina_m0_lora_train.py
import os
import sys
import math
import torch
import numpy as np
from torch import nn
from torch.utils.data import DataLoader
from PIL import Image
import wandb
import clip
from tqdm import tqdm
from transformers import AutoProcessor, AutoConfig
from peft import LoraConfig, TaskType, get_peft_model

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.append(project_root)
from dataset import MSRVTT_Dataset
from jina_modeling import JinaVLForRanking, formatting_prompts_func

CONFIG = {
    "data_path": "./imgflip_data/msrvtt",
    "image_path": "./imgflip_data/images",
    "train_csv": "train_ids.csv",
    "train_json": "train_data.json",
    "train_emotion_json": "train_emotion.json",
    "batch_size": 128,
    "lr": 1e-5,  # 降低默认学习率，避免梯度爆炸（特别是只训练 score 头时）
    "epochs": 5,
    "save_dir": "./checkpoints_rerank",
    "image_max_side": 256,
    "jina_micro_batch": 64,
    "qwen_min_side": 32,
    "max_text_len": 2048,
}

def compute_features(clip_model, dataloader, device):
    text_feats = []
    image_feats = []
    upvotes = []
    captions = []
    clip_model.eval()
    # 确保获取CLIP模型的实际设备
    try:
        clip_dev = next(clip_model.parameters()).device
    except (StopIteration, AttributeError):
        clip_dev = torch.device("cpu")
    with torch.no_grad():
        for batch in dataloader:
            # 确保数据在CLIP模型所在的设备上
            images = batch["image"].to(clip_dev)
            query_ids = batch["query_ids"].to(clip_dev)
            img = clip_model.encode_image(images)
            txt = clip_model.encode_text(query_ids)
            img = img / img.norm(dim=1, keepdim=True).clamp(min=1e-6)
            txt = txt / txt.norm(dim=1, keepdim=True).clamp(min=1e-6)
            image_feats.append(img.cpu())
            text_feats.append(txt.cpu())
            upvotes.extend(batch["upvote"].cpu().numpy().tolist())
            captions.extend(batch["query_text"])
    text_matrix = torch.cat(text_feats, dim=0)
    image_matrix = torch.cat(image_feats, dim=0)
    return text_matrix, image_matrix, np.array(upvotes, dtype=np.float32), captions

def ranknet_loss(scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    s = scores.unsqueeze(0)
    y = labels.unsqueeze(0)
    diff = s.transpose(0, 1) - s
    rel = (y.transpose(0, 1) - y) > 0
    pos = torch.nonzero(rel, as_tuple=False)
    if pos.size(0) == 0:
        return scores.sum() * 0.0
    d = diff[pos[:, 0], pos[:, 1]]
    # 数值稳定性：使用 softplus 替代 log1p(exp(-x))
    return torch.nn.functional.softplus(-d).mean()

def lambdarank_loss(scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    labels = labels.to(scores.device)
    n = scores.size(0)
    if n == 0:
        return torch.tensor(0.0, device=scores.device, dtype=scores.dtype, requires_grad=True)
    # 检查是否有正样本
    if (labels > 0).sum() == 0:
        # 如果没有正样本，返回一个小的损失值，但不会产生梯度
        return scores.sum() * 0.0
    # 数值稳定性：限制 labels 范围，避免 pow(2, labels) 溢出
    labels_clamped = torch.clamp(labels, max=10.0)  # 2^10 = 1024，足够大
    gains = torch.pow(2.0, labels_clamped) - 1.0
    order = torch.argsort(scores, descending=True)
    ranks = torch.empty_like(order, dtype=torch.long)
    ranks[order] = torch.arange(n, device=scores.device, dtype=torch.long)
    discounts = 1.0 / torch.log2(ranks.float() + 2.0)
    ideal_order = torch.argsort(labels, descending=True)
    ideal_ranks = torch.empty_like(ideal_order, dtype=torch.long)
    ideal_ranks[ideal_order] = torch.arange(n, device=scores.device, dtype=torch.long)
    ideal_discounts = 1.0 / torch.log2(ideal_ranks.float() + 2.0)
    idcg = (gains[ideal_order] * ideal_discounts[ideal_order]).sum()
    idcg = torch.clamp(idcg, min=1e-8)
    s = scores.unsqueeze(0)
    g = gains.unsqueeze(0)
    d = discounts.unsqueeze(0)
    sd = s.transpose(0, 1) - s
    rel = (labels.unsqueeze(0).transpose(0, 1) - labels.unsqueeze(0)) > 0
    pos = torch.nonzero(rel, as_tuple=False)
    if pos.size(0) == 0:
        # 如果没有正样本对，返回0损失（应该在调用前检查）
        return scores.sum() * 0.0
    gi = g[:, pos[:, 0]].squeeze(0)
    gj = g[:, pos[:, 1]].squeeze(0)
    di = d[:, pos[:, 0]].squeeze(0)
    dj = d[:, pos[:, 1]].squeeze(0)
    delta_ndcg = torch.abs((gi - gj) * (di - dj)) / idcg
    # 数值稳定性：限制 delta_ndcg 范围
    delta_ndcg = torch.clamp(delta_ndcg, max=100.0)
    sdiff = sd[pos[:, 0], pos[:, 1]]
    # 检查是否有inf或nan值，如果有则返回一个小的损失值
    if torch.isinf(delta_ndcg).any() or torch.isnan(delta_ndcg).any():
        return torch.tensor(0.0, device=scores.device, dtype=scores.dtype, requires_grad=True)
    # 数值稳定性：使用 softplus 替代 log1p(exp(-x))，它们数学上等价但更稳定
    # log1p(exp(-x)) = softplus(-x) = log(1 + exp(-x))
    # 当 x 很大的负数时，exp(-x) 会溢出，但 softplus 会自动处理
    loss_per_pair = delta_ndcg * torch.nn.functional.softplus(-sdiff)
    loss_val = loss_per_pair.mean()
    # 最终检查损失值是否有效
    if torch.isnan(loss_val) or torch.isinf(loss_val):
        return torch.tensor(0.0, device=scores.device, dtype=scores.dtype, requires_grad=True)
    return loss_val

def build_image_paths(dataset, image_root):
    paths = []
    for item in dataset.data:
        vid = item["video_id"]
        name = vid if "." in vid else f"{vid}.jpg"
        paths.append(os.path.join(image_root, name))
    return paths

def _build_image_path(image_dir, vid):
    """构建图片路径（与推理脚本对齐）"""
    return os.path.join(image_dir, vid if "." in vid else f"{vid}.jpg")

def _safe_clip_indices(arr, max_len):
    """安全裁剪索引（与推理脚本对齐）"""
    try:
        a = np.array(arr, dtype=np.int64)
    except Exception:
        return arr
    if max_len is None or int(max_len) <= 0:
        return a
    mask = (a >= 0) & (a < int(max_len))
    return a[mask]

def score_batch(model, processor, texts, images, device, requires_grad=False):
    msgs = []
    imgs_list = []
    from PIL import Image as _I
    max_side_cfg = int(CONFIG.get("image_max_side", 448))
    min_side_cfg = int(CONFIG.get("qwen_min_side", 28))
    mb = int(CONFIG.get("jina_micro_batch", 0))
    for t, img in zip(texts, images):
        if img is None:
            img = _I.new("RGB", (224, 224), (0, 0, 0))
        else:
            w, h = img.size
            # First ensure minimum side
            if min(w, h) < min_side_cfg:
                scale_up = float(min_side_cfg) / float(min(w, h))
                w = int(round(w * scale_up))
                h = int(round(h * scale_up))
                img = img.resize((w, h), Image.BICUBIC)
            # Then clamp maximum side
            max_side = max_side_cfg
            if max(w, h) > max_side:
                scale_down = float(max_side) / float(max(w, h))
                w = int(round(w * scale_down))
                h = int(round(h * scale_down))
                img = img.resize((w, h), Image.BICUBIC)
            # Final pad if rounding still violates min side
            w, h = img.size
            if min(w, h) < min_side_cfg:
                w2 = max(w, min_side_cfg)
                h2 = max(h, min_side_cfg)
                canvas = _I.new("RGB", (w2, h2), (0, 0, 0))
                canvas.paste(img, ((w2 - w) // 2, (h2 - h) // 2))
                img = canvas
        imgs_list.append(img)
    
    # 检查模型设备，确保模型在正确的设备上
    try:
        mdl_dev = next(model.parameters()).device
    except StopIteration:
        # 如果模型没有参数（不应该发生），使用传入的 device
        mdl_dev = device
    
    # 如果模型不在 GPU 上，但 device 是 CUDA，需要将模型移到 GPU
    # 这通常发生在使用 use_jina_cache 优化时
    if device.type == "cuda" and mdl_dev.type != "cuda":
        print(f"[score_batch] Warning: Model is on {mdl_dev} but device is {device}. Moving model to {device}...")
        model = model.to(device)
        mdl_dev = device
    
    outs = []
    if mb and mb > 0:
        for i in range(0, len(imgs_list), mb):
            # 使用官方的 formatting_prompts_func 格式化 prompt（与官方实现对齐）
            batch_texts = []
            batch_images = []
            for j in range(i, min(i + mb, len(texts))):
                prompt_text = formatting_prompts_func(
                    query=texts[j],
                    doc="",  # doc 是图像，在 prompt 中用占位符
                    query_type='text',
                    doc_type='image'
                )
                batch_texts.append(prompt_text)
                batch_images.append(imgs_list[j])
            inputs = processor(text=batch_texts, images=batch_images, return_tensors="pt", padding=True, truncation=True, max_length=int(CONFIG.get("max_text_len", 512)))
            inputs = {k: v.to(mdl_dev) for k, v in inputs.items()}
            try:
                tok_id = int(getattr(model, "score_token_id", 100))
                bsz = inputs["input_ids"].size(0)
                inputs["input_ids"] = torch.cat([inputs["input_ids"], torch.full((bsz, 1), tok_id, device=inputs["input_ids"].device)], dim=1)
                inputs["attention_mask"] = torch.cat([inputs["attention_mask"], torch.ones((bsz, 1), device=inputs["attention_mask"].device)], dim=1)
            except Exception:
                pass
            # 根据requires_grad决定是否使用梯度，直接使用 JinaVLForRanking 的前向得到分数
            if requires_grad:
                if mdl_dev.type == "cuda":
                    from torch import amp as _amp
                    with _amp.autocast("cuda", dtype=torch.bfloat16):
                        with torch.set_grad_enabled(True):
                            logits = model(**inputs)
                else:
                    with torch.set_grad_enabled(True):
                        logits = model(**inputs)
            else:
                with torch.inference_mode():
                    if mdl_dev.type == "cuda":
                        from torch import amp as _amp
                        with _amp.autocast("cuda", dtype=torch.bfloat16):
                            logits = model(**inputs)
                    else:
                        logits = model(**inputs)
            outs.append(logits if isinstance(logits, torch.Tensor) else torch.as_tensor(logits, device=mdl_dev))
            # 清理中间变量（与推理脚本对齐，但训练时不清理以保持梯度）
            if not requires_grad:
                del inputs, logits
                if mdl_dev.type == "cuda":
                    torch.cuda.empty_cache()
        out = torch.cat(outs, dim=0)
    else:
        # 使用官方的 formatting_prompts_func 格式化 prompt（与官方实现对齐）
        batch_texts = []
        for j in range(len(texts)):
            prompt_text = formatting_prompts_func(
                query=texts[j],
                doc="",  # doc 是图像，在 prompt 中用占位符
                query_type='text',
                doc_type='image'
            )
            batch_texts.append(prompt_text)
        inputs = processor(text=batch_texts, images=imgs_list, return_tensors="pt", padding=True, truncation=True, max_length=int(CONFIG.get("max_text_len", 512)))
        inputs = {k: v.to(mdl_dev) for k, v in inputs.items()}
        try:
            tok_id = int(getattr(model, "score_token_id", 100))
            bsz = inputs["input_ids"].size(0)
            inputs["input_ids"] = torch.cat([inputs["input_ids"], torch.full((bsz, 1), tok_id, device=inputs["input_ids"].device)], dim=1)
            inputs["attention_mask"] = torch.cat([inputs["attention_mask"], torch.ones((bsz, 1), device=inputs["attention_mask"].device)], dim=1)
        except Exception:
            pass
        # 根据requires_grad决定是否使用梯度，直接使用 JinaVLForRanking 的前向得到分数
        if requires_grad:
            if mdl_dev.type == "cuda":
                from torch import amp as _amp
                with _amp.autocast("cuda", dtype=torch.bfloat16):
                    with torch.set_grad_enabled(True):
                        out = model(**inputs)
            else:
                with torch.set_grad_enabled(True):
                    out = model(**inputs)
        else:
            with torch.inference_mode():
                if mdl_dev.type == "cuda":
                    from torch import amp as _amp
                    with _amp.autocast("cuda", dtype=torch.bfloat16):
                        out = model(**inputs)
                else:
                    out = model(**inputs)
        # 清理中间变量（与推理脚本对齐，但训练时不清理以保持梯度）
        if not requires_grad:
            del inputs
            if mdl_dev.type == "cuda":
                torch.cuda.empty_cache()
    # 返回值格式：训练时返回tensor（需要梯度），评估时返回numpy（与推理脚本对齐）
    if requires_grad:
        return out if isinstance(out, torch.Tensor) else torch.as_tensor(out, device=mdl_dev)
    else:
        return out.detach().cpu().view(-1) if isinstance(out, torch.Tensor) else torch.as_tensor(out, device="cpu").view(-1)

def format_eval_metrics(eval_metrics, epoch=None, prefix=""):
    """格式化评估指标，使其更易读
    
    Args:
        eval_metrics: 包含 R@k 和 P@k 指标的字典
        epoch: 当前epoch（可选）
        prefix: 前缀字符串（如 "Initial Eval"）
    """
    ks = [1, 3, 5, 10, 50, 100]
    
    # 构建表格格式的输出
    lines = []
    if prefix:
        lines.append(f"\n{'=' * 80}")
        lines.append(f"{prefix} Evaluation Results" + (f" (Epoch {epoch})" if epoch is not None else ""))
        lines.append(f"{'=' * 80}")
    else:
        lines.append(f"\n{'=' * 80}")
        lines.append(f"Evaluation Results" + (f" (Epoch {epoch})" if epoch is not None else ""))
        lines.append(f"{'=' * 80}")
    
    # 表头
    lines.append(f"{'Metric':<12} | {'R@1':>8} | {'R@3':>8} | {'R@5':>8} | {'R@10':>9} | {'R@50':>9} | {'R@100':>10}")
    lines.append("-" * 80)
    
    # Recall 行
    recall_values = [f"{eval_metrics.get(f'R@{k}', 0.0):>7.2f}%" for k in ks]
    lines.append(f"{'Recall':<12} | {recall_values[0]:>8} | {recall_values[1]:>8} | {recall_values[2]:>8} | {recall_values[3]:>9} | {recall_values[4]:>9} | {recall_values[5]:>10}")
    
    # Precision 行
    precision_values = [f"{eval_metrics.get(f'P@{k}', 0.0):>7.2f}%" for k in ks]
    lines.append(f"{'Precision':<12} | {precision_values[0]:>8} | {precision_values[1]:>8} | {precision_values[2]:>8} | {precision_values[3]:>9} | {precision_values[4]:>9} | {precision_values[5]:>10}")
    
    lines.append("=" * 80)
    
    return "\n".join(lines)

def evaluate_recalls_jina(captions, image_paths, model, processor, topk_base, precomputed_topk, train_text_matrix, train_image_matrix, image_captions=None, jina_eval_cache=None):
    """评估函数（与推理脚本对齐）
    
    Args:
        captions: query captions (queries)
        image_paths: image paths (image pool)
        image_captions: captions for each image in the image pool (用于构建gt索引)
        jina_eval_cache: 可选的评估阶段 Jina 特征缓存，格式与训练缓存相同
    """
    # 确保模型处于eval模式
    model.eval()
    
    # 检查是否使用评估缓存
    use_eval_cache = jina_eval_cache is not None
    eval_feature_map = None
    eval_candidate_map = None
    eval_is_positive_map = None
    if use_eval_cache:
        # 构建评估缓存的特征映射、候选图像索引映射和正样本标记映射
        eval_feature_map = {}
        eval_candidate_map = {}
        eval_is_positive_map = {}
        if "features" in jina_eval_cache and "query_indices" in jina_eval_cache:
            candidate_indices_list = jina_eval_cache.get("candidate_indices", None)
            is_positive_list = jina_eval_cache.get("is_positive", None)
            for idx, (feat_list, q_idx_list) in enumerate(zip(
                jina_eval_cache["features"],
                jina_eval_cache["query_indices"]
            )):
                q_idx_val = int(q_idx_list[0].item() if isinstance(q_idx_list, torch.Tensor) else q_idx_list[0])
                if q_idx_val not in eval_feature_map:
                    eval_feature_map[q_idx_val] = feat_list
                    if candidate_indices_list is not None and idx < len(candidate_indices_list):
                        eval_candidate_map[q_idx_val] = candidate_indices_list[idx]
                    if is_positive_list is not None and idx < len(is_positive_list):
                        eval_is_positive_map[q_idx_val] = is_positive_list[idx]
            print(f"[Eval Cache] Loaded {len(eval_feature_map)} cached query features for evaluation")
            if eval_candidate_map:
                print(f"[Eval Cache] Loaded candidate indices for {len(eval_candidate_map)} queries")
            if eval_is_positive_map:
                total_positives = sum(p.sum().item() for p in eval_is_positive_map.values())
                print(f"[Eval Cache] Loaded positive labels for {len(eval_is_positive_map)} queries, total {total_positives} positives")
            print(f"[Eval Cache] Using cached features and labels, skipping model forward pass and caption_to_indices construction")
    
    # 如果使用评估缓存，不需要构建 caption_to_indices（正样本信息已在缓存中）
    # 否则，需要构建 caption_to_indices 用于 ground truth
    if not use_eval_cache:
        from collections import defaultdict
        # caption_to_indices 应该基于 image pool 的 captions 构建，索引是 images 的索引
        # 如果没有提供 image_captions，尝试从 image_paths 推断（假设每个 image 对应一个 caption）
        if image_captions is None:
            # 如果没有提供，假设 image_captions 与 captions 相同（这种情况应该不会发生，但为了兼容性）
            print("[Eval] Warning: image_captions not provided, using captions as fallback (may cause incorrect evaluation)")
            image_captions = captions
        caption_to_indices = defaultdict(list)
        for idx, cap in enumerate(image_captions):
            caption_to_indices[cap].append(idx)
    else:
        caption_to_indices = None  # 使用缓存时不需要
    total = len(captions)
    recalls = {}
    precisions = {}
    need_sim = precomputed_topk is None
    if need_sim:
        sim = train_text_matrix.cpu().numpy() @ train_image_matrix.cpu().numpy().T
    
    # 检查模型设备，确保模型在正确的设备上
    # 如果使用评估缓存，不需要整个模型，只需要 score head
    if not use_eval_cache:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 如果模型不在 GPU 上，但 CUDA 可用，需要将模型移到 GPU
        if torch.cuda.is_available() and device.type != "cuda":
            print(f"[evaluate_recalls_jina] Warning: Model is on {device} but CUDA is available. Moving model to GPU...")
            model = model.to("cuda")
            device = torch.device("cuda")
    else:
        # 使用评估缓存时，只需要 score head 在 GPU 上
        try:
            if hasattr(model, "score"):
                device = next(model.score.parameters()).device
            else:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        except StopIteration:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # k值列表与推理脚本对齐（包含3）
    ks = [1, 3, 5, 10, 50, 100]
    # 优化：改变循环顺序，每个query只处理一次，然后计算所有k值的指标
    # 初始化所有k值的统计
    recalls = {f"R@{k}": 0.0 for k in ks}
    precisions = {f"P@{k}": 0.0 for k in ks}
    hits_dict = {k: 0 for k in ks}
    acc_dict = {k: 0.0 for k in ks}
    
    # 添加进度条（tqdm已在文件开头导入）
    for q_idx in tqdm(range(total), desc="Evaluating"):
        # 如果使用评估缓存，使用缓存中存储的候选图像索引
        if use_eval_cache and eval_candidate_map is not None and q_idx in eval_candidate_map:
            # 使用缓存中的候选图像索引
            base_topk = eval_candidate_map[q_idx].cpu().numpy()
            base_topk = _safe_clip_indices(base_topk, len(image_paths))
            if len(base_topk) == 0:
                continue
        else:
            # 正常流程：动态选择候选图像
            if need_sim:
                k_base = min(topk_base, train_image_matrix.shape[0])
                base_topk = np.argpartition(-sim[q_idx], range(k_base))[:k_base]
            else:
                base_topk = np.array(precomputed_topk[q_idx], dtype=np.int64)
                k_base = base_topk.shape[0]
            # 使用_safe_clip_indices进行安全检查（与推理脚本对齐）
            base_topk = _safe_clip_indices(base_topk, len(image_paths))
            if len(base_topk) == 0:
                continue
        
        # 如果使用评估缓存，不需要加载图像和计算特征
        if use_eval_cache and eval_feature_map is not None and q_idx in eval_feature_map:
            # 使用缓存的特征，直接通过 score head 计算分数
            cached_features = eval_feature_map[q_idx]  # [N, hidden_dim]
            
            # 验证特征数量与候选图像数量一致
            if cached_features.shape[0] != len(base_topk):
                # 这种情况通常发生在：
                # 1. 评估时使用了 --eval_subset_image_pool，导致图像池被裁剪
                # 2. 某些图像路径在评估时不存在，导致 image_paths 长度小于缓存生成时
                # 3. 数据集发生了变化（图像被删除或移动）
                # 代码会自动对齐到较小的数量，但可能影响评估结果的准确性
                if cached_features.shape[0] > len(base_topk):
                    print(f"[Eval Cache] Warning: Cached features count ({cached_features.shape[0]}) > candidate count ({len(base_topk)}) for query {q_idx}. "
                          f"This may be due to image pool size mismatch. Using {len(base_topk)} candidates.")
                else:
                    print(f"[Eval Cache] Warning: Cached features count ({cached_features.shape[0]}) < candidate count ({len(base_topk)}) for query {q_idx}. "
                          f"This should not happen. Using {cached_features.shape[0]} candidates.")
                # 如果数量不一致，使用较小的数量
                min_count = min(cached_features.shape[0], len(base_topk))
                cached_features = cached_features[:min_count]
                base_topk = base_topk[:min_count]
            
            # 获取 score head
            if hasattr(model, "base_model") and hasattr(model.base_model, "model"):
                score_head = model.base_model.model.score
            elif hasattr(model, "score"):
                score_head = model.score
            else:
                raise RuntimeError("Score head not found in model")
            
            # 确保 score head 在正确的设备上
            if next(score_head.parameters()).device.type != "cuda" and device.type == "cuda":
                score_head = score_head.to(device)
            
            # 获取 score head 的 dtype，确保特征与模型 dtype 匹配
            score_head_dtype = next(score_head.parameters()).dtype
            
            # 通过 score head 计算分数
            with torch.no_grad():
                features_input = cached_features.to(device).to(dtype=score_head_dtype)
                scores = score_head(features_input).squeeze(-1)
                # numpy 不支持 bfloat16，需要先转换为 float32
                scores = scores.cpu().float().numpy()
        else:
            # 使用_build_image_path构建图片路径（与推理脚本对齐）
            imgs = []
            for idx_i in base_topk.tolist():
                if idx_i < len(image_paths):
                    p = image_paths[idx_i]
                else:
                    # 如果image_paths是路径列表，直接使用；否则需要从dataset构建
                    p = image_paths[idx_i] if isinstance(image_paths, list) and idx_i < len(image_paths) else None
                if p and os.path.exists(p):
                    try:
                        imgs.append(Image.open(p).convert("RGB"))
                    except Exception:
                        imgs.append(Image.new("RGB", (224, 224)))
                else:
                    imgs.append(Image.new("RGB", (224, 224)))
            texts = [captions[q_idx]] * len(imgs)
            # 使用score_batch（评估模式，不需要梯度）
            # 确保在评估模式下，模型不会更新参数
            with torch.no_grad():
                scores = score_batch(model, processor, texts, imgs, device, requires_grad=False)
            if isinstance(scores, torch.Tensor):
                scores = scores.numpy()
        # 及时清理图片对象（与推理脚本对齐）
        if not use_eval_cache:
            del imgs
        
        order = np.argsort(-scores)
        reranked_topk = base_topk[order]
        
        # 如果使用评估缓存，直接使用缓存中的正样本标记
        if use_eval_cache and eval_is_positive_map is not None and q_idx in eval_is_positive_map:
            # 使用缓存中的正样本标记
            is_positive = eval_is_positive_map[q_idx].cpu().numpy()
            # 确保长度一致
            if len(is_positive) != len(reranked_topk):
                min_len = min(len(is_positive), len(reranked_topk))
                is_positive = is_positive[:min_len]
                reranked_topk = reranked_topk[:min_len]
            
            # 根据排序后的顺序重新排列正样本标记
            is_positive_reranked = is_positive[order[:len(is_positive)]]
            
            # 对每个k值计算指标
            for k in ks:
                k_eff = min(k, len(reranked_topk))
                if k_eff == 0:
                    continue
                # 检查 top-k 中是否有正样本
                if is_positive_reranked[:k_eff].any():
                    hits_dict[k] += 1
                # 计算 precision
                correct = is_positive_reranked[:k_eff].sum()
                acc_dict[k] += correct / float(k_eff)
        else:
            # 正常流程：使用 caption_to_indices 构建 ground truth
            if caption_to_indices is None:
                print(f"[Eval] Warning: caption_to_indices not available for query {q_idx}, skipping")
                continue
            gt = caption_to_indices[captions[q_idx]]
            # 对每个k值计算指标（只处理一次，计算所有k值）
            for k in ks:
                k_eff = min(k, len(base_topk))
                if k_eff == 0:
                    continue
                topk_used = reranked_topk[:k_eff]
                if any(i in gt for i in topk_used):
                    hits_dict[k] += 1
                correct = sum(1 for i in topk_used if i in gt)
                acc_dict[k] += correct / float(k_eff)
        # 每处理一定数量的查询后清理一次缓存（与推理脚本对齐）
        if (q_idx + 1) % 10 == 0 and device.type == "cuda":
            torch.cuda.empty_cache()
    # 计算最终的recall和precision
    for k in ks:
        recalls[f"R@{k}"] = hits_dict[k] / total * 100.0
        precisions[f"P@{k}"] = acc_dict[k] / total * 100.0
    return {**recalls, **precisions}

def train(args=None):
    force_device = None
    if args is not None and getattr(args, "device", None):
        force_device = str(getattr(args, "device")).lower()
    if torch.cuda.is_available():
        if force_device and force_device not in (None, "", "auto", "cuda"):
            # 处理 cuda:0, cuda:1 等格式
            if force_device.startswith("cuda:"):
                device_idx = int(force_device.split(":")[1])
                # 当使用CUDA_VISIBLE_DEVICES时，设备索引会被重新映射
                # 所以直接使用索引即可
                device = torch.device(f"cuda:{device_idx}")
            else:
                device = torch.device(force_device)
            try:
                torch.cuda.set_device(device)
            except Exception:
                pass
        else:
            device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("medium")
    except Exception:
        pass
    try:
        ep = str(os.environ.get("HF_ENDPOINT", "")).strip()
        if not (ep.startswith("http://") or ep.startswith("https://")):
            os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    except Exception:
        pass
    device_clip = torch.device("cpu")
    clip_model, preprocess = clip.load("ViT-B/32", device=device_clip)
    if args is not None:
        CONFIG.update({
            "data_path": args.data_path,
            "image_path": args.image_path,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "lr": args.lr,
            "save_dir": args.save_dir,
        })
        topk_base = getattr(args, "topk_base", 100)
        extra_negs = max(0, getattr(args, "extra_negatives", 0))
        candidate_mode = getattr(args, "candidate_mode", "topk_plus_pos")
        model_name = getattr(args, "model_name", "jinaai/jina-reranker-m0")
        lora_r = int(getattr(args, "lora_r", 8))
        lora_alpha = int(getattr(args, "lora_alpha", 16))
        lora_dropout = float(getattr(args, "lora_dropout", 0.05))
        loss_type = getattr(args, "loss_type", "lambdarank")
        label_mode = getattr(args, "label_mode", "inv_rank")
        train_sample_limit = int(getattr(args, "train_sample_limit", 0))
        train_sample_mode = str(getattr(args, "train_sample_mode", "first")).lower()
        train_sample_seed = int(getattr(args, "train_sample_seed", 0))
        CONFIG["image_max_side"] = int(getattr(args, "image_max_side", CONFIG.get("image_max_side", 256)))
        CONFIG["jina_micro_batch"] = int(getattr(args, "jina_micro_batch", CONFIG.get("jina_micro_batch", 4)))
    else:
        topk_base = 100
        extra_negs = 0
        candidate_mode = "topk_plus_pos"
        model_name = "jinaai/jina-reranker-m0"
        lora_r = 8
        lora_alpha = 16
        lora_dropout = 0.05
        loss_type = "lambdarank"
        label_mode = "inv_rank"
        train_sample_limit = 0
    train_csv_path = os.path.join(CONFIG["data_path"], CONFIG["train_csv"])
    train_json_path = os.path.join(CONFIG["data_path"], CONFIG["train_json"])
    train_emotion_json_path = os.path.join(CONFIG["data_path"], CONFIG["train_emotion_json"])
    train_dataset = MSRVTT_Dataset(
        csv_path=train_csv_path,
        json_path=train_json_path,
        features_path=CONFIG["image_path"],
        emotion_json_path=train_emotion_json_path,
        clip_preprocess=preprocess,
        bert_tokenizer=None,
        max_words=32,
        is_train=True,
        load_image=True,
    )
    # 训练数据采样：先计算采样索引，但不立即采样dataset（为了保存完整特征缓存）
    train_sample_indices = None  # 保存采样索引，用于从缓存中正确截取
    train_dataset_full = train_dataset  # 保存完整dataset的引用
    if train_sample_limit and train_sample_limit > 0:
        try:
            total_train = len(getattr(train_dataset, "data"))
            n_lim = min(int(train_sample_limit), total_train)
            print(f"[Train] Sampling queries: mode={train_sample_mode}, seed={train_sample_seed}, n={n_lim}/{total_train}")
            if train_sample_mode == "random":
                rng = np.random.RandomState(train_sample_seed)
                indices = rng.choice(total_train, size=n_lim, replace=False)
                indices = sorted(indices.tolist())  # 保持顺序以便后续处理
                train_sample_indices = np.array(indices, dtype=np.int64)  # 保存索引用于从缓存截取
                print(f"[Train] Random sampling: will select {n_lim} queries from {total_train} total queries (indices saved for cache)")
            else:
                train_sample_indices = np.arange(n_lim, dtype=np.int64)  # first 模式就是前 n 个
                print(f"[Train] First-N sampling: will select first {n_lim} queries from {total_train} total queries")
        except Exception as e:
            print(f"[Train] Warning: Failed to prepare sampling: {e}")
            import traceback
            traceback.print_exc()
    
    # 先使用完整dataset计算特征（如果缓存不存在），这样缓存中保存的是完整特征
    # 如果缓存存在，则直接加载完整特征，然后根据采样索引截取
    cache_train_path = os.path.join(CONFIG["save_dir"], "train_cache.pt")
    if os.path.exists(cache_train_path):
        print(f"[Cache] Loading training cache from: {cache_train_path}")
        obj = torch.load(cache_train_path, map_location="cpu", weights_only=False)
        text_matrix_full = obj["text"].cpu()
        image_matrix_full = obj["image"].cpu()
        upvotes_full = obj.get("upvotes", None)
        captions_full = obj["captions"]
        topk_cached = obj.get("topk_indices", None)
        topk_base_cached = obj.get("topk_base", None)
        print(f"[Cache] Loaded training cache: text_matrix shape={text_matrix_full.shape}, image_matrix shape={image_matrix_full.shape}, "
              f"captions={len(captions_full)}, topk_cached={'Yes' if topk_cached is not None else 'No'}")
        # 如果 topk_cached 存在且 topk_base 匹配，不需要 sim；否则需要计算 sim
        sim = None
        try:
            if train_sample_limit and train_sample_limit > 0:
                # 如果使用了采样，topk_cached 不能直接使用（索引会变化），需要重新计算
                topk_cached = None
                topk_base_cached = None
                sim = None  # 采样后需要重新计算 sim
            elif topk_cached is not None and topk_base_cached == topk_base:
                # 如果 topk_cached 存在且 topk_base 匹配，不需要 sim
                sim = None
            else:
                # 如果 topk_cached 不存在或不匹配，需要计算 sim（但可以预先计算以提高效率）
                print(f"[Cache] Precomputing similarity matrix (topk_cached not available or topk_base mismatch)")
                sim = (text_matrix_full.cpu().numpy() @ image_matrix_full.cpu().numpy().T)
        except Exception as e:
            print(f"[Cache] Warning: Error in sim computation logic: {e}")
            sim = None
        # 如果使用了 train_sample_limit，需要根据采样索引从完整缓存中截取
        if train_sample_limit and train_sample_limit > 0 and train_sample_indices is not None:
            try:
                # 从完整缓存中根据采样索引截取
                max_idx = train_sample_indices.max()
                cache_size = text_matrix_full.shape[0]
                if max_idx < cache_size:
                    text_matrix = text_matrix_full[train_sample_indices]
                    captions = [captions_full[int(i)] for i in train_sample_indices.tolist()]
                    # upvotes 应该与 image_matrix 对应（完整 image pool），而不是与 text_matrix 对应
                    # 因为训练循环中使用 upvotes[i]，其中 i 是 image 索引
                    upvotes = upvotes_full  # 保持完整，与 image_matrix 对应
                    image_matrix = image_matrix_full  # image pool 保持完整
                    # 采样后，预计算 sim 矩阵以提高训练效率
                    print(f"[Train] Precomputing similarity matrix for sampled queries...")
                    sim = (text_matrix.cpu().numpy() @ image_matrix.cpu().numpy().T)
                    print(f"[Train] Applied sampling to cached features: {len(train_sample_indices)}/{cache_size} queries selected, image pool={image_matrix.shape[0]} (full)")
                else:
                    print(f"[Train] Warning: Sample indices exceed cache size, using first N instead")
                    n_q = min(int(train_sample_limit), text_matrix_full.shape[0])
                    text_matrix = text_matrix_full[:n_q]
                    captions = captions_full[:n_q]
                    # upvotes 应该与 image_matrix 对应（完整 image pool）
                    upvotes = upvotes_full  # 保持完整，与 image_matrix 对应
                    image_matrix = image_matrix_full
                    # 预计算 sim 矩阵以提高训练效率
                    print(f"[Train] Precomputing similarity matrix for sampled queries...")
                    sim = (text_matrix.cpu().numpy() @ image_matrix.cpu().numpy().T)
            except Exception as e:
                print(f"[Train] Warning: Failed to apply sampling to cached features: {e}")
                import traceback
                traceback.print_exc()
                # 降级到简单截取
                n_q = min(int(train_sample_limit), text_matrix_full.shape[0])
                text_matrix = text_matrix_full[:n_q]
                captions = captions_full[:n_q]
                if isinstance(upvotes_full, np.ndarray):
                    upvotes = upvotes_full[:n_q]
                else:
                    upvotes = upvotes_full
                image_matrix = image_matrix_full
        else:
            # 没有采样限制，使用完整特征
            text_matrix = text_matrix_full
            captions = captions_full
            image_matrix = image_matrix_full
            upvotes = upvotes_full
    else:
        # 缓存不存在，使用完整dataset计算完整特征并保存
        print(f"[Cache] Computing training features (cache not found at {cache_train_path})")
        print(f"[Cache] Computing full features (will save complete cache for future use)")
        loader_full = DataLoader(train_dataset_full, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=0)
        image_paths_train_full = build_image_paths(train_dataset_full, CONFIG["image_path"])
        text_matrix_full, image_matrix_full, upvotes_full, captions_full = compute_features(clip_model, loader_full, device)
        topk_cached = None
        topk_base_cached = None
        # 预先计算 sim 矩阵（如果后续需要）
        sim = (text_matrix_full.cpu().numpy() @ image_matrix_full.cpu().numpy().T)
        # 计算 topk_indices（与 precompute_indices.py 对齐）
        try:
            k_base = min(topk_base, sim.shape[1])
            topk_list = [np.argpartition(-sim[q], range(k_base))[:k_base] for q in range(sim.shape[0])]
            topk_py = [arr.tolist() for arr in topk_list]
            print(f"[Cache] Computed topk_indices: topk_base={topk_base}, queries={len(topk_py)}")
        except Exception as e:
            print(f"[Cache] Warning: Failed to compute topk_indices: {e}")
            topk_py = None
        # 保存完整训练缓存（不采样，与 precompute_indices.py 对齐）
        try:
            os.makedirs(CONFIG["save_dir"], exist_ok=True)
            cache_data = {
                "text": text_matrix_full.cpu(),
                "image": image_matrix_full.cpu(),
                "upvotes": upvotes_full,
                "captions": captions_full,
                "topk_base": topk_base,
                "topk_indices": topk_py
            }
            torch.save(cache_data, cache_train_path)
            print(f"[Cache] Saved complete training cache to: {cache_train_path}")
            print(f"[Cache] Cache contents: text_matrix shape={text_matrix_full.shape}, image_matrix shape={image_matrix_full.shape}, "
                  f"captions={len(captions_full)}")
        except Exception as e:
            print(f"[Cache] Warning: Failed to save training cache: {e}")
            import traceback
            traceback.print_exc()
        # 如果使用了采样，从完整特征中截取
        if train_sample_limit and train_sample_limit > 0 and train_sample_indices is not None:
            try:
                max_idx = train_sample_indices.max()
                cache_size = text_matrix_full.shape[0]
                if max_idx < cache_size:
                    text_matrix = text_matrix_full[train_sample_indices]
                    captions = [captions_full[int(i)] for i in train_sample_indices.tolist()]
                    # upvotes 应该与 image_matrix 对应（完整 image pool），而不是与 text_matrix 对应
                    # 因为训练循环中使用 upvotes[i]，其中 i 是 image 索引
                    upvotes = upvotes_full  # 保持完整，与 image_matrix 对应
                    image_matrix = image_matrix_full  # image pool 保持完整
                    # 采样后，预计算 sim 矩阵以提高训练效率
                    print(f"[Train] Precomputing similarity matrix for sampled queries...")
                    sim = (text_matrix.cpu().numpy() @ image_matrix.cpu().numpy().T)
                    print(f"[Train] Applied sampling to computed features: {len(train_sample_indices)}/{cache_size} queries selected, image pool={image_matrix.shape[0]} (full)")
                else:
                    print(f"[Train] Warning: Sample indices exceed computed size, using first N instead")
                    n_q = min(int(train_sample_limit), text_matrix_full.shape[0])
                    text_matrix = text_matrix_full[:n_q]
                    captions = captions_full[:n_q]
                    # upvotes 应该与 image_matrix 对应（完整 image pool）
                    upvotes = upvotes_full  # 保持完整，与 image_matrix 对应
                    image_matrix = image_matrix_full
                    # 预计算 sim 矩阵以提高训练效率
                    print(f"[Train] Precomputing similarity matrix for sampled queries...")
                    sim = (text_matrix.cpu().numpy() @ image_matrix.cpu().numpy().T)
            except Exception as e:
                print(f"[Train] Warning: Failed to apply sampling to computed features: {e}")
                import traceback
                traceback.print_exc()
                # 降级到简单截取
                n_q = min(int(train_sample_limit), text_matrix_full.shape[0])
                text_matrix = text_matrix_full[:n_q]
                captions = captions_full[:n_q]
                if isinstance(upvotes_full, np.ndarray):
                    upvotes = upvotes_full[:n_q]
                else:
                    upvotes = upvotes_full
                image_matrix = image_matrix_full
        else:
            # 没有采样限制，使用完整特征
            text_matrix = text_matrix_full
            captions = captions_full
            image_matrix = image_matrix_full
            upvotes = upvotes_full
    # image_paths_train应该基于完整dataset构建（完整image pool），而不是采样后的
    # 因为训练时需要在完整的image pool中检索
    image_paths_train = build_image_paths(train_dataset_full, CONFIG["image_path"])
    subset_images_n = len(image_paths_train)
    print(f"[Train] Image pool: {subset_images_n} images (full pool, not sampled)")
    
    # 如果使用了采样，对dataset进行采样（仅用于某些统计信息，不影响训练）
    if train_sample_limit and train_sample_limit > 0 and train_sample_indices is not None:
        try:
            # 对dataset进行采样（仅用于某些统计信息）
            if train_sample_mode == "random":
                train_dataset.data = [train_dataset_full.data[i] for i in train_sample_indices.tolist()]
            else:
                train_dataset.data = train_dataset_full.data[:len(train_sample_indices)]
            print(f"[Train] Applied sampling to dataset: {len(train_dataset.data)} queries (for stats only, image pool remains full)")
        except Exception as e:
            print(f"[Train] Warning: Failed to apply sampling to dataset: {e}")
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    try:
        if hasattr(processor, "tokenizer"):
            processor.tokenizer.padding_side = "left"
    except Exception:
        pass
    # ============ 检查是否使用 Jina 缓存 ============
    # 需要在模型加载之前确定，以便优化显存使用
    use_jina_cache = bool(getattr(args, "use_jina_cache", False)) if args is not None else False
    
    # 使用官方 JinaVLForRanking 结构作为基础模型
    # 使用模型原生的 bfloat16（范围与 float32 相同，训练更稳定，无需 GradScaler）
    cfg = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    base_model = JinaVLForRanking.from_pretrained(
        model_name,
        config=cfg,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        low_cpu_mem_usage=False,
    )
    # 微调策略：
    # - score 层（排序头 MLP）：必须训练（任务特定）
    # - LoRA（LLM 层）：可选训练（适应新领域）
    train_lora = bool(getattr(args, "train_lora", True)) if args is not None else True
    
    def _count_trainable(m):
        return sum(int(p.requires_grad) for p in m.parameters())
    def _count_trainable_by_name(m):
        """统计各部分的可训练参数"""
        stats = {"lora": 0, "score": 0, "other": 0}
        for name, p in m.named_parameters():
            if p.requires_grad:
                if "lora" in name.lower():
                    stats["lora"] += p.numel()
                elif "score" in name.lower():
                    stats["score"] += p.numel()
                else:
                    stats["other"] += p.numel()
        return stats
    
    if train_lora:
        # 注入 LoRA 并训练
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"]
        peft_cfg = LoraConfig(r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, task_type=TaskType.CAUSAL_LM, target_modules=target_modules)
        model = get_peft_model(base_model, peft_cfg)
        model = model.to(device)
        print("[Model] LoRA enabled: training LLM layers with LoRA")
        
        # 检查 LoRA 参数是否成功注入
        lora_cnt = _count_trainable(model)
        if lora_cnt == 0:
            print({"warning": "No trainable LoRA params found, retrying with auto target_modules"})
            peft_cfg = LoraConfig(r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, task_type=TaskType.CAUSAL_LM, target_modules=None)
            model = get_peft_model(base_model, peft_cfg).to(device)
    else:
        # 不使用 LoRA，直接使用基础模型
        # 优化：如果使用 jina_cache，不需要整个模型在 GPU 上，只需要 score head
        if use_jina_cache:
            # 只加载 score head 到 GPU，节省显存
            model = base_model  # 保持模型在 CPU
            # 冻结所有 LLM 参数
            for param in model.parameters():
                param.requires_grad = False
            # 只将 score head 移到 GPU
            if hasattr(model, "score"):
                model.score = model.score.to(device)
                model.score = model.score.float()
                for param in model.score.parameters():
                    param.requires_grad = True
            print("[Model] LoRA disabled: only training score head")
            print("[Model] Using jina_cache: only score head loaded to GPU (base model stays on CPU)")
        else:
            # 不使用缓存时，需要整个模型在 GPU 上用于特征提取
            model = base_model.to(device)
            # 冻结所有 LLM 参数
            for param in model.parameters():
                param.requires_grad = False
            print("[Model] LoRA disabled: only training score head")
    
    # score 层（排序头）始终可训练（无论是否使用 LoRA）
    # jina-reranker-m0 的 score 层是任务特定的排序头，必须微调
    # 注意：如果使用 jina_cache 且 train_lora=False，score head 已经在上面处理了
    if not (use_jina_cache and not train_lora):
        # 只有在不使用 jina_cache 或使用 LoRA 时才需要处理 score head
        if train_lora and hasattr(model, "base_model") and hasattr(model.base_model, "model"):
            # PEFT 包装后，原始模型在 model.base_model.model
            base = model.base_model.model
        else:
            base = model
        if hasattr(base, "score"):
            # 将 score head 转换为 float32，获得更高的权重更新精度
            # （bfloat16 只有 7 位尾数，float32 有 23 位，对可训练参数更好）
            base.score = base.score.float()
            for param in base.score.parameters():
                param.requires_grad = True
            print("[Model] Score head (MLP) is trainable (float32 for better precision)")
        else:
            print("[Model] Warning: score head not found in model")
    
    trainable_cnt = _count_trainable(model)
    trainable_stats = _count_trainable_by_name(model)
    print(f"[Model] Trainable params: {trainable_cnt:,} (LoRA: {trainable_stats['lora']:,}, score: {trainable_stats['score']:,}, other: {trainable_stats['other']:,})")
    
    if trainable_cnt == 0:
        raise RuntimeError("No trainable params found. Please check model configuration.")
    
    # ============ 加载预提取的 Jina 特征缓存（可选，仅用于 score-only 训练）============
    # use_jina_cache 已在上面定义
    jina_cache_data = None
    jina_cache_path = os.path.join(CONFIG["save_dir"], "jina_features_cache.pt")
    
    if use_jina_cache:
        if not train_lora:
            # 只训练 score 层时可以使用预提取的特征缓存
            if os.path.exists(jina_cache_path):
                print(f"[Jina Cache] Loading pre-extracted features from: {jina_cache_path}")
                try:
                    jina_cache_data = torch.load(jina_cache_path, map_location="cpu", weights_only=False)
                    
                    # 验证缓存文件格式
                    required_keys = ["features", "labels", "query_indices"]
                    if not all(key in jina_cache_data for key in required_keys):
                        raise ValueError(f"Cache file missing required keys. Expected: {required_keys}, Got: {list(jina_cache_data.keys())}")
                    
                    print(f"[Jina Cache] Loaded {len(jina_cache_data['features'])} training samples")
                    print(f"[Jina Cache] Features: {sum(f.shape[0] for f in jina_cache_data['features'])} samples")
                    print(f"[Jina Cache] Using pre-extracted features for score-only training (much faster!)")
                except Exception as e:
                    print(f"[Jina Cache] ERROR loading cache: {e}")
                    print(f"[Jina Cache] Will fall back to on-the-fly feature extraction")
                    jina_cache_data = None
            else:
                print(f"[Jina Cache] Cache file not found: {jina_cache_path}")
                print(f"[Jina Cache] Please run precompute_jina_features.py first, or set --use_jina_cache=False")
                print(f"[Jina Cache] Falling back to on-the-fly feature extraction")
                jina_cache_data = None
        else:
            print(f"[Jina Cache] Warning: --use_jina_cache is set but train_lora=True")
            print(f"[Jina Cache] Pre-extracted cache can only be used when train_lora=False (score-only training)")
            print(f"[Jina Cache] Ignoring --use_jina_cache, will use on-the-fly feature extraction")
            jina_cache_data = None
    
    try:
        if args is not None and getattr(args, "grad_checkpointing", False):
            if hasattr(model, "gradient_checkpointing_enable"):
                model.gradient_checkpointing_enable()
            if hasattr(model, "config"):
                try:
                    setattr(model.config, "use_cache", False)
                except Exception:
                    pass
        if hasattr(model, "enable_input_require_grads"):
            try:
                model.enable_input_require_grads()
            except Exception:
                pass
    except Exception:
        pass
    # 可选：从 checkpoint 恢复（支持 LoRA 和 score-only 两种格式）
    try:
        if args is not None and getattr(args, "resume_path", ""):
            rp = str(getattr(args, "resume_path"))
            if rp and os.path.exists(rp):
                print(f"[Checkpoint] Loading checkpoint from: {rp}")
                obj_resume = torch.load(rp, map_location=device, weights_only=False)
                ps = obj_resume.get("peft", None)
                model_name_resume = obj_resume.get("model_name", "unknown")
                train_lora_resume = obj_resume.get("train_lora", True)  # 默认 True（兼容旧 checkpoint）
                
                if ps:
                    if train_lora and train_lora_resume:
                        # 当前使用 LoRA，checkpoint 也有 LoRA：直接加载
                        model.load_state_dict(ps, strict=False)
                        print(f"[Checkpoint] Loaded LoRA + score checkpoint")
                    elif not train_lora and not train_lora_resume:
                        # 当前不使用 LoRA，checkpoint 也没有 LoRA：直接加载 score
                        model_state = model.state_dict()
                        loaded_count = 0
                        for name, param in ps.items():
                            if name in model_state:
                                model_state[name].copy_(param)
                                loaded_count += 1
                        print(f"[Checkpoint] Loaded score-only checkpoint ({loaded_count} params)")
                    elif train_lora and not train_lora_resume:
                        # 当前使用 LoRA，但 checkpoint 只有 score：只加载 score 部分
                        model_state = model.state_dict()
                        loaded_count = 0
                        for name, param in ps.items():
                            # score 参数在 PEFT 模型中的路径是 base_model.model.score.*
                            peft_name = f"base_model.model.{name}" if not name.startswith("base_model") else name
                            if peft_name in model_state:
                                model_state[peft_name].copy_(param)
                                loaded_count += 1
                            elif name in model_state:
                                model_state[name].copy_(param)
                                loaded_count += 1
                        model.load_state_dict(model_state, strict=False)
                        print(f"[Checkpoint] Loaded score from score-only checkpoint into LoRA model ({loaded_count} params)")
                    else:
                        # 当前不使用 LoRA，但 checkpoint 有 LoRA：只加载 score 部分
                        model_state = model.state_dict()
                        loaded_count = 0
                        for name, param in ps.items():
                            # 从 PEFT 路径提取原始名称
                            orig_name = name.replace("base_model.model.", "") if "base_model.model." in name else name
                            if orig_name in model_state and "score" in orig_name:
                                model_state[orig_name].copy_(param)
                                loaded_count += 1
                        print(f"[Checkpoint] Loaded score from LoRA checkpoint ({loaded_count} params)")
                    
                    print(f"[Checkpoint] Successfully loaded checkpoint:")
                    print(f"  - Path: {rp}")
                    print(f"  - Model name: {model_name_resume}")
                    print(f"  - Checkpoint train_lora: {train_lora_resume}")
                    print(f"  - Current train_lora: {train_lora}")
                    total_params = sum(p.numel() for p in ps.values())
                    print(f"  - Total parameters in checkpoint: {total_params:,}")
                else:
                    print(f"[Checkpoint] Warning: No 'peft' key found in checkpoint {rp}")
            else:
                print(f"[Checkpoint] Warning: Checkpoint path does not exist: {rp}")
    except Exception as e:
        print(f"[Checkpoint] Error loading checkpoint: {e}")
        import traceback
        traceback.print_exc()
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=CONFIG["lr"])
    # bfloat16 不需要 GradScaler（范围与 float32 相同，不会溢出）
    sch_type = getattr(args, "scheduler", "cosine_warmup") if args is not None else "cosine_warmup"
    scheduler = None
    global_step = 0
    total_queries_prefetch = text_matrix.shape[0]
    total_steps = CONFIG["epochs"] * max(1, total_queries_prefetch)
    if sch_type == "cosine_warmup":
        warmup = int(getattr(args, "warmup_steps", max(1, int(0.05 * total_steps)))) if args is not None else max(1, int(0.05 * total_steps))
        def lr_lambda(step):
            if step < warmup:
                return float(step + 1) / float(max(1, warmup))
            progress = float(step - warmup) / float(max(1, total_steps - warmup))
            return 0.5 * (1.0 + math.cos(math.pi * min(1.0, max(0.0, progress))))
        scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_lambda)
    elif sch_type == "cosine":
        tmax = CONFIG["epochs"]
        min_lr = float(getattr(args, "min_lr", 0.0)) if args is not None else 0.0
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=tmax, eta_min=min_lr)
    elif sch_type == "step":
        step_size = int(getattr(args, "step_size", 1)) if args is not None else 1
        gamma = float(getattr(args, "gamma", 0.5)) if args is not None else 0.5
        scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=step_size, gamma=gamma)
    elif sch_type == "plateau":
        gamma = float(getattr(args, "gamma", 0.5)) if args is not None else 0.5
        patience = int(getattr(args, "patience", 1)) if args is not None else 1
        min_lr = float(getattr(args, "min_lr", 0.0)) if args is not None else 0.0
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=gamma, patience=patience, min_lr=min_lr)
    best_r1 = -1.0
    best_r5 = -1.0
    best_r10 = -1.0
    keep_all_best = bool(getattr(args, "keep_all_best", False))
    if args and args.wandb_project:
        wandb.init(project=args.wandb_project, entity=getattr(args, 'wandb_entity', None), config=vars(args), mode=getattr(args, 'wandb_mode', 'disabled'))
        wandb.watch(model, log=None)
    from collections import defaultdict
    caption_to_indices = defaultdict(list)
    for idx, cap in enumerate(captions):
        caption_to_indices[cap].append(idx)
    
    # ============ 构建 query_idx -> 主贴图片索引的映射 ============
    # 用于在训练时确保主贴图片排第一
    query_to_main_post_idx = {}
    if hasattr(train_dataset_full, "data"):
        # 构建 video_id -> image索引的映射
        vid_to_img_idx = {}
        for img_idx, img_path in enumerate(image_paths_train):
            # 从路径中提取video_id
            img_name = os.path.basename(img_path)
            vid = img_name.replace(".jpg", "") if img_name.endswith(".jpg") else img_name
            vid_to_img_idx[vid] = img_idx
        
        # 构建 query_idx -> 主贴图片索引的映射
        for q_idx, item in enumerate(train_dataset_full.data):
            if q_idx < len(captions):  # 确保索引在范围内
                main_post_vid = item.get("video_id", None)
                if main_post_vid:
                    # 处理可能的.jpg后缀
                    vid_key = main_post_vid.replace(".jpg", "") if main_post_vid.endswith(".jpg") else main_post_vid
                    if vid_key in vid_to_img_idx:
                        query_to_main_post_idx[q_idx] = vid_to_img_idx[vid_key]
        print(f"[Train] Built query_to_main_post_idx mapping: {len(query_to_main_post_idx)}/{len(captions)} queries have main post image mapping")
    
    # ============ 训练前初始评估（可选）============
    eval_before_train = bool(getattr(args, "eval_before_train", False)) if args is not None else False
    if eval_before_train:
        print("=" * 80)
        print("[Initial Eval] Running evaluation with original (untrained) model...")
        print("=" * 80)
        
        # 准备测试数据（与训练后评估使用相同的逻辑）
        test_json_path = os.path.join(CONFIG["data_path"], "test_data.json")
        test_emotion_json_path = os.path.join(CONFIG["data_path"], "test_emotion.json")
        test_dataset = MSRVTT_Dataset(
            csv_path=test_json_path,
            json_path=test_json_path,
            features_path=CONFIG["image_path"],
            emotion_json_path=test_emotion_json_path,
            clip_preprocess=preprocess,
            bert_tokenizer=None,
            max_words=32,
            is_train=False,
            load_image=True,
        )
        test_loader = DataLoader(test_dataset, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=0)
        image_paths_test = build_image_paths(test_dataset, CONFIG["image_path"])
        cache_test_path = os.path.join(CONFIG["save_dir"], "test_cache.pt")
        
        if os.path.exists(cache_test_path):
            print(f"[Initial Eval Cache] Loading test cache from: {cache_test_path}")
            obj_t = torch.load(cache_test_path, map_location="cpu", weights_only=False)
            t_text_full = obj_t["text"].cpu()
            t_image_full = obj_t["image"].cpu()
            t_caps_full = obj_t["captions"]
            topk_test = obj_t.get("topk_indices", None)
            print(f"[Initial Eval Cache] Loaded test cache: text_matrix shape={t_text_full.shape}, image_matrix shape={t_image_full.shape}, "
                  f"captions={len(t_caps_full)}, topk_cached={'Yes' if topk_test is not None else 'No'}")
            # 如果使用了采样，从完整缓存中截取
            if args is not None and getattr(args, "eval_sample_limit", 0):
                total_eval = len(t_caps_full)
                n_eval = min(int(getattr(args, "eval_sample_limit", 0)), total_eval)
                mode = str(getattr(args, "eval_sample_mode", "first")).lower()
                seed = int(getattr(args, "eval_sample_seed", 0))
                print(f"[Initial Eval] Sampling queries: mode={mode}, seed={seed}, n={n_eval}/{total_eval}")
                if mode == "random":
                    rng = np.random.RandomState(seed)
                    indices = rng.choice(total_eval, size=n_eval, replace=False)
                    print(f"[Initial Eval] Random sampling: selected {n_eval} queries from {total_eval} total queries")
                else:
                    indices = np.arange(n_eval, dtype=np.int64)
                    print(f"[Initial Eval] First-N sampling: selected first {n_eval} queries from {total_eval} total queries")
                if isinstance(t_text_full, torch.Tensor):
                    t_text = t_text_full[indices]
                else:
                    t_text = t_text_full[indices]
                t_caps = [t_caps_full[int(i)] for i in indices.tolist()]
                t_image = t_image_full  # image pool 保持完整
                topk_test = None
                print(f"[Initial Eval] Applied sampling to cached features: {len(indices)}/{total_eval} queries selected, image pool={t_image.shape[0]} (full)")
            else:
                # 没有采样限制，使用完整特征
                t_text = t_text_full
                t_caps = t_caps_full
                t_image = t_image_full
        else:
            # 缓存不存在，使用完整dataset计算完整特征并保存
            print(f"[Initial Eval Cache] Computing test features (cache not found at {cache_test_path})")
            print(f"[Initial Eval Cache] Computing full test features (will save complete cache for future use)")
            t_text_full, t_image_full, _, t_caps_full = compute_features(clip_model, test_loader, device)
            topk_test = None
            # 保存完整测试缓存（不采样）
            try:
                os.makedirs(CONFIG["save_dir"], exist_ok=True)
                cache_data = {
                    "text": t_text_full.cpu(),
                    "image": t_image_full.cpu(),
                    "captions": t_caps_full
                }
                torch.save(cache_data, cache_test_path)
                print(f"[Initial Eval Cache] Saved complete test cache to: {cache_test_path}")
                print(f"[Initial Eval Cache] Cache contents: text_matrix shape={t_text_full.shape}, image_matrix shape={t_image_full.shape}, "
                      f"captions={len(t_caps_full)}")
            except Exception as e:
                print(f"[Initial Eval Cache] Warning: Failed to save test cache: {e}")
                import traceback
                traceback.print_exc()
            # 如果使用了采样，从完整特征中截取
            if args is not None and getattr(args, "eval_sample_limit", 0):
                total_eval = len(t_caps_full)
                n_eval = min(int(getattr(args, "eval_sample_limit", 0)), total_eval)
                mode = str(getattr(args, "eval_sample_mode", "first")).lower()
                seed = int(getattr(args, "eval_sample_seed", 0))
                print(f"[Initial Eval] Sampling queries: mode={mode}, seed={seed}, n={n_eval}/{total_eval}")
                if mode == "random":
                    rng = np.random.RandomState(seed)
                    indices = rng.choice(total_eval, size=n_eval, replace=False)
                    print(f"[Initial Eval] Random sampling: selected {n_eval} queries from {total_eval} total queries")
                else:
                    indices = np.arange(n_eval, dtype=np.int64)
                    print(f"[Initial Eval] First-N sampling: selected first {n_eval} queries from {total_eval} total queries")
                if isinstance(t_text_full, torch.Tensor):
                    t_text = t_text_full[indices]
                else:
                    t_text = t_text_full[indices]
                t_caps = [t_caps_full[int(i)] for i in indices.tolist()]
                t_image = t_image_full  # image pool 保持完整
                print(f"[Initial Eval] Applied sampling to computed features: {len(indices)}/{total_eval} queries selected, image pool={t_image.shape[0]} (full)")
            else:
                # 没有采样限制，使用完整特征
                t_text = t_text_full
                t_caps = t_caps_full
                t_image = t_image_full
        
        subset_pool = bool(getattr(args, "eval_subset_image_pool", False))
        if subset_pool:
            print(f"[Initial Eval] Subsetting image pool to match selected queries")
            caps_full = [item["query"] for item in test_dataset.data]
            selected = set(t_caps)
            keep_idx = [i for i, c in enumerate(caps_full) if c in selected]
            if isinstance(t_image, torch.Tensor):
                t_image = t_image[keep_idx]
            if isinstance(image_paths_test, list):
                image_paths_test = [image_paths_test[i] for i in keep_idx]
            print(f"[Initial Eval] Image pool: {len(keep_idx)}/{len(caps_full)} images kept")
        else:
            print(f"[Initial Eval] Image pool: keeping all {len(image_paths_test) if isinstance(image_paths_test, list) else t_image.shape[0]} images (full pool)")
        
        # 执行初始评估
        model.eval()
        
        # 检查是否有评估缓存
        jina_eval_cache_path = os.path.join(CONFIG["save_dir"], "jina_features_eval_cache.pt")
        has_eval_cache = use_jina_cache and not train_lora and os.path.exists(jina_eval_cache_path)
        
        # 检查模型设备
        model_device_before_eval = next(model.parameters()).device
        model_was_on_cpu = model_device_before_eval.type == "cpu"
        
        if model_was_on_cpu and use_jina_cache and not train_lora:
            if has_eval_cache:
                # 如果有评估缓存，只需要确保 score head 在 GPU
                if hasattr(model, "score"):
                    if next(model.score.parameters()).device.type != "cuda":
                        model.score = model.score.to(device)
                    print(f"[Initial Eval] Using evaluation cache: only score head on GPU, base model stays on CPU")
                else:
                    print(f"[Initial Eval] Warning: Score head not found, moving entire model to GPU")
                    model = model.to(device)
            else:
                # 没有评估缓存，需要整个模型在 GPU 上
                print(f"[Initial Eval] No evaluation cache found. Moving entire model to GPU for evaluation...")
                model = model.to(device)
                print(f"[Initial Eval] Model moved to {device} for evaluation")
        
        # 获取 image pool 的 captions（用于构建 gt 索引）
        image_captions_test = [item["query"] for item in test_dataset.data]
        
        # 检查是否有评估阶段的 Jina 缓存
        jina_eval_cache = None
        if use_jina_cache and not train_lora and os.path.exists(jina_eval_cache_path):
            try:
                print(f"[Initial Eval Cache] Loading evaluation cache from: {jina_eval_cache_path}")
                jina_eval_cache = torch.load(jina_eval_cache_path, map_location="cpu", weights_only=False)
                if "features" in jina_eval_cache and "query_indices" in jina_eval_cache:
                    print(f"[Initial Eval Cache] Loaded evaluation cache with {len(jina_eval_cache['features'])} queries")
                    print(f"[Initial Eval Cache] Using cached features for evaluation (no need to move model to GPU)")
                else:
                    print(f"[Initial Eval Cache] Warning: Evaluation cache format incorrect, falling back to model inference")
                    jina_eval_cache = None
            except Exception as e:
                print(f"[Initial Eval Cache] Error loading evaluation cache: {e}, falling back to model inference")
                jina_eval_cache = None
        elif use_jina_cache and not train_lora:
            print(f"[Initial Eval Cache] Evaluation cache not found: {jina_eval_cache_path}")
            print(f"[Initial Eval Cache] To use evaluation cache, run precompute script for test set")
            print(f"[Initial Eval Cache] Falling back to model inference (will move model to GPU)")
        
        initial_eval_metrics = evaluate_recalls_jina(
            t_caps, image_paths_test, model, processor, 
            topk_base=topk_base, precomputed_topk=topk_test, 
            train_text_matrix=t_text, train_image_matrix=t_image, 
            image_captions=image_captions_test,
            jina_eval_cache=jina_eval_cache
        )
        
        # 使用格式化函数输出更易读的结果
        print(format_eval_metrics(initial_eval_metrics, epoch=0, prefix="[Initial Eval]"))
        
        if wandb.run is not None:
            wandb.log({
                "initial_R@1": initial_eval_metrics["R@1"], "initial_P@1": initial_eval_metrics["P@1"],
                "initial_R@5": initial_eval_metrics["R@5"], "initial_P@5": initial_eval_metrics["P@5"],
                "initial_R@10": initial_eval_metrics["R@10"], "initial_P@10": initial_eval_metrics["P@10"],
                "initial_R@50": initial_eval_metrics["R@50"], "initial_P@50": initial_eval_metrics["P@50"],
                "initial_R@100": initial_eval_metrics["R@100"], "initial_P@100": initial_eval_metrics["P@100"],
                "epoch": 0
            })
        
        # 评估完成后，如果之前模型在 CPU，可以选择移回 CPU 以节省显存
        # 但为了训练，我们保持模型在 GPU 上（如果训练需要）
        if model_was_on_cpu and use_jina_cache and not train_lora:
            # 如果使用 jina_cache 且只训练 score head，模型可以保持在 CPU
            # 但 score head 应该在 GPU 上（如果训练需要）
            pass  # 保持当前状态，训练时会处理
    
    for epoch in range(CONFIG["epochs"]):
        model.train()
        total_loss = 0.0
        count = 0
        total_queries = text_matrix.shape[0]
        num_images = min(image_matrix.shape[0], len(image_paths_train))
        
        # 如果使用预提取的特征缓存，需要构建特征索引映射
        jina_feature_map = None
        cached_query_indices = None  # 缓存中实际存在的 query 索引列表
        if jina_cache_data is not None:
            # 构建 query_idx -> (features, labels) 的映射
            # 每个 query 可能有多个样本（不同的候选组合），取第一个匹配的
            jina_feature_map = {}
            for feat_list, label_list, q_idx_list in zip(
                jina_cache_data["features"],
                jina_cache_data["labels"],
                jina_cache_data["query_indices"]
            ):
                # q_idx_list 是一个 tensor，所有元素都是同一个 query_idx
                q_idx_val = int(q_idx_list[0].item() if isinstance(q_idx_list, torch.Tensor) else q_idx_list[0])
                # 只保存第一个匹配的（如果同一个 query 有多个，后续可以优化）
                if q_idx_val not in jina_feature_map:
                    jina_feature_map[q_idx_val] = (feat_list, label_list)
            
            # 获取缓存中所有 query 索引，并排序
            cached_query_indices = sorted(jina_feature_map.keys())
            print(f"[Jina Cache] Built feature map for {len(jina_feature_map)} queries")
            print(f"[Jina Cache] Query indices in cache: {cached_query_indices[:10]}... (showing first 10)")
            print(f"[Jina Cache] Using cached features, will only process queries in cache")
        
        # 确定要遍历的 queries
        if jina_cache_data is not None and cached_query_indices is not None:
            # 只遍历缓存中存在的 queries
            queries_to_process = cached_query_indices
            print(f"[Training] Using cached queries only: {len(queries_to_process)} queries from cache")
        else:
            # 正常流程：遍历所有 queries
            queries_to_process = range(total_queries)
            print(f"[Training] Processing all queries: {total_queries} queries")
        
        for q_idx in tqdm(queries_to_process, desc=f"jina_m0_lora epoch {epoch+1}"):
            # 如果使用预提取的特征缓存，跳过候选选择和标签构建
            if jina_cache_data is not None and jina_feature_map is not None and q_idx in jina_feature_map:
                # 直接使用缓存中的特征和标签，跳过候选选择
                pass  # 将在后面直接使用缓存的特征
            else:
                # 正常流程：选择候选并构建标签
                if topk_cached is not None and (topk_base_cached == topk_base):
                    base_topk = np.array(topk_cached[q_idx], dtype=np.int64)
                    k_base = base_topk.shape[0]
                else:
                    if sim is None:
                        k_base = min(topk_base, num_images)
                        t = text_matrix[q_idx].cpu().numpy()
                        sims_q = (t[None, :] @ image_matrix.cpu().numpy().T).squeeze(0)
                        base_topk = np.argpartition(-sims_q, range(k_base))[:k_base]
                    else:
                        k_base = min(topk_base, sim.shape[1])
                        base_topk = np.argpartition(-sim[q_idx], range(k_base))[:k_base]
                pos_all = np.array(caption_to_indices[captions[q_idx]], dtype=np.int64)
                if candidate_mode == "topk":
                    core_set = base_topk
                elif candidate_mode == "pos_all":
                    core_set = pos_all
                else:
                    core_set = np.unique(np.concatenate([base_topk, pos_all]))
                all_indices = np.arange(num_images)
                mask = np.ones_like(all_indices, dtype=bool)
                mask[core_set] = False
                candidates_neg = all_indices[mask]
                if extra_negs > 0 and candidates_neg.size > 0:
                    sel = np.random.choice(candidates_neg, size=min(extra_negs, candidates_neg.size), replace=False)
                    train_indices = np.concatenate([core_set, sel])
                else:
                    train_indices = core_set
                if train_indices.size == 0:
                    continue
                train_indices = train_indices[(train_indices >= 0) & (train_indices < len(image_paths_train))]
                if train_indices.size == 0:
                    continue
                gt = set(caption_to_indices[captions[q_idx]])
                pos_list = [(i, upvotes[i]) for i in train_indices if i in gt and (isinstance(upvotes, (list, np.ndarray)) and i < len(upvotes)) and upvotes[i] > 0]
                # 如果没有正样本，跳过这个训练样本，避免学习错误信号
                if len(pos_list) == 0:
                    continue
                labels = np.zeros(train_indices.shape[0], dtype=np.float32)
                if label_mode == "inv_rank":
                    # 确保主贴图片排第一：如果主贴图片在pos_list中，优先排到第一位
                    main_post_idx = query_to_main_post_idx.get(q_idx, None)
                    sorted_pos = []
                    if main_post_idx is not None:
                        # 查找主贴图片是否在pos_list中
                        main_post_in_list = None
                        other_pos = []
                        for item in pos_list:
                            if item[0] == main_post_idx:
                                main_post_in_list = item
                            else:
                                other_pos.append(item)
                        # 如果找到主贴图片，把它排到第一位
                        if main_post_in_list is not None:
                            sorted_pos = [main_post_in_list] + sorted(other_pos, key=lambda x: x[1], reverse=True)
                        else:
                            # 主贴图片不在pos_list中，按upvotes排序
                            sorted_pos = sorted(pos_list, key=lambda x: x[1], reverse=True)
                    else:
                        # 没有主贴图片映射，按upvotes排序
                        sorted_pos = sorted(pos_list, key=lambda x: x[1], reverse=True)
                    for rank, (i, _) in enumerate(sorted_pos, start=1):
                        w = 1.0 / float(rank)
                        idx_in_arr = np.where(train_indices == i)[0][0]
                        labels[idx_in_arr] = float(w)
                elif label_mode == "raw_log":
                    # 确保主贴图片排第一：如果主贴图片在pos_list中，优先排到第一位
                    main_post_idx = query_to_main_post_idx.get(q_idx, None)
                    sorted_pos_list = pos_list
                    if main_post_idx is not None:
                        # 查找主贴图片是否在pos_list中
                        main_post_in_list = None
                        other_pos = []
                        for item in pos_list:
                            if item[0] == main_post_idx:
                                main_post_in_list = item
                            else:
                                other_pos.append(item)
                        # 如果找到主贴图片，把它排到第一位
                        if main_post_in_list is not None:
                            sorted_pos_list = [main_post_in_list] + sorted(other_pos, key=lambda x: x[1], reverse=True)
                        else:
                            # 主贴图片不在pos_list中，按upvotes排序
                            sorted_pos_list = sorted(pos_list, key=lambda x: x[1], reverse=True)
                    else:
                        # 没有主贴图片映射，按upvotes排序
                        sorted_pos_list = sorted(pos_list, key=lambda x: x[1], reverse=True)
                    vals = np.array([u for _, u in sorted_pos_list], dtype=np.float32)
                    vals = np.log1p(vals)
                    for (i, _), s in zip(sorted_pos_list, vals):
                        idx_in_arr = np.where(train_indices == i)[0][0]
                        labels[idx_in_arr] = float(s)
                else:
                    # softmax模式：确保主贴图片排第一
                    main_post_idx = query_to_main_post_idx.get(q_idx, None)
                    sorted_pos_list = pos_list
                    if main_post_idx is not None:
                        # 查找主贴图片是否在pos_list中
                        main_post_in_list = None
                        other_pos = []
                        for item in pos_list:
                            if item[0] == main_post_idx:
                                main_post_in_list = item
                            else:
                                other_pos.append(item)
                        # 如果找到主贴图片，把它排到第一位
                        if main_post_in_list is not None:
                            sorted_pos_list = [main_post_in_list] + sorted(other_pos, key=lambda x: x[1], reverse=True)
                        else:
                            # 主贴图片不在pos_list中，按upvotes排序
                            sorted_pos_list = sorted(pos_list, key=lambda x: x[1], reverse=True)
                    else:
                        # 没有主贴图片映射，按upvotes排序
                        sorted_pos_list = sorted(pos_list, key=lambda x: x[1], reverse=True)
                    vals = np.array([u for _, u in sorted_pos_list], dtype=np.float32)
                    vals = np.log1p(vals)
                    exps = np.exp(vals - vals.max())
                    denom = exps.sum()
                    sm = exps / (denom if denom > 0 else 1.0)
                    for (i, _), s in zip(sorted_pos_list, sm):
                        idx_in_arr = np.where(train_indices == i)[0][0]
                        labels[idx_in_arr] = float(s)
            # 如果使用预提取的特征缓存，直接使用特征通过 score head 计算分数
            # 注意：当使用缓存时，queries_to_process 只包含缓存中的 queries，所以这里应该总是 True
            if jina_cache_data is not None and jina_feature_map is not None and q_idx in jina_feature_map:
                # 使用预提取的特征（跳过候选选择和图像加载）
                matched_features, matched_labels = jina_feature_map[q_idx]
                
                # 检查是否有有效的正样本对
                if matched_labels.sum().item() == 0:
                    # 如果没有正样本，跳过
                    continue
                
                # 直接通过 score head 计算分数（不需要通过整个模型）
                if train_lora and hasattr(model, "base_model") and hasattr(model.base_model, "model"):
                    score_head = model.base_model.model.score
                else:
                    score_head = model.score
                
                # 优化：特征分批移动到 GPU，避免一次性占用太多显存
                # 如果特征很大，可以分批处理
                # 获取 score head 的 dtype，确保特征与模型 dtype 匹配
                score_head_dtype = next(score_head.parameters()).dtype
                features_input = matched_features.to(device).to(dtype=score_head_dtype)
                
                # 通过 score head 计算分数
                preds = score_head(features_input).squeeze(-1)
                target_t = matched_labels.to(device).float()
            else:
                # 正常流程：使用 score_batch 提取特征并计算分数
                # 如果使用缓存，这个分支理论上不应该被执行（因为只遍历缓存中的 queries）
                if use_jina_cache and not train_lora:
                    # 这不应该发生，因为 queries_to_process 只包含缓存中的 queries
                    print(f"[Error] Query {q_idx} not in jina_cache, but was in queries_to_process. This is a bug.")
                    continue
                
                imgs = [Image.open(image_paths_train[i]).convert("RGB") if os.path.exists(image_paths_train[i]) else Image.new("RGB", (224, 224)) for i in train_indices]
                texts = [captions[q_idx]] * len(imgs)
                target_t = torch.from_numpy(labels).to(device).float()
                preds = score_batch(model, processor, texts, imgs, device, requires_grad=True)
            # 检查是否有有效的正样本对用于计算损失
            has_positive_pair = (target_t > 0).any() and (target_t == 0).any()
            if not has_positive_pair:
                # 如果没有正负样本对，跳过这个样本
                continue
            if loss_type == "ranknet":
                loss = ranknet_loss(preds, target_t)
            elif loss_type == "lambdarank":
                loss = lambdarank_loss(preds, target_t)
            else:
                loss = torch.nn.functional.mse_loss(preds, target_t)
            # 检查损失是否有效（非零、非NaN、非Inf）
            if torch.isnan(loss) or torch.isinf(loss) or loss.item() == 0.0:
                print(f"Warning: Skipping sample {q_idx+1} due to invalid loss before backward: loss={loss.item()}")
                continue
            opt.zero_grad()
            loss.backward()
            # 计算梯度范数（裁剪前）
            grad_norm_before = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], max_norm=float('inf'))
            # 梯度裁剪
            grad_norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], max_norm=1.0)
            # 检查梯度是否有inf/nan，如果有则跳过这个batch
            if torch.isnan(grad_norm) or torch.isinf(grad_norm) or torch.isnan(grad_norm_before) or torch.isinf(grad_norm_before):
                print(f"Warning: Gradient exploded at sample {q_idx+1}, grad_norm_before_clip={grad_norm_before:.6f}, skipping...")
                opt.zero_grad()
                continue
            # 每 100 个样本打印一次梯度范数，用于监控
            if (count + 1) % 100 == 0:
                print(f"  [Debug] grad_norm_before_clip={grad_norm_before:.4f}, grad_norm_after_clip={grad_norm:.4f}")
            opt.step()
            # 检查模型参数是否有inf/nan，如果有则尝试恢复或停止训练
            has_invalid_params = False
            for p in model.parameters():
                if p.requires_grad and (torch.isnan(p).any() or torch.isinf(p).any()):
                    has_invalid_params = True
                    break
            if has_invalid_params:
                print(f"CRITICAL: Model parameters contain inf/nan after step {q_idx+1}!")
                print(f"  This usually means:")
                print(f"  1. Learning rate too high (try --lr=1e-5 or lower)")
                print(f"  2. Gradient explosion (grad_norm before clip may be too large)")
                print(f"  3. Numerical instability in loss function")
                print(f"Stopping training to prevent wasted computation.")
                raise RuntimeError(f"Model parameters exploded at step {q_idx+1}. Try reducing learning rate.")
            if scheduler is not None and sch_type == "cosine_warmup":
                scheduler.step()
                global_step += 1
            loss_val = float(loss.item())
            # 再次检查损失值是否有效（防止inf被累加）
            if not (math.isnan(loss_val) or math.isinf(loss_val)):
                total_loss += loss_val
                count += 1
            else:
                # 如果损失是inf/nan，跳过这个样本，不累加，并打印警告
                print(f"Warning: Skipping sample {q_idx+1} due to invalid loss: {loss_val}")
                continue
            # 每100个样本打印一次损失
            if count % 100 == 0:
                avg_loss_so_far = total_loss / count
                print(f"Epoch {epoch+1}, Sample {count}/{total_queries}, Loss: {loss.item():.6f}, Avg Loss: {avg_loss_so_far:.6f}")
                # 可选：记录epoch内的loss到wandb（默认关闭，避免日志过多）
                # 如果需要更细粒度的监控，可以设置 wandb_log_interval > 0
                if wandb.run is not None:
                    log_interval = int(getattr(args, "wandb_log_interval", 0)) if args is not None else 0
                    if log_interval > 0 and count % log_interval == 0:
                        wandb.log({
                            "train_loss_step": avg_loss_so_far,
                            "train_loss_current": loss.item(),
                            "step": count,
                            "epoch": epoch + (count / total_queries),  # 浮点数epoch，表示epoch内的进度
                            "lr": opt.param_groups[0]["lr"]
                        })
        avg_loss = total_loss / max(count, 1)
        # 检查最终平均损失是否有效
        if math.isnan(avg_loss) or math.isinf(avg_loss):
            print(f"Warning: Epoch {epoch+1} completed with invalid average loss: {avg_loss}")
            print(f"  Total loss: {total_loss}, Count: {count}")
            # 如果平均损失是inf/nan，尝试找出问题
            if count > 0:
                print(f"  This suggests some loss values were inf/nan but were not properly filtered.")
        print(f"Epoch {epoch+1} completed. Total samples: {count}, Average loss: {avg_loss:.6f}")
        if wandb.run is not None:
            wandb.log({"train_loss": avg_loss, "epoch": epoch+1, "lr": opt.param_groups[0]["lr"], "train_samples": count})
        if scheduler is not None and sch_type in ("cosine", "step"):
            scheduler.step()
        if scheduler is not None and sch_type == "plateau":
            scheduler.step(avg_loss)
        try:
            os.makedirs(CONFIG["save_dir"], exist_ok=True)
            pre_suffix = f"_jina_m0_lora_{loss_type}_{label_mode}"
            checkpoint_path = os.path.join(CONFIG["save_dir"], f"jina_m0_lora_epoch_{epoch+1}_pre_eval{pre_suffix}.pt")
            
            # 只保存可训练的参数（score 层 + 可选的 LoRA）
            if train_lora:
                # 使用 PEFT 时，state_dict 已经只包含 LoRA + score
                trainable_state = model.state_dict()
            else:
                # 不使用 PEFT 时，只保存 score 层参数
                trainable_state = {}
                for name, param in model.named_parameters():
                    if param.requires_grad:
                        trainable_state[name] = param.data.clone()
            
            checkpoint_data = {
                "peft": trainable_state,
                "model_name": model_name,
                "train_lora": train_lora,  # 记录是否使用了 LoRA
            }
            torch.save(checkpoint_data, checkpoint_path)
            # 统计保存的参数数量
            total_params = sum(p.numel() for p in checkpoint_data["peft"].values())
            print(f"[Checkpoint] Saved checkpoint (pre-eval) for epoch {epoch+1}:")
            print(f"  - Path: {checkpoint_path}")
            print(f"  - Model name: {model_name}")
            print(f"  - train_lora: {train_lora}")
            print(f"  - Total parameters: {total_params:,}")
            print(f"  - File size: {os.path.getsize(checkpoint_path) / (1024*1024):.2f} MB")
        except Exception as e:
            print(f"[Checkpoint] Error saving checkpoint: {e}")
            import traceback
            traceback.print_exc()
        test_json_path = os.path.join(CONFIG["data_path"], "test_data.json")
        test_emotion_json_path = os.path.join(CONFIG["data_path"], "test_emotion.json")
        test_dataset = MSRVTT_Dataset(
            csv_path=test_json_path,
            json_path=test_json_path,
            features_path=CONFIG["image_path"],
            emotion_json_path=test_emotion_json_path,
            clip_preprocess=preprocess,
            bert_tokenizer=None,
            max_words=32,
            is_train=False,
            load_image=True,
        )
        test_loader = DataLoader(test_dataset, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=0)
        image_paths_test = build_image_paths(test_dataset, CONFIG["image_path"])
        cache_test_path = os.path.join(CONFIG["save_dir"], "test_cache.pt")
        if os.path.exists(cache_test_path):
            print(f"[Cache] Loading test cache from: {cache_test_path}")
            obj_t = torch.load(cache_test_path, map_location="cpu", weights_only=False)
            t_text_full = obj_t["text"].cpu()
            t_image_full = obj_t["image"].cpu()
            t_caps_full = obj_t["captions"]
            topk_test = obj_t.get("topk_indices", None)
            print(f"[Cache] Loaded test cache: text_matrix shape={t_text_full.shape}, image_matrix shape={t_image_full.shape}, "
                  f"captions={len(t_caps_full)}, topk_cached={'Yes' if topk_test is not None else 'No'}")
            # 如果使用了采样，从完整缓存中截取
            if args is not None and getattr(args, "eval_sample_limit", 0):
                total_eval = len(t_caps_full)
                n_eval = min(int(getattr(args, "eval_sample_limit", 0)), total_eval)
                mode = str(getattr(args, "eval_sample_mode", "first")).lower()
                seed = int(getattr(args, "eval_sample_seed", 0))
                print(f"[Eval] Sampling queries: mode={mode}, seed={seed}, n={n_eval}/{total_eval}")
                if mode == "random":
                    rng = np.random.RandomState(seed)
                    indices = rng.choice(total_eval, size=n_eval, replace=False)
                    print(f"[Eval] Random sampling: selected {n_eval} queries from {total_eval} total queries")
                else:
                    indices = np.arange(n_eval, dtype=np.int64)
                    print(f"[Eval] First-N sampling: selected first {n_eval} queries from {total_eval} total queries")
                if isinstance(t_text_full, torch.Tensor):
                    t_text = t_text_full[indices]
                else:
                    t_text = t_text_full[indices]
                t_caps = [t_caps_full[int(i)] for i in indices.tolist()]
                t_image = t_image_full  # image pool 保持完整
                topk_test = None
                print(f"[Eval] Applied sampling to cached features: {len(indices)}/{total_eval} queries selected, image pool={t_image.shape[0]} (full)")
            else:
                # 没有采样限制，使用完整特征
                t_text = t_text_full
                t_caps = t_caps_full
                t_image = t_image_full
        else:
            # 缓存不存在，使用完整dataset计算完整特征并保存
            print(f"[Cache] Computing test features (cache not found at {cache_test_path})")
            print(f"[Cache] Computing full test features (will save complete cache for future use)")
            t_text_full, t_image_full, _, t_caps_full = compute_features(clip_model, test_loader, device)
            topk_test = None
            # 保存完整测试缓存（不采样）
            try:
                os.makedirs(CONFIG["save_dir"], exist_ok=True)
                cache_data = {
                    "text": t_text_full.cpu(),
                    "image": t_image_full.cpu(),
                    "captions": t_caps_full
                }
                torch.save(cache_data, cache_test_path)
                print(f"[Cache] Saved complete test cache to: {cache_test_path}")
                print(f"[Cache] Cache contents: text_matrix shape={t_text_full.shape}, image_matrix shape={t_image_full.shape}, "
                      f"captions={len(t_caps_full)}")
            except Exception as e:
                print(f"[Cache] Warning: Failed to save test cache: {e}")
                import traceback
                traceback.print_exc()
            # 如果使用了采样，从完整特征中截取
            if args is not None and getattr(args, "eval_sample_limit", 0):
                total_eval = len(t_caps_full)
                n_eval = min(int(getattr(args, "eval_sample_limit", 0)), total_eval)
                mode = str(getattr(args, "eval_sample_mode", "first")).lower()
                seed = int(getattr(args, "eval_sample_seed", 0))
                print(f"[Eval] Sampling queries: mode={mode}, seed={seed}, n={n_eval}/{total_eval}")
                if mode == "random":
                    rng = np.random.RandomState(seed)
                    indices = rng.choice(total_eval, size=n_eval, replace=False)
                    print(f"[Eval] Random sampling: selected {n_eval} queries from {total_eval} total queries")
                else:
                    indices = np.arange(n_eval, dtype=np.int64)
                    print(f"[Eval] First-N sampling: selected first {n_eval} queries from {total_eval} total queries")
                if isinstance(t_text_full, torch.Tensor):
                    t_text = t_text_full[indices]
                else:
                    t_text = t_text_full[indices]
                t_caps = [t_caps_full[int(i)] for i in indices.tolist()]
                t_image = t_image_full  # image pool 保持完整
                print(f"[Eval] Applied sampling to computed features: {len(indices)}/{total_eval} queries selected, image pool={t_image.shape[0]} (full)")
            else:
                # 没有采样限制，使用完整特征
                t_text = t_text_full
                t_caps = t_caps_full
                t_image = t_image_full
            subset_pool = bool(getattr(args, "eval_subset_image_pool", False))
            if subset_pool:
                print(f"[Eval] Subsetting image pool to match selected queries")
                caps_full = [item["query"] for item in test_dataset.data]
                selected = set(t_caps)
                keep_idx = [i for i, c in enumerate(caps_full) if c in selected]
                if isinstance(t_image, torch.Tensor):
                    t_image = t_image[keep_idx]
                if isinstance(image_paths_test, list):
                    image_paths_test = [image_paths_test[i] for i in keep_idx]
                print(f"[Eval] Image pool: {len(keep_idx)}/{len(caps_full)} images kept")
            else:
                print(f"[Eval] Image pool: keeping all {len(image_paths_test) if isinstance(image_paths_test, list) else t_image.shape[0]} images (full pool)")
        # 评估前确保模型处于eval模式，并检查模型参数是否有效
        model.eval()
        
        # 检查是否有评估缓存
        jina_eval_cache_path = os.path.join(CONFIG["save_dir"], "jina_features_eval_cache.pt")
        has_eval_cache = use_jina_cache and not train_lora and os.path.exists(jina_eval_cache_path)
        
        # 检查模型设备：如果使用 use_jina_cache 优化，模型可能在 CPU 上
        # 如果使用评估缓存，只需要 score head 在 GPU；否则需要整个模型在 GPU
        model_device_before_eval = next(model.parameters()).device
        model_was_on_cpu = model_device_before_eval.type == "cpu"
        
        if model_was_on_cpu and use_jina_cache and not train_lora:
            if has_eval_cache:
                # 如果有评估缓存，只需要确保 score head 在 GPU
                if hasattr(model, "score"):
                    if next(model.score.parameters()).device.type != "cuda":
                        model.score = model.score.to(device)
                    print(f"[Eval] Using evaluation cache: only score head on GPU, base model stays on CPU")
                else:
                    print(f"[Eval] Warning: Score head not found, moving entire model to GPU")
                    model = model.to(device)
            else:
                # 没有评估缓存，需要整个模型在 GPU 上
                print(f"[Eval] No evaluation cache found. Moving entire model to GPU for evaluation...")
                model = model.to(device)
                print(f"[Eval] Model moved to {device} for evaluation")
        
        # 检查模型参数是否有inf/nan
        has_invalid_params = False
        for p in model.parameters():
            if p.requires_grad and (torch.isnan(p).any() or torch.isinf(p).any()):
                has_invalid_params = True
                print(f"Warning: Model parameters contain inf/nan before evaluation. Evaluation results may be invalid.")
                break
        if has_invalid_params:
            print(f"Skipping evaluation due to invalid model parameters.")
            eval_metrics = {f"R@{k}": 0.0 for k in [1, 3, 5, 10, 50, 100]}
            eval_metrics.update({f"P@{k}": 0.0 for k in [1, 3, 5, 10, 50, 100]})
        else:
            # 获取 image pool 的 captions（用于构建 gt 索引）
            image_captions_test = [item["query"] for item in test_dataset.data]
            
            # 检查是否有评估阶段的 Jina 缓存
            jina_eval_cache = None
            jina_eval_cache_path = os.path.join(CONFIG["save_dir"], "jina_features_eval_cache.pt")
            if use_jina_cache and not train_lora and os.path.exists(jina_eval_cache_path):
                try:
                    print(f"[Eval Cache] Loading evaluation cache from: {jina_eval_cache_path}")
                    jina_eval_cache = torch.load(jina_eval_cache_path, map_location="cpu", weights_only=False)
                    if "features" in jina_eval_cache and "query_indices" in jina_eval_cache:
                        print(f"[Eval Cache] Loaded evaluation cache with {len(jina_eval_cache['features'])} queries")
                        print(f"[Eval Cache] Using cached features for evaluation (no need to move model to GPU)")
                    else:
                        print(f"[Eval Cache] Warning: Evaluation cache format incorrect, falling back to model inference")
                        jina_eval_cache = None
                except Exception as e:
                    print(f"[Eval Cache] Error loading evaluation cache: {e}, falling back to model inference")
                    jina_eval_cache = None
            elif use_jina_cache and not train_lora:
                print(f"[Eval Cache] Evaluation cache not found: {jina_eval_cache_path}")
                print(f"[Eval Cache] To use evaluation cache, run precompute script for test set")
                print(f"[Eval Cache] Falling back to model inference (will move model to GPU)")
            
            eval_metrics = evaluate_recalls_jina(
                t_caps, image_paths_test, model, processor, 
                topk_base=topk_base, precomputed_topk=topk_test, 
                train_text_matrix=t_text, train_image_matrix=t_image, 
                image_captions=image_captions_test,
                jina_eval_cache=jina_eval_cache
            )
        
        # 评估完成后，如果之前模型在 CPU，可以选择移回 CPU 以节省显存
        # 但为了下一轮训练，我们保持模型在 GPU 上（如果下一轮训练也需要）
        # 如果需要节省显存，可以在这里移回 CPU
        # if model_was_on_cpu and use_jina_cache and not train_lora:
        #     print(f"[Eval] Moving model back to CPU to save GPU memory...")
        #     model = model.to("cpu")
        #     # 只保留 score head 在 GPU
        #     if hasattr(model, "score"):
        #         model.score = model.score.to(device)
        # 使用格式化函数输出更易读的结果
        print(format_eval_metrics(eval_metrics, epoch=epoch+1))
        if wandb.run is not None:
            wandb.log({
                "R@1": eval_metrics["R@1"], "P@1": eval_metrics["P@1"],
                "R@5": eval_metrics["R@5"], "P@5": eval_metrics["P@5"],
                "R@10": eval_metrics["R@10"], "P@10": eval_metrics["P@10"],
                "R@50": eval_metrics["R@50"], "P@50": eval_metrics["P@50"],
                "R@100": eval_metrics["R@100"], "P@100": eval_metrics["P@100"],
                "epoch": epoch+1
            })
        # os.makedirs(CONFIG["save_dir"], exist_ok=True)
        # suffix = f"_jina_m0_lora_{loss_type}_{label_mode}"
        # if eval_metrics["R@1"] > best_r1:
        #     best_r1 = eval_metrics["R@1"]
        #     torch.save({"peft": model.state_dict(), "head": head.state_dict(), "model_name": model_name}, os.path.join(CONFIG["save_dir"], f"jina_m0_lora_best_R@1{suffix}.pt"))
        #     if keep_all_best:
        #         torch.save({"peft": model.state_dict(), "head": head.state_dict(), "model_name": model_name}, os.path.join(CONFIG["save_dir"], f"jina_m0_lora_best_R@1_epoch_{epoch+1}{suffix}.pt"))
        #     if wandb.run is not None:
        #         wandb.log({"best_R@1": best_r1})
        # if eval_metrics["R@5"] > best_r5:
        #     best_r5 = eval_metrics["R@5"]
        #     torch.save({"peft": model.state_dict(), "head": head.state_dict(), "model_name": model_name}, os.path.join(CONFIG["save_dir"], f"jina_m0_lora_best_R@5{suffix}.pt"))
        #     if keep_all_best:
        #         torch.save({"peft": model.state_dict(), "head": head.state_dict(), "model_name": model_name}, os.path.join(CONFIG["save_dir"], f"jina_m0_lora_best_R@5_epoch_{epoch+1}{suffix}.pt"))
        #     if wandb.run is not None:
        #         wandb.log({"best_R@5": best_r5})
        # if eval_metrics["R@10"] > best_r10:
        #     best_r10 = eval_metrics["R@10"]
        #     torch.save({"peft": model.state_dict(), "head": head.state_dict(), "model_name": model_name}, os.path.join(CONFIG["save_dir"], f"jina_m0_lora_best_R@10{suffix}.pt"))
        #     if keep_all_best:
        #         torch.save({"peft": model.state_dict(), "head": head.state_dict(), "model_name": model_name}, os.path.join(CONFIG["save_dir"], f"jina_m0_lora_best_R@10_epoch_{epoch+1}{suffix}.pt"))
        #     if wandb.run is not None:
        #         wandb.log({"best_R@10": best_r10})
        # torch.save({"peft": model.state_dict(), "head": head.state_dict(), "model_name": model_name}, os.path.join(CONFIG["save_dir"], f"jina_m0_lora_epoch_{epoch+1}{suffix}.pt"))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default=CONFIG["data_path"])
    parser.add_argument("--image_path", type=str, default=CONFIG["image_path"])
    parser.add_argument("--batch_size", type=int, default=CONFIG["batch_size"])
    parser.add_argument("--epochs", type=int, default=CONFIG["epochs"])
    parser.add_argument("--lr", type=float, default=CONFIG["lr"])
    parser.add_argument("--save_dir", type=str, default=CONFIG["save_dir"])
    parser.add_argument("--wandb_project", type=str, default="")
    parser.add_argument("--wandb_entity", type=str, default="")
    parser.add_argument("--wandb_mode", type=str, default="disabled")
    parser.add_argument("--wandb_log_interval", type=int, default=0, help="Log training loss to wandb every N steps (0 = only log at end of epoch, recommended)")
    parser.add_argument("--topk_base", type=int, default=100)
    parser.add_argument("--extra_negatives", type=int, default=0)
    parser.add_argument("--candidate_mode", type=str, default="topk_plus_pos")
    parser.add_argument("--keep_all_best", action="store_true")
    parser.add_argument("--scheduler", type=str, default=None)
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument("--step_size", type=int, default=1)
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--patience", type=int, default=1)
    parser.add_argument("--min_lr", type=float, default=0.0)
    parser.add_argument("--loss_type", type=str, default="lambdarank")
    parser.add_argument("--label_mode", type=str, default="inv_rank")
    parser.add_argument("--model_name", type=str, default="jinaai/jina-reranker-m0")
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--train_lora", type=lambda x: x.lower() in ('true', '1', 'yes'), default=False, help="Whether to train LoRA parameters in LLM layers (default: True). Score head is always trained.")
    parser.add_argument("--train_sample_limit", type=int, default=0, help="Number of training queries to use (0 = all queries, sampling only applies to queries; image pool stays full)")
    parser.add_argument("--train_sample_mode", type=str, default="random", choices=["first", "random"], help="Sampling mode for training queries: 'first' for first N queries, 'random' for random sampling")
    parser.add_argument("--train_sample_seed", type=int, default=0, help="Random seed for training query sampling (only used when train_sample_mode='random')")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--eval_sample_limit", type=int, default=200, help="Number of queries to evaluate (0 = all queries, sampling only applies to queries; image pool stays full unless --eval_subset_image_pool)")
    parser.add_argument("--eval_sample_mode", type=str, default="random", choices=["first", "random"], help="Sampling mode: 'first' for first N queries, 'random' for random sampling")
    parser.add_argument("--eval_sample_seed", type=int, default=114, help="Random seed for query sampling (only used when eval_sample_mode='random')")
    parser.add_argument("--eval_subset_image_pool", action="store_true", help="If set, subset image pool to only include images matching selected queries (default: keep full image pool)")
    parser.add_argument("--image_max_side", type=int, default=256)
    parser.add_argument("--jina_micro_batch", type=int, default=64)
    parser.add_argument("--grad_checkpointing", action="store_true")
    parser.add_argument("--qwen_min_side", type=int, default=32)
    parser.add_argument("--resume_path", type=str, default="")
    parser.add_argument("--use_jina_cache", action="store_true", 
                        help="Use pre-extracted Jina features cache for score-only training (requires train_lora=False). "
                             "Much faster than on-the-fly feature extraction. Cache should be generated by precompute_jina_features.py first.")
    parser.add_argument("--eval_before_train", action="store_true",
                        help="Run evaluation with original (untrained) model before training starts. "
                             "Useful to see baseline performance before training.")
    args = parser.parse_args()
    train(args)
