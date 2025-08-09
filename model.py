import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertTokenizer
import clip
import numpy as np
from co_attention import Co_attention_block
class CLIP4Meme(nn.Module):
    """
    Model refactored to align with the paper's QI and QC branches,
    while maintaining a two-stage training structure.
    """
    def __init__(self,
                 pretrained_clip_name: str = "ViT-B/32",
                 bert_model_name: str = 'bert-base-chinese'):
        super(CLIP4Meme, self).__init__()
        self.clip, self.clip_preprocess = clip.load(pretrained_clip_name, device='cpu', jit=False)
        self.bert_tokenizer = BertTokenizer.from_pretrained(bert_model_name)
        self.bert_model = BertModel.from_pretrained(bert_model_name)
        clip_feature_dim = self.clip.visual.output_dim
        bert_feature_dim = self.bert_model.config.hidden_size
        self.emotion_projection = nn.Linear(bert_feature_dim, clip_feature_dim)
        self.co_attention = Co_attention_block(
            num_attention_heads=8, hidden_size=clip_feature_dim, dropout_rate=0.1
        )
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def get_bert_tokenizer(self):
        return self.bert_tokenizer

    def encode_image(self, image):
        return self.clip.encode_image(image.type(self.clip.dtype))

    def encode_text_pooled(self, text_ids):
        return self.clip.encode_text(text_ids)

    def encode_text_tokens(self, text_ids):
        x = self.clip.token_embedding(text_ids).type(self.clip.dtype)
        x = x + self.clip.positional_embedding.type(self.clip.dtype)
        x = x.permute(1, 0, 2)
        x = self.clip.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.clip.ln_final(x).type(self.clip.dtype)
        attention_mask = (1.0 - (text_ids == 0).float().unsqueeze(1).unsqueeze(2)) * -10000.0
        return x, attention_mask
        
    def encode_emotion(self, emotion_ids, attention_mask):
        outputs = self.bert_model(input_ids=emotion_ids, attention_mask=attention_mask)
        masked_output = outputs.last_hidden_state * attention_mask.unsqueeze(-1)
        mean_pooled = masked_output.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)
        return mean_pooled

    def forward(self,
                query_ids,
                candidate_ids=None, # Stage 1 does not need this for computation
                image=None,
                emotion_ids=None,
                emotion_mask=None,
                training_stage: str = 'stage1_qi'):

        # ====================================================================
        # 阶段一 (Stage 1): 训练 QI (Query-Image) 相似度
        # ====================================================================
        if training_stage == 'stage1_qi':
            if image is None:
                raise ValueError("Image tensor must be provided for QI training.")
            
            # 1. 提取基础特征
            image_feat = self.encode_image(image)
            query_feat_pooled = self.encode_text_pooled(query_ids)
            # *** 关键修改: 现在获取 Query 的 token-level 特征用于融合 ***
            query_feat_tokens, query_mask = self.encode_text_tokens(query_ids)

            # 2. [可选] 融合情感特征到图像特征
            if emotion_ids is not None:
                with torch.no_grad():
                    emotion_feat_bert = self.encode_emotion(emotion_ids, emotion_mask)
                projected_emotion_feat = self.emotion_projection(emotion_feat_bert.type(self.emotion_projection.weight.dtype))
                image_feat = image_feat + projected_emotion_feat
            
            # 3. 跨模态融合 (Cross-Modal Fusion)
            # *** 关键修改: Co-attention 现在融合 Image 和 Query ***
            image_feat_seq = F.normalize(image_feat, dim=-1).unsqueeze(1)
            image_mask_for_attention = torch.ones(image_feat_seq.size(0), 1, 1, 1).to(image.device)
            
            fused_representation, _ = self.co_attention(
                image_feat_seq, image_mask_for_attention,
                query_feat_tokens, query_mask # 使用 query 特征进行交互
            )
            
            # 4. 计算相似度
            # 论文描述: "...计算主题文本特征 e_q 与跨模态融合特征 cross(...) 之间的QI 相似度。"
            # 这意味着我们将原始的 query 特征与融合后的表示进行比较。
            final_fused_feat = fused_representation.squeeze(1)
            
            query_feat_pooled = F.normalize(query_feat_pooled, dim=-1)
            final_fused_feat = F.normalize(final_fused_feat, dim=-1)
            
            sim_matrix = torch.matmul(query_feat_pooled, final_fused_feat.t()) * self.logit_scale.exp()
            
            return sim_matrix

        # ====================================================================
        # 阶段二 (Stage 2): 训练 QC (Query-Context) 相似度
        # ====================================================================
        elif training_stage == 'stage2_qc':
            if candidate_ids is None:
                raise ValueError("Candidate text tensor must be provided for QC training.")
                
            # 1. 提取 Query 和 Context 特征
            query_feat_pooled = self.encode_text_pooled(query_ids)
            candidate_feat_pooled = self.encode_text_pooled(candidate_ids)
            
            # 2. 计算相似度
            query_feat_pooled = F.normalize(query_feat_pooled, dim=-1)
            candidate_feat_pooled = F.normalize(candidate_feat_pooled, dim=-1)
            sim_matrix = torch.matmul(query_feat_pooled, candidate_feat_pooled.t()) * self.logit_scale.exp()

            return sim_matrix
        
        else:
            raise ValueError(f"Unknown training_stage: {training_stage}")