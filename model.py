# model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertTokenizer
import clip
import numpy as np
from co_attention import Co_attention_block
class CLIP4Meme(nn.Module):
    """
    Model with a STABLE architecture for Stage 1 (QI) training.
    """
    def __init__(self,
                 pretrained_clip_name: str = "ViT-B/32",
                 bert_model_name: str = 'bert-base-chinese'):
        super(CLIP4Meme, self).__init__()
        # 模型组件初始化 (与之前版本相同)
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

    # 各个编码器函数 (与之前版本相同)
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
                candidate_ids, # Stage 1 now requires this for computation
                image,
                emotion_ids=None,
                emotion_mask=None,
                training_stage: str = 'stage1_qi',
                return_features: bool = False):

        # ====================================================================
        # 阶段一 (Stage 1): 训练 QI (Query-Image) 相似度 [稳定版架构]
        # ====================================================================
        if training_stage == 'stage1_qi':
            # 1. 提取所有需要的特征
            image_feat = self.encode_image(image)
            query_feat_pooled = self.encode_text_pooled(query_ids)
            # *** 关键修改: 现在获取 Context (candidate_ids) 的 token-level 特征用于融合 ***
            candidate_feat_tokens, candidate_mask = self.encode_text_tokens(candidate_ids)

            # 2. [可选] 融合情感特征到图像特征
            if emotion_ids is not None:
                with torch.no_grad():
                    emotion_feat_bert = self.encode_emotion(emotion_ids, emotion_mask)
                projected_emotion_feat = self.emotion_projection(emotion_feat_bert.type(self.emotion_projection.weight.dtype))
                image_feat = image_feat + projected_emotion_feat
            
            # 3. 跨模态融合 (Cross-Modal Fusion)
            # *** 关键修改: Co-attention 现在融合 Image 和 Context (图片描述) ***
            # 这是更稳定、更合理的架构，避免了信息泄漏。
            image_feat_seq = F.normalize(image_feat, dim=-1).unsqueeze(1)
            image_mask_for_attention = torch.ones(image_feat_seq.size(0), 1, 1, 1).to(image.device)
            
            fused_image_feat, _ = self.co_attention(
                image_feat_seq, image_mask_for_attention,
                candidate_feat_tokens, candidate_mask # 使用 context 特征进行交互
            )
            
            # 4. 计算相似度
            # 现在，我们用独立的 Query 特征去匹配融合后的“图像+上下文”特征。
            final_fused_feat = fused_image_feat.squeeze(1)
            
            query_feat_pooled = F.normalize(query_feat_pooled, dim=-1)
            final_fused_feat = F.normalize(final_fused_feat, dim=-1)
            
            # 如果只需要特征，直接返回
            if return_features:
                return query_feat_pooled, final_fused_feat
            
            sim_matrix = torch.matmul(query_feat_pooled, final_fused_feat.t()) * self.logit_scale.exp()
            
            return sim_matrix

        # ====================================================================
        # 阶段二 (Stage 2): 训练 QC (Query-Context) 相似度 [逻辑不变]
        # ====================================================================
        elif training_stage == 'stage2_qc':
            query_feat_pooled = self.encode_text_pooled(query_ids)
            candidate_feat_pooled = self.encode_text_pooled(candidate_ids)
            
            query_feat_pooled = F.normalize(query_feat_pooled, dim=-1)
            candidate_feat_pooled = F.normalize(candidate_feat_pooled, dim=-1)
            
            # 如果只需要特征，直接返回
            if return_features:
                return query_feat_pooled, candidate_feat_pooled
                
            sim_matrix = torch.matmul(query_feat_pooled, candidate_feat_pooled.t()) * self.logit_scale.exp()

            return sim_matrix
        
        else:
            raise ValueError(f"Unknown training_stage: {training_stage}")

    def inference_fusion(self, 
                        query_ids, 
                        candidate_ids, 
                        image, 
                        emotion_ids=None, 
                        emotion_mask=None,
                        alpha: float = 0.6,
                        temperature: float = 0.07):
        """
        推理时的融合方法：同时利用QI和QC分支，加权得到最终结果
        
        Args:
            alpha: QI分支的权重，QC分支权重为(1-alpha) (默认0.6)
            temperature: 温度参数，控制相似度分布的尖锐程度
        """
        with torch.no_grad():
            # 1. 计算QI分支的相似度
            qi_sim = self.forward(
                query_ids=query_ids,
                candidate_ids=candidate_ids,
                image=image,
                emotion_ids=emotion_ids,
                emotion_mask=emotion_mask,
                training_stage='stage1_qi'
            )
            
            # 2. 计算QC分支的相似度
            qc_sim = self.forward(
                query_ids=query_ids,
                candidate_ids=candidate_ids,
                image=image,  # QC不需要图片，但为了保持接口一致
                emotion_ids=emotion_ids,
                emotion_mask=emotion_mask,
                training_stage='stage2_qc'
            )
            
            # 3. 加权融合
            # 注意：这里使用logits进行加权，然后重新应用temperature
            # Final = α × QI + (1-α) × QC
            fused_sim = (alpha * qi_sim + (1 - alpha) * qc_sim) / temperature
            
            return fused_sim, qi_sim, qc_sim