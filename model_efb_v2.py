# model_efb_v2.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertConfig, BertTokenizer
import clip
import numpy as np
import math

# ====================================================================================
#  从 clip4clip 移植过来的 Co-attention 模块及其依赖项
#  我们在这里复用这些成熟的模块来实现 图像-上下文 交互
# ====================================================================================

class BertBiAttention(nn.Module):
    def __init__(self, num_attention_heads, hidden_size, dropout_rate=0.1):
        super(BertBiAttention, self).__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError(f"The hidden size ({hidden_size}) is not a multiple of the number of attention heads ({num_attention_heads})")
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query1 = nn.Linear(hidden_size, self.all_head_size)
        self.key1 = nn.Linear(hidden_size, self.all_head_size)
        self.value1 = nn.Linear(hidden_size, self.all_head_size)
        self.dropout1 = nn.Dropout(dropout_rate)

        self.query2 = nn.Linear(hidden_size, self.all_head_size)
        self.key2 = nn.Linear(hidden_size, self.all_head_size)
        self.value2 = nn.Linear(hidden_size, self.all_head_size)
        self.dropout2 = nn.Dropout(dropout_rate)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, input_tensor1, attention_mask1, input_tensor2, attention_mask2):
        # input_tensor1: image_feat [B, 1, D]
        # input_tensor2: context_feat [B, N_tokens, D]
        
        mixed_query_layer1 = self.query1(input_tensor1)
        mixed_key_layer1 = self.key1(input_tensor1)
        mixed_value_layer1 = self.value1(input_tensor1)
        query_layer1 = self.transpose_for_scores(mixed_query_layer1)
        key_layer1 = self.transpose_for_scores(mixed_key_layer1)
        value_layer1 = self.transpose_for_scores(mixed_value_layer1)

        mixed_query_layer2 = self.query2(input_tensor2)
        mixed_key_layer2 = self.key2(input_tensor2)
        mixed_value_layer2 = self.value2(input_tensor2)
        query_layer2 = self.transpose_for_scores(mixed_query_layer2)
        key_layer2 = self.transpose_for_scores(mixed_key_layer2)
        value_layer2 = self.transpose_for_scores(mixed_value_layer2)

        # Context-to-Image Attention
        attention_scores1 = torch.matmul(query_layer2, key_layer1.transpose(-1, -2))
        attention_scores1 = attention_scores1 / math.sqrt(self.attention_head_size)
        attention_scores1 = attention_scores1 + attention_mask1 # mask is [B, 1, 1, 1] for image
        attention_probs1 = nn.Softmax(dim=-1)(attention_scores1)
        attention_probs1 = self.dropout1(attention_probs1)
        context_layer1 = torch.matmul(attention_probs1, value_layer1) # Fused Image -> Context
        
        # Image-to-Context Attention
        attention_scores2 = torch.matmul(query_layer1, key_layer2.transpose(-1, -2))
        attention_scores2 = attention_scores2 / math.sqrt(self.attention_head_size)
        attention_scores2 = attention_scores2 + attention_mask2 # mask is [B, 1, 1, N_tokens] for context
        attention_probs2 = nn.Softmax(dim=-1)(attention_scores2)
        attention_probs2 = self.dropout2(attention_probs2)
        context_layer2 = torch.matmul(attention_probs2, value_layer2) # Fused Context -> Image

        # Reshape back
        context_layer1 = context_layer1.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape1 = context_layer1.size()[:-2] + (self.all_head_size,)
        context_layer1 = context_layer1.view(*new_context_layer_shape1)

        context_layer2 = context_layer2.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape2 = context_layer2.size()[:-2] + (self.all_head_size,)
        context_layer2 = context_layer2.view(*new_context_layer_shape2)
        
        return context_layer1, context_layer2

class BertBiOutput(nn.Module):
    def __init__(self, hidden_size, dropout_rate=0.1):
        super(BertBiOutput, self).__init__()
        self.dense1 = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm1 = nn.LayerNorm(hidden_size, eps=1e-12)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dense2 = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm2 = nn.LayerNorm(hidden_size, eps=1e-12)
        self.dropout2 = nn.Dropout(dropout_rate)

    def forward(self, hidden_states1, input_tensor1, hidden_states2, input_tensor2):
        hidden_states1 = self.dense1(hidden_states1)
        hidden_states1 = self.dropout1(hidden_states1)
        hidden_states1 = self.LayerNorm1(hidden_states1 + input_tensor1)

        hidden_states2 = self.dense2(hidden_states2)
        hidden_states2 = self.dropout2(hidden_states2)
        hidden_states2 = self.LayerNorm2(hidden_states2 + input_tensor2)
        return hidden_states1, hidden_states2

class Co_attention_block(nn.Module):
    def __init__(self, num_attention_heads, hidden_size, dropout_rate=0.1):
        super(Co_attention_block, self).__init__()
        self.biattention = BertBiAttention(num_attention_heads, hidden_size, dropout_rate)
        self.biOutput = BertBiOutput(hidden_size, dropout_rate)
        # In clip4clip, there are FFN layers here. We can simplify by omitting them
        # or add them for more capacity. For equivalence, we should add them.
        self.v_intermediate = nn.Sequential(nn.Linear(hidden_size, hidden_size * 4), nn.GELU())
        self.v_output = nn.Linear(hidden_size * 4, hidden_size)
        self.v_LayerNorm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.v_dropout = nn.Dropout(dropout_rate)
        
        self.t_intermediate = nn.Sequential(nn.Linear(hidden_size, hidden_size * 4), nn.GELU())
        self.t_output = nn.Linear(hidden_size * 4, hidden_size)
        self.t_LayerNorm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.t_dropout = nn.Dropout(dropout_rate)

    def forward(self, image_input, image_mask, context_input, context_mask):
        # Bi-directional attention
        img_to_ctx, ctx_to_img = self.biattention(image_input, image_mask, context_input, context_mask)
        
        # Residual connection and LayerNorm (Attention part)
        attention_output_img, attention_output_ctx = self.biOutput(ctx_to_img, image_input, img_to_ctx, context_input)

        # FFN part
        intermediate_output_img = self.v_intermediate(attention_output_img)
        layer_output_img = self.v_output(intermediate_output_img)
        layer_output_img = self.v_dropout(layer_output_img)
        layer_output_img = self.v_LayerNorm(layer_output_img + attention_output_img)
        
        intermediate_output_ctx = self.t_intermediate(attention_output_ctx)
        layer_output_ctx = self.t_output(intermediate_output_ctx)
        layer_output_ctx = self.t_dropout(layer_output_ctx)
        layer_output_ctx = self.t_LayerNorm(layer_output_ctx + attention_output_ctx)
        
        return layer_output_img, layer_output_ctx

# ====================================================================================
#  EFB-V2 主模型
# ====================================================================================
class EFB_V3(nn.Module):
    def __init__(self,
                 pretrained_clip_name: str = "ViT-B/32",
                 bert_model_name: str = 'bert-base-chinese'):
        super(EFB_V3, self).__init__()

        # --- 1. 加载预训练模型 (与V2相同) ---
        self.clip, self.clip_preprocess = clip.load(pretrained_clip_name, device='cpu', jit=False)
        self.bert_tokenizer = BertTokenizer.from_pretrained(bert_model_name)
        self.bert_model = BertModel.from_pretrained(bert_model_name)

        clip_feature_dim = self.clip.visual.output_dim
        bert_feature_dim = self.bert_model.config.hidden_size
        
        # --- 2. 定义自定义模块 (与V2相同) ---
        self.emotion_projection = nn.Linear(bert_feature_dim, clip_feature_dim)
        self.co_attention = Co_attention_block(
            num_attention_heads=8,
            hidden_size=clip_feature_dim,
            dropout_rate=0.1
        )
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
    def get_bert_tokenizer(self):
        return self.bert_tokenizer
    # --- 特征编码函数 (与V2相同) ---
    def encode_image(self, image):
        return self.clip.encode_image(image.type(self.clip.dtype))

    def encode_text_final_feature(self, text_tokens):
        return self.clip.encode_text(text_tokens)

    def encode_text_all_tokens(self, text_tokens):
        x = self.clip.token_embedding(text_tokens).type(self.clip.dtype)
        x = x + self.clip.positional_embedding.type(self.clip.dtype)
        x = x.permute(1, 0, 2)
        x = self.clip.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.clip.ln_final(x).type(self.clip.dtype)
        attention_mask = (text_tokens != 0).float()
        return x, attention_mask
        
    def encode_emotion(self, emotion_token_ids, attention_mask):
        outputs = self.bert_model(input_ids=emotion_token_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state
        masked_output = last_hidden_state * attention_mask.unsqueeze(-1)
        mean_pooled = masked_output.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)
        return mean_pooled

    def forward(self, image, query_tokens, context_tokens, emotion_token_ids, emotion_attention_mask, use_emotion_fusion: bool = False):
        
        # === 步骤 1: 特征提取 ===
        image_feat = self.encode_image(image) # Shape: [B, D]
        query_feat_final = self.encode_text_final_feature(query_tokens) # Shape: [B, D]
        context_feat_tokens, context_mask_raw = self.encode_text_all_tokens(context_tokens) # Shape: [B, N, D]

        # === 步骤 2: [开关点] 图像特征增强 ===
        initial_image_feat = image_feat
        if use_emotion_fusion:
            # 如果开启，计算情感特征并融合
            with torch.no_grad(): # BERT部分可以冻结以节省计算
                emotion_feat_bert = self.encode_emotion(emotion_token_ids, emotion_attention_mask)
            
            projected_emotion_feat = self.emotion_projection(emotion_feat_bert.type(self.emotion_projection.weight.dtype))
            
            # 使用加法融合，概念上等同于“添加第四通道”信息
            initial_image_feat = initial_image_feat + projected_emotion_feat
            
        # 归一化初始图像特征
        initial_image_feat = F.normalize(initial_image_feat, dim=-1)

        # === 步骤 3: 图像-上下文交互 (统一流程) ===
        # 无论是否融合了情感，后续流程完全相同
        
        # 1. 将图像特征视为长度为1的序列
        image_feat_seq = initial_image_feat.unsqueeze(1) # Shape: [B, 1, D]
        # 创建用于Co-attention的mask
        image_mask_for_attention = torch.ones(image_feat_seq.size(0), 1, 1, 1).to(image_feat_seq.device)
        context_mask_for_attention = (1.0 - context_mask_raw.unsqueeze(1).unsqueeze(2)) * -10000.0

        # 2. Co-attention 交互
        fused_image_feat, _ = self.co_attention( # 我们只需要交互后的图像特征
            image_feat_seq, image_mask_for_attention,
            context_feat_tokens, context_mask_for_attention
        )
        
        # 3. 聚合/选择最终的图像特征
        final_image_feat = fused_image_feat.squeeze(1) # Shape: [B, D]
        final_image_feat = F.normalize(final_image_feat, dim=-1)

        # === 步骤 4: 最终相似度计算 ===
        # 归一化查询特征
        query_feat_final = F.normalize(query_feat_final, dim=-1)
        
        # 计算相似度
        sim_matrix = torch.matmul(query_feat_final, final_image_feat.t()) * self.logit_scale.exp()

        return sim_matrix