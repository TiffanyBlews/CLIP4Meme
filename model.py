# model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertConfig
import clip
import numpy as np

class EFB(nn.Module):
    """
    Emotion-Fusion-Based (EFB) Model for Meme-Text Retrieval.
    
    This model implements the architecture described in the technical document.
    It uses a dual-branch approach (QI and QC similarity) to retrieve images
    based on a text query.
    """
    def __init__(self,
                 pretrained_clip_name: str = "ViT-B/32",
                 bert_model_name: str = 'bert-base-chinese',
                 sim_header: str = "seqTransf",
                 interaction: str = "wti",
                 text_pool_type: str = "transf_avg"):
        super(EFB, self).__init__()

        # --- 1. Load Pretrained Models ---
        # Load CLIP model for vision and text encoding
        self.clip, self.clip_preprocess = clip.load(pretrained_clip_name, device='cpu', jit=False)
        
        # Load BERT model for emotion encoding
        self.bert_tokenizer = None # Will be initialized later with the dataloader
        bert_config = BertConfig.from_pretrained(bert_model_name, output_hidden_states=True)
        self.bert_model = BertModel.from_pretrained(bert_model_name, config=bert_config)

        # Get feature dimensions
        clip_feature_dim = self.clip.visual.output_dim
        bert_feature_dim = self.bert_model.config.hidden_size

        # --- 2. Define Custom Modules ---
        # Linear projection layer to match BERT's output dim to CLIP's
        self.emotion_projection = nn.Linear(bert_feature_dim, clip_feature_dim)
        
        # Cross-modal Transformer for QI branch fusion
        self.sim_header = sim_header
        if self.sim_header == "seqTransf":
            self.cross_modal_transformer = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=clip_feature_dim,
                    nhead=4,
                    dim_feedforward=clip_feature_dim * 2,
                    dropout=0.1,
                    activation='relu'
                ),
                num_layers=2
            )
            
        # Learnable temperature parameter for scaling logits
        # Initialize with a smaller value to prevent numerical instability
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def get_bert_tokenizer(self):
        """Lazy loader for BERT tokenizer to be used in the Dataset."""
        if self.bert_tokenizer is None:
            from transformers import BertTokenizer
            self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
        return self.bert_tokenizer

    def encode_image(self, image):
        # CLIP image encoding
        return self.clip.visual(image.type(self.clip.dtype))

    def encode_text(self, text):
        # CLIP text encoding
        return self.clip.encode_text(text)

    def encode_emotion(self, emotion_token_ids, attention_mask):
        # BERT emotion encoding
        # We take the mean of the last hidden state's token embeddings
        outputs = self.bert_model(input_ids=emotion_token_ids, attention_mask=attention_mask)
        # Average pooling of the last hidden state
        last_hidden_state = outputs.last_hidden_state
        masked_output = last_hidden_state * attention_mask.unsqueeze(-1)
        mean_pooled = masked_output.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)
        return mean_pooled

    def forward(self, image, query, context, emotion_input_ids, emotion_attention_mask):
        """
        Performs a forward pass through the EFB model.

        Args:
            image (torch.Tensor): Preprocessed image tensor.
            query (torch.Tensor): Tokenized query text (caption/gt).
            context (torch.Tensor): Tokenized context text (Gemini description).
            emotion_input_ids (torch.Tensor): Tokenized emotion labels from BERT.
            emotion_attention_mask (torch.Tensor): Attention mask for emotion labels.

        Returns:
            dict: A dictionary containing QI and QC similarity matrices.
        """
        # --- Feature Extraction ---
        # e_i: Image features from CLIP
        image_feat = self.encode_image(image)
        
        # e_q: Query features from CLIP
        query_feat = self.encode_text(query)
        
        # e_c: Context features from CLIP
        context_feat = self.encode_text(context)

        # e_u: Emotion features from BERT
        emotion_feat_bert = self.encode_emotion(emotion_input_ids, emotion_attention_mask)
        # Project emotion features to match CLIP's dimension
        emotion_feat = self.emotion_projection(emotion_feat_bert)

        # Normalize features
        image_feat = F.normalize(image_feat, dim=-1)
        query_feat = F.normalize(query_feat, dim=-1)
        context_feat = F.normalize(context_feat, dim=-1)
        emotion_feat = F.normalize(emotion_feat, dim=-1)

        # --- Emotion Fusion ---
        # Fuse emotion features with image features via addition
        # This creates the enhanced image representation overline(e_u+i)
        fused_image_feat = F.normalize(image_feat + emotion_feat, dim=-1)

        # --- Similarity Calculation ---
        
        # QC (Query-Context) Similarity Branch
        # Straightforward cosine similarity between query and Gemini-generated context
        sim_qc = torch.matmul(query_feat, context_feat.t()) * self.logit_scale.exp()

        # QI (Query-Image) Similarity Branch
        if self.sim_header == "seqTransf":
            # Use cross-modal Transformer for advanced fusion
            # Prepare sequence: [query_feat, fused_image_feat]
            # Add a dimension for the sequence length
            seq_input = torch.stack([query_feat, fused_image_feat], dim=0) # Shape: [2, BatchSize, Dim]
            
            # Transformer expects (SeqLen, Batch, Dim)
            fused_output = self.cross_modal_transformer(seq_input)
            
            # Extract refined features
            refined_query_feat = F.normalize(fused_output[0], dim=-1)
            refined_image_feat = F.normalize(fused_output[1], dim=-1)

            sim_qi = torch.matmul(refined_query_feat, refined_image_feat.t()) * self.logit_scale.exp()
        else:
            # Default to simple cosine similarity if no transformer
            sim_qi = torch.matmul(query_feat, fused_image_feat.t()) * self.logit_scale.exp()
            
        # Clamp similarity values to prevent extreme values
        sim_qc = torch.clamp(sim_qc, min=-100, max=100)
        sim_qi = torch.clamp(sim_qi, min=-100, max=100)
            
        return {"sim_qi": sim_qi, "sim_qc": sim_qc}