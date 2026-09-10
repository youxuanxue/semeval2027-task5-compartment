"""
Heavy Cross-Encoder Architecture (DeBERTa-v3 / GBERT / RoBERTa)
Integrates constituent representation with ablation contextual contrast.
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig


class HeavyCompositionalityCrossEncoder(nn.Module):
    def __init__(self, model_name: str, task_type: str = "nn", dropout: float = 0.25):
        super().__init__()
        self.task_type = task_type
        self.config = AutoConfig.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name, config=self.config)
        hidden_size = self.config.hidden_size

        self.dropout = nn.Dropout(dropout)
        
        # Dual-Pooling projection: CLS + Mean-Pooled + LayerNorm
        self.pre_head = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        out_dim = 2 if task_type == "nn" else 1
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, out_dim)
        )

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        if token_type_ids is not None and "token_type_ids" in self.encoder.forward.__code__.co_varnames:
            outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        else:
            outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)

        last_hidden = outputs.last_hidden_state  # [B, L, H]
        cls_rep = last_hidden[:, 0, :]           # [B, H]

        mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
        sum_embeddings = torch.sum(last_hidden * mask_expanded, 1)
        sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
        mean_rep = sum_embeddings / sum_mask    # [B, H]

        features = torch.cat([cls_rep, mean_rep], dim=1) # [B, 2*H]
        projected = self.pre_head(features)
        logits = self.head(projected)
        return logits
