"""
Agent Charlie: Cross-Encoder Architecture for Task 5
Jointly predicts modifier & head compositionality scores (for NN)
or single score (for PV) using contextualized target representations and pairwise ranking loss.
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig


class CompositionalityCrossEncoder(nn.Module):
    def __init__(self, model_name: str, task_type: str = "nn", dropout: float = 0.2):
        super().__init__()
        self.task_type = task_type
        self.config = AutoConfig.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name, config=self.config)
        hidden_size = self.config.hidden_size

        self.dropout = nn.Dropout(dropout)
        
        if task_type == "nn":
            # Predicts 2 values: Mod score and Head score
            self.head = nn.Sequential(
                nn.Linear(hidden_size * 2, hidden_size),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, 2)
            )
        else:
            # Predicts 1 value: overall PV score
            self.head = nn.Sequential(
                nn.Linear(hidden_size * 2, hidden_size),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, 1)
            )

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        if token_type_ids is not None and "token_type_ids" in self.encoder.forward.__code__.co_varnames:
            outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        else:
            outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)

        # Use CLS token representation + Mean pooling across valid tokens
        last_hidden = outputs.last_hidden_state  # [B, L, H]
        cls_rep = last_hidden[:, 0, :]           # [B, H]

        mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
        sum_embeddings = torch.sum(last_hidden * mask_expanded, 1)
        sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
        mean_rep = sum_embeddings / sum_mask    # [B, H]

        combined = torch.cat([cls_rep, mean_rep], dim=1) # [B, 2*H]
        logits = self.head(self.dropout(combined))
        return logits
