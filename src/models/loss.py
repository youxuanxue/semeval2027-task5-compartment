"""
Hybrid Ranking & Regression Loss for Spearman Rho Optimization
Directly optimizes relative pairwise ordering within targets while anchoring absolute scale.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PairwiseSpearmanLoss(nn.Module):
    def __init__(self, rank_weight: float = 0.6, reg_weight: float = 0.4, margin: float = 0.1):
        super().__init__()
        self.rank_weight = rank_weight
        self.reg_weight = reg_weight
        self.margin = margin
        self.reg_loss = nn.SmoothL1Loss()

    def forward(self, preds: torch.Tensor, targets: torch.Tensor):
        """
        preds: [B, D]
        targets: [B, D] (where D is 2 for NN, 1 for PV)
        """
        # 1. Smooth L1 regression loss for value anchoring
        l_reg = self.reg_loss(preds, targets)

        # 2. Pairwise ranking loss across batch
        batch_size, dims = preds.size()
        if batch_size < 2:
            return l_reg

        l_rank = torch.tensor(0.0, device=preds.device)
        for d in range(dims):
            p_d = preds[:, d]   # [B]
            t_d = targets[:, d] # [B]

            diff_p = p_d.unsqueeze(1) - p_d.unsqueeze(0) # [B, B]
            diff_t = t_d.unsqueeze(1) - t_d.unsqueeze(0) # [B, B]

            sign_t = torch.sign(diff_t)
            # Only consider non-zero target differences
            valid_mask = (torch.abs(diff_t) > 0.05).float()

            # Hinge loss: max(0, -sign_t * diff_p + margin)
            hinge = F.relu(-sign_t * diff_p + self.margin) * valid_mask
            num_valid = torch.clamp(valid_mask.sum(), min=1.0)
            l_rank = l_rank + (hinge.sum() / num_valid)

        l_rank = l_rank / dims
        return self.reg_weight * l_reg + self.rank_weight * l_rank
