"""
Agent Charlie & Bravo: V2 Model Trainer with Prototype In-Filling
Fine-tunes Cross-Encoder with Lexical Prototype Ablation Features and Pairwise Ranking Loss.
Saves optimal weights for production trial inference.
"""

import os
import sys
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.loader import Task5Dataset
from src.metrics.evaluator import evaluate_predictions
from src.models.heavy_encoder import HeavyCompositionalityCrossEncoder
from src.models.loss import PairwiseSpearmanLoss
from scripts.train_cv import EnhancedSemEvalDataset


def train_production_model(
    tsv_path: str,
    task_type: str,
    lang: str,
    model_name: str,
    save_path: str,
    epochs: int = 3,
    batch_size: int = 16,
    lr: float = 2.5e-5,
):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"\n[*] Training Production V2 Model for [{task_type.upper()}, {lang.upper()}] on {device}")
    
    ds = Task5Dataset(tsv_path, task_type=task_type)
    folds = ds.group_k_fold(k=5, seed=42)
    train_data, val_data = folds[0]  # Use Fold 1 validation for early stopping
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    train_loader = DataLoader(
        EnhancedSemEvalDataset(train_data, tokenizer, task_type, lang),
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        EnhancedSemEvalDataset(val_data, tokenizer, task_type, lang),
        batch_size=batch_size,
        shuffle=False,
    )

    model = HeavyCompositionalityCrossEncoder(model_name, task_type=task_type).to(device)
    criterion = PairwiseSpearmanLoss(rank_weight=0.75, reg_weight=0.25, margin=0.15)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps * 0.1), num_training_steps=total_steps)

    best_rho = -1.0

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            targets = batch["target_scores"].to(device)

            optimizer.zero_grad()
            preds = model(input_ids, attention_mask)
            loss = criterion(preds, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validation
        model.eval()
        val_p, val_t, val_names = [], [], []
        with torch.no_grad():
            for batch in val_loader:
                p = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
                val_p.append(p.cpu().numpy())
                val_t.append(batch["target_scores"].numpy())
                val_names.extend(batch["target"])

        p_arr = np.concatenate(val_p, axis=0)
        t_arr = np.concatenate(val_t, axis=0)

        if task_type == "nn":
            m1 = evaluate_predictions(val_names, t_arr[:, 0], p_arr[:, 0])
            m2 = evaluate_predictions(val_names, t_arr[:, 1], p_arr[:, 1])
            epoch_rho = (m1["spearman_rho"] + m2["spearman_rho"]) / 2.0
            print(f"  Epoch {epoch}/{epochs} - Train Loss: {train_loss:.4f} | Mod Rho: {m1['spearman_rho']:.4f} | Head Rho: {m2['spearman_rho']:.4f} | Avg Rho: {epoch_rho:.4f}")
        else:
            m = evaluate_predictions(val_names, t_arr[:, 0], p_arr[:, 0])
            epoch_rho = m["spearman_rho"]
            print(f"  Epoch {epoch}/{epochs} - Train Loss: {train_loss:.4f} | PV Rho: {epoch_rho:.4f}")

        if epoch_rho > best_rho:
            best_rho = epoch_rho
            torch.save(model.state_dict(), save_path)
            print(f"    [+] Saved optimal model checkpoint (Best Rho: {best_rho:.4f}) -> {save_path}")

    return best_rho


def main():
    os.makedirs("checkpoints/v2_models", exist_ok=True)
    # Train production model for EN-NN
    train_production_model(
        tsv_path="data/train/en-nn-train.tsv",
        task_type="nn",
        lang="en",
        model_name="google-bert/bert-base-uncased",
        save_path="checkpoints/v2_models/en_nn_best.pt",
        epochs=3,
        batch_size=16,
    )
    # Train production model for EN-PV
    train_production_model(
        tsv_path="data/train/en-pv-train.tsv",
        task_type="pv",
        lang="en",
        model_name="google-bert/bert-base-uncased",
        save_path="checkpoints/v2_models/en_pv_best.pt",
        epochs=3,
        batch_size=16,
    )


if __name__ == "__main__":
    main()
