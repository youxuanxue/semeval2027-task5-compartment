"""
Agent Charlie & Echo: Cross-Validation Orchestrator
Executes complete 5-Fold training, saves out-of-fold (OOF) predictions and model checkpoints.
"""

import os
import sys
import argparse
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.loader import Task5Dataset
from src.metrics.evaluator import evaluate_predictions
from src.models.heavy_encoder import HeavyCompositionalityCrossEncoder
from src.models.loss import PairwiseSpearmanLoss
from src.features.ablation import generate_ablation_pairs


class EnhancedSemEvalDataset(Dataset):
    def __init__(self, records, tokenizer, task_type="nn", lang="en", max_length=160):
        self.records = records
        self.tokenizer = tokenizer
        self.task_type = task_type
        self.lang = lang
        self.max_length = max_length

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        item = self.records[idx]
        ablation = generate_ablation_pairs(item, self.task_type, self.lang)

        if self.task_type == "nn":
            # Formulate rich relational prompt including prototypes
            text_a = f"Compound: {item['target']} | Modifier: {item['mod']} (e.g. {ablation['mod_sub']}) | Head: {item['head']} (e.g. {ablation['head_sub']})"
            target_scores = torch.tensor([item["mod_score"], item["head_score"]], dtype=torch.float)
        else:
            text_a = f"Particle Verb: {item['target']} | Verb: {item['base']} (e.g. {ablation['base_sub']}) | Particle: {item['particle']}"
            target_scores = torch.tensor([item["score"]], dtype=torch.float)

        text_b = item["context"]
        encoded = self.tokenizer(
            text_a,
            text_b,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "target_scores": target_scores,
            "target": item["target"],
            "id": item["id"],
        }


def run_5_fold_training(
    tsv_path: str,
    task_type: str,
    lang: str,
    model_name: str,
    batch_size: int = 16,
    epochs: int = 3,
    lr: float = 2.5e-5,
    save_dir: str = "checkpoints/heavy_model",
):
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print("\n=======================================================")
    print(f"[*] Launching 5-Fold Training: {tsv_path} ({task_type.upper()}, {lang.upper()})")
    print(f"[*] Backbone: {model_name} | Device: {device}")
    print("=======================================================")

    ds = Task5Dataset(tsv_path, task_type=task_type)
    folds = ds.group_k_fold(k=5, seed=42)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    oof_preds = []
    oof_targets = []
    oof_target_names = []
    fold_scores = []

    for fold_idx, (train_data, val_data) in enumerate(folds, 1):
        print(f"\n--- Fold {fold_idx}/5 (Train: {len(train_data)}, Val: {len(val_data)}) ---")
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
        criterion = PairwiseSpearmanLoss(rank_weight=0.7, reg_weight=0.3, margin=0.15)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

        total_steps = len(train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps * 0.1), num_training_steps=total_steps)

        best_rho = -1.0
        best_val_preds = None

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

            # Evaluate
            model.eval()
            val_p = []
            val_t = []
            val_names = []
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
            else:
                m = evaluate_predictions(val_names, t_arr[:, 0], p_arr[:, 0])
                epoch_rho = m["spearman_rho"]

            print(f"  Epoch {epoch}/{epochs} | Loss: {train_loss:.4f} | Val Rho: {epoch_rho:.4f}")
            if epoch_rho > best_rho:
                best_rho = epoch_rho
                best_val_preds = p_arr
                # Save best fold weights
                torch.save(model.state_dict(), f"{save_dir}/{lang}_{task_type}_fold{fold_idx}.pt")

        fold_scores.append(best_rho)
        oof_preds.append(best_val_preds)
        oof_targets.append(np.concatenate([b["target_scores"].numpy() for b in val_loader], axis=0))
        oof_target_names.extend([t for b in val_loader for t in b["target"]])
        print(f"  [+] Fold {fold_idx} Best Rho: {best_rho:.4f}")

    print("\n[✓] 5-Fold Cross-Validation Completed!")
    print(f"    Per-fold Rhos: {[round(s, 4) for s in fold_scores]}")
    print(f"    Mean CV Spearman Rho: {np.mean(fold_scores):.4f} (±{np.std(fold_scores):.4f})")
    return fold_scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tsv", type=str, default="data/train/en-nn-train.tsv")
    parser.add_argument("--task_type", type=str, default="nn")
    parser.add_argument("--lang", type=str, default="en")
    parser.add_argument("--model_name", type=str, default="google-bert/bert-base-uncased")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()

    run_5_fold_training(
        tsv_path=args.tsv,
        task_type=args.task_type,
        lang=args.lang,
        model_name=args.model_name,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )
