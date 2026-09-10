"""
Agent Charlie & Bravo: Cross-Encoder Baseline Training Script
Runs 1-fold or 5-fold CV with Pairwise Ranking Loss.
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
from src.models.cross_encoder import CompositionalityCrossEncoder
from src.models.loss import PairwiseSpearmanLoss


class SemEvalDataset(Dataset):
    def __init__(self, records, tokenizer, task_type="nn", max_length=128):
        self.records = records
        self.tokenizer = tokenizer
        self.task_type = task_type
        self.max_length = max_length

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        item = self.records[idx]
        if self.task_type == "nn":
            text_a = f"Compound: {item['target']} | Modifier: {item['mod']} | Head: {item['head']}"
            target_scores = torch.tensor([item["mod_score"], item["head_score"]], dtype=torch.float)
        else:
            text_a = f"Particle Verb: {item['target']} | Base: {item['base']} | Particle: {item['particle']}"
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


def train_epoch(model, dataloader, optimizer, scheduler, criterion, device):
    model.train()
    total_loss = 0.0
    for batch in dataloader:
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

        total_loss += loss.item()
    return total_loss / len(dataloader)


def evaluate(model, dataloader, device, task_type="nn"):
    model.eval()
    all_preds = []
    all_targets = []
    all_target_names = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            preds = model(input_ids, attention_mask)

            all_preds.append(preds.cpu().numpy())
            all_targets.append(batch["target_scores"].numpy())
            all_target_names.extend(batch["target"])

    preds = np.concatenate(all_preds, axis=0)
    targets = np.concatenate(all_targets, axis=0)

    if task_type == "nn":
        mod_metrics = evaluate_predictions(all_target_names, targets[:, 0], preds[:, 0])
        head_metrics = evaluate_predictions(all_target_names, targets[:, 1], preds[:, 1])
        avg_rho = (mod_metrics["spearman_rho"] + head_metrics["spearman_rho"]) / 2.0
        return {
            "avg_spearman_rho": avg_rho,
            "mod_metrics": mod_metrics,
            "head_metrics": head_metrics,
        }
    else:
        pv_metrics = evaluate_predictions(all_target_names, targets[:, 0], preds[:, 0])
        return {
            "avg_spearman_rho": pv_metrics["spearman_rho"],
            "pv_metrics": pv_metrics,
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tsv", type=str, default="data/train/en-nn-train.tsv")
    parser.add_argument("--task_type", type=str, default="nn")
    parser.add_argument("--model_name", type=str, default="prajjwal1/bert-tiny") # Lightweight for fast testing
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--fold", type=int, default=1)
    args = parser.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"[*] Commencing training on device: {device}")
    print(f"[*] Task: {args.task_type} | Data: {args.tsv} | Backbone: {args.model_name}")

    ds = Task5Dataset(args.tsv, task_type=args.task_type)
    folds = ds.group_k_fold(k=5, seed=42)
    train_data, val_data = folds[args.fold - 1]
    print(f"[*] Fold {args.fold}: Train={len(train_data)}, Val={len(val_data)}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    train_loader = DataLoader(SemEvalDataset(train_data, tokenizer, task_type=args.task_type), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(SemEvalDataset(val_data, tokenizer, task_type=args.task_type), batch_size=args.batch_size, shuffle=False)

    model = CompositionalityCrossEncoder(args.model_name, task_type=args.task_type).to(device)
    criterion = PairwiseSpearmanLoss(rank_weight=0.6, reg_weight=0.4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps * 0.1), num_training_steps=total_steps)

    for epoch in range(1, args.epochs + 1):
        loss = train_epoch(model, train_loader, optimizer, scheduler, criterion, device)
        val_res = evaluate(model, val_loader, device, task_type=args.task_type)
        print(f"Epoch {epoch}/{args.epochs} - Train Loss: {loss:.4f} | Val Avg Spearman Rho: {val_res['avg_spearman_rho']:.4f}")
        if args.task_type == "nn":
            print(f"  -> Mod Rho: {val_res['mod_metrics']['spearman_rho']:.4f} (RMSE: {val_res['mod_metrics']['rmse']:.4f})")
            print(f"  -> Head Rho: {val_res['head_metrics']['spearman_rho']:.4f} (RMSE: {val_res['head_metrics']['rmse']:.4f})")

    print("[✓] Baseline Cross-Encoder execution completed successfully!")


if __name__ == "__main__":
    main()
