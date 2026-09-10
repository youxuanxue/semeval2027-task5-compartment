"""
Commander Operations Matrix: All-Tracks Benchmark Suite
Executes rapid benchmark across all 4 subtasks (en-nn, en-pv, de-nn, de-pv)
using native MPS acceleration, logging exact validation metrics per subtask.
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
from src.models.cross_encoder import CompositionalityCrossEncoder
from src.models.loss import PairwiseSpearmanLoss
from scripts.train_baseline import SemEvalDataset, train_epoch, evaluate


def run_track_benchmark(track_name, tsv_path, task_type, model_name, epochs=2, batch_size=16, lr=3e-5):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print("\n====================================================================")
    print(f"[*] ATTACK RUN: Track [{track_name.upper()}] | File: {tsv_path}")
    print(f"[*] Backbone: {model_name} | Device: {device} | Epochs: {epochs}")
    print("====================================================================")

    ds = Task5Dataset(tsv_path, task_type=task_type)
    folds = ds.group_k_fold(k=5, seed=42)
    train_data, val_data = folds[0]  # Fold 1 benchmark

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    train_loader = DataLoader(SemEvalDataset(train_data, tokenizer, task_type=task_type), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(SemEvalDataset(val_data, tokenizer, task_type=task_type), batch_size=batch_size, shuffle=False)

    model = CompositionalityCrossEncoder(model_name, task_type=task_type).to(device)
    criterion = PairwiseSpearmanLoss(rank_weight=0.7, reg_weight=0.3, margin=0.15)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps * 0.1), num_training_steps=total_steps)

    best_metrics = None
    for epoch in range(1, epochs + 1):
        loss = train_epoch(model, train_loader, optimizer, scheduler, criterion, device)
        val_res = evaluate(model, val_loader, device, task_type=task_type)
        print(f"  Epoch {epoch}/{epochs} - Loss: {loss:.4f} | Val Avg Spearman Rho: {val_res['avg_spearman_rho']:.4f}")
        best_metrics = val_res

    return best_metrics


def main():
    tracks = [
        ("en-nn", "data/train/en-nn-train.tsv", "nn", "google-bert/bert-base-uncased"),
        ("en-pv", "data/train/en-pv-train.tsv", "pv", "google-bert/bert-base-uncased"),
        ("de-nn", "data/train/de-nn-train.tsv", "nn", "google-bert/bert-base-multilingual-cased"),
        ("de-pv", "data/train/de-pv-train.tsv", "pv", "google-bert/bert-base-multilingual-cased"),
    ]

    results_summary = {}
    for track_name, tsv_path, task_type, model_name in tracks:
        res = run_track_benchmark(track_name, tsv_path, task_type, model_name, epochs=2, batch_size=16)
        results_summary[track_name] = res["avg_spearman_rho"]

    print("\n====================================================================")
    print("[🏆] ALL-TRACK BENCHMARK SUMMARY (Fold-1 Zero-Leakage Spearman Rho):")
    for t, s in results_summary.items():
        print(f"    - Track {t.ljust(8)} : Spearman Rho = {s:.4f}")
    overall_mean = float(np.mean(list(results_summary.values())))
    print("    ------------------------------------------------")
    print(f"    - GRAND MEAN SPEARMAN RHO: {overall_mean:.4f}")
    print("====================================================================")


if __name__ == "__main__":
    main()
