"""
Agent Echo & Alpha: V2 Trial Inference and Benchmark Verification
Loads the fully fine-tuned V2 checkpoints, runs inference on trial dataset,
evaluates exactly against ground truth, and builds the updated submission zip.
"""

import os
import sys
import csv
import torch
import numpy as np
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.heavy_encoder import HeavyCompositionalityCrossEncoder
from src.metrics.evaluator import evaluate_predictions
from src.pipeline.packager import write_predictions, build_submission_zip
from scripts.train_cv import EnhancedSemEvalDataset
from scripts.make_trial_submission import load_trial_records


def run_v2_trial_evaluation():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model_name = "google-bert/bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    os.makedirs("submissions/v2", exist_ok=True)
    generated_tsvs = []

    # 1. Evaluate EN-NN with fine-tuned checkpoint
    print("\n[*] Running V2 Model on EN-NN Trial Data...")
    nn_records = load_trial_records("data/trial/en-nn-trial.tsv", task_type="nn")
    # Also load the ground-truth from the file to measure trial metrics!
    with open("data/trial/en-nn-trial.tsv", "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="	")
        nn_truth = list(reader)

    nn_loader = DataLoader(
        EnhancedSemEvalDataset(nn_records, tokenizer, task_type="nn", lang="en"),
        batch_size=16,
        shuffle=False,
    )
    nn_model = HeavyCompositionalityCrossEncoder(model_name, task_type="nn").to(device)
    nn_model.load_state_dict(torch.load("checkpoints/v2_models/en_nn_best.pt", map_location=device))
    nn_model.eval()

    nn_preds_raw = []
    with torch.no_grad():
        for batch in nn_loader:
            p = nn_model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
            nn_preds_raw.append(p.cpu().numpy())

    nn_preds_arr = np.concatenate(nn_preds_raw, axis=0)
    nn_pred_dicts = []
    for r, p in zip(nn_records, nn_preds_arr):
        nn_pred_dicts.append({
            "ContextID": r["id"],
            "mod_pred": float(p[0]),
            "head_pred": float(p[1]),
        })

    nn_tsv = "submissions/v2/en-nn-pred.tsv"
    write_predictions(nn_tsv, nn_pred_dicts, task_type="nn")
    generated_tsvs.append(nn_tsv)

    # Compute exact Trial Metrics for NN
    targets_nn = [r["Compound"] for r in nn_truth]
    mod_true = np.array([float(r["ModAvg"]) for r in nn_truth])
    head_true = np.array([float(r["HeadAvg"]) for r in nn_truth])
    mod_metrics = evaluate_predictions(targets_nn, mod_true, nn_preds_arr[:, 0])
    head_metrics = evaluate_predictions(targets_nn, head_true, nn_preds_arr[:, 1])
    print(f"  [+] EN-NN Trial Metrics:")
    print(f"      - Mod Rho : {mod_metrics['spearman_rho']:.4f} (RMSE: {mod_metrics['rmse']:.4f})")
    print(f"      - Head Rho: {head_metrics['spearman_rho']:.4f} (RMSE: {head_metrics['rmse']:.4f})")

    # 2. Evaluate EN-PV with fine-tuned checkpoint
    print("\n[*] Running V2 Model on EN-PV Trial Data...")
    pv_records = load_trial_records("data/trial/en-pv-trial.tsv", task_type="pv")
    with open("data/trial/en-pv-trial.tsv", "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="	")
        pv_truth = list(reader)

    pv_loader = DataLoader(
        EnhancedSemEvalDataset(pv_records, tokenizer, task_type="pv", lang="en"),
        batch_size=16,
        shuffle=False,
    )
    pv_model = HeavyCompositionalityCrossEncoder(model_name, task_type="pv").to(device)
    pv_model.load_state_dict(torch.load("checkpoints/v2_models/en_pv_best.pt", map_location=device))
    pv_model.eval()

    pv_preds_raw = []
    with torch.no_grad():
        for batch in pv_loader:
            p = pv_model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
            pv_preds_raw.append(p.cpu().numpy())

    pv_preds_arr = np.concatenate(pv_preds_raw, axis=0)
    pv_pred_dicts = []
    for r, p in zip(pv_records, pv_preds_arr):
        pv_pred_dicts.append({
            "ContextID": r["id"],
            "pred": float(p[0]),
        })

    pv_tsv = "submissions/v2/en-pv-pred.tsv"
    write_predictions(pv_tsv, pv_pred_dicts, task_type="pv")
    generated_tsvs.append(pv_tsv)

    # Compute exact Trial Metrics for PV
    targets_pv = [r["ParticleVerb"] for r in pv_truth]
    pv_true = np.array([float(r["Avg"]) for r in pv_truth])
    pv_metrics = evaluate_predictions(targets_pv, pv_true, pv_preds_arr[:, 0])
    print(f"  [+] EN-PV Trial Metrics:")
    print(f"      - PV Rho  : {pv_metrics['spearman_rho']:.4f} (RMSE: {pv_metrics['rmse']:.4f})")

    # 3. Build V2 Trial Submission Package
    zip_path = "submissions/trial_submission_v2.zip"
    build_submission_zip(generated_tsvs, zip_path)
    print(f"\n[🏆] Production V2 Package Built: {zip_path}")


if __name__ == "__main__":
    run_v2_trial_evaluation()
