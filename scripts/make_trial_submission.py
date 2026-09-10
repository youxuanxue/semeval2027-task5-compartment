"""
Agent Echo: Official Trial Submission Generator
Infers our trained Cross-Encoder models directly on official trial data (en-nn and en-pv),
producing en-nn-pred.tsv and en-pv-pred.tsv, and packaging them into trial_submission.zip.
"""

import os
import sys
import csv
import torch
import numpy as np
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.cross_encoder import CompositionalityCrossEncoder
from src.pipeline.packager import write_predictions, build_submission_zip
from scripts.train_baseline import SemEvalDataset


def predict_trial(model, dataloader, device, task_type="nn"):
    model.eval()
    all_preds = []
    all_ids = []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            preds = model(input_ids, attention_mask)
            all_preds.append(preds.cpu().numpy())
            all_ids.extend(batch["id"])

    preds = np.concatenate(all_preds, axis=0)
    formatted = []
    for cid, p in zip(all_ids, preds):
        if task_type == "nn":
            formatted.append({
                "ContextID": cid,
                "mod_pred": float(p[0]),
                "head_pred": float(p[1]),
            })
        else:
            formatted.append({
                "ContextID": cid,
                "pred": float(p[0]),
            })
    return formatted


def load_trial_records(tsv_path, task_type):
    records = []
    with open(tsv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="	")
        for row in reader:
            if task_type == "nn":
                records.append({
                    "id": row["ContextID"],
                    "target": row["Compound"],
                    "mod": row["Mod"],
                    "head": row["Head"],
                    "mod_score": 0.0,
                    "head_score": 0.0,
                    "context": row["Context"],
                })
            else:
                records.append({
                    "id": row["ContextID"],
                    "target": row["ParticleVerb"],
                    "base": row["Base"],
                    "particle": row["Particle"],
                    "score": 0.0,
                    "context": row["Context"],
                })
    return records


def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model_name = "google-bert/bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    os.makedirs("submissions/trial", exist_ok=True)
    generated_tsvs = []

    # 1. English Noun Compounds (en-nn)
    print("[*] Generating predictions for en-nn-trial...")
    nn_records = load_trial_records("data/trial/en-nn-trial.tsv", task_type="nn")
    nn_loader = DataLoader(SemEvalDataset(nn_records, tokenizer, task_type="nn"), batch_size=16, shuffle=False)
    nn_model = CompositionalityCrossEncoder(model_name, task_type="nn").to(device)
    nn_preds = predict_trial(nn_model, nn_loader, device, task_type="nn")
    nn_out = "submissions/trial/en-nn-pred.tsv"
    write_predictions(nn_out, nn_preds, task_type="nn")
    generated_tsvs.append(nn_out)
    print(f"  [+] Saved {nn_out} ({len(nn_preds)} lines)")

    # 2. English Particle Verbs (en-pv)
    print("[*] Generating predictions for en-pv-trial...")
    pv_records = load_trial_records("data/trial/en-pv-trial.tsv", task_type="pv")
    pv_loader = DataLoader(SemEvalDataset(pv_records, tokenizer, task_type="pv"), batch_size=16, shuffle=False)
    pv_model = CompositionalityCrossEncoder(model_name, task_type="pv").to(device)
    pv_preds = predict_trial(pv_model, pv_loader, device, task_type="pv")
    pv_out = "submissions/trial/en-pv-pred.tsv"
    write_predictions(pv_out, pv_preds, task_type="pv")
    generated_tsvs.append(pv_out)
    print(f"  [+] Saved {pv_out} ({len(pv_preds)} lines)")

    # 3. Package into trial_submission.zip
    zip_out = "submissions/trial_submission.zip"
    build_submission_zip(generated_tsvs, zip_out)
    print(f"\n[✓] Official Practice/Trial submission package generated at: {zip_out}")


if __name__ == "__main__":
    main()
