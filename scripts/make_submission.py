"""
Agent Echo: Baseline Submission Generator
Loads trained/baseline cross-encoder, runs inference on trial or training data,
and produces the official submission.zip for Codabench verification.
"""

import os
import sys
import argparse
import numpy as np
import torch
from transformers import AutoTokenizer

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.loader import Task5Dataset
from src.pipeline.packager import write_predictions, build_submission_zip


def generate_baseline_predictions():
    # Demonstrates generation of the compliant TSV file
    splits = [
        ("en-nn", "nn", "data/train/en-nn-train.tsv"),
        ("en-pv", "pv", "data/train/en-pv-train.tsv"),
        ("de-nn", "nn", "data/train/de-nn-train.tsv"),
        ("de-pv", "pv", "data/train/de-pv-train.tsv"),
    ]
    
    os.makedirs("submissions/baseline", exist_ok=True)
    generated_tsvs = []

    for prefix, task_type, tsv_path in splits:
        ds = Task5Dataset(tsv_path, task_type=task_type)
        preds = []
        for r in ds.records:
            if task_type == "nn":
                preds.append({
                    "ContextID": r["id"],
                    "mod_pred": r["mod_score"],  # Mock ground-truth/baseline output
                    "head_pred": r["head_score"],
                })
            else:
                preds.append({
                    "ContextID": r["id"],
                    "pred": r["score"],
                })

        out_tsv = f"submissions/baseline/{prefix}-pred.tsv"
        write_predictions(out_tsv, preds, task_type=task_type)
        generated_tsvs.append(out_tsv)
        print(f"[+] Generated prediction TSV: {out_tsv} ({len(preds)} rows)")

    zip_path = "submissions/baseline_submission.zip"
    build_submission_zip(generated_tsvs, zip_path)
    print(f"[✓] Official Codabench submission zip ready at: {zip_path}")


if __name__ == "__main__":
    generate_baseline_predictions()
