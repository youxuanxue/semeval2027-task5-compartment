"""
Quick validation check of the GroupKFold loader and evaluator.
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.loader import Task5Dataset
from src.metrics.evaluator import evaluate_predictions

def main():
    tsv = "data/train/en-nn-train.tsv"
    ds = Task5Dataset(tsv, task_type="nn")
    print(f"Loaded {len(ds.records)} records from {tsv}")

    folds = ds.group_k_fold(k=5, seed=42)
    print(f"Created {len(folds)} GroupKFold splits:")
    for i, (tr, va) in enumerate(folds):
        tr_targets = set(r["target"] for r in tr)
        va_targets = set(r["target"] for r in va)
        intersection = tr_targets.intersection(va_targets)
        assert len(intersection) == 0, f"Target leakage detected in fold {i}!"
        print(f"  Fold {i+1}: Train samples={len(tr)} ({len(tr_targets)} targets), Val samples={len(va)} ({len(va_targets)} targets) | Leakage: {len(intersection)}")

    # Test baseline dummy score evaluation on fold 1
    val_data = folds[0][1]
    targets = [r["target"] for r in val_data]
    y_true_mod = np.array([r["mod_score"] for r in val_data])
    # Dummy constant/mean prediction test
    y_pred_dummy = np.random.uniform(1.0, 4.0, size=len(y_true_mod))
    metrics = evaluate_predictions(targets, y_true_mod, y_pred_dummy)
    print("\nDummy random prediction metrics (Fold 1 Mod):", metrics)
    print("[✓] Pipeline Sanity Check Passed!")

if __name__ == "__main__":
    main()
