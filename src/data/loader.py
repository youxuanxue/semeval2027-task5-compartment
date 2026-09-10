"""
Data loading and Group-K-Fold splitting pipeline.
Strict target-level isolation ensures no leakage across folds.
"""

import csv
import random
from typing import Dict, List, Tuple
import numpy as np


class Task5Dataset:
    def __init__(self, tsv_path: str, task_type: str):
        """
        task_type: 'nn' (noun compound) or 'pv' (particle verb)
        """
        self.tsv_path = tsv_path
        self.task_type = task_type
        self.records = []
        self._load()

    def _load(self):
        with open(self.tsv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="	")
            for row in reader:
                if self.task_type == "nn":
                    record = {
                        "id": row["ContextID"],
                        "target": row["Compound"],
                        "mod": row["Mod"],
                        "head": row["Head"],
                        "mod_score": float(row["ModAvg"]),
                        "head_score": float(row["HeadAvg"]),
                        "context": row["Context"],
                    }
                else:  # pv
                    record = {
                        "id": row["ContextID"],
                        "target": row["ParticleVerb"],
                        "base": row["Base"],
                        "particle": row["Particle"],
                        "score": float(row["Avg"]),
                        "context": row["Context"],
                    }
                self.records.append(record)

    def group_k_fold(self, k: int = 5, seed: int = 42) -> List[Tuple[List[dict], List[dict]]]:
        """
        Splits dataset into k folds grouped strictly by target (word level).
        Guarantees that targets in val_fold never appear in train_fold.
        """
        # Group records by target
        target_to_records: Dict[str, List[dict]] = {}
        for r in self.records:
            t = r["target"]
            target_to_records.setdefault(t, []).append(r)

        unique_targets = list(target_to_records.keys())
        rng = random.Random(seed)
        rng.shuffle(unique_targets)

        # Distribute targets evenly across k buckets
        target_folds: List[List[str]] = [[] for _ in range(k)]
        fold_counts = [0] * k
        for t in unique_targets:
            # allocate to fold with lowest record count
            min_idx = int(np.argmin(fold_counts))
            target_folds[min_idx].append(t)
            fold_counts[min_idx] += len(target_to_records[t])

        folds = []
        for i in range(k):
            val_targets = set(target_folds[i])
            val_data = [r for t in val_targets for r in target_to_records[t]]
            train_data = [r for t, recs in target_to_records.items() if t not in val_targets for r in recs]
            folds.append((train_data, val_data))

        return folds
