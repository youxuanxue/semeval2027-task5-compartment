"""
Official Metric Evaluator for SemEval-2027 Task 5 (ComPartMent)
Calculates:
1. Spearman's rho for ranking correlation (Primary Metric)
2. Context variance correlation (Sensitivity to contextual variability)
3. RMSE (Error magnitude)
"""

import math
from typing import Dict, List, Tuple
import numpy as np


def rank_data(a: np.ndarray) -> np.ndarray:
    """Assign ranks to data, dealing with ties by averaging."""
    sorter = np.argsort(a)
    inv = np.empty(sorter.size, dtype=np.intp)
    inv[sorter] = np.arange(sorter.size)

    a_sorted = a[sorter]
    obs = np.r_[True, a_sorted[1:] != a_sorted[:-1]]
    dense = obs.cumsum()[inv]

    # cumulative counts of each unique value
    count = np.r_[np.nonzero(obs)[0], len(a)]
    return 0.5 * (count[dense] + count[dense - 1] + 1)


def spearman_rho(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Spearman rank-order correlation coefficient."""
    if len(x) < 2:
        return 0.0
    rx = rank_data(x)
    ry = rank_data(y)
    vx = rx - np.mean(rx)
    vy = ry - np.mean(ry)
    denom = np.sqrt(np.sum(vx**2) * np.sum(vy**2))
    if denom == 0:
        return 0.0
    return float(np.sum(vx * vy) / denom)


def rmse_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Root Mean Squared Error."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def evaluate_predictions(
    targets: List[str],
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    """
    Computes all three official evaluation metrics:
    - spearman_rho (ranking)
    - context_var_rho (correlation of standard deviations across contexts per target)
    - rmse (absolute scale error)
    """
    overall_rho = spearman_rho(y_true, y_pred)
    overall_rmse = rmse_score(y_true, y_pred)

    # Compute context variance correlation per target
    target_groups: Dict[str, Tuple[List[float], List[float]]] = {}
    for t, true_val, pred_val in zip(targets, y_true, y_pred):
        if t not in target_groups:
            target_groups[t] = ([], [])
        target_groups[t][0].append(float(true_val))
        target_groups[t][1].append(float(pred_val))

    true_stds = []
    pred_stds = []
    for t, (t_vals, p_vals) in target_groups.items():
        if len(t_vals) >= 2:
            true_stds.append(np.std(t_vals, ddof=1) if len(t_vals) > 1 else 0.0)
            pred_stds.append(np.std(p_vals, ddof=1) if len(p_vals) > 1 else 0.0)

    if len(true_stds) >= 2:
        context_var_rho = spearman_rho(np.array(true_stds), np.array(pred_stds))
    else:
        context_var_rho = 0.0

    return {
        "spearman_rho": overall_rho,
        "context_var_rho": context_var_rho,
        "rmse": overall_rmse,
    }
