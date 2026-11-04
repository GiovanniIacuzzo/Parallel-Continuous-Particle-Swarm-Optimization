"""Paired non-parametric inference helpers for optimizer experiments."""
from __future__ import annotations

from typing import Any

import numpy as np


def holm_adjust(p_values: list[float]) -> list[float]:
    """Holm step-down family-wise adjusted p-values, returned in input order."""
    if not p_values:
        return []
    p = np.asarray(p_values, dtype=float)
    if np.any(~np.isfinite(p)) or np.any((p < 0.0) | (p > 1.0)):
        raise ValueError("p-values must be finite and lie in [0, 1].")
    order = np.argsort(p)
    adjusted_sorted = np.empty_like(p)
    running = 0.0
    m = len(p)
    for rank, idx in enumerate(order):
        candidate = min(1.0, (m - rank) * p[idx])
        running = max(running, candidate)
        adjusted_sorted[rank] = running
    adjusted = np.empty_like(p)
    for rank, idx in enumerate(order):
        adjusted[idx] = adjusted_sorted[rank]
    return adjusted.tolist()


def paired_wilcoxon(a: list[float] | np.ndarray,
                     b: list[float] | np.ndarray) -> dict[str, Any]:
    """Two-sided paired Wilcoxon signed-rank test and matched-pairs effect size.

    The effect is the matched-pairs rank-biserial correlation. The location
    estimate is the median paired difference A-B. Zero differences are retained
    in the median but omitted from the signed-rank sum using zero_method='wilcox'.
    """
    try:
        from scipy.stats import rankdata, wilcoxon
    except ImportError as exc:  # pragma: no cover - dependency is declared in [dev]
        raise RuntimeError("Paired inference requires SciPy; install with `pip install -e .[dev]`.") from exc

    x = np.asarray(a, dtype=float)
    y = np.asarray(b, dtype=float)
    if x.shape != y.shape or x.ndim != 1 or x.size < 2:
        raise ValueError("Paired samples must be one-dimensional, equal-length and contain >=2 pairs.")
    if np.any(~np.isfinite(x)) or np.any(~np.isfinite(y)):
        raise ValueError("Paired samples must contain only finite values.")

    differences = x - y
    nonzero = differences[differences != 0.0]
    if nonzero.size == 0:
        statistic, p_value, rank_biserial = 0.0, 1.0, 0.0
    else:
        result = wilcoxon(x, y, alternative="two-sided", zero_method="wilcox", method="auto")
        statistic = float(result.statistic)
        p_value = float(result.pvalue)
        ranks = rankdata(np.abs(nonzero), method="average")
        positive = float(np.sum(ranks[nonzero > 0.0]))
        negative = float(np.sum(ranks[nonzero < 0.0]))
        rank_biserial = (positive - negative) / (positive + negative)

    return {
        "n_pairs": int(x.size),
        "median_paired_difference_A_minus_B": float(np.median(differences)),
        "mean_paired_difference_A_minus_B": float(np.mean(differences)),
        "wilcoxon_statistic": statistic,
        "p_value": p_value,
        "matched_pairs_rank_biserial": float(rank_biserial),
    }
