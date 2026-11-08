#!/usr/bin/env python3
"""Create the descriptive BBOB and CEC rank plots from external_rank_summary.csv.

The official COCO ECDF/runtime-to-target plots must still be generated with
cocopp; this script only plots the repository's supplementary rank summaries.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plot_results.common import COLORS, configure_style, read_csv, save_figure


def _one_plot(rows: list[dict[str, str]], *, metric: str, suite: str,
              output: Path, filename: str, campaign: str | None) -> list[Path]:
    selected = [r for r in rows if r.get("metric") == metric and suite.lower() in r.get("suite", "").lower()]
    if campaign:
        selected = [r for r in selected if r.get("campaign") == campaign]
    campaigns = sorted({r.get("campaign", "") for r in selected})
    if len(campaigns) != 1:
        if not campaigns:
            raise ValueError(f"Nessun record per suite={suite}, metric={metric}")
        raise ValueError(f"Trovate più campagne {campaigns}; specificare --{suite}-campaign")
    selected.sort(key=lambda r: (float(r["mean_rank_function_balanced"]), r.get("method", "")))
    methods = [r["method"] for r in selected]
    means = np.asarray([float(r["mean_rank_function_balanced"]) for r in selected])
    lows = np.asarray([float(r["bootstrap_95_low_function_cluster"]) for r in selected])
    highs = np.asarray([float(r["bootstrap_95_high_function_cluster"]) for r in selected])
    if not np.all(np.isfinite(means)):
        raise ValueError(f"Ranghi non finiti per {campaigns[0]}")
    ci_valid = np.isfinite(lows) & np.isfinite(highs)
    errors = np.vstack([np.maximum(0.0, means[ci_valid] - lows[ci_valid]),
                        np.maximum(0.0, highs[ci_valid] - means[ci_valid])])
    configure_style()
    fig, ax = plt.subplots(figsize=(7.0, max(3.0, 0.31 * len(methods) + 0.8)))
    y = np.arange(len(methods))
    # Highlight the CPSO entry without implying that lower rank alone is a
    # complete anytime-performance assessment.
    colors = [COLORS["red"] if "cpso" in method.lower() else COLORS["blue"] for method in methods]
    ax.barh(y, means, color=colors, alpha=0.88, height=0.64)
    if np.any(ci_valid):
        ax.errorbar(means[ci_valid], y[ci_valid], xerr=errors, fmt="none", ecolor=COLORS["navy"], capsize=2.1, linewidth=0.85)
    ax.set_yticks(y, methods)
    ax.invert_yaxis()
    ax.set_xlabel("Function-balanced average rank (lower is better)")
    ax.set_title(f"{suite}: paired final-performance ranks")
    ax.grid(axis="x")
    ax.set_axisbelow(True)
    ax.axvline(1.0, color=COLORS["grey"], linestyle=":", linewidth=0.7)
    n_groups = selected[0].get("complete_problem_seed_groups", "?")
    n_functions = selected[0].get("function_clusters", "?")
    ci_note = "95% function-cluster intervals shown where defined" if np.any(ci_valid) else "function-cluster intervals unavailable for this sample"
    fig.text(0.5, 0.005,
             f"Campaign: {campaigns[0]} · complete matched problem-seed groups: {n_groups} · function clusters: {n_functions}; {ci_note}. Descriptive ranks; not a substitute for official anytime metrics.",
             ha="center", fontsize=6.1, color=COLORS["grey"])
    fig.tight_layout(rect=(0, 0.055, 1, 1))
    return list(save_figure(fig, output, filename))


def generate(summary_csv: Path, output: Path, *, bbob_campaign: str | None = None,
             cec_campaign: str | None = None) -> dict[str, list[Path] | str]:
    try:
        rows = read_csv(summary_csv, required=("campaign", "suite", "metric", "method", "mean_rank_function_balanced",
                                              "bootstrap_95_low_function_cluster", "bootstrap_95_high_function_cluster"))
    except (FileNotFoundError, ValueError) as exc:
        print(f"[SKIP] external ranks: {exc}")
        return {"input": str(exc)}
    result: dict[str, list[Path] | str] = {}
    tasks = [
        ("bbob", lambda: _one_plot(rows, metric="final_best_value", suite="bbob", output=output,
                                    filename="rank_cpso_bbob_stageC_pilot_v1_final_best_value", campaign=bbob_campaign)),
        ("cec", lambda: _one_plot(rows, metric="best_at_120000_FE", suite="cec2013-lsgo", output=output,
                                   filename="rank_cpso_cec2013_stageC_pilot_v1_best_at_120000_FE", campaign=cec_campaign)),
    ]
    for name, fn in tasks:
        try:
            paths = fn()
            result[name] = paths
            print(f"[OK] {name} rank plot")
            for path in paths:
                print(f"     {path}")
        except (ValueError, FileNotFoundError) as exc:
            result[name] = str(exc)
            print(f"[SKIP] {name}: {exc}")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=Path("paper/figures/external_pilots/external_rank_summary.csv"))
    parser.add_argument("--output", type=Path, default=Path("paper/figures/external_pilots"))
    parser.add_argument("--bbob-campaign", type=str, default=None)
    parser.add_argument("--cec-campaign", type=str, default=None)
    args = parser.parse_args()
    result = generate(args.summary, args.output, bbob_campaign=args.bbob_campaign, cec_campaign=args.cec_campaign)
    return 0 if any(isinstance(value, list) for value in result.values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
