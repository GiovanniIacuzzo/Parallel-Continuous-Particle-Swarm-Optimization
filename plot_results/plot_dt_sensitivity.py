#!/usr/bin/env python3
"""Plot the analysed Delta-t profile from dt_family_profile.csv."""
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

from plot_results.common import COLORS, configure_style, read_csv, read_json, save_figure


def generate(analysis_dir: Path, output: Path) -> list[Path]:
    source = analysis_dir / "dt_family_profile.csv"
    rows = read_csv(source, required=("dt", "median_family_effect_log10", "seed_bootstrap_95_low", "seed_bootstrap_95_high"))
    rows.sort(key=lambda r: float(r["dt"]))
    x = np.asarray([float(r["dt"]) for r in rows], dtype=float)
    y = np.asarray([float(r["median_family_effect_log10"]) for r in rows], dtype=float)
    low = np.asarray([float(r["seed_bootstrap_95_low"]) for r in rows], dtype=float)
    high = np.asarray([float(r["seed_bootstrap_95_high"]) for r in rows], dtype=float)
    if not np.all(np.isfinite(np.concatenate((x, y, low, high)))):
        raise ValueError(f"Valori non finiti in {source}")
    if np.any(x <= 0) or len(np.unique(x)) != len(x):
        raise ValueError("I valori dt devono essere positivi e non duplicati per l'asse logaritmico")
    validation = read_json(analysis_dir / "validation.json")
    reference_dt = float(validation["reference_dt"]) if "reference_dt" in validation else None
    configure_style()
    fig, ax = plt.subplots(figsize=(6.8, 3.65))
    ax.fill_between(x, low, high, color=COLORS["blue"], alpha=0.16, linewidth=0,
                    label="Paired-seed bootstrap 95% interval")
    ax.plot(x, y, marker="o", markersize=4.2, linewidth=1.6, color=COLORS["navy"],
            label="Median across selected benchmark functions")
    ax.axhline(0.0, color=COLORS["grey"], linewidth=0.85, linestyle="--")
    if reference_dt is not None:
        ax.axvline(reference_dt, color=COLORS["red"], linewidth=0.9, linestyle=":",
                   label=fr"Reference $\Delta t={reference_dt:g}$")
    ax.set_xscale("log")
    ax.set_xticks(x, [f"{v:g}" for v in x])
    ax.set_xlabel(r"CPSO interval length, $\Delta t$ (log scale)")
    ax.set_ylabel(r"Median paired $\Delta\log_{10}(f)$")
    evidence = validation.get("evidence_class", "not specified")
    suffix = " — exploratory pilot" if evidence == "exploratory" else ""
    ax.set_title("Sensitivity profile for the selected benchmark set" + suffix)
    ax.grid(axis="y")
    ax.legend(frameon=False, fontsize=6.8)
    note = "Interval is conditional on the selected benchmark functions and dimensions; it is not a population-level CI."
    fig.text(0.5, 0.005, note, ha="center", fontsize=6.2, color=COLORS["grey"])
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    paths = list(save_figure(fig, output, "fig_dt_sensitivity_family_profile"))
    print(f"[OK] Delta-t profile from {source}")
    for path in paths:
        print(f"     {path}")
    return paths


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dir", type=Path, default=Path("results/analysis/cpso_dt_sensitivity_m2_screening"))
    parser.add_argument("--output", type=Path, default=Path("paper/figures/dt_m2_screening"))
    args = parser.parse_args()
    try:
        generate(args.analysis_dir, args.output)
    except (FileNotFoundError, ValueError) as exc:
        print(f"[SKIP] {exc}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
