#!/usr/bin/env python3
"""Analyze the 100-point CPSO dt grid without altering experimental results.

Reports relative residual at a common FE budget and empirical attainment of
predeclared relative-improvement thresholds. The global optimum of the six
non-negative native objectives is f*=0. These are empirical search-efficiency
metrics; they are not a proof of a mathematical basin of attraction.

Run from repository root:
python analysis/analyze_dt_sweep_100.py \
  --input results/raw/cpso_dt_sweep_100_m2 \
  --output-dir results/analysis/cpso_dt_sweep_100_m2
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analysis.analyze_dt_sensitivity import load_and_validate  # noqa: E402


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"No rows for {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def geometric_edges(values: list[float]) -> np.ndarray:
    v = np.array(values, dtype=float)
    log = np.log(v)
    mid = (log[:-1] + log[1:]) / 2
    edges = np.r_[log[0] - (mid[0] - log[0]), mid,
                  log[-1] + (log[-1] - mid[-1])]
    return np.exp(edges)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    from cpso_serial.output_policy import results_output
    args.output_dir = results_output(args.output_dir)

    config, records, validation = load_and_validate(args.input)
    dts = [float(v) for v in validation["dt_values"]]
    dims = [int(v) for v in validation["dimensions"]]
    benchmarks = list(validation["benchmarks"])
    seeds = list(validation["seeds"])
    if len(dts) != 100:
        raise ValueError(f"This analyzer requires 100 dt values, found {len(dts)}")
    if not (abs(dts[0] - 0.01) <= 1e-10 and abs(dts[-1] - 30.0) <= 1e-10):
        raise ValueError("Unexpected dt interval; expected [0.01,30]")
    if len(records) != len(dts) * len(dims) * len(benchmarks) * len(seeds):
        raise ValueError("Incomplete factorial experiment")

    bucket: dict[tuple[str, int, float], list[float]] = defaultdict(list)
    last_significant: dict[tuple[str, int, float], list[int]] = defaultdict(list)
    for rec in records:
        h = np.asarray(rec["best_history"], dtype=float)
        if h.ndim != 1 or len(h) != int(config["iterations"]) + 1:
            raise ValueError(f"Incorrect best_history length in {rec['run_key']}")
        if not np.all(np.isfinite(h)) or np.any(np.diff(h) > 1e-10 * np.maximum(1.0, np.abs(h[:-1]))):
            raise ValueError(f"History invalid or nonmonotone in {rec['run_key']}")
        first, final = float(h[0]), float(rec["best_value"])
        if first <= 1e-14:
            raise ValueError(f"Near-zero initial gap; cannot normalize {rec['run_key']}")
        # All six objectives have a known zero global minimum. Clip at numerical
        # resolution, but do not label near-optimal objectives as success.
        rho = min(1.0, max(1e-16, final / first))
        key = (str(rec["benchmark"]), int(rec["dimension"]), float(rec["dt"]))
        bucket[key].append(rho)
        # Most recent iteration that improved the incumbent by >1e-6 in
        # relative terms. This is an empirical stagnation diagnostic, not a
        # local stability theorem. First history entry is iteration zero.
        step_gain = (h[:-1] - h[1:]) / np.maximum(np.abs(h[:-1]), 1e-12)
        significant = np.flatnonzero(step_gain > 1e-6)
        last_significant[key].append(int(significant[-1]) + 1 if significant.size else 0)

    per_problem = []
    bykey = {}
    for benchmark in benchmarks:
        for dim in dims:
            for dt in dts:
                v = np.array(bucket[(benchmark, dim, dt)], dtype=float)
                if v.size != len(seeds):
                    raise ValueError(f"Missing paired seeds for {benchmark},D={dim},dt={dt:g}")
                stalls = np.asarray(last_significant[(benchmark, dim, dt)], dtype=int)
                if stalls.size != len(seeds):
                    raise ValueError(f"Missing stagnation data for {benchmark},D={dim},dt={dt:g}")
                row = dict(
                    benchmark=benchmark, dimension=dim, dt=dt, n_seeds=int(v.size),
                    median_residual_fraction=float(np.median(v)),
                    q25_residual_fraction=float(np.quantile(v, 0.25)),
                    q75_residual_fraction=float(np.quantile(v, 0.75)),
                    median_log10_residual=float(np.median(np.log10(v))),
                    median_last_significant_iteration=float(np.median(stalls)),
                    fraction_stalled_before_20pct_budget=float(np.mean(
                        stalls < 0.2 * int(config["iterations"]))),
                    fraction_90percent_reduction=float(np.mean(v <= .1)),
                    fraction_99percent_reduction=float(np.mean(v <= .01)),
                )
                bykey[(benchmark,dim,dt)] = row
                per_problem.append(row)

    global_rows = []
    for dt in dts:
        cells = [bykey[(b,dim,dt)] for b in benchmarks for dim in dims]
        global_rows.append(dict(
            dt=dt,
            median_across_cells_log10_residual=float(np.median(
                [r["median_log10_residual"] for r in cells])),
            fraction_cells_90pct_reduction_in_4of5_seeds=float(np.mean(
                [r["fraction_90percent_reduction"] >= 0.8 for r in cells])),
            fraction_cells_99pct_reduction_in_4of5_seeds=float(np.mean(
                [r["fraction_99percent_reduction"] >= 0.8 for r in cells])),
        ))

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "dt100_per_problem.csv", per_problem)
    write_csv(out / "dt100_global_profile.csv", global_rows)
    figdir = out / "figures"
    figdir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({"pdf.fonttype":42, "ps.fonttype":42, "font.size":9,
                         "axes.spines.top":False, "axes.spines.right":False})

    # Heatmap: use true logarithmic dt boundaries rather than equally spaced
    # sample indices; this accurately reflects uneven sampling density.
    labels = [f"{b}, D={dim}" for dim in dims for b in benchmarks]
    matrix = np.array([[bykey[(b, dim, dt)]["median_log10_residual"]
                         for dt in dts] for dim in dims for b in benchmarks])
    fig, ax = plt.subplots(figsize=(10.4, 5.2))
    mesh = ax.pcolormesh(geometric_edges(dts), np.arange(len(labels)+1),
                         matrix, shading="flat", cmap="viridis_r")
    ax.set_xscale("log")
    ax.set_xlim(dts[0], dts[-1])
    ax.set_xticks([.01,.03,.1,.3,1,3,10,30])
    ax.set_xticklabels(["0.01","0.03","0.1","0.3","1","3","10","30"])
    ax.set_yticks(np.arange(len(labels))+.5, labels)
    ax.invert_yaxis()
    ax.set_xlabel(r"CPSO interval length $\Delta t$ (logarithmic axis)")
    ax.set_title("CPSO sensitivity: residual objective gap at a fixed FE budget")
    fig.colorbar(mesh, ax=ax, pad=.02, label=r"Median $\log_{10}[(f_B-f^*)/(f_0-f^*)]$")
    fig.tight_layout()
    fig.savefig(figdir / "fig_dt100_heatmap.pdf", bbox_inches="tight")
    plt.close(fig)

    # Each problem/dimension gets its own curve, avoiding hidden cross-function
    # scale effects and permitting inspection of nonmonotone dt regions.
    for dim in dims:
        fig, ax = plt.subplots(figsize=(8.5,4.8))
        for b in benchmarks:
            y = [bykey[(b,dim,dt)]["median_residual_fraction"] for dt in dts]
            ax.plot(dts, y, linewidth=1.5, label=b)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(dts[0], dts[-1])
        ax.set_xlabel(r"$\Delta t$ (logarithmic axis)")
        ax.set_ylabel(r"Median remaining initial objective gap $\rho_B$")
        ax.set_title(f"CPSO dt sensitivity — dimension {dim} (5 paired seeds)")
        ax.grid(True, alpha=.22)
        ax.legend(fontsize=8, ncol=3, loc="best", frameon=False)
        fig.tight_layout()
        fig.savefig(figdir / f"fig_dt100_residual_D{dim}.pdf", bbox_inches="tight")
        plt.close(fig)

    # Stagnation timing: isolate early freeze-out of progress as dt increases.
    fig, axes = plt.subplots(len(dims),1,figsize=(9, 3.3*len(dims)),sharex=True)
    if len(dims)==1:
        axes=[axes]
    for ax,dim in zip(axes,dims):
        for b in benchmarks:
            ys=[bykey[(b,dim,dt)]["median_last_significant_iteration"] for dt in dts]
            ax.plot(dts,ys,linewidth=1.3,label=b)
        ax.set_xscale("log")
        ax.set_ylim(0,int(config["iterations"])*1.02)
        ax.set_ylabel(f"D={dim}: last significant\nimprovement (iteration)")
        ax.grid(True,alpha=.22)
    axes[0].legend(fontsize=7.7,ncol=3,loc="upper center",frameon=False)
    axes[-1].set_xlabel(r"$\Delta t$ (logarithmic axis)")
    fig.suptitle("Stagnation onset: last >1e-6 relative incumbent improvement",y=.99)
    fig.tight_layout()
    fig.savefig(figdir / "fig_dt100_stagnation.pdf",bbox_inches="tight")
    plt.close(fig)

    # Empirical success by relative improvement, not convergence to global optimum.
    fig, axes = plt.subplots(len(dims), 1, figsize=(9, 3.3*len(dims)), sharex=True)
    if len(dims)==1:
        axes=[axes]
    for ax,dim in zip(axes,dims):
        for b in benchmarks:
            y=[bykey[(b,dim,dt)]["fraction_90percent_reduction"] for dt in dts]
            ax.plot(dts,y,label=b,linewidth=1.35)
        ax.set_xscale("log")
        ax.set_ylim(-.05,1.05)
        ax.set_ylabel(f"D={dim}: fraction of runs\nwith >=90% gap reduction")
        ax.grid(True,alpha=.22)
    axes[0].legend(fontsize=7.7, ncol=3, loc="upper center", frameon=False)
    axes[-1].set_xlabel(r"$\Delta t$ (logarithmic axis)")
    fig.suptitle("Empirical relative-attainment profiles (five seeds per cell)",y=.99)
    fig.tight_layout()
    fig.savefig(figdir / "fig_dt100_relative_attainment.pdf",bbox_inches="tight")
    plt.close(fig)

    print(f"Verified {len(records)} records: {len(dts)} dt values, {len(benchmarks)} functions, {len(dims)} dimensions, {len(seeds)} seeds")
    print("Note: 90% relative reduction is not proof of convergence to the global optimum.")
    print(f"Saved CSV summaries and vector PDF figures to: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
