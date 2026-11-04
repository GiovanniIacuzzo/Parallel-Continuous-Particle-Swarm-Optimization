#!/usr/bin/env python3
"""Validate and plot a verified OpenMP Star strong-scaling dataset."""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plot_style import PALETTE, configure_style


def ci_mean(values: np.ndarray) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    n = len(values)
    mean = float(values.mean())
    if n < 2:
        return mean, mean, mean
    try:
        from scipy.stats import t
        q = float(t.ppf(0.975, n - 1))
    except Exception:
        q = 1.96
    half = q * float(values.std(ddof=1)) / math.sqrt(n)
    return mean, mean - half, mean + half


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def validate(rows: list[dict[str, str]], input_dir: Path) -> dict[str, Any]:
    if not rows:
        raise ValueError("No rows in scaling_runs.csv")
    required = {"workers_requested", "actual_threads", "seed", "topology", "sync_interval",
                "function_evaluations", "elapsed_seconds", "compute_seconds", "coordination_seconds",
                "best_value", "openmp_enabled", "population_size", "iterations", "dimension", "repetition"}
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"Scaling file lacks required fields: {sorted(missing)}")
    if any(r["topology"].lower() != "star" for r in rows):
        raise ValueError("Strong scaling requires Star topology only.")
    if any(int(r["sync_interval"]) != 1 for r in rows):
        raise ValueError("Strong scaling requires sync_interval=1.")
    if any(int(r["function_evaluations"]) != int(r["population_size"]) * (int(r["iterations"]) + 1)
           for r in rows):
        raise ValueError("FE accounting mismatch: expected population_size * (iterations + 1).")
    if any(float(r["coordination_seconds"]) < 0.0 or float(r["elapsed_seconds"]) <= 0.0
           or float(r["coordination_seconds"]) > float(r["elapsed_seconds"]) for r in rows):
        raise ValueError("Invalid timing fields: coordination must be non-negative and no greater than elapsed time.")
    if any(r["openmp_enabled"].lower() != "true" for r in rows):
        raise ValueError("Invalid strong-scaling dataset: one or more runs were built without OpenMP.")
    if any(int(r["workers_requested"]) != int(r["actual_threads"]) for r in rows):
        raise ValueError("Invalid strong-scaling dataset: actual OpenMP team size differs from requested P.")
    row_keys = [(int(r["workers_requested"]), int(r["seed"])) for r in rows]
    if len(row_keys) != len(set(row_keys)):
        raise ValueError("Duplicate worker/seed runs detected in strong-scaling CSV.")
    workers = sorted({int(r["workers_requested"]) for r in rows})
    if len(workers) < 2 or workers[0] != 1:
        raise ValueError("Need a P=1 baseline and at least one parallel worker count.")
    seeds_by_p = {p: {int(r["seed"]) for r in rows if int(r["workers_requested"]) == p} for p in workers}
    baseline_seeds = seeds_by_p[1]
    if not baseline_seeds or any(x != baseline_seeds for x in seeds_by_p.values()):
        raise ValueError("The seed sets must be complete and paired across worker counts.")
    fes = {int(r["function_evaluations"]) for r in rows}
    pop = {int(r["population_size"]) for r in rows}
    dims = {int(r["dimension"]) for r in rows}
    if len(fes) != 1 or len(pop) != 1 or len(dims) != 1:
        raise ValueError("Dimension, global population and FE budget must remain fixed across P.")

    values = {(int(r["seed"]), int(r["workers_requested"])): float(r["best_value"]) for r in rows}
    # Validate full best-so-far histories, not only terminal values.
    history_by_seed: dict[tuple[int, int], list[float]] = {}
    for row in rows:
        p = int(row["workers_requested"])
        rep = int(row["repetition"])
        seed = int(row["seed"])
        raw_path = input_dir / f"star_p{p}_run{rep:02d}.json"
        if not raw_path.is_file():
            raise FileNotFoundError(f"Missing raw run file required for trajectory audit: {raw_path}")
        raw = json.loads(raw_path.read_text(encoding="utf-8"))
        history = [float(x) for x in raw.get("history", [])]
        expected_len = int(row["iterations"]) + 1
        if len(history) != expected_len:
            raise ValueError(f"History length mismatch for {raw_path}: {len(history)} != {expected_len}.")
        if abs(history[-1] - float(row["best_value"])) > 1e-12:
            raise ValueError(f"Final history value disagrees with CSV for {raw_path}.")
        if any(b > a + 1e-12 for a, b in zip(history, history[1:])):
            raise ValueError(f"Best-so-far history is not monotone in {raw_path}.")
        history_by_seed[(seed, p)] = history
    diffs = []
    for p in workers:
        md = max(abs(values[(s, p)] - values[(s, 1)]) for s in baseline_seeds)
        diffs.append({"workers": p, "max_abs_best_value_difference_vs_P1": md})
    invariant = all(d["max_abs_best_value_difference_vs_P1"] <= 1e-10 for d in diffs)
    full_history_equivalent = True
    max_history_difference = 0.0
    for seed in baseline_seeds:
        baseline_history = np.asarray(history_by_seed[(seed, 1)], dtype=float)
        for p in workers:
            difference = float(np.max(np.abs(np.asarray(history_by_seed[(seed, p)], dtype=float) - baseline_history)))
            max_history_difference = max(max_history_difference, difference)
            if difference > 1e-10:
                full_history_equivalent = False
    if not invariant or not full_history_equivalent:
        raise ValueError("Star full-trajectory equivalence check failed across worker counts; inspect before scaling analysis.")
    return {
        "workers": workers,
        "repetitions": len(baseline_seeds),
        "dimension": next(iter(dims)),
        "population_size": next(iter(pop)),
        "function_evaluations": next(iter(fes)),
        "trajectory_equivalent": True,
        "full_best_so_far_histories_equivalent": full_history_equivalent,
        "maximum_absolute_history_difference": max_history_difference,
        "paired_outcome_differences": diffs,
    }


def summarize(rows: list[dict[str, str]], validation: dict[str, Any]) -> list[dict[str, Any]]:
    workers = validation["workers"]
    seeds = sorted({int(r["seed"]) for r in rows})
    time = {(int(r["seed"]), int(r["workers_requested"])): float(r["elapsed_seconds"]) for r in rows}
    best = {(int(r["seed"]), int(r["workers_requested"])): float(r["best_value"]) for r in rows}
    coordination = {(int(r["seed"]), int(r["workers_requested"])): float(r["coordination_seconds"]) for r in rows}
    elapsed = {(int(r["seed"]), int(r["workers_requested"])): float(r["elapsed_seconds"]) for r in rows}
    t1 = np.asarray([time[(s, 1)] for s in seeds])
    summaries = []
    for p in workers:
        tp = np.asarray([time[(s, p)] for s in seeds])
        vals = np.asarray([best[(s, p)] for s in seeds])
        elapsed_mean, elapsed_low, elapsed_high = ci_mean(tp)
        paired_speedups = t1 / tp
        speedup, speedup_low, speedup_high = ci_mean(paired_speedups)
        efficiencies = paired_speedups / p
        efficiency, efficiency_low, efficiency_high = ci_mean(efficiencies)
        best_mean, best_low, best_high = ci_mean(vals)
        fractions = np.asarray([coordination[(s, p)] / elapsed[(s, p)] for s in seeds])
        frac_mean, frac_low, frac_high = ci_mean(fractions)
        summaries.append({
            "workers": p,
            "actual_threads": p,
            "n": len(seeds),
            "elapsed_mean_s": elapsed_mean,
            "elapsed_sd_s": float(tp.std(ddof=1)) if len(tp) > 1 else 0.0,
            "elapsed_median_s": float(np.median(tp)),
            "elapsed_ci95_low_s": elapsed_low,
            "elapsed_ci95_high_s": elapsed_high,
            "paired_speedup_mean": speedup,
            "paired_speedup_ci95_low": speedup_low,
            "paired_speedup_ci95_high": speedup_high,
            "paired_efficiency_mean": efficiency,
            "paired_efficiency_ci95_low": efficiency_low,
            "paired_efficiency_ci95_high": efficiency_high,
            "best_value_mean": best_mean,
            "best_value_ci95_low": best_low,
            "best_value_ci95_high": best_high,
            "coordination_fraction_mean": frac_mean,
            "coordination_fraction_ci95_low": frac_low,
            "coordination_fraction_ci95_high": frac_high,
        })
    return summaries


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot(summary: list[dict[str, Any]], output: Path) -> None:
    configure_style()
    output.mkdir(parents=True, exist_ok=True)
    p = np.asarray([r["workers"] for r in summary])

    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    y = np.asarray([r["elapsed_mean_s"] for r in summary])
    lo = np.maximum(0.0, y - np.asarray([r["elapsed_ci95_low_s"] for r in summary]))
    hi = np.asarray([r["elapsed_ci95_high_s"] for r in summary]) - y
    ax.errorbar(p, y, yerr=np.vstack([lo, hi]), fmt="o-", capsize=3,
                color=PALETTE["navy"], label="Mean wall-clock time (95% CI)")
    ax.set_xscale("log", base=2)
    ax.set_xticks(p, [str(x) for x in p])
    ax.set_xlabel("OpenMP threads")
    ax.set_ylabel("Wall-clock time (s)")
    ax.set_ylim(bottom=0)
    ax.set_title("CPSO strong scaling: Star topology")
    ax.grid(True, which="both")
    ax.legend(frameon=False)
    fig.savefig(output / "fig_04_strong_scaling_wall_time.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    speed = np.asarray([r["paired_speedup_mean"] for r in summary])
    slo = np.maximum(0.0, speed - np.asarray([r["paired_speedup_ci95_low"] for r in summary]))
    shi = np.asarray([r["paired_speedup_ci95_high"] for r in summary]) - speed
    ax.plot(p, p, "--", color=PALETTE["grey"], label="Ideal linear speedup")
    ax.errorbar(p, speed, yerr=np.vstack([slo, shi]), fmt="o-", capsize=3,
                color=PALETTE["blue"], label="Paired speedup (95% CI)")
    ax.set_xscale("log", base=2)
    ax.set_xticks(p, [str(x) for x in p])
    ax.set_xlabel("OpenMP threads")
    ax.set_ylabel("Speedup")
    ax.set_title("CPSO strong-scaling speedup")
    ax.grid(True, which="both")
    ax.legend(frameon=False)
    fig.savefig(output / "fig_05_strong_scaling_speedup.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    eff = np.asarray([r["paired_efficiency_mean"] for r in summary])
    elo = np.maximum(0.0, eff - np.asarray([r["paired_efficiency_ci95_low"] for r in summary]))
    ehi = np.asarray([r["paired_efficiency_ci95_high"] for r in summary]) - eff
    ax.axhline(1.0, linestyle="--", color=PALETTE["grey"], label="Ideal efficiency")
    ax.errorbar(p, eff, yerr=np.vstack([elo, ehi]), fmt="o-", capsize=3,
                color=PALETTE["teal"], label="Paired parallel efficiency (95% CI)")
    ax.set_xscale("log", base=2)
    ax.set_xticks(p, [str(x) for x in p])
    ax.set_xlabel("OpenMP threads")
    ax.set_ylabel("Parallel efficiency")
    ax.set_ylim(bottom=0)
    ax.set_title("CPSO strong-scaling efficiency")
    ax.grid(True, which="both")
    ax.legend(frameon=False)
    fig.savefig(output / "fig_06_strong_scaling_efficiency.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    fraction = np.asarray([r["coordination_fraction_mean"] for r in summary])
    flo = np.maximum(0.0, fraction - np.asarray([r["coordination_fraction_ci95_low"] for r in summary]))
    fhi = np.asarray([r["coordination_fraction_ci95_high"] for r in summary]) - fraction
    ax.errorbar(p, fraction, yerr=np.vstack([flo, fhi]), fmt="o-", capsize=3,
                color=PALETTE["gold"], label="Explicit coordination / elapsed time (95% CI)")
    ax.set_xscale("log", base=2)
    ax.set_xticks(p, [str(x) for x in p])
    ax.set_xlabel("OpenMP threads")
    ax.set_ylabel("Explicit coordination-time fraction")
    ax.set_ylim(bottom=0)
    ax.set_title("CPSO strong scaling: explicit coordination overhead")
    ax.grid(True, which="both")
    ax.legend(frameon=False)
    fig.savefig(output / "fig_07_strong_scaling_coordination_fraction.pdf")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Directory containing scaling_runs.csv.")
    parser.add_argument("--output-dir", default="results/strong_scaling_analysis", type=Path)
    parser.add_argument("--figures-dir", default="figures/strong_scaling", type=Path)
    args = parser.parse_args()
    rows = load_rows(args.input / "scaling_runs.csv")
    validation = validate(rows, args.input)
    summary = summarize(rows, validation)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "validation.json").write_text(json.dumps(validation, indent=2), encoding="utf-8")
    write_csv(args.output_dir / "strong_scaling_summary.csv", summary)
    plot(summary, args.figures_dir)
    print(f"Validated {len(rows)} OpenMP runs; generated 4 vector PDF figures.")


if __name__ == "__main__":
    main()
