#!/usr/bin/env python3
"""Validate paired dt-sensitivity runs and create vector PDF figures."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plot_style import PALETTE, configure_style
from paired_stats import holm_adjust, paired_wilcoxon


def dt_tag(value: float) -> str:
    return format(value, ".8g").replace("-", "m").replace(".", "p")


def ci_mean(values: np.ndarray) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    mean = float(np.mean(values))
    if len(values) < 2:
        return mean, mean, mean
    try:
        from scipy.stats import t
        q = float(t.ppf(0.975, len(values) - 1))
    except Exception:
        q = 1.96
    half = q * float(np.std(values, ddof=1)) / math.sqrt(len(values))
    return mean, mean - half, mean + half


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def validate(rows: list[dict[str, str]], input_dir: Path) -> tuple[dict[str, Any], dict[float, list[list[float]]]]:
    if not rows:
        raise ValueError("No dt-sensitivity rows found.")
    required = {"dt", "repetition", "seed", "dimension", "population_size", "iterations",
                "function_evaluations", "topology", "workers_requested", "actual_threads",
                "sync_interval", "best_value", "elapsed_seconds", "openmp_enabled"}
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"dt-sensitivity CSV lacks required columns: {sorted(missing)}")
    dts = sorted({float(r["dt"]) for r in rows})
    seeds = sorted({int(r["seed"]) for r in rows})
    if len(dts) < 2 or len(seeds) < 2:
        raise ValueError("At least two dt values and two paired seeds are required.")
    if any(dt <= 0 for dt in dts):
        raise ValueError("All dt values must be positive.")
    if not any(abs(dt - 1.0) <= 1e-12 for dt in dts):
        raise ValueError("The predeclared reference interval dt=1.0 must be included in the sensitivity sweep.")
    if any(r["topology"] != "star" or int(r["sync_interval"]) != 1 for r in rows):
        raise ValueError("The dt study is defined for Star with sync_interval=1.")
    cells = [(float(r["dt"]), int(r["seed"])) for r in rows]
    if len(cells) != len(set(cells)):
        raise ValueError("Duplicate (dt, seed) entries detected.")
    for dt in dts:
        found = {int(r["seed"]) for r in rows if float(r["dt"]) == dt}
        if found != set(seeds):
            raise ValueError(f"Seed pairing is incomplete for dt={dt:g}.")
    fes = {int(r["function_evaluations"]) for r in rows}
    pop = {int(r["population_size"]) for r in rows}
    iterations = {int(r["iterations"]) for r in rows}
    dims = {int(r["dimension"]) for r in rows}
    workers = {int(r["workers_requested"]) for r in rows}
    actual = {int(r["actual_threads"]) for r in rows}
    openmp_ok = all(r["openmp_enabled"].lower() == "true" for r in rows)
    if any(int(r["function_evaluations"]) != int(r["population_size"]) * (int(r["iterations"]) + 1)
           for r in rows):
        raise ValueError("Objective-evaluation accounting mismatch.")
    if len(fes) != 1 or len(pop) != 1 or len(iterations) != 1 or len(dims) != 1 or len(workers) != 1:
        raise ValueError("The dt sweep must keep dimension, population, iterations, FEs and worker count fixed.")

    histories: dict[float, list[list[float]]] = {dt: [] for dt in dts}
    for row in rows:
        dt = float(row["dt"])
        rep = int(row["repetition"])
        raw_file = input_dir / "runs" / f"dt_{dt_tag(dt)}_run{rep:02d}.json"
        if not raw_file.is_file():
            raise FileNotFoundError(f"Missing raw run file: {raw_file}")
        raw = json.loads(raw_file.read_text(encoding="utf-8"))
        hist = [float(x) for x in raw.get("history", [])]
        if len(hist) != next(iter(iterations)) + 1:
            raise ValueError(f"History length mismatch in {raw_file}.")
        if abs(hist[-1] - float(row["best_value"])) > 1e-12:
            raise ValueError(f"CSV and raw JSON final values differ in {raw_file}.")
        if any(b > a + 1e-12 for a, b in zip(hist, hist[1:])):
            raise ValueError(f"Best-so-far history is not monotone in {raw_file}.")
        histories[dt].append(hist)
    validation = {
        "dt_values": dts,
        "repetitions": len(seeds),
        "paired_seeds": seeds,
        "dimension": next(iter(dims)),
        "population_size": next(iter(pop)),
        "iterations": next(iter(iterations)),
        "function_evaluations": next(iter(fes)),
        "workers_requested": next(iter(workers)),
        "actual_threads_unique": sorted(actual),
        "openmp_enabled_all": openmp_ok,
        "parallel_team_match": all(int(r["actual_threads"]) == int(r["workers_requested"]) for r in rows),
    }
    return validation, histories


def summarize(rows: list[dict[str, str]], validation: dict[str, Any]) -> list[dict[str, Any]]:
    summary = []
    for dt in validation["dt_values"]:
        group = [r for r in rows if float(r["dt"]) == dt]
        values = np.asarray([float(r["best_value"]) for r in group])
        times = np.asarray([float(r["elapsed_seconds"]) for r in group])
        vm, vlo, vhi = ci_mean(values)
        tm, tlo, thi = ci_mean(times)
        summary.append({
            "dt": dt, "n": len(group), "best_mean": vm,
            "best_sd": float(values.std(ddof=1)), "best_median": float(np.median(values)),
            "best_q25": float(np.quantile(values, .25)), "best_q75": float(np.quantile(values, .75)),
            "best_ci95_low": vlo, "best_ci95_high": vhi,
            "elapsed_mean_s": tm, "elapsed_ci95_low_s": tlo, "elapsed_ci95_high_s": thi,
        })
    return summary



def paired_dt_tests(rows: list[dict[str, str]], validation: dict[str, Any]) -> list[dict[str, Any]]:
    """Compare every tested dt against the predeclared dt=1 reference by seed."""
    reference_dt = min(validation["dt_values"], key=lambda v: abs(v - 1.0))
    values = {(float(r["dt"]), int(r["seed"])): float(r["best_value"]) for r in rows}
    tests: list[dict[str, Any]] = []
    for dt in validation["dt_values"]:
        if dt == reference_dt:
            continue
        seeds = validation["paired_seeds"]
        result = paired_wilcoxon(
            [values[(dt, seed)] for seed in seeds],
            [values[(reference_dt, seed)] for seed in seeds],
        )
        tests.append({"dt": dt, "reference_dt": reference_dt, **result})
    adjusted = holm_adjust([float(row["p_value"]) for row in tests])
    for row, p_adj in zip(tests, adjusted):
        row["p_value_holm"] = p_adj
    return tests

def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def make_figures(rows: list[dict[str, str]], validation: dict[str, Any], summary: list[dict[str, Any]],
                 histories: dict[float, list[list[float]]], output: Path) -> None:
    configure_style()
    output.mkdir(parents=True, exist_ok=True)
    dts = validation["dt_values"]

    # Mean final objective with t-based 95% CI.
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    x = np.asarray(dts)
    y = np.asarray([r["best_mean"] for r in summary])
    lo = np.maximum(0.0, y - np.asarray([r["best_ci95_low"] for r in summary]))
    hi = np.asarray([r["best_ci95_high"] for r in summary]) - y
    ax.errorbar(x, y, yerr=np.vstack([lo, hi]), fmt="o-", capsize=3, color=PALETTE["navy"])
    ax.set_xlabel("CPSO interval length, $\\Delta t$")
    ax.set_ylabel("Mean final objective value (95% CI)")
    ax.set_title("Sensitivity of CPSO solution quality to interval length")
    ax.set_xticks(dts)
    ax.set_ylim(bottom=0)
    ax.grid(True)
    fig.savefig(output / "fig_13_dt_sensitivity_mean_quality.pdf")
    plt.close(fig)

    # Individual observations, IQR and median.
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    rng = np.random.default_rng(20261104)
    for pos, dt in enumerate(dts, start=1):
        values = np.asarray([float(r["best_value"]) for r in rows if float(r["dt"]) == dt])
        jitter = rng.uniform(-0.10, 0.10, size=len(values))
        color = PALETTE["blue"]
        ax.scatter(pos + jitter, values, s=18, color=color, alpha=0.68,
                   edgecolor="white", linewidth=0.3, zorder=3)
        q25, med, q75 = np.quantile(values, [0.25, 0.5, 0.75])
        ax.vlines(pos, q25, q75, color=PALETTE["teal"], linewidth=4, zorder=2)
        ax.scatter(pos, med, marker="_", color="black", s=130, linewidth=1.7, zorder=4)
    ax.set_xticks(np.arange(1, len(dts) + 1), [f"{v:g}" for v in dts])
    ax.set_xlabel("CPSO interval length, $\\Delta t$")
    ax.set_ylabel("Final objective value")
    ax.set_title("Final objective distribution across interval lengths")
    ax.set_ylim(bottom=0)
    ax.grid(True, axis="y")
    fig.savefig(output / "fig_14_dt_sensitivity_distribution.pdf")
    plt.close(fig)

    # Convergence profiles for up to five representative dt values.
    indices = np.unique(np.linspace(0, len(dts) - 1, min(5, len(dts))).round().astype(int))
    selected = [dts[i] for i in indices]
    fig, ax = plt.subplots(figsize=(6.6, 3.8))
    colors = [PALETTE["navy"], PALETTE["blue"], PALETTE["teal"], PALETTE["gold"], PALETTE["red"]]
    fes = np.arange(1, validation["iterations"] + 2) * validation["population_size"]
    for dt, color in zip(selected, colors):
        h = np.asarray(histories[dt], dtype=float)
        med = np.median(h, axis=0)
        q25, q75 = np.quantile(h, [0.25, 0.75], axis=0)
        ax.plot(fes, med, color=color, label=fr"$\Delta t={dt:g}$")
        ax.fill_between(fes, q25, q75, color=color, alpha=.12, linewidth=0)
    ax.set_xlabel("Objective-function evaluations")
    ax.set_ylabel("Best-so-far objective value")
    ax.set_title("CPSO convergence for representative interval lengths")
    ax.grid(True)
    ax.legend(frameon=False)
    fig.savefig(output / "fig_15_dt_sensitivity_convergence.pdf")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Directory with dt_sensitivity_runs.csv and raw runs.")
    parser.add_argument("--output-dir", default="results/dt_sensitivity_analysis", type=Path)
    parser.add_argument("--figures-dir", default="figures/dt_sensitivity", type=Path)
    args = parser.parse_args()
    rows = load_rows(args.input / "dt_sensitivity_runs.csv")
    validation, histories = validate(rows, args.input)
    summary = summarize(rows, validation)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "validation.json").write_text(json.dumps(validation, indent=2), encoding="utf-8")
    write_csv(args.output_dir / "dt_sensitivity_summary.csv", summary)
    write_csv(args.output_dir / "paired_dt_tests.csv", paired_dt_tests(rows, validation))
    make_figures(rows, validation, summary, histories, args.figures_dir)
    print(f"Validated {len(rows)} paired dt-sensitivity runs; generated 3 vector PDF figures.")


if __name__ == "__main__":
    main()
