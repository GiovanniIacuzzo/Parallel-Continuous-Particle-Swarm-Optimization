#!/usr/bin/env python3
"""Analyze paired island-topology runs and create publication-style vector figures."""
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
from paired_stats import holm_adjust, paired_wilcoxon


def ci_mean(x: np.ndarray) -> tuple[float, float, float]:
    x = np.asarray(x, dtype=float)
    mean = float(np.mean(x))
    if len(x) < 2:
        return mean, mean, mean
    try:
        from scipy.stats import t
        q = float(t.ppf(0.975, len(x) - 1))
    except Exception:
        q = 1.96
    h = q * float(np.std(x, ddof=1)) / math.sqrt(len(x))
    return mean, mean - h, mean + h


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def validate(rows: list[dict[str, str]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("No topology rows found.")
    required = {"topology", "workers_requested", "actual_threads", "sync_interval", "seed",
                "function_evaluations", "population_size", "dimension", "best_value",
                "elapsed_seconds", "coordination_seconds", "openmp_enabled", "iterations", "repetition"}
    missing_columns = required - set(rows[0])
    if missing_columns:
        raise ValueError(f"Topology results lack required columns: {sorted(missing_columns)}")
    seeds = sorted({int(r["seed"]) for r in rows})
    present_topologies = {r["topology"] for r in rows}
    if present_topologies != {"independent", "star", "ring"}:
        raise ValueError("The planned topology comparison requires independent, star and ring results.")
    topologies = ["independent", "star", "ring"]
    intervals = sorted({int(r["sync_interval"]) for r in rows})
    workers = sorted({int(r["workers_requested"]) for r in rows})
    if len(seeds) < 2:
        raise ValueError("At least two paired seeds are required; 30 are recommended for publication.")
    if len(workers) != 1:
        raise ValueError("This figure set is designed for one fixed island count; run separate studies for each P.")
    iterations = {int(r["iterations"]) for r in rows}
    if len(iterations) != 1:
        raise ValueError("Iteration count must remain fixed across every topology/interval cell.")
    if any(int(r["function_evaluations"]) != int(r["population_size"]) * (int(r["iterations"]) + 1)
           for r in rows):
        raise ValueError("Objective-evaluation accounting mismatch.")
    row_keys = [(r["topology"], int(r["workers_requested"]), int(r["sync_interval"]), int(r["seed"])) for r in rows]
    if len(row_keys) != len(set(row_keys)):
        raise ValueError("Duplicate topology/worker/interval/seed rows detected.")
    # Paired-seed completeness for every experimental cell.
    cells = {(t, p, k) for t in topologies for p in workers for k in intervals}
    missing = []
    for t, p, k in cells:
        got = {int(r["seed"]) for r in rows
               if r["topology"] == t and int(r["workers_requested"]) == p and int(r["sync_interval"]) == k}
        if got != set(seeds):
            missing.append((t, p, k, sorted(set(seeds) - got)))
    if missing:
        raise ValueError(f"Incomplete seed pairing in topology results: {missing[:3]}")
    # Independent islands ignore the exchange interval by definition. Verify that the
    # raw algorithmic outcomes respect this invariant for every worker count and seed.
    independent = {(int(r["workers_requested"]), int(r["seed"]), int(r["sync_interval"])): float(r["best_value"])
                   for r in rows if r["topology"] == "independent"}
    for p in workers:
        for seed in seeds:
            reference = independent[(p, seed, intervals[0])]
            if any(abs(independent[(p, seed, interval)] - reference) > 1e-12 for interval in intervals[1:]):
                raise ValueError("Independent topology results depend on sync_interval, which should be ignored.")
    if 1 not in intervals:
        raise ValueError("The topology study must include sync_interval=1 for the convergence comparison.")
    fes = {int(r["function_evaluations"]) for r in rows}
    pop = {int(r["population_size"]) for r in rows}
    dim = {int(r["dimension"]) for r in rows}
    actual_threads = {int(r["actual_threads"]) for r in rows}
    openmp = all(r["openmp_enabled"].lower() == "true" for r in rows)
    worker_match = all(int(r["actual_threads"]) == int(r["workers_requested"]) for r in rows)
    if len(fes) != 1 or len(pop) != 1 or len(dim) != 1:
        raise ValueError("Global population, dimension and FE budget must be fixed in the topology comparison.")
    return {"topologies": topologies, "sync_intervals": intervals, "workers": workers,
            "seeds": seeds, "n_per_cell": len(seeds), "dimension": next(iter(dim)),
            "population_size": next(iter(pop)), "function_evaluations": next(iter(fes)),
            "openmp_enabled_all": openmp, "actual_threads_unique": sorted(actual_threads),
            "requested_teams_match_actual": worker_match,
            "parallel_timing_valid": openmp and worker_match and max(workers) > 1}


def summarize(rows: list[dict[str, str]], validation: dict[str, Any]) -> list[dict[str, Any]]:
    result = []
    for topology in validation["topologies"]:
        for p in validation["workers"]:
            for interval in validation["sync_intervals"]:
                group = [r for r in rows if r["topology"] == topology
                         and int(r["workers_requested"]) == p and int(r["sync_interval"]) == interval]
                f = np.asarray([float(r["best_value"]) for r in group])
                t = np.asarray([float(r["elapsed_seconds"]) for r in group])
                c = np.asarray([float(r["coordination_seconds"]) for r in group])
                fm, flo, fhi = ci_mean(f)
                tm, tlo, thi = ci_mean(t)
                cm, clo, chi = ci_mean(c)
                result.append({
                    "topology": topology, "workers_requested": p, "sync_interval": interval,
                    "n": len(group), "best_value_mean": fm, "best_value_sd": float(f.std(ddof=1)),
                    "best_value_median": float(np.median(f)), "best_value_q25": float(np.quantile(f, .25)),
                    "best_value_q75": float(np.quantile(f, .75)),
                    "best_value_ci95_low": flo, "best_value_ci95_high": fhi,
                    "elapsed_mean_s": tm, "elapsed_sd_s": float(t.std(ddof=1)),
                    "elapsed_ci95_low_s": tlo, "elapsed_ci95_high_s": thi,
                    "coordination_mean_s": cm, "coordination_ci95_low_s": clo,
                    "coordination_ci95_high_s": chi,
                    "openmp_enabled_all": all(r["openmp_enabled"].lower() == "true" for r in group),
                    "actual_threads_unique": sorted({int(r["actual_threads"]) for r in group}),
                })
    return result


def load_histories(input_dir: Path, rows: list[dict[str, str]]) -> dict[tuple[str, int, int], list[list[float]]]:
    # The runner stores individual JSON files in input_dir/runs.
    histories: dict[tuple[str, int, int], list[list[float]]] = defaultdict(list)
    histories_by_seed: dict[tuple[str, int, int, int], list[float]] = {}
    for r in rows:
        topology = r["topology"]
        p = int(r["workers_requested"])
        interval = int(r["sync_interval"])
        rep = int(r["repetition"])
        seed = int(r["seed"])
        file = input_dir / "runs" / f"{topology}_p{p}_sync{interval}_run{rep:02d}.json"
        if not file.is_file():
            raise FileNotFoundError(f"Missing raw run file: {file}")
        d = json.loads(file.read_text(encoding="utf-8"))
        history = [float(x) for x in d["history"]]
        if len(history) != int(r["iterations"]) + 1:
            raise ValueError(f"History length mismatch in {file}.")
        if abs(history[-1] - float(r["best_value"])) > 1e-12:
            raise ValueError(f"Final history value disagrees with CSV in {file}.")
        if any(b > a + 1e-12 for a, b in zip(history, history[1:])):
            raise ValueError(f"Best-so-far history is not monotone in {file}.")
        histories[(topology, p, interval)].append(history)
        histories_by_seed[(topology, p, interval, seed)] = history

    # Check the complete Independent trajectories, not only their final values.
    intervals = sorted({int(r["sync_interval"]) for r in rows})
    workers = sorted({int(r["workers_requested"]) for r in rows})
    seeds = sorted({int(r["seed"]) for r in rows})
    for p in workers:
        for seed in seeds:
            reference = histories_by_seed[("independent", p, intervals[0], seed)]
            for interval in intervals[1:]:
                candidate = histories_by_seed[("independent", p, interval, seed)]
                if len(candidate) != len(reference) or any(
                    abs(a - b) > 1e-12 for a, b in zip(reference, candidate)
                ):
                    raise ValueError("Independent full trajectories depend on sync_interval; expected exact invariance.")
    return histories



def paired_tests(rows: list[dict[str, str]], validation: dict[str, Any]) -> list[dict[str, Any]]:
    """Paired Wilcoxon tests for final objective values; Holm-adjusted globally."""
    pairs = [("independent", "star"), ("independent", "ring"), ("star", "ring")]
    tests: list[dict[str, Any]] = []
    for p in validation["workers"]:
        for interval in validation["sync_intervals"]:
            cell = [r for r in rows if int(r["workers_requested"]) == p
                    and int(r["sync_interval"]) == interval]
            values = {(r["topology"], int(r["seed"])): float(r["best_value"]) for r in cell}
            seeds = validation["seeds"]
            for topology_a, topology_b in pairs:
                result = paired_wilcoxon(
                    [values[(topology_a, seed)] for seed in seeds],
                    [values[(topology_b, seed)] for seed in seeds],
                )
                tests.append({
                    "workers": p, "sync_interval": interval,
                    "topology_A": topology_a, "topology_B": topology_b,
                    **result,
                })
    adjusted = holm_adjust([float(row["p_value"]) for row in tests])
    for row, p_adj in zip(tests, adjusted):
        row["p_value_holm_global"] = p_adj
    return tests

def write_csv(path: Path, data: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(data[0]))
        writer.writeheader()
        writer.writerows(data)


def plot(validation: dict[str, Any], rows: list[dict[str, str]], summary: list[dict[str, Any]], histories: dict[tuple[str, int, int], list[list[float]]],
         output: Path) -> None:
    configure_style()
    output.mkdir(parents=True, exist_ok=True)
    p = validation["workers"][0]
    topologies = validation["topologies"]
    intervals = validation["sync_intervals"]

    # Figure: convergence profiles at tau=1, with median and IQR.
    fig, ax = plt.subplots(figsize=(6.6, 3.8))
    for topology in topologies:
        hs = np.asarray(histories[(topology, p, 1)], dtype=float)
        x = np.arange(1, hs.shape[1] + 1) * validation["population_size"]
        med = np.median(hs, axis=0)
        q1 = np.quantile(hs, .25, axis=0)
        q3 = np.quantile(hs, .75, axis=0)
        color = PALETTE.get(topology, PALETTE["navy"])
        ax.plot(x, med, color=color, label=topology.capitalize())
        ax.fill_between(x, q1, q3, color=color, alpha=.15, linewidth=0)
    ax.set_xlabel("Objective-function evaluations")
    ax.set_ylabel("Best-so-far objective value")
    ax.set_title(f"Island-topology comparison ($D={validation['dimension']}$, $P={p}$, $\\tau=1$)")
    ax.grid(True)
    ax.legend(frameon=False)
    fig.savefig(output / "fig_08_topology_convergence_tau1.pdf")
    plt.close(fig)

    # Figure: show every final observation; do not reduce the distribution to quartiles only.
    fig, ax = plt.subplots(figsize=(6.3, 3.8))
    rng = np.random.default_rng(20261104)
    for position, topology in enumerate(topologies, start=1):
        vals = np.asarray([float(r["best_value"]) for r in rows
                           if r["topology"] == topology
                           and int(r["workers_requested"]) == p
                           and int(r["sync_interval"]) == 1])
        color = PALETTE.get(topology, PALETTE["navy"])
        jitter = rng.uniform(-0.085, 0.085, size=len(vals))
        ax.scatter(position + jitter, vals, s=20, color=color, alpha=0.72,
                   edgecolor="white", linewidth=0.35, zorder=3)
        med = float(np.median(vals))
        q25, q75 = np.quantile(vals, [0.25, 0.75])
        ax.vlines(position, q25, q75, color=color, linewidth=4, alpha=0.85, zorder=2)
        ax.scatter(position, med, marker="_", color="black", s=140, linewidth=1.8, zorder=4)
    ax.set_xlim(0.5, len(topologies) + 0.5)
    ax.set_xticks(np.arange(1, len(topologies) + 1), [t.capitalize() for t in topologies])
    ax.set_ylabel("Final objective value")
    ax.set_title(f"Final solution quality by topology ($P={p}$, $\\tau=1$)")
    ax.grid(True, axis="y")
    fig.savefig(output / "fig_09_final_quality_by_topology.pdf")
    plt.close(fig)

    # Figure: mean solution quality versus exchange interval.
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for topology in topologies:
        group = sorted([r for r in summary if r["topology"] == topology
                        and int(r["workers_requested"]) == p], key=lambda r: int(r["sync_interval"]))
        xx = np.asarray([int(r["sync_interval"]) for r in group])
        yy = np.asarray([r["best_value_mean"] for r in group])
        lo = np.maximum(0.0, yy - np.asarray([r["best_value_ci95_low"] for r in group]))
        hi = np.asarray([r["best_value_ci95_high"] for r in group]) - yy
        color = PALETTE.get(topology, PALETTE["navy"])
        ax.errorbar(xx, yy, yerr=np.vstack([lo, hi]), marker="o", capsize=3,
                    color=color, label=topology.capitalize())
    ax.set_xlabel("Exchange interval, $\\tau$ (generations)")
    ax.set_ylabel("Mean final objective value (95% CI)")
    ax.set_ylim(bottom=0)
    ax.set_title(f"Effect of migration frequency ($P={p}$)")
    ax.set_xticks(intervals)
    ax.grid(True)
    ax.legend(frameon=False)
    fig.savefig(output / "fig_10_topology_vs_exchange_interval.pdf")
    plt.close(fig)

    # Figure: end-to-end runtime vs exchange interval; interpret coordination only as in-memory work.
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for topology in topologies:
        group = sorted([r for r in summary if r["topology"] == topology
                        and int(r["workers_requested"]) == p], key=lambda r: int(r["sync_interval"]))
        xx = np.asarray([int(r["sync_interval"]) for r in group])
        yy = np.asarray([r["elapsed_mean_s"] for r in group])
        lo = np.maximum(0.0, yy - np.asarray([r["elapsed_ci95_low_s"] for r in group]))
        hi = np.asarray([r["elapsed_ci95_high_s"] for r in group]) - yy
        color = PALETTE.get(topology, PALETTE["navy"])
        ax.errorbar(xx, yy, yerr=np.vstack([lo, hi]), marker="o", capsize=3,
                    color=color, label=topology.capitalize())
    ax.set_xlabel("Exchange interval, $\\tau$ (generations)")
    ax.set_ylabel("Mean wall-clock time (s; 95% CI)")
    ax.set_ylim(bottom=0)
    ax.set_title(f"Runtime versus exchange interval ($P={p}$)")
    if not validation["parallel_timing_valid"]:
        ax.text(0.5, 0.02, "Serial fallback or thread-team mismatch: descriptive timing only",
                transform=ax.transAxes, ha="center", va="bottom", fontsize=7,
                color=PALETTE["red"], weight="bold")
        fig.subplots_adjust(bottom=0.22)
    ax.set_xticks(intervals)
    ax.grid(True)
    ax.legend(frameon=False)
    fig.savefig(output / "fig_11_topology_runtime_vs_interval.pdf")
    plt.close(fig)

    # Explicit in-process coordination fraction (not network communication latency).
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for topology in topologies:
        group_rows = [r for r in rows if r["topology"] == topology
                      and int(r["workers_requested"]) == p]
        xs = []
        means = []
        lows = []
        highs = []
        for interval in intervals:
            cell = [float(r["coordination_seconds"]) / float(r["elapsed_seconds"])
                    for r in group_rows if int(r["sync_interval"]) == interval]
            mean, low, high = ci_mean(np.asarray(cell, dtype=float))
            xs.append(interval); means.append(mean); lows.append(low); highs.append(high)
        y = np.asarray(means)
        color = PALETTE.get(topology, PALETTE["navy"])
        ax.errorbar(xs, y, yerr=np.vstack([y - np.asarray(lows), np.asarray(highs) - y]),
                    marker="o", capsize=3, color=color, label=topology.capitalize())
    ax.set_xlabel("Exchange interval, $\\tau$ (generations)")
    ax.set_ylabel("Explicit coordination / wall-clock time (95% CI)")
    ax.set_title(f"In-process coordination fraction ($P={p}$)")
    ax.set_xticks(intervals)
    ax.set_ylim(bottom=0)
    ax.grid(True)
    ax.legend(frameon=False)
    fig.savefig(output / "fig_12_topology_coordination_fraction.pdf")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Topology study output directory.")
    parser.add_argument("--output-dir", default="results/topology_analysis", type=Path)
    parser.add_argument("--figures-dir", default="figures/topology", type=Path)
    args = parser.parse_args()
    rows = load_rows(args.input / "topology_runs.csv")
    validation = validate(rows)
    summary = summarize(rows, validation)
    histories = load_histories(args.input, rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "validation.json").write_text(json.dumps(validation, indent=2), encoding="utf-8")
    write_csv(args.output_dir / "topology_summary.csv", summary)
    tests = paired_tests(rows, validation)
    write_csv(args.output_dir / "paired_topology_tests.csv", tests)
    plot(validation, rows, summary, histories, args.figures_dir)
    if validation["parallel_timing_valid"]:
        print(f"Validated {len(rows)} paired topology runs with actual OpenMP teams; generated 5 vector PDF figures.")
    else:
        print("WARNING: results are not valid for parallel-runtime claims (OpenMP/team-size verification failed).")
        print(f"Validated {len(rows)} algorithmic topology runs; runtime values are descriptive only.")


if __name__ == "__main__":
    main()
