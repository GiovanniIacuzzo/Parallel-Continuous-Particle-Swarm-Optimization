#!/usr/bin/env python3
"""Audit the first macOS results archive and create reproducible vector figures.

This script deliberately distinguishes algorithmic/serial evidence from strong-
scaling evidence. The supplied results are rejected as a strong-scaling dataset
when OpenMP was unavailable or the actual worker team differed from the request.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import math
import re
import statistics
import zipfile
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plot_style import PALETTE, configure_style


def read_archive(archive: Path) -> tuple[list[dict[str, str]], dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    with zipfile.ZipFile(archive) as zf:
        names = set(zf.namelist())
        base = "results/strong_scaling_500d/"
        csv_name = base + "scaling_runs.csv"
        cfg_name = base + "experiment_configuration.json"
        env_name = "results/rastrigin_500d_star_environment.json"
        for name in (csv_name, cfg_name, env_name):
            if name not in names:
                raise FileNotFoundError(f"Required archive member is missing: {name}")
        rows = list(csv.DictReader(io.StringIO(zf.read(csv_name).decode("utf-8"))))
        config = json.loads(zf.read(cfg_name))
        environment = json.loads(zf.read(env_name))
        json_results: list[dict[str, Any]] = []
        pattern = re.compile(r"results/strong_scaling_500d/star_p1_run\d+\.json$")
        for name in sorted(n for n in names if pattern.match(n)):
            result = json.loads(zf.read(name))
            result["_archive_member"] = name
            json_results.append(result)
    return rows, config, environment, json_results


def mean_ci(values: np.ndarray, confidence: float = 0.95) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    n = len(values)
    mean = float(np.mean(values))
    if n < 2:
        return mean, mean, mean
    sd = float(np.std(values, ddof=1))
    try:
        from scipy.stats import t
        critical = float(t.ppf((1.0 + confidence) / 2.0, n - 1))
    except Exception:
        critical = 1.96
    half = critical * sd / math.sqrt(n)
    return mean, mean - half, mean + half


def summarize(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    workers = sorted({int(r["workers"]) for r in rows})
    out = []
    for p in workers:
        group = [r for r in rows if int(r["workers"]) == p]
        elapsed = np.array([float(r["elapsed_seconds"]) for r in group])
        best = np.array([float(r["best_value"]) for r in group])
        mean_t, lo_t, hi_t = mean_ci(elapsed)
        mean_f, lo_f, hi_f = mean_ci(best)
        out.append({
            "workers_requested": p,
            "runs": len(group),
            "actual_threads_unique": sorted({
                int(json.loads(r.get("actual_threads", "1")) if r.get("actual_threads") else 1)
                for r in group
            }),
            "openmp_enabled_all": all(r["openmp_enabled"].lower() == "true" for r in group),
            "function_evaluations_unique": sorted({int(r["function_evaluations"]) for r in group}),
            "elapsed_mean_s": mean_t,
            "elapsed_sd_s": float(np.std(elapsed, ddof=1)) if len(elapsed) > 1 else 0.0,
            "elapsed_median_s": float(np.median(elapsed)),
            "elapsed_q25_s": float(np.quantile(elapsed, 0.25)),
            "elapsed_q75_s": float(np.quantile(elapsed, 0.75)),
            "elapsed_ci95_low_s": lo_t,
            "elapsed_ci95_high_s": hi_t,
            "best_mean": mean_f,
            "best_sd": float(np.std(best, ddof=1)) if len(best) > 1 else 0.0,
            "best_median": float(np.median(best)),
            "best_q25": float(np.quantile(best, 0.25)),
            "best_q75": float(np.quantile(best, 0.75)),
            "best_ci95_low": lo_f,
            "best_ci95_high": hi_f,
        })
    return out


def paired_seed_audit(rows: list[dict[str, str]], run_json: list[dict[str, Any]]) -> dict[str, Any]:
    requested_workers = sorted({int(r["workers"]) for r in rows})
    by_seed_worker = {(int(r["seed"]), int(r["workers"])): float(r["best_value"]) for r in rows}
    seeds = sorted({int(r["seed"]) for r in rows})
    p1 = {int(r["seed"]): float(r["best_value"]) for r in rows if int(r["workers"]) == 1}
    paired_diffs = []
    all_pairs_match = True
    for p in requested_workers:
        diffs = [abs(by_seed_worker[(seed, p)] - p1[seed]) for seed in seeds]
        maximum = max(diffs) if diffs else 0.0
        paired_diffs.append({"workers_requested": p, "max_abs_final_value_difference_vs_P1": maximum})
        if maximum > 1e-12:
            all_pairs_match = False

    histories_by_seed = {int(d["seed"]): d["history"] for d in run_json}
    p1_history_complete = len(histories_by_seed) == len(seeds)
    history_lengths = sorted({len(h) for h in histories_by_seed.values()})
    history_monotone = all(
        all(b <= a + 1e-12 for a, b in zip(h, h[1:]))
        for h in histories_by_seed.values()
    )
    openmp_enabled_all = all(r["openmp_enabled"].lower() == "true" for r in rows)
    actual_team_matches_request = all(
        int(r.get("max_openmp_threads", "1")) == int(r["workers"]) for r in rows
    )
    # The legacy CSV does not contain actual_threads. The JSON files do.
    # Its max_openmp_threads field is the runtime maximum, not the actual team;
    # infer team execution from the per-run environment plus the user-provided
    # environment report (which explicitly says OpenMP was unavailable).
    strong_scaling_valid = openmp_enabled_all and actual_team_matches_request and len(requested_workers) > 1
    return {
        "requested_worker_counts": requested_workers,
        "seeds_per_worker_count": len(seeds),
        "seed_sets_complete_and_equal": all(
            {int(r["seed"]) for r in rows if int(r["workers"]) == p} == set(seeds)
            for p in requested_workers
        ),
        "all_seed_paired_final_values_equal_to_P1_within_1e-12": all_pairs_match,
        "paired_final_value_max_differences": paired_diffs,
        "P1_history_file_count": len(run_json),
        "P1_history_count_matches_seed_count": p1_history_complete,
        "history_lengths": history_lengths,
        "all_P1_best_so_far_histories_monotone": history_monotone,
        "openmp_enabled_all_runs": openmp_enabled_all,
        "runtime_thread_capacity_matches_requested_workers": actual_team_matches_request,
        "strong_scaling_valid": strong_scaling_valid,
        "strong_scaling_reason": (
            "OpenMP was disabled in the recorded environment; all runs therefore used the serial fallback."
            if not openmp_enabled_all else
            "The recorded runtime team size did not match the requested worker count."
            if not actual_team_matches_request else
            "Valid according to available metadata; inspect affinity and hardware reporting before publication."
        ),
    }


def save_summary_csv(path: Path, summary: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)


def make_figures(
    output_preliminary: Path,
    output_diagnostic: Path,
    p1_runs: list[dict[str, Any]],
    rows: list[dict[str, str]],
    summary: list[dict[str, Any]],
) -> None:
    configure_style()
    output_preliminary.mkdir(parents=True, exist_ok=True)
    output_diagnostic.mkdir(parents=True, exist_ok=True)

    # Figure 1: P=1 serial baseline, median with interquartile band.
    histories = np.asarray([r["history"] for r in p1_runs], dtype=float)
    fes = np.asarray(p1_runs[0]["history_fes"], dtype=int)
    med = np.median(histories, axis=0)
    q25 = np.quantile(histories, 0.25, axis=0)
    q75 = np.quantile(histories, 0.75, axis=0)
    fig, ax = plt.subplots(figsize=(6.65, 3.75))
    ax.plot(fes, med, color=PALETTE["navy"], label=f"Median best-so-far ($n={len(p1_runs)}$)")
    ax.fill_between(fes, q25, q75, color=PALETTE["blue"], alpha=0.20, linewidth=0,
                    label="25th-75th percentile")
    ax.set_xlabel("Objective-function evaluations")
    ax.set_ylabel("Best-so-far objective value")
    ax.set_title("CPSO on shifted Rastrigin ($D=500$): serial baseline")
    ax.grid(True, which="major", axis="both")
    ax.legend(frameon=False, loc="upper right")
    fig.savefig(output_preliminary / "fig_01_serial_convergence_rastrigin_500d.pdf")
    plt.close(fig)

    # Figure 2: final best values for the valid single-thread baseline.
    final_values = np.asarray([float(r["best_value"]) for r in rows if int(r["workers"]) == 1])
    fig, ax = plt.subplots(figsize=(5.7, 3.7))
    violin = ax.violinplot([final_values], positions=[1], widths=0.58,
                           showmeans=False, showmedians=True, showextrema=False)
    for body in violin["bodies"]:
        body.set_facecolor(PALETTE["teal"])
        body.set_edgecolor(PALETTE["navy"])
        body.set_alpha(0.30)
    violin["cmedians"].set_color(PALETTE["navy"])
    rng = np.random.default_rng(20261104)
    jitter = rng.uniform(-0.07, 0.07, size=len(final_values))
    ax.scatter(1 + jitter, final_values, s=19, color=PALETTE["blue"], alpha=0.75,
               edgecolor="white", linewidth=0.35, zorder=3)
    ax.set_xlim(0.55, 1.45)
    ax.set_xticks([1], [f"Star, $P=1$ ($n={len(final_values)}$)"])
    ax.set_ylabel("Final objective value")
    ax.set_title("Final objective distribution: serial baseline")
    ax.grid(True, axis="y")
    fig.savefig(output_preliminary / "fig_02_final_objective_distribution_rastrigin_500d.pdf")
    plt.close(fig)

    # Figure 3: end-to-end serial baseline wall-clock distribution.
    serial_times = np.asarray([float(r["elapsed_seconds"]) for r in rows if int(r["workers"]) == 1])
    fig, ax = plt.subplots(figsize=(5.7, 3.7))
    parts = ax.violinplot([serial_times], positions=[1], widths=0.58,
                          showmeans=False, showmedians=True, showextrema=False)
    for body in parts["bodies"]:
        body.set_facecolor(PALETTE["gold"])
        body.set_edgecolor(PALETTE["navy"])
        body.set_alpha(0.30)
    parts["cmedians"].set_color(PALETTE["navy"])
    ax.scatter(1 + rng.uniform(-0.07, 0.07, len(serial_times)), serial_times,
               s=19, color=PALETTE["gold"], alpha=0.85,
               edgecolor="white", linewidth=0.35, zorder=3)
    ax.set_xlim(0.55, 1.45)
    ax.set_xticks([1], [f"Serial execution ($n={len(serial_times)}$)"])
    ax.set_ylabel("Wall-clock time (s)")
    ax.set_title("CPSO runtime distribution: serial baseline")
    ax.grid(True, axis="y")
    fig.savefig(output_preliminary / "fig_03_serial_runtime_distribution_rastrigin_500d.pdf")
    plt.close(fig)

    # Diagnostic 1: requested worker count vs actual threads in the archived environment.
    worker = np.asarray([s["workers_requested"] for s in summary], dtype=int)
    # This archive has environment-level max_openmp_threads=1 and openmp_enabled=false.
    actual = np.ones_like(worker)
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    ax.plot(worker, worker, linestyle="--", color=PALETTE["grey"],
            linewidth=1.0, label="Requested = actual (ideal)")
    ax.plot(worker, actual, marker="o", color=PALETTE["red"],
            label="Actual OpenMP threads (recorded)")
    ax.set_xlabel("Requested worker count")
    ax.set_ylabel("OpenMP threads")
    ax.set_title("Runtime verification: requested vs actual OpenMP threads")
    ax.set_xscale("log", base=2)
    ax.set_xticks(worker, [str(x) for x in worker])
    ax.set_ylim(0, max(worker) + 5)
    ax.grid(True)
    ax.legend(frameon=False)
    fig.subplots_adjust(bottom=0.22)
    fig.text(0.14, 0.035, "DIAGNOSTIC ONLY - OpenMP disabled; not a strong-scaling result",
             fontsize=8, color=PALETTE["red"], weight="bold")
    fig.savefig(output_diagnostic / "diagnostic_01_requested_vs_actual_threads.pdf")
    plt.close(fig)

    # Diagnostic 2: timing by requested P, explicitly labelled serial fallback.
    grouped = [[float(r["elapsed_seconds"]) for r in rows if int(r["workers"]) == p]
               for p in worker]
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    bp = ax.boxplot(grouped, positions=np.arange(1, len(worker) + 1), widths=0.58,
                    patch_artist=True, showfliers=False)
    for patch in bp["boxes"]:
        patch.set_facecolor(PALETTE["light"])
        patch.set_edgecolor(PALETTE["blue"])
    for line in bp["medians"]:
        line.set_color(PALETTE["navy"])
        line.set_linewidth(1.5)
    ax.set_xticks(np.arange(1, len(worker) + 1), [str(x) for x in worker])
    ax.set_xlabel("Requested worker count (no OpenMP team was created)")
    ax.set_ylabel("Wall-clock time (s)")
    ax.set_title("Timing audit of the serial-fallback runs")
    ax.grid(True, axis="y")
    fig.subplots_adjust(bottom=0.22)
    fig.text(0.14, 0.035, "DIAGNOSTIC ONLY - do not interpret as speedup or efficiency",
             fontsize=8, color=PALETTE["red"], weight="bold")
    fig.savefig(output_diagnostic / "diagnostic_02_serial_fallback_timings.pdf")
    plt.close(fig)


def build_report(rows: list[dict[str, str]], config: dict[str, Any], environment: dict[str, Any],
                 paired: dict[str, Any], summary: list[dict[str, Any]]) -> str:
    p1 = next(s for s in summary if s["workers_requested"] == 1)
    times = [float(r["elapsed_seconds"]) for r in rows if int(r["workers"]) == 1]
    vals = [float(r["best_value"]) for r in rows if int(r["workers"]) == 1]
    return f'''# Audit of the first CPSO-HPC experiment archive

**Archive:** `results_received_2026-11-04.zip`  
**Assessment:** preliminary serial baseline and reproducibility check; **not a valid strong-scaling dataset**.

## 1. Experiment contents

- Runs: {len(rows)}
- Benchmark: {config['problem']['name']}; dimension: {config['problem']['dimension']}
- Topology: Star; synchronization interval: {config['algorithm']['sync_interval']}
- Global population: {config['algorithm']['population_size']}
- Iterations: {config['algorithm']['iterations']}
- Function evaluations per run: {sorted({int(r['function_evaluations']) for r in rows})}
- Requested worker counts: {paired['requested_worker_counts']}
- Repetitions per requested worker count: {paired['seeds_per_worker_count']}

## 2. Critical execution finding

The saved environment reports `openmp_enabled = {environment['openmp_enabled']}` and
`max_openmp_threads = {environment['max_openmp_threads']}` on `{environment['platform']}`.
The result records do not contain a per-run OpenMP team size, but the environment explicitly
reports that OpenMP was unavailable and the runtime maximum was one thread. Consequently,
the worker-count sweep did not execute parallel teams and **must not be used to calculate
or claim strong speedup, parallel efficiency, or scalability**.

The apparent runtimes by requested worker count are retained only as a diagnostic. They
are not parallel timings. The correct interpretation is that 240 serial executions were
performed with different logical island partitions.

## 3. Valid single-thread baseline

For the 30 runs with requested P=1:

- Mean wall time: {float(np.mean(times)):.6f} s
- Standard deviation: {float(np.std(times, ddof=1)):.6f} s
- Median wall time: {float(np.median(times)):.6f} s
- Mean final objective: {float(np.mean(vals)):.6f}
- Standard deviation of final objective: {float(np.std(vals, ddof=1)):.6f}
- Median final objective: {float(np.median(vals)):.6f}
- Exact FEs per run: {int(rows[0]['function_evaluations'])}

These figures establish a serial reference for this benchmark/configuration only. They do
not establish general optimization performance, nor do they validate the topology variants.

## 4. Reproducibility and accounting

- Seed sets complete and equal across requested worker counts: `{paired['seed_sets_complete_and_equal']}`
- Paired final values equal to P=1 within 1e-12: `{paired['all_seed_paired_final_values_equal_to_P1_within_1e-12']}`
- P=1 history file count matches seed count: `{paired['P1_history_count_matches_seed_count']}`
- All P=1 best-so-far histories are monotone: `{paired['all_P1_best_so_far_histories_monotone']}`
- History lengths: `{paired['history_lengths']}`

The equality of paired Star outcomes is consistent with the fixed-seed, sync-every-iteration
invariance of the current algorithm. It is a useful implementation consistency check, not a
parallel performance measurement.

## 5. Decision for the manuscript

1. Retain this archive as the versioned serial fallback/reproducibility dataset.
2. Use only the P=1 values for the initial single-thread convergence and runtime plots.
3. Exclude all worker-sweep timing from claims or plots about parallel performance.
4. Re-run the Star strong-scaling protocol with an OpenMP-enabled build; the new runner fails
   closed if OpenMP is disabled or the actual team size differs from the requested P.
5. Run the topology study separately. Ring and Independent alter information flow and are
   algorithmic variants, not merely thread-count variants of Star.
'''


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path, help="Original results.zip archive.")
    parser.add_argument("--output-dir", default="results/preliminary_serial_fallback", type=Path)
    parser.add_argument("--figures-dir", default="figures", type=Path)
    args = parser.parse_args()

    rows, config, environment, p1_runs = read_archive(args.archive)
    if not rows:
        raise RuntimeError("No scaling rows found in the supplied results archive.")
    summary = summarize(rows)
    paired = paired_seed_audit(rows, p1_runs)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_summary_csv(args.output_dir / "summary_by_requested_workers.csv", summary)
    (args.output_dir / "audit.json").write_text(json.dumps({
        "environment": environment,
        "experiment_configuration": {
            "benchmark": config["problem"]["name"],
            "dimension": config["problem"]["dimension"],
            "population_size": config["algorithm"]["population_size"],
            "iterations": config["algorithm"]["iterations"],
            "topology": config["algorithm"]["topology"],
            "sync_interval": config["algorithm"]["sync_interval"],
            "sweep": config["sweep"],
        },
        "paired_seed_audit": paired,
        "summary": summary,
    }, indent=2), encoding="utf-8")
    (args.output_dir / "results_audit.md").write_text(
        build_report(rows, config, environment, paired, summary), encoding="utf-8"
    )
    make_figures(
        args.figures_dir / "preliminary",
        args.figures_dir / "diagnostics",
        p1_runs,
        rows,
        summary,
    )
    print(f"Audited {len(rows)} runs; strong_scaling_valid={paired['strong_scaling_valid']}")
    print(f"Summary: {args.output_dir / 'results_audit.md'}")
    print(f"Figures: {args.figures_dir}")


if __name__ == "__main__":
    main()
