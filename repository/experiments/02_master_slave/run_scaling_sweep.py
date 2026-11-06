#!/usr/bin/env python3
"""Run paired strong-scaling experiments for the synchronous master--slave CPSO.

The ``sphere_work`` objective is a controlled synthetic arithmetic workload,
not a standard optimization benchmark. It is intended to estimate the break-even
region where parallel objective evaluation can amortize task scheduling costs.
Every measured parallel run is checked against the serial result for exact
best-value, best-position, best-history, and function-evaluation equivalence.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import math
import os
import platform
import random
import statistics
import sys
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

from cpso_serial import (  # noqa: E402
    CpsoConfig,
    native_build_info,
    native_version,
    run_master_slave,
    run_serial,
)


BENCHMARK_BOUNDS = {
    "sphere": 5.12,
    "rastrigin": 5.12,
    "rosenbrock": 5.0,
    "ackley": 32.768,
    "griewank": 600.0,
    "salomon": 100.0,
    "sphere_work": 5.12,
}


def parse_int_list(value: str, *, name: str, minimum: int = 1) -> list[int]:
    try:
        result = sorted({int(part.strip()) for part in value.split(",") if part.strip()})
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{name} must be comma-separated integers") from exc
    if not result or any(item < minimum for item in result):
        raise argparse.ArgumentTypeError(f"{name} values must be >= {minimum}")
    return result


def quantile(values: list[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = probability * (len(ordered) - 1)
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return ordered[low]
    weight = position - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def bootstrap_median_ci(
    values: list[float], *, seed: int, repetitions: int = 4000
) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    if len(values) == 1:
        return values[0], values[0]
    rng = random.Random(seed)
    estimates = [
        statistics.median(rng.choices(values, k=len(values)))
        for _ in range(repetitions)
    ]
    low = quantile(estimates, 0.025)
    high = quantile(estimates, 0.975)
    return low, high


def exact_match(reference: Any, candidate: Any) -> bool:
    return (
        reference.best_value == candidate.best_value
        and reference.best_position == candidate.best_position
        and reference.best_history == candidate.best_history
        and reference.function_evaluations == candidate.function_evaluations
        and reference.iterations_completed == candidate.iterations_completed
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--objective", choices=tuple(BENCHMARK_BOUNDS), default="sphere_work")
    parser.add_argument("--dimensions", type=lambda v: parse_int_list(v, name="dimensions"), default=[32])
    parser.add_argument("--populations", type=lambda v: parse_int_list(v, name="populations"), default=[32])
    parser.add_argument(
        "--work-iterations",
        type=lambda v: parse_int_list(v, name="work-iterations", minimum=0),
        default=None,
        help="Per-evaluation recurrence count; used only with sphere_work (0 means no extra loop)",
    )
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--seeds", type=int, default=3,
                        help="Number of paired seeds for each configuration")
    parser.add_argument("--seed-start", type=int, default=11000)
    parser.add_argument("--workers", type=lambda v: parse_int_list(v, name="workers"), default=[2, 4])
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--bootstrap-repetitions", type=int, default=4000)
    args = parser.parse_args()

    if args.iterations < 0 or args.seeds < 1 or args.warmups < 0:
        parser.error("iterations/warmups must be non-negative and seeds must be positive")
    if args.seed_start < 0 or args.seed_start + args.seeds - 1 > (1 << 64) - 1:
        parser.error("seed range must fit an unsigned 64-bit integer")
    if args.bootstrap_repetitions < 100:
        parser.error("bootstrap-repetitions must be >= 100")
    if args.work_iterations is None:
        args.work_iterations = [0, 5000, 20000] if args.objective == "sphere_work" else [0]
    elif args.objective != "sphere_work" and args.work_iterations != [0]:
        parser.error("nonzero work-iterations are only supported for objective=sphere_work")
    if args.objective != "sphere_work":
        args.work_iterations = [0]
    if args.output is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        args.output = ROOT / "results" / "raw" / "master_slave_sweep" / f"run_{stamp}"
    return args


def build_config(dimension: int, population: int, iterations: int, seed: int, bound: float) -> CpsoConfig:
    return CpsoConfig(
        lower_bounds=(-bound,) * dimension,
        upper_bounds=(bound,) * dimension,
        population_size=population,
        iterations=iterations,
        seed=seed,
    )


def result_row(
    *, result: Any, mode: str, workers: int, objective: str, work_iterations: int,
    dimension: int, population: int, iterations: int, seed: int,
) -> dict[str, Any]:
    elapsed = result.elapsed_seconds
    evaluations = result.function_evaluations
    residual = elapsed - result.evaluation_wall_seconds - result.propagation_seconds
    return {
        "objective": objective,
        "work_iterations": work_iterations,
        "dimension": dimension,
        "population_size": population,
        "iterations": iterations,
        "seed": seed,
        "mode": mode,
        "workers": workers,
        "function_evaluations": evaluations,
        "best_value": result.best_value,
        "elapsed_seconds": elapsed,
        "evaluation_batch_wall_seconds": result.evaluation_wall_seconds,
        "propagation_seconds": result.propagation_seconds,
        "residual_seconds": max(0.0, residual),
        "evaluations_per_second": evaluations / elapsed if elapsed > 0.0 else None,
        "propagation_fraction": result.propagation_seconds / elapsed if elapsed > 0.0 else None,
    }


def group_key(row: dict[str, Any]) -> tuple[int, int, int]:
    return int(row["dimension"]), int(row["population_size"]), int(row["work_iterations"])


def main() -> int:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    bound = BENCHMARK_BOUNDS[args.objective]
    rows: list[dict[str, Any]] = []
    pairs: dict[tuple[int, int, int, int], list[tuple[Any, Any]]] = {}

    configs = [
        (dimension, population, work)
        for dimension in args.dimensions
        for population in args.populations
        for work in args.work_iterations
    ]

    for dimension, population, work in configs:
        # Warm-up each path once per scenario; warm-up samples are not recorded.
        for warmup_index in range(args.warmups):
            warmup_seed = (args.seed_start + args.seeds + 1009 * warmup_index) & ((1 << 64) - 1)
            warmup_config = build_config(dimension, population, args.iterations, warmup_seed, bound)
            run_serial(warmup_config, args.objective, work_iterations=work)
            for worker_count in args.workers:
                run_master_slave(
                    warmup_config, args.objective, workers=worker_count, work_iterations=work
                )

        for seed in range(args.seed_start, args.seed_start + args.seeds):
            config = build_config(dimension, population, args.iterations, seed, bound)
            conditions = [("serial", 1)] + [("master_slave", worker) for worker in args.workers]
            shuffle_seed = seed ^ (dimension << 17) ^ (population << 5) ^ work ^ 0x4350534F
            random.Random(shuffle_seed).shuffle(conditions)
            results: dict[tuple[str, int], Any] = {}
            for mode, worker_count in conditions:
                if mode == "serial":
                    result = run_serial(config, args.objective, work_iterations=work)
                else:
                    result = run_master_slave(
                        config, args.objective, workers=worker_count, work_iterations=work
                    )
                results[(mode, worker_count)] = result
                rows.append(result_row(
                    result=result, mode=mode, workers=worker_count,
                    objective=args.objective, work_iterations=work,
                    dimension=dimension, population=population,
                    iterations=args.iterations, seed=seed,
                ))

            serial_result = results[("serial", 1)]
            for worker_count in args.workers:
                parallel_result = results[("master_slave", worker_count)]
                if not exact_match(serial_result, parallel_result):
                    raise RuntimeError(
                        "Serial/master-slave algorithmic output mismatch: "
                        f"dimension={dimension}, population={population}, work={work}, "
                        f"seed={seed}, workers={worker_count}"
                    )
                key = (dimension, population, work, worker_count)
                pairs.setdefault(key, []).append((serial_result, parallel_result))

    csv_path = args.output / "scaling_runs.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary_rows: list[dict[str, Any]] = []
    for (dimension, population, work, worker_count), pair_values in sorted(pairs.items()):
        total_speedups = [s.elapsed_seconds / p.elapsed_seconds
                          for s, p in pair_values if p.elapsed_seconds > 0.0]
        eval_speedups = [s.evaluation_wall_seconds / p.evaluation_wall_seconds
                         for s, p in pair_values if p.evaluation_wall_seconds > 0.0]
        efficiencies = [speedup / worker_count for speedup in total_speedups]
        serial_times = [s.elapsed_seconds for s, _ in pair_values]
        parallel_times = [p.elapsed_seconds for _, p in pair_values]
        ci_low, ci_high = bootstrap_median_ci(
            total_speedups,
            seed=dimension ^ (population << 7) ^ (work << 1) ^ worker_count,
            repetitions=args.bootstrap_repetitions,
        )
        median_speedup = statistics.median(total_speedups) if total_speedups else None
        speedup_mad = (
            statistics.median(abs(value - median_speedup) for value in total_speedups)
            if median_speedup is not None else None
        )
        summary_rows.append({
            "dimension": dimension,
            "population_size": population,
            "work_iterations": work,
            "workers": worker_count,
            "paired_runs": len(pair_values),
            "median_serial_seconds": statistics.median(serial_times),
            "median_parallel_seconds": statistics.median(parallel_times),
            "median_total_speedup": median_speedup,
            "total_speedup_mad": speedup_mad,
            "total_speedup_bootstrap_ci95_low": ci_low,
            "total_speedup_bootstrap_ci95_high": ci_high,
            "median_evaluation_batch_speedup": statistics.median(eval_speedups) if eval_speedups else None,
            "median_parallel_efficiency": statistics.median(efficiencies) if efficiencies else None,
            "median_serial_evaluations_per_second": statistics.median(
                [s.function_evaluations / s.elapsed_seconds for s, _ in pair_values if s.elapsed_seconds > 0]
            ),
            "median_parallel_evaluations_per_second": statistics.median(
                [p.function_evaluations / p.elapsed_seconds for _, p in pair_values if p.elapsed_seconds > 0]
            ),
            "exact_output_equivalence": True,
        })

    summary_path = args.output / "scaling_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "release": native_version(),
        "native_build_info": native_build_info(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "machine": platform.machine(),
        "logical_cpu_count": os.cpu_count(),
        "objective": args.objective,
        "objective_semantics": (
            "Synthetic Sphere arithmetic workload: Sphere(x) plus a negligible deterministic positive term; "
            "work_iterations sets recurrence count per objective evaluation. Not a standard benchmark."
            if args.objective == "sphere_work" else "Standard built-in analytic objective."
        ),
        "configuration": {
            "dimensions": args.dimensions,
            "populations": args.populations,
            "iterations": args.iterations,
            "work_iterations": args.work_iterations,
            "seeds": args.seeds,
            "seed_start": args.seed_start,
            "workers": args.workers,
            "warmups": args.warmups,
            "bootstrap_repetitions": args.bootstrap_repetitions,
        },
        "total_runs": len(rows),
        "paired_configurations": len(summary_rows),
        "equivalence_pairs_verified": len(pairs) * args.seeds,
        "exact_output_equivalence_verified_for_all_pairs": True,
        "timing_definitions": {
            "elapsed_seconds": "End-to-end native optimizer wall time; includes worker-pool creation and shutdown in master-slave mode.",
            "evaluation_batch_wall_seconds": "Sum of wall-clock durations of each complete evaluation batch, including dispatch and waiting.",
            "propagation_seconds": "Sequential particle propagation time as instrumented by the native engine.",
            "residual_seconds": "Non-negative residual elapsed minus evaluation-batch wall and propagation; includes state updates and other orchestration, not a direct measurement of scheduler overhead.",
        },
        "speedup_definition": "Paired serial elapsed_seconds / paired master-slave elapsed_seconds, same seed/configuration/objective work.",
        "statistical_summary": "Median paired speedup, MAD, and percentile bootstrap 95% interval for the median; exploratory with small seed counts.",
        "caveats": [
            "Run on an otherwise idle machine with a fixed power mode where possible.",
            "Record the Mac model, Apple Silicon chip, macOS version, compiler and build type from metadata.",
            "Repeat the full sweep on the target Mac before making any hardware-performance claim.",
            "The synthetic objective estimates a controlled arithmetic break-even region; it does not replace COCO/CEC objective evaluations.",
            "Logical CPU count does not imply homogeneous Apple Silicon cores or linear scaling.",
        ],
    }
    metadata_path = args.output / "scaling_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    print(f"Raw runs: {csv_path}")
    print(f"Paired summary: {summary_path}")
    print(f"Metadata: {metadata_path}")
    print(f"Exact serial/master-slave equivalence verified for {len(pairs) * args.seeds} worker/seed pairs.")
    for item in summary_rows:
        ci = (item["total_speedup_bootstrap_ci95_low"], item["total_speedup_bootstrap_ci95_high"])
        print(
            f"D={item['dimension']:>4}, N={item['population_size']:>4}, "
            f"work={item['work_iterations']:>7}, workers={item['workers']:>2}: "
            f"median speedup={item['median_total_speedup']:.3f}x "
            f"(bootstrap 95% CI {ci[0]:.3f}, {ci[1]:.3f})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
