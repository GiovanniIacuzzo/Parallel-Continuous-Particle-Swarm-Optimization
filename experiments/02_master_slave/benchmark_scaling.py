#!/usr/bin/env python3
"""Paired serial/master--slave timing experiment for the built-in objectives.

The script verifies exact algorithmic-output equivalence for every seed before
writing the timing records. It is an exploratory local benchmark driver, not a
replacement for COCO/CEC or a publication-grade performance protocol.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import platform
import random
import statistics
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

from cpso_serial import CpsoConfig, native_build_info, native_version, run_master_slave, run_serial  # noqa: E402


def parse_workers(value: str) -> list[int]:
    try:
        workers = sorted({int(item.strip()) for item in value.split(",") if item.strip()})
    except ValueError as exc:
        raise argparse.ArgumentTypeError("workers must be comma-separated integers") from exc
    if not workers or any(worker < 1 or worker > 4096 for worker in workers):
        raise argparse.ArgumentTypeError("workers must be in [1,4096]")
    return workers


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", default="rastrigin",
                        choices=("sphere", "rastrigin", "rosenbrock", "ackley", "griewank", "salomon"))
    parser.add_argument("--dimension", type=int, default=64)
    parser.add_argument("--population", type=int, default=96)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--seeds", type=int, default=5,
                        help="Number of paired seeds, starting at --seed-start")
    parser.add_argument("--seed-start", type=int, default=1000)
    parser.add_argument("--workers", type=parse_workers, default=parse_workers("2,4"))
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "raw" / "master_slave")
    args = parser.parse_args()
    if args.dimension < 1 or args.population < 1 or args.iterations < 0 or args.seeds < 1 or args.warmups < 0:
        parser.error("dimension/population/seeds must be positive; iterations/warmups non-negative")
    if args.seed_start < 0 or args.seed_start + args.seeds - 1 > (1 << 64) - 1:
        parser.error("seed range must fit an unsigned 64-bit integer")
    return args


def exact_match(reference: Any, candidate: Any) -> bool:
    return (
        reference.best_value == candidate.best_value
        and reference.best_position == candidate.best_position
        and reference.best_history == candidate.best_history
        and reference.function_evaluations == candidate.function_evaluations
    )


def main() -> int:
    args = parse_args()
    bounds = {
        "sphere": 5.12,
        "rastrigin": 5.12,
        "rosenbrock": 5.0,
        "ackley": 32.768,
        "griewank": 600.0,
        "salomon": 100.0,
    }[args.benchmark]

    def make_config(seed: int) -> CpsoConfig:
        return CpsoConfig(
            lower_bounds=(-bounds,) * args.dimension,
            upper_bounds=(bounds,) * args.dimension,
            population_size=args.population,
            iterations=args.iterations,
            seed=seed,
        )

    # Warm all requested code paths, but discard these runs from the dataset.
    for warmup in range(args.warmups):
        config = make_config(args.seed_start + warmup)
        run_serial(config, args.benchmark)
        for worker_count in args.workers:
            run_master_slave(config, args.benchmark, workers=worker_count)

    rows: list[dict[str, Any]] = []
    paired_speedups: dict[int, list[float]] = {worker: [] for worker in args.workers}
    paired_eval_speedups: dict[int, list[float]] = {worker: [] for worker in args.workers}

    for seed in range(args.seed_start, args.seed_start + args.seeds):
        config = make_config(seed)
        conditions = [("serial", 1)] + [("master_slave", worker) for worker in args.workers]
        random.Random(seed ^ 0xC5_50_2026).shuffle(conditions)
        results: dict[tuple[str, int], Any] = {}

        for mode, worker_count in conditions:
            if mode == "serial":
                result = run_serial(config, args.benchmark)
            else:
                result = run_master_slave(config, args.benchmark, workers=worker_count)
            results[(mode, worker_count)] = result
            rows.append({
                "seed": seed,
                "mode": mode,
                "workers": worker_count,
                "benchmark": args.benchmark,
                "dimension": args.dimension,
                "population_size": args.population,
                "iterations": args.iterations,
                "function_evaluations": result.function_evaluations,
                "best_value": result.best_value,
                "elapsed_seconds": result.elapsed_seconds,
                "evaluation_wall_seconds": result.evaluation_wall_seconds,
                "propagation_seconds": result.propagation_seconds,
            })

        reference = results[("serial", 1)]
        for worker_count in args.workers:
            candidate = results[("master_slave", worker_count)]
            if not exact_match(reference, candidate):
                raise RuntimeError(
                    f"Output mismatch for seed={seed}, workers={worker_count}; "
                    "parallel mode is not equivalent to the serial reference"
                )
            if candidate.elapsed_seconds > 0.0:
                paired_speedups[worker_count].append(
                    reference.elapsed_seconds / candidate.elapsed_seconds
                )
            if candidate.evaluation_wall_seconds > 0.0:
                paired_eval_speedups[worker_count].append(
                    reference.evaluation_wall_seconds / candidate.evaluation_wall_seconds
                )

    args.output.mkdir(parents=True, exist_ok=True)
    csv_path = args.output / "master_slave_runs.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summaries = {}
    for worker_count in args.workers:
        total = paired_speedups[worker_count]
        evaluation = paired_eval_speedups[worker_count]
        summaries[str(worker_count)] = {
            "paired_total_time_speedup_mean": statistics.mean(total) if total else None,
            "paired_total_time_speedup_median": statistics.median(total) if total else None,
            "paired_evaluation_wall_speedup_mean": statistics.mean(evaluation) if evaluation else None,
            "paired_evaluation_wall_speedup_median": statistics.median(evaluation) if evaluation else None,
        }

    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "native_version": native_version(),
        "native_build_info": native_build_info(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "machine": platform.machine(),
        "logical_cpu_count": __import__("os").cpu_count(),
        "configuration": {
            "benchmark": args.benchmark,
            "dimension": args.dimension,
            "population_size": args.population,
            "iterations": args.iterations,
            "seed_start": args.seed_start,
            "seeds": args.seeds,
            "workers": args.workers,
            "warmups": args.warmups,
        },
        "exact_output_equivalence_verified": True,
        "speedup_definition": "paired serial elapsed_seconds / master_slave elapsed_seconds",
        "evaluation_wall_definition": "sum of objective-batch wall times, including dispatch and wait overhead",
        "summary_by_worker_count": summaries,
        "interpretation_warning": (
            "Exploratory local measurement. Cheap built-in objectives may be dominated by scheduling overhead. "
            "Do not claim general scalability or hardware-independent speedup from this experiment alone."
        ),
    }
    json_path = args.output / "master_slave_metadata.json"
    json_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    print(f"Wrote {csv_path}")
    print(f"Wrote {json_path}")
    print("Exact best value, best position, best history and FE count matched serial for every seed.")
    for worker, summary in summaries.items():
        print(
            f"workers={worker}: median total-time speedup="
            f"{summary['paired_total_time_speedup_median']:.4f}x"
            if summary["paired_total_time_speedup_median"] is not None
            else f"workers={worker}: speedup unavailable"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
