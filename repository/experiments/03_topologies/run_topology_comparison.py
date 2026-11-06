from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import platform
import random
import sys
from typing import Iterable

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

from cpso_serial import (  # noqa: E402
    CpsoConfig,
    native_build_info,
    native_version,
    run_master_slave,
    run_serial,
)

TOPOLOGIES = ("global", "ring", "grid_von_neumann")
BENCHMARKS = ("sphere", "rastrigin", "rosenbrock", "ackley", "griewank", "salomon")


def factor_grid(population_size: int) -> tuple[int, int]:
    """Return the factor pair closest to a square, with rows <= columns."""
    rows = math.isqrt(population_size)
    while rows > 1 and population_size % rows != 0:
        rows -= 1
    columns = population_size // rows
    if rows < 2 or columns < 2:
        raise ValueError("a non-degenerate 2D grid requires a composite population with both dimensions >= 2")
    return rows, columns


def benchmark_bounds(
    name: str, dimension: int, symmetric_bound: float | None
) -> tuple[list[float], list[float]]:
    if symmetric_bound is not None:
        return [-symmetric_bound] * dimension, [symmetric_bound] * dimension
    ranges = {
        "sphere": (-5.12, 5.12),
        "rastrigin": (-5.12, 5.12),
        "rosenbrock": (-5.0, 10.0),
        "ackley": (-32.768, 32.768),
        "griewank": (-600.0, 600.0),
        "salomon": (-100.0, 100.0),
    }
    lower, upper = ranges[name]
    return [lower] * dimension, [upper] * dimension


def _write_row(writer: csv.DictWriter, handle, row: dict) -> None:
    writer.writerow(row)
    handle.flush()


def run_experiment(args: argparse.Namespace) -> Path:
    grid_rows, grid_columns = args.grid_rows, args.grid_columns
    if grid_rows is None and grid_columns is None:
        grid_rows, grid_columns = factor_grid(args.population)
    elif grid_rows is None or grid_columns is None:
        raise ValueError("--grid-rows and --grid-columns must be supplied together")
    if grid_rows < 2 or grid_columns < 2:
        raise ValueError("grid topology requires both dimensions >= 2")
    if grid_rows * grid_columns != args.population:
        raise ValueError("grid_rows * grid_columns must equal --population")
    if args.workers < 0 or args.workers > 4096:
        raise ValueError("--workers must be in [0,4096]; 0 selects serial execution")
    if args.seeds < 1 or args.iterations < 0 or args.population < 1:
        raise ValueError("seeds/population must be positive and iterations non-negative")
    if not args.dimensions or any(dimension < 1 for dimension in args.dimensions):
        raise ValueError("all dimensions must be positive")
    if "rosenbrock" in args.benchmarks and any(dimension < 2 for dimension in args.dimensions):
        raise ValueError("Rosenbrock requires dimension >= 2")
    unknown = sorted(set(args.benchmarks) - set(BENCHMARKS))
    if unknown:
        raise ValueError(f"Unsupported benchmark(s): {unknown}")

    seeds = [args.seed_start + offset for offset in range(args.seeds)]
    if any(seed < 0 or seed > (1 << 64) - 1 for seed in seeds):
        raise ValueError("seed range must fit unsigned 64-bit integers")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = args.output_dir or ROOT / "results" / "raw" / "topology_comparison" / timestamp
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=False)

    execution_mode = "serial" if args.workers == 0 else "master_slave"
    worker_count = 1 if args.workers == 0 else args.workers
    total_runs = len(args.benchmarks) * len(args.dimensions) * len(seeds) * len(TOPOLOGIES)
    metadata = {
        "schema_version": 1,
        "native_version": native_version(),
        "native_build_info": native_build_info(),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "logical_cpu_count": os.cpu_count(),
        "benchmark_suite": list(args.benchmarks),
        "dimensions": list(args.dimensions),
        "population_size": args.population,
        "iterations": args.iterations,
        "seed_start": args.seed_start,
        "seeds": seeds,
        "topologies": list(TOPOLOGIES),
        "grid_shape": [grid_rows, grid_columns],
        "grid_definition": "toroidal von-Neumann 4-neighborhood plus self; duplicate indices removed",
        "ring_definition": "periodic radius-1 ring: predecessor, self, successor",
        "global_definition": "incumbent population-wide personal best; retained on equal objective values",
        "local_tie_break": "smallest particle index among equal personal-best values",
        "execution_mode": execution_mode,
        "worker_count": worker_count,
        "objective_bounds": "benchmark-specific standard box unless --bound overrides with a symmetric box",
        "success_tolerance": args.success_tolerance,
        "paired_design": "each seed is run with all three topologies; execution order is deterministically shuffled per seed and dimension",
        "interpretation": "algorithmic topology comparison; no speedup claim is implied by this experiment",
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )

    run_fields = [
        "run_id", "benchmark", "dimension", "population_size", "iterations", "seed",
        "topology", "grid_rows", "grid_columns", "execution_mode", "worker_count", "run_order",
        "best_value", "best_position", "success_tolerance", "success", "function_evaluations",
        "elapsed_seconds", "evaluation_wall_seconds", "propagation_seconds",
        "topology_setup_seconds", "social_selection_seconds",
    ]
    convergence_fields = [
        "run_id", "benchmark", "dimension", "seed", "topology", "iteration", "best_value"
    ]
    run_counter = 0
    with (
        (output_dir / "runs.csv").open("w", newline="", encoding="utf-8") as runs_handle,
        (output_dir / "convergence.csv").open("w", newline="", encoding="utf-8") as convergence_handle,
    ):
        runs_writer = csv.DictWriter(runs_handle, fieldnames=run_fields)
        convergence_writer = csv.DictWriter(convergence_handle, fieldnames=convergence_fields)
        runs_writer.writeheader()
        convergence_writer.writeheader()
        runs_handle.flush()
        convergence_handle.flush()

        for benchmark in args.benchmarks:
            for dimension in args.dimensions:
                lower, upper = benchmark_bounds(benchmark, dimension, args.bound)
                for seed in seeds:
                    order = list(TOPOLOGIES)
                    random.Random(seed ^ 0xC05A71 ^ dimension).shuffle(order)
                    for run_order, topology_name in enumerate(order):
                        topology_kwargs: dict[str, int | str] = {"topology": topology_name}
                        if topology_name == "grid_von_neumann":
                            topology_kwargs.update(grid_rows=grid_rows, grid_columns=grid_columns)
                        config = CpsoConfig(
                            lower_bounds=lower,
                            upper_bounds=upper,
                            population_size=args.population,
                            iterations=args.iterations,
                            seed=seed,
                            topology=str(topology_kwargs["topology"]),
                            grid_rows=int(topology_kwargs["grid_rows"]) if "grid_rows" in topology_kwargs else None,
                            grid_columns=int(topology_kwargs["grid_columns"]) if "grid_columns" in topology_kwargs else None,
                        )
                        if args.workers == 0:
                            result = run_serial(config, benchmark)
                        else:
                            result = run_master_slave(config, benchmark, workers=args.workers)

                        expected_evaluations = args.population * (args.iterations + 1)
                        if result.function_evaluations != expected_evaluations:
                            raise RuntimeError(
                                f"Unexpected FE count for {benchmark}/{topology_name}/seed={seed}: "
                                f"{result.function_evaluations} != {expected_evaluations}"
                            )
                        if len(result.best_history) != args.iterations + 1:
                            raise RuntimeError("Native result has an unexpected best-history length")
                        if not all(math.isfinite(value) for value in result.best_history):
                            raise RuntimeError("Non-finite value in best-so-far history")
                        if any(a < b for a, b in zip(result.best_history, result.best_history[1:])):
                            raise RuntimeError("Best-so-far history must be non-increasing")
                        if result.best_value != result.best_history[-1]:
                            raise RuntimeError("Final objective value differs from final history entry")

                        run_counter += 1
                        run_id = f"{benchmark}_d{dimension}_s{seed}_{topology_name}"
                        _write_row(runs_writer, runs_handle, {
                            "run_id": run_id,
                            "benchmark": benchmark,
                            "dimension": dimension,
                            "population_size": args.population,
                            "iterations": args.iterations,
                            "seed": seed,
                            "topology": topology_name,
                            "grid_rows": grid_rows if topology_name == "grid_von_neumann" else "",
                            "grid_columns": grid_columns if topology_name == "grid_von_neumann" else "",
                            "execution_mode": execution_mode,
                            "worker_count": worker_count,
                            "run_order": run_order,
                            "best_value": format(result.best_value, ".17g"),
                            "best_position": json.dumps(list(result.best_position), separators=(",", ":")),
                            "success_tolerance": args.success_tolerance,
                            "success": int(result.best_value <= args.success_tolerance),
                            "function_evaluations": result.function_evaluations,
                            "elapsed_seconds": format(result.elapsed_seconds, ".17g"),
                            "evaluation_wall_seconds": format(result.evaluation_wall_seconds, ".17g"),
                            "propagation_seconds": format(result.propagation_seconds, ".17g"),
                            "topology_setup_seconds": format(result.topology_setup_seconds, ".17g"),
                            "social_selection_seconds": format(result.social_selection_seconds, ".17g"),
                        })
                        for iteration, value in enumerate(result.best_history):
                            _write_row(convergence_writer, convergence_handle, {
                                "run_id": run_id,
                                "benchmark": benchmark,
                                "dimension": dimension,
                                "seed": seed,
                                "topology": topology_name,
                                "iteration": iteration,
                                "best_value": format(value, ".17g"),
                            })
                        print(
                            f"[{run_counter}/{total_runs}] {benchmark} d={dimension} "
                            f"seed={seed} topology={topology_name}: "
                            f"f_best={result.best_value:.8g} time={result.elapsed_seconds:.4g}s",
                            flush=True,
                        )

    (output_dir / "README.md").write_text(
        "# CPSO particle-topology comparison\n\n"
        "This directory contains paired runs for global-best, periodic ring, and "
        "toroidal von-Neumann grid social topologies. Each topology receives the "
        "same benchmark, dimension, population, iteration budget, CPSO parameters, "
        "and seed. The topology execution order is deterministically shuffled.\n\n"
        "- `metadata.json`: complete run configuration and environment.\n"
        "- `runs.csv`: one row per optimizer run.\n"
        "- `convergence.csv`: best-so-far trace for each run and iteration.\n\n"
        "This is an algorithmic topology comparison. It is not a speedup experiment. "
        "Use equal FE budgets for quality comparisons and analyze elapsed times as "
        "secondary observations only.\n",
        encoding="utf-8",
    )
    return output_dir


def parse_csv_list(value: str, allowed: Iterable[str], argument_name: str) -> list[str]:
    items = [item.strip() for item in value.split(",") if item.strip()]
    invalid = sorted(set(items) - set(allowed))
    if not items or invalid:
        raise argparse.ArgumentTypeError(
            f"{argument_name} must contain values from {sorted(allowed)}; invalid={invalid}"
        )
    if len(items) != len(set(items)):
        raise argparse.ArgumentTypeError(f"{argument_name} must not contain duplicates")
    return items


def parse_dimensions(value: str) -> list[int]:
    try:
        dimensions = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("dimensions must be comma-separated integers") from exc
    if not dimensions:
        raise argparse.ArgumentTypeError("at least one dimension is required")
    if len(dimensions) != len(set(dimensions)):
        raise argparse.ArgumentTypeError("dimensions must not contain duplicates")
    return dimensions


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Paired comparison of global-best, ring, and toroidal grid CPSO social topologies"
    )
    parser.add_argument(
        "--benchmarks",
        type=lambda value: parse_csv_list(value, BENCHMARKS, "benchmarks"),
        default=["sphere", "rastrigin", "rosenbrock", "ackley"],
    )
    parser.add_argument(
        "--dimensions", type=parse_dimensions, default=[10],
        help="comma-separated dimensions, e.g. 10,30,50,100",
    )
    parser.add_argument("--dimension", dest="dimensions", type=lambda value: [int(value)], help=argparse.SUPPRESS)
    parser.add_argument("--population", type=int, default=40)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--seeds", type=int, default=30)
    parser.add_argument("--seed-start", type=int, default=1)
    parser.add_argument("--grid-rows", type=int, default=None)
    parser.add_argument("--grid-columns", type=int, default=None)
    parser.add_argument(
        "--workers", type=int, default=0,
        help="0 uses the serial executor; positive values use master--slave for every topology",
    )
    parser.add_argument(
        "--bound", type=float, default=None,
        help="override benchmark-specific bounds with [-bound,+bound]",
    )
    parser.add_argument("--success-tolerance", type=float, default=1e-8)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    if args.bound is not None and (not math.isfinite(args.bound) or args.bound <= 0.0):
        parser.error("--bound must be finite and positive")
    if not math.isfinite(args.success_tolerance) or args.success_tolerance < 0.0:
        parser.error("--success-tolerance must be finite and non-negative")
    try:
        output = run_experiment(args)
    except (ValueError, RuntimeError, OSError) as exc:
        parser.error(str(exc))
    print(f"\nRaw topology comparison written to: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
