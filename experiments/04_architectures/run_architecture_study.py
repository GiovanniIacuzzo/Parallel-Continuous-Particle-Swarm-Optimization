#!/usr/bin/env python3
"""Run and analyze a paired study of CPSO parallel execution architectures.

The script is intentionally a single entry point: it executes configured runs,
checkpoints completed runs to JSONL, creates CSV tables, performs descriptive
paired bootstrap analysis, and generates publication-style figures.

Architectures:
  serial              synchronous serial reference
  master_slave        synchronous; parallel objective evaluation only
  particle_parallel   synchronous; parallel particle initialization/propagation
                      and parallel objective evaluation
  asynchronous        event-driven particle updates with immediate publication
  islands             independent sub-swarms run concurrently with optional
                      synchronous ring/fully-connected best-to-worst migration

The master_slave and particle_parallel modes are required to match the serial
reference exactly for the same topology, seed, and optimization parameters.
Asynchronous and island modes intentionally define different search dynamics.
"""
from __future__ import annotations

import argparse
import bisect
import csv
import hashlib
import json
import math
import os
import platform
import random
import statistics
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from checkpoint_io import atomic_write_json, read_jsonl_checkpoint  # noqa: E402
from environment_metadata import collect_environment_metadata  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
PYTHON_ROOT = ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

from cpso_serial import (  # noqa: E402
    CpsoConfig,
    native_build_info,
    native_version,
    run_asynchronous,
    run_island_model,
    run_master_slave,
    run_particle_parallel,
    run_serial,
)

SUPPORTED_ARCHITECTURES = {
    "serial", "master_slave", "particle_parallel", "asynchronous", "islands"
}
SUPPORTED_BENCHMARKS = {
    "sphere", "rastrigin", "rosenbrock", "ackley", "griewank", "salomon", "sphere_work", "inverse_diffusion"
}
LOG_FLOOR = 1.0e-16

DEFAULT_BOUNDS: dict[str, tuple[float, float]] = {
    "sphere": (-100.0, 100.0),
    "rastrigin": (-5.12, 5.12),
    "rosenbrock": (-5.0, 10.0),
    "ackley": (-32.768, 32.768),
    "griewank": (-600.0, 600.0),
    "salomon": (-100.0, 100.0),
    "sphere_work": (-100.0, 100.0),
    "inverse_diffusion": (-2.0, 2.0),
}


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def stable_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def source_fingerprint() -> str:
    """Hash the native/Python implementation and the unified study driver."""
    digest = hashlib.sha256()
    roots = (
        ROOT / "cpp",
        ROOT / "python",
        ROOT / "experiments" / "04_architectures",
    )
    source_suffixes = {".py", ".cpp", ".hpp", ".h"}
    files = [
        path for base in roots if base.exists() for path in base.rglob("*")
        if path.is_file() and path.suffix in source_suffixes
        and "__pycache__" not in path.parts
    ]
    files.extend(path for path in (ROOT / "CMakeLists.txt", ROOT / "pyproject.toml") if path.is_file())
    for path in sorted(set(files), key=lambda item: item.relative_to(ROOT).as_posix()):
        relative = path.relative_to(ROOT).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "little"))
        digest.update(relative)
        content = path.read_bytes()
        digest.update(len(content).to_bytes(8, "little"))
        digest.update(content)
    return digest.hexdigest()


def quantile(values: list[float], probability: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def bootstrap_median_interval(
    values: list[float], seed: int, replicates: int = 2000
) -> tuple[float, float]:
    if not values:
        return math.nan, math.nan
    if len(values) == 1:
        return values[0], values[0]
    rng = random.Random(seed)
    n = len(values)
    medians = [statistics.median(values[rng.randrange(n)] for _ in range(n))
               for _ in range(replicates)]
    return quantile(medians, 0.025), quantile(medians, 0.975)


def grid_shape(population: int, requested: list[int] | None) -> tuple[int, int]:
    if requested is not None:
        if (
            not isinstance(requested, (list, tuple))
            or len(requested) != 2
            or any(isinstance(value, bool) or not isinstance(value, int) for value in requested)
        ):
            raise ValueError("grid_shape must contain two integer values [rows, columns]")
        rows, columns = requested
        if rows < 2 or columns < 2 or rows * columns != population:
            raise ValueError("grid_shape must contain integers >=2 whose product equals population_size")
        return rows, columns
    # Choose the factor pair closest to a square, deterministic for each N.
    best_rows = 1
    for candidate in range(1, int(math.sqrt(population)) + 1):
        if population % candidate == 0:
            best_rows = candidate
    rows = best_rows
    columns = population // rows
    if rows < 2 or columns < 2:
        raise ValueError("grid topology needs a population with a non-trivial factorization")
    return rows, columns


def validate_study(study: dict[str, Any]) -> None:
    required = ("benchmarks", "dimensions", "population_size", "iterations", "seeds", "methods")
    missing = [key for key in required if key not in study]
    if missing:
        raise ValueError(f"Missing study configuration keys: {missing}")
    if not isinstance(study["benchmarks"], list) or not study["benchmarks"] or any(
        not isinstance(benchmark, str) or benchmark not in SUPPORTED_BENCHMARKS
        for benchmark in study["benchmarks"]
    ):
        raise ValueError(f"benchmarks must be selected from {sorted(SUPPORTED_BENCHMARKS)}")
    if len(set(study["benchmarks"])) != len(study["benchmarks"]):
        raise ValueError("benchmarks must not contain duplicates")
    if not isinstance(study["dimensions"], list) or not study["dimensions"] or any(
        isinstance(d, bool) or not isinstance(d, int) or d < 1 for d in study["dimensions"]
    ):
        raise ValueError("dimensions must be positive integers")
    if len(set(study["dimensions"])) != len(study["dimensions"]):
        raise ValueError("dimensions must not contain duplicates")
    if "rosenbrock" in study["benchmarks"] and any(d < 2 for d in study["dimensions"]):
        raise ValueError("Rosenbrock requires dimension >= 2")
    population_size = study["population_size"]
    iterations = study["iterations"]
    if isinstance(population_size, bool) or not isinstance(population_size, int) or population_size < 2:
        raise ValueError("population_size must be an integer >=2")
    if isinstance(iterations, bool) or not isinstance(iterations, int) or iterations < 0:
        raise ValueError("iterations must be a non-negative integer")
    if not isinstance(study["seeds"], list) or not study["seeds"] or any(
        isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 or seed > (1 << 64) - 1
        for seed in study["seeds"]
    ) or len(set(study["seeds"])) != len(study["seeds"]):
        raise ValueError("seeds must be a non-empty list of unique non-negative unsigned 64-bit integers")
    if not isinstance(study["methods"], list) or not study["methods"]:
        raise ValueError("methods must be a non-empty list of method objects")

    names: set[str] = set()
    serial_topologies: set[str] = set()
    serial_topology_counts: dict[str, int] = defaultdict(int)
    compared_topologies: set[str] = set()
    for method in study["methods"]:
        if not isinstance(method, dict):
            raise ValueError("Every method configuration must be a JSON object")
        name = method.get("name")
        architecture = method.get("architecture")
        topology = method.get("topology", "global")
        if not isinstance(name, str) or not name.strip() or name in names:
            raise ValueError("Every method needs a unique non-empty name")
        names.add(name)
        if architecture not in SUPPORTED_ARCHITECTURES:
            raise ValueError(f"Unsupported architecture {architecture!r}; choose from {sorted(SUPPORTED_ARCHITECTURES)}")
        if topology not in (
            "global", "ring", "grid_von_neumann", "wheel", "random",
            "small_world", "scale_free",
        ):
            raise ValueError(f"Unsupported particle topology {topology!r}")
        if architecture == "serial":
            serial_topologies.add(topology)
            serial_topology_counts[topology] += 1
        else:
            compared_topologies.add(topology)
        if topology == "grid_von_neumann":
            grid_shape(population_size, method.get("grid_shape", study.get("grid_shape")))
        if architecture in {"master_slave", "particle_parallel", "asynchronous"}:
            workers = method.get("workers")
            if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
                raise ValueError(f"Method {name}: workers must be a positive integer")
        if architecture == "islands":
            island_count = method.get("islands")
            workers = method.get("workers")
            if isinstance(island_count, bool) or not isinstance(island_count, int) or island_count < 2:
                raise ValueError(f"Method {name}: islands must be an integer >=2")
            if isinstance(workers, bool) or not isinstance(workers, int) or not 1 <= workers <= island_count:
                raise ValueError(f"Method {name}: workers must lie in [1,islands]")
            if topology == "grid_von_neumann":
                raise ValueError("The current island backend does not support a grid inner topology")
            if int(study["population_size"]) % island_count != 0:
                raise ValueError(f"Method {name}: population_size must divide evenly across islands")
            local_population = int(study["population_size"]) // island_count
            if local_population < 2:
                raise ValueError(f"Method {name}: each island must contain at least two particles")
            degree = int(method.get("topology_degree", study.get("topology_degree", 4)))
            if topology == "wheel" and local_population < 2:
                raise ValueError(f"Method {name}: wheel inner topology requires at least two particles per island")
            if topology == "random" and not 1 <= degree < local_population:
                raise ValueError(f"Method {name}: random inner topology requires 1 <= degree < island population")
            if topology == "small_world" and not 2 <= degree < local_population:
                raise ValueError(f"Method {name}: small-world inner topology degree must be smaller than island population")
            if topology == "scale_free" and local_population < degree // 2 + 1:
                raise ValueError(f"Method {name}: scale-free inner topology has too few particles per island")
            if method.get("migration_topology", "ring") not in {"ring", "fully_connected"}:
                raise ValueError(f"Method {name}: migration_topology must be ring or fully_connected")
            migration_interval = method.get("migration_interval", 0)
            if (
                isinstance(migration_interval, bool)
                or not isinstance(migration_interval, int)
                or migration_interval < 0
            ):
                raise ValueError(f"Method {name}: migration_interval must be a non-negative integer")
    check_equivalence = study.get("check_equivalence", True)
    if not isinstance(check_equivalence, bool):
        raise ValueError("check_equivalence must be a boolean")
    if check_equivalence:
        duplicate_baselines = sorted(topology for topology, count in serial_topology_counts.items() if count != 1)
        if duplicate_baselines:
            raise ValueError(
                "Include exactly one serial reference per particle-social topology; "
                f"duplicate baselines for {duplicate_baselines}"
            )
        missing_baselines = sorted(compared_topologies - serial_topologies)
        if missing_baselines:
            raise ValueError(
                "The paired architecture study needs one serial reference for every compared "
                f"particle-social topology; missing baselines for {missing_baselines}"
            )
    work_iterations_value = study.get("work_iterations", 0)
    if isinstance(work_iterations_value, bool) or not isinstance(work_iterations_value, int):
        raise ValueError("work_iterations must be an integer")
    work_iterations = work_iterations_value
    if work_iterations < 0 or work_iterations > 100_000_000:
        raise ValueError("work_iterations must lie in [0,100000000]")
    if not set(study["benchmarks"]) & {"sphere_work", "inverse_diffusion"} and work_iterations != 0:
        raise ValueError("work_iterations is only valid with a controlled-cost objective")
    if "inverse_diffusion" in study["benchmarks"] and (
        work_iterations < 16 or work_iterations > 100_000 or work_iterations % 4 != 0
    ):
        raise ValueError("inverse_diffusion requires a multiple-of-four work_iterations in [16,100000]")

    # Validate every objective/dimension/method combination during dry-runs too,
    # so malformed bounds, grid layouts, or CPSO coefficients fail before a
    # campaign is launched.
    for benchmark in study["benchmarks"]:
        for dimension in study["dimensions"]:
            for method in study["methods"]:
                make_cpso_config(study, benchmark, dimension, study["seeds"][0], method).validate()


def make_cpso_config(study: dict[str, Any], benchmark: str, dimension: int, seed: int,
                     method: dict[str, Any]) -> CpsoConfig:
    bounds_map = study.get("bounds", {})
    default_lower, default_upper = DEFAULT_BOUNDS[benchmark]
    selected = bounds_map.get(benchmark, [default_lower, default_upper])
    if len(selected) != 2 or not float(selected[0]) < float(selected[1]):
        raise ValueError(f"Invalid bounds for {benchmark}: {selected}")
    lower = (float(selected[0]),) * dimension
    upper = (float(selected[1]),) * dimension
    topology = method.get("topology", "global")
    rows = columns = None
    if topology == "grid_von_neumann":
        rows, columns = grid_shape(int(study["population_size"]), method.get("grid_shape", study.get("grid_shape")))
    return CpsoConfig(
        lower_bounds=lower,
        upper_bounds=upper,
        population_size=int(study["population_size"]),
        iterations=int(study["iterations"]),
        seed=int(seed),
        mu_max=float(study.get("mu_max", 0.9)),
        mu_min=float(study.get("mu_min", 0.4)),
        cognitive=float(study.get("cognitive", 2.05)),
        social=float(study.get("social", 2.05)),
        dt=float(study.get("dt", 0.7)),
        initial_velocity_fraction=float(study.get("initial_velocity_fraction", 0.1)),
        boundary_restitution=float(study.get("boundary_restitution", 0.5)),
        topology=topology,
        grid_rows=rows,
        grid_columns=columns,
        topology_degree=int(method.get("topology_degree", study.get("topology_degree", 4))),
        topology_rewiring_probability=float(
            method.get("topology_rewiring_probability", study.get("topology_rewiring_probability", 0.15))
        ),
    )


def invoke_method(config: CpsoConfig, benchmark: str, method: dict[str, Any], work_iterations: int):
    architecture = method["architecture"]
    effective_work_iterations = work_iterations if benchmark in {"sphere_work", "inverse_diffusion"} else 0
    if architecture == "serial":
        return run_serial(config, benchmark, work_iterations=effective_work_iterations)
    if architecture == "master_slave":
        return run_master_slave(
            config, benchmark, workers=int(method["workers"]), work_iterations=effective_work_iterations
        )
    if architecture == "particle_parallel":
        return run_particle_parallel(
            config, benchmark, workers=int(method["workers"]), work_iterations=effective_work_iterations
        )
    if architecture == "asynchronous":
        return run_asynchronous(
            config, benchmark, workers=int(method["workers"]), work_iterations=effective_work_iterations
        )
    if architecture == "islands":
        return run_island_model(
            config,
            benchmark,
            islands=int(method["islands"]),
            workers=int(method["workers"]),
            migration_interval=int(method.get("migration_interval", 0)),
            migration_topology=method.get("migration_topology", "ring"),
            work_iterations=effective_work_iterations,
        )
    raise ValueError(f"Unknown architecture {architecture!r}")


def method_record(method: dict[str, Any]) -> dict[str, Any]:
    return {
        "method": method["name"],
        "architecture": method["architecture"],
        "topology": method.get("topology", "global"),
        "workers": int(method.get("workers", 1)),
        "island_count": int(method.get("islands", 1)),
        "migration_interval": int(method.get("migration_interval", 0)),
        "migration_topology": method.get("migration_topology", "none"),
        "topology_degree": int(method.get("topology_degree", 4)),
        "topology_rewiring_probability": float(method.get("topology_rewiring_probability", 0.15)),
    }


def run_key(study_hash: str, benchmark: str, dimension: int, seed: int,
            method: dict[str, Any]) -> str:
    return stable_hash({
        "study_hash": study_hash,
        "benchmark": benchmark,
        "dimension": dimension,
        "seed": seed,
        "method": method,
    })


def validate_result(result: Any, population: int, iterations: int, method: dict[str, Any]) -> None:
    expected_fe = population * (iterations + 1)
    if result.function_evaluations != expected_fe:
        raise RuntimeError(
            f"{method['name']}: FE mismatch: got {result.function_evaluations}, expected {expected_fe}"
        )
    if not result.best_history or len(result.best_history) != len(result.history_evaluations):
        raise RuntimeError(f"{method['name']}: invalid best-history/FE-axis lengths")
    if result.history_evaluations[0] != population or result.history_evaluations[-1] != expected_fe:
        raise RuntimeError(f"{method['name']}: history FE axis has invalid endpoints")
    if any(b < a for a, b in zip(result.history_evaluations, result.history_evaluations[1:])):
        raise RuntimeError(f"{method['name']}: history FE axis is not monotonic")
    if any(b > a for a, b in zip(result.best_history, result.best_history[1:])):
        raise RuntimeError(f"{method['name']}: best-so-far history is not non-increasing")


def evaluations_to_target(
    history_evaluations: tuple[int, ...] | list[int],
    best_history: tuple[float, ...] | list[float],
    target: float,
) -> int | None:
    if len(history_evaluations) != len(best_history):
        raise ValueError("history_evaluations and best_history must have equal lengths")
    return next(
        (int(fe) for fe, value in zip(history_evaluations, best_history)
         if float(value) <= float(target)),
        None,
    )


def result_to_record(study_hash: str, benchmark: str, dimension: int, seed: int,
                     method: dict[str, Any], result: Any, success_target: float) -> dict[str, Any]:
    target_fe = evaluations_to_target(result.history_evaluations, result.best_history, success_target)
    record = {
        "run_key": run_key(study_hash, benchmark, dimension, seed, method),
        "benchmark": benchmark,
        "dimension": dimension,
        "population_size": result.function_evaluations // (result.iterations_completed + 1)
            if result.iterations_completed + 1 else 0,
        "iterations": result.iterations_completed,
        "seed": seed,
        **method_record(method),
        "best_value": result.best_value,
        "best_position": list(result.best_position),
        "best_history": list(result.best_history),
        "history_evaluations": list(result.history_evaluations),
        "function_evaluations": result.function_evaluations,
        "success_target": success_target,
        "success": bool(result.best_value <= success_target),
        "evaluations_to_target": target_fe,
        "elapsed_seconds": result.elapsed_seconds,
        "evaluation_wall_seconds": result.evaluation_wall_seconds,
        "propagation_wall_seconds": result.propagation_seconds,
        "topology_setup_seconds": result.topology_setup_seconds,
        "social_selection_seconds": result.social_selection_seconds,
        "initialization_seconds": result.initialization_seconds,
        "propagation_worker_seconds": result.propagation_worker_seconds,
        "evaluation_worker_seconds": result.evaluation_worker_seconds,
        "migration_seconds": result.migration_seconds,
        "async_update_wall_seconds": result.async_update_wall_seconds,
        "migrations_performed": result.migrations_performed,
        "reported_worker_count": result.worker_count,
        "reported_island_count": result.island_count,
    }
    return record


def execute_study(study: dict[str, Any], output: Path, no_plots: bool = False,
                  dry_run: bool = False) -> Path:
    validate_study(study)
    study_hash = stable_hash(study)
    code_hash = source_fingerprint()

    tasks: list[tuple[str, int, int, dict[str, Any]]] = []
    for benchmark in study["benchmarks"]:
        for dimension in study["dimensions"]:
            for seed in study["seeds"]:
                for method in study["methods"]:
                    tasks.append((benchmark, int(dimension), int(seed), method))

    run_file = output / "runs.jsonl"
    if dry_run:
        # A dry-run is read-only and independent of the compiled native library.
        completed_records, checkpoint_state = read_jsonl_checkpoint(run_file, repair=False)
        completed_keys = {item["run_key"] for item in completed_records}
        suffix_note = (
            f"; {checkpoint_state['discarded_bytes']} bytes of an incomplete final record "
            "will be discarded on resume"
            if checkpoint_state["discarded_partial_record"] else ""
        )
        print(
            f"Study contains {len(tasks)} runs; {len(completed_keys)} already completed{suffix_note}."
        )
        return output

    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "run_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("study_hash") != study_hash:
            raise ValueError("Output directory already contains a different study manifest; use a new directory")
        if manifest.get("source_sha256") != code_hash:
            raise ValueError("Source code differs from the original study; use a new output directory")
    else:
        manifest = {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "study_hash": study_hash,
            "source_sha256": code_hash,
            "study": study,
            "native_version": native_version(),
            "native_build_info": native_build_info(),
            "platform": platform.platform(),
            "python": sys.version,
            "logical_cpu_count": os.cpu_count(),
            "environment": collect_environment_metadata(
                native_version=native_version(),
                native_build_info=native_build_info(),
            ),
            "driver": str(Path(__file__).relative_to(ROOT)),
        }
        atomic_write_json(manifest_path, manifest)

    checkpoint_records, checkpoint_state = read_jsonl_checkpoint(run_file, repair=True)
    completed: dict[str, dict[str, Any]] = {item["run_key"]: item for item in checkpoint_records}
    if checkpoint_state["discarded_partial_record"]:
        print(
            "Recovered checkpoint: discarded an incomplete final JSONL record "
            f"({checkpoint_state['discarded_bytes']} bytes)."
        )
    elif checkpoint_state["added_final_newline"]:
        print("Normalized the final checkpoint record with a terminating newline.")

    work_iterations = int(study.get("work_iterations", 0))
    target_map = study.get("success_targets", {})
    target_default = float(study.get("success_target_default", 1.0e-8))
    check_equivalence = bool(study.get("check_equivalence", True))

    # Interleave architecture variants within each (benchmark,D,seed) case to
    # avoid always measuring one method first under warmer/cooler conditions.
    grouped: dict[tuple[str, int, int], list[tuple[str, int, int, dict[str, Any]]]] = defaultdict(list)
    for task in tasks:
        grouped[(task[0], task[1], task[2])].append(task)
    ordered_tasks: list[tuple[str, int, int, dict[str, Any]]] = []
    for case, case_tasks in sorted(grouped.items()):
        benchmark, dimension, seed = case
        shuffle_seed = (seed * 1_000_003 + dimension * 97 + sum(ord(c) for c in benchmark)) & 0xFFFFFFFF
        random.Random(shuffle_seed).shuffle(case_tasks)
        ordered_tasks.extend(case_tasks)

    print(f"Architecture study: {len(tasks)} requested runs; {len(completed)} already complete.")
    with run_file.open("a", encoding="utf-8") as stream:
        for index, (benchmark, dimension, seed, method) in enumerate(ordered_tasks, start=1):
            key = run_key(study_hash, benchmark, dimension, seed, method)
            if key in completed:
                continue
            config = make_cpso_config(study, benchmark, dimension, seed, method)
            started = time.perf_counter()
            result = invoke_method(config, benchmark, method, work_iterations)
            validate_result(result, config.population_size, config.iterations, method)
            target = float(target_map.get(benchmark, target_default))
            recorded_method = {
                **method,
                "topology_degree": config.topology_degree,
                "topology_rewiring_probability": config.topology_rewiring_probability,
            }
            record = result_to_record(study_hash, benchmark, dimension, seed, recorded_method, result, target)
            record["run_wall_seconds_driver"] = time.perf_counter() - started
            stream.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
            completed[key] = record

            print(
                f"[{index}/{len(tasks)}] {benchmark} D={dimension} seed={seed} "
                f"{method['name']}: f={result.best_value:.6g}, "
                f"FE={result.function_evaluations}, wall={result.elapsed_seconds:.4f}s"
            )

    records = list(completed.values())
    if check_equivalence:
        verify_exact_equivalence(records)
    materialize_tables(records, output)
    analyze_records(
        records, output, no_plots=no_plots,
        equivalence_verified=check_equivalence,
    )
    return output


def verify_exact_equivalence(records: list[dict[str, Any]]) -> None:
    seen: set[tuple[str, int, int, str]] = set()
    for record in records:
        key = (record["benchmark"], int(record["dimension"]), int(record["seed"]), record["method"])
        if key in seen:
            raise ValueError(f"Duplicate completed run in JSONL: {key}")
        seen.add(key)

    for record in records:
        if record["architecture"] not in {"master_slave", "particle_parallel"}:
            continue
        baseline_candidates = [
            candidate for candidate in records
            if candidate["benchmark"] == record["benchmark"]
            and candidate["dimension"] == record["dimension"]
            and candidate["seed"] == record["seed"]
            and candidate["architecture"] == "serial"
            and candidate["topology"] == record["topology"]
        ]
        if len(baseline_candidates) != 1:
            raise ValueError(
                f"Expected exactly one serial baseline for {record['benchmark']} D={record['dimension']} "
                f"seed={record['seed']} topology={record['topology']}; found {len(baseline_candidates)}"
            )
        baseline = baseline_candidates[0]
        for field in ("best_value", "best_position", "best_history", "history_evaluations", "function_evaluations"):
            if record[field] != baseline[field]:
                raise RuntimeError(
                    f"Exact synchronous equivalence failed for {record['method']} vs serial "
                    f"at {record['benchmark']} D={record['dimension']} seed={record['seed']} field={field}"
                )


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def materialize_tables(records: list[dict[str, Any]], output: Path) -> None:
    run_fields = [
        "run_key", "benchmark", "dimension", "population_size", "iterations", "seed",
        "method", "architecture", "topology", "workers", "island_count", "migration_interval",
        "migration_topology", "topology_degree", "topology_rewiring_probability",
        "best_value", "success_target", "success", "evaluations_to_target",
        "function_evaluations", "elapsed_seconds", "evaluation_wall_seconds", "propagation_wall_seconds",
        "topology_setup_seconds", "social_selection_seconds", "initialization_seconds",
        "propagation_worker_seconds", "evaluation_worker_seconds", "migration_seconds",
        "async_update_wall_seconds", "migrations_performed", "reported_worker_count",
        "reported_island_count", "run_wall_seconds_driver",
    ]
    write_csv(output / "runs.csv", run_fields, records)

    convergence_rows: list[dict[str, Any]] = []
    for record in records:
        for fe, best in zip(record["history_evaluations"], record["best_history"]):
            convergence_rows.append({
                "run_key": record["run_key"],
                "benchmark": record["benchmark"],
                "dimension": record["dimension"],
                "seed": record["seed"],
                "method": record["method"],
                "architecture": record["architecture"],
                "topology": record["topology"],
                "evaluations_completed": fe,
                "best_value": best,
            })
    write_csv(
        output / "convergence.csv",
        ["run_key", "benchmark", "dimension", "seed", "method", "architecture", "topology",
         "evaluations_completed", "best_value"],
        convergence_rows,
    )


def analyze_records(
    records: list[dict[str, Any]],
    output: Path,
    no_plots: bool = False,
    *,
    equivalence_verified: bool = True,
) -> None:
    groups: dict[tuple[str, int, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[(record["benchmark"], int(record["dimension"]), record["method"])].append(record)

    summary_rows: list[dict[str, Any]] = []
    for (benchmark, dimension, method), rows in sorted(groups.items()):
        values = [float(row["best_value"]) for row in rows]
        elapsed = [float(row["elapsed_seconds"]) for row in rows]
        value_median = statistics.median(values)
        summary_rows.append({
            "benchmark": benchmark,
            "dimension": dimension,
            "method": method,
            "architecture": rows[0]["architecture"],
            "topology": rows[0]["topology"],
            "n_seeds": len(values),
            "best_median": value_median,
            "best_q25": quantile(values, 0.25),
            "best_q75": quantile(values, 0.75),
            "best_mad": statistics.median(abs(value - value_median) for value in values),
            "success_rate": sum(bool(row["success"]) for row in rows) / len(rows),
            "ert_evaluations": (
                sum(int(row["evaluations_to_target"])
                    if row.get("evaluations_to_target") is not None
                    else int(row["function_evaluations"]) for row in rows)
                / sum(bool(row["success"]) for row in rows)
                if any(bool(row["success"]) for row in rows) else math.inf
            ),
            "median_elapsed_seconds": statistics.median(elapsed),
            "median_evaluation_wall_seconds": statistics.median(float(row["evaluation_wall_seconds"]) for row in rows),
            "median_propagation_wall_seconds": statistics.median(float(row["propagation_wall_seconds"]) for row in rows),
            "median_async_update_wall_seconds": statistics.median(float(row["async_update_wall_seconds"]) for row in rows),
            "median_migrations_performed": statistics.median(float(row["migrations_performed"]) for row in rows),
        })
    write_csv(output / "summary.csv", list(summary_rows[0].keys()), summary_rows)

    # Pair all methods with the serial baseline of the same topology and seed.
    comparison_rows: list[dict[str, Any]] = []
    for record in records:
        if record["architecture"] == "serial":
            continue
        baseline_candidates = [
            row for row in records
            if row["benchmark"] == record["benchmark"]
            and row["dimension"] == record["dimension"]
            and row["seed"] == record["seed"]
            and row["architecture"] == "serial"
            and row["topology"] == record["topology"]
        ]
        if len(baseline_candidates) != 1:
            continue
        baseline = baseline_candidates[0]
        delta = math.log10(max(float(record["best_value"]), LOG_FLOOR)) - math.log10(
            max(float(baseline["best_value"]), LOG_FLOOR)
        )
        comparison_rows.append({
            "benchmark": record["benchmark"],
            "dimension": record["dimension"],
            "seed": record["seed"],
            "method": record["method"],
            "architecture": record["architecture"],
            "topology": record["topology"],
            "delta_log10_best_vs_serial": delta,
            "serial_elapsed_seconds": baseline["elapsed_seconds"],
            "method_elapsed_seconds": record["elapsed_seconds"],
            "paired_speedup": (
                float(baseline["elapsed_seconds"]) / float(record["elapsed_seconds"])
                if equivalence_verified
                and record["architecture"] in {"master_slave", "particle_parallel"}
                and float(baseline["elapsed_seconds"]) > 0.0
                and float(record["elapsed_seconds"]) > 0.0 else ""
            ),
            "runtime_ratio_interpretation": (
                "same synchronous algorithm: exact trajectory verified; valid paired speedup"
                if equivalence_verified
                and record["architecture"] in {"master_slave", "particle_parallel"}
                else "same synchronous backend but exact trajectory not verified: speedup withheld"
                if record["architecture"] in {"master_slave", "particle_parallel"}
                else "different search dynamics: do not label as speedup"
            ),
        })
    comparison_fields = list(comparison_rows[0].keys()) if comparison_rows else [
        "benchmark", "dimension", "seed", "method", "architecture", "topology",
        "delta_log10_best_vs_serial", "serial_elapsed_seconds", "method_elapsed_seconds",
        "paired_speedup", "runtime_ratio_interpretation",
    ]
    write_csv(output / "paired_comparisons.csv", comparison_fields, comparison_rows)

    paired_groups: dict[tuple[str, int, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in comparison_rows:
        paired_groups[(row["benchmark"], int(row["dimension"]), row["method"], row["architecture"])].append(row)
    paired_summary_rows: list[dict[str, Any]] = []
    for (benchmark, dimension, method, architecture), rows in sorted(paired_groups.items()):
        deltas = [float(row["delta_log10_best_vs_serial"]) for row in rows]
        ci_low, ci_high = bootstrap_median_interval(
            deltas,
            seed=20261104 + dimension + sum(ord(c) for c in benchmark + method),
        )
        speedups = [float(row["paired_speedup"]) for row in rows if row["paired_speedup"] != ""]
        paired_summary_rows.append({
            "benchmark": benchmark,
            "dimension": dimension,
            "method": method,
            "architecture": architecture,
            "n_paired_seeds": len(deltas),
            "median_delta_log10_best": statistics.median(deltas),
            "bootstrap_ci95_low": ci_low,
            "bootstrap_ci95_high": ci_high,
            "median_speedup_if_algorithm_equivalent": statistics.median(speedups) if speedups else "",
            "speedup_n": len(speedups),
            "interpretation": "negative objective delta favors method; intervals descriptive, not p-values",
        })
    paired_summary_fields = list(paired_summary_rows[0].keys()) if paired_summary_rows else [
        "benchmark", "dimension", "method", "architecture", "n_paired_seeds",
        "median_delta_log10_best", "bootstrap_ci95_low", "bootstrap_ci95_high",
        "median_speedup_if_algorithm_equivalent", "speedup_n", "interpretation",
    ]
    write_csv(output / "paired_summary.csv", paired_summary_fields, paired_summary_rows)

    if not no_plots:
        make_figures(records, output)

    report = [
        "# CPSO parallel architecture study",
        "",
        f"Completed runs: {len(records)}",
        f"Benchmarks: {', '.join(sorted({r['benchmark'] for r in records}))}",
        f"Dimensions: {', '.join(str(v) for v in sorted({int(r['dimension']) for r in records}))}",
        f"Methods: {', '.join(sorted({r['method'] for r in records}))}",
        "",
        "## Interpretation rules",
        "",
        "- `master_slave` and `particle_parallel` are synchronous variants; their results must be exactly equal to the serial reference for the same topology, seed, and FE budget. Only their paired wall-time ratio is labelled speedup.",
        "- `asynchronous` uses event-driven updates and publishes personal-best information after each completed evaluation. Its history is sampled after every committed event; completion order is schedule-dependent when multiple workers are used.",
        "- `islands` divides the fixed total population across independent sub-swarms. Migration is synchronized at generation barriers and does not consume additional function evaluations. Island variants change the information flow and therefore the search dynamics.",
        "- Quality comparisons are paired by benchmark, dimension, topology, and seed. Bootstrap intervals are descriptive uncertainty intervals, not significance tests or p-values.",
        "- `evaluations_to_target` is the first FE count at which the configured target is attained; missing values are censored at the run FE budget. `ert_evaluations` is total successful/failed FE cost divided by the number of successes, and is infinite when no run reaches the target.",
        "- `evaluation_worker_seconds` and `propagation_worker_seconds` are sums over task durations and can exceed wall-clock time under parallel execution.",
        "- `evaluation_wall_seconds` is the wall time of synchronous evaluation batches; for asynchronous runs, objective calls during the event loop are represented by `evaluation_worker_seconds` while `async_update_wall_seconds` covers the complete asynchronous update scheduler.",
        "- `sphere_work` is a synthetic execution-cost instrument, not an optimization benchmark suite.",
        "",
        "Do not interpret runtime differences between asynchronous/island methods and synchronous serial as speedup: these methods have different search dynamics. No COCO/CEC claims are implied by runs on built-in functions.",
    ]
    if not no_plots:
        report.append("Figures are in `figures/` and are exported as PNG, PDF, and SVG.")
    (output / "ARCHITECTURE_STUDY.md").write_text("\n".join(report) + "\n", encoding="utf-8")


def _boxplot_with_labels(ax: Any, values: list[list[float]], labels: list[str], **kwargs: Any) -> Any:
    """Use the boxplot label keyword supported by the installed Matplotlib version."""
    import matplotlib

    # The keyword was renamed in Matplotlib 3.9; it is accepted through
    # **kwargs in some releases, so signature introspection is insufficient.
    version = tuple(int(part.split(".")[0]) for part in matplotlib.__version__.split(".")[:2])
    label_keyword = "tick_labels" if version >= (3, 9) else "labels"
    return ax.boxplot(values, **{label_keyword: labels}, **kwargs)


def make_figures(records: list[dict[str, Any]], output: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("Figure generation requires matplotlib; install the project 'analysis' extra") from exc

    figure_dir = output / "figures"
    figure_dir.mkdir(exist_ok=True)
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        grouped[(row["benchmark"], int(row["dimension"]))].append(row)

    for (benchmark, dimension), rows in sorted(grouped.items()):
        by_method: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            by_method[row["method"]].append(row)
        methods = sorted(by_method)

        # FE-axis median convergence curves. Histories are step-held between
        # observations, putting generation-synchronous and event-driven runs on
        # the same objective-evaluation axis.
        x_values = sorted({
            int(fe) for row in rows for fe in row["history_evaluations"]
        })
        if len(x_values) > 2500:
            stride = math.ceil(len(x_values) / 2500)
            x_values = x_values[::stride]
            if x_values[-1] != max(int(fe) for row in rows for fe in row["history_evaluations"]):
                x_values.append(max(int(fe) for row in rows for fe in row["history_evaluations"]))

        fig, ax = plt.subplots(figsize=(8.0, 5.0), constrained_layout=True)
        for method in methods:
            method_runs = by_method[method]
            run_axes = [list(map(int, row["history_evaluations"])) for row in method_runs]
            run_histories = [list(map(float, row["best_history"])) for row in method_runs]
            medians: list[float] = []
            q25: list[float] = []
            q75: list[float] = []
            for fe in x_values:
                current = []
                for axis, history in zip(run_axes, run_histories):
                    index = bisect.bisect_right(axis, fe) - 1
                    if index >= 0:
                        current.append(max(history[index], LOG_FLOOR))
                if current:
                    medians.append(statistics.median(current))
                    q25.append(quantile(current, 0.25))
                    q75.append(quantile(current, 0.75))
            ax.plot(x_values[:len(medians)], medians, label=method)
            ax.fill_between(x_values[:len(q25)], q25, q75, alpha=0.12)
        ax.set_yscale("log")
        ax.set_xlabel("Objective function evaluations completed")
        ax.set_ylabel("Median best-so-far objective (log scale)")
        ax.set_title(f"CPSO architecture comparison — {benchmark}, D={dimension}")
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(fontsize=8)
        stem = f"{benchmark}_d{dimension}_convergence_fe"
        for ext in ("png", "pdf", "svg"):
            fig.savefig(figure_dir / f"{stem}.{ext}", dpi=220 if ext == "png" else None)
        plt.close(fig)

        # Final quality distributions.
        fig, ax = plt.subplots(figsize=(8.0, 5.0), constrained_layout=True)
        data = [
            [math.log10(max(float(row["best_value"]), LOG_FLOOR)) for row in by_method[method]]
            for method in methods
        ]
        _boxplot_with_labels(ax, data, methods, showmeans=True)
        ax.set_ylabel("log10(max(final objective, 1e-16))")
        ax.set_title(f"Final solution quality — {benchmark}, D={dimension}")
        ax.tick_params(axis="x", labelrotation=35)
        ax.grid(True, axis="y", alpha=0.25)
        stem = f"{benchmark}_d{dimension}_final_quality"
        for ext in ("png", "pdf", "svg"):
            fig.savefig(figure_dir / f"{stem}.{ext}", dpi=220 if ext == "png" else None)
        plt.close(fig)

        # Wall-clock distributions are descriptive only; algorithmically
        # different methods are not assigned a speedup label.
        fig, ax = plt.subplots(figsize=(8.0, 5.0), constrained_layout=True)
        time_data = [[float(row["elapsed_seconds"]) for row in by_method[method]] for method in methods]
        _boxplot_with_labels(ax, time_data, methods, showmeans=True)
        ax.set_ylabel("Run wall time (s; includes worker-pool lifecycle)")
        ax.set_title(f"Runtime distributions — {benchmark}, D={dimension}")
        ax.tick_params(axis="x", labelrotation=35)
        ax.grid(True, axis="y", alpha=0.25)
        stem = f"{benchmark}_d{dimension}_runtime"
        for ext in ("png", "pdf", "svg"):
            fig.savefig(figure_dir / f"{stem}.{ext}", dpi=220 if ext == "png" else None)
        plt.close(fig)

        # Speedup is plotted only for synchronous variants that are required to
        # preserve exactly the serial trajectory.
        serial_rows = [row for row in rows if row["architecture"] == "serial"]
        serial_by_seed_topology = {
            (int(row["seed"]), row["topology"]): float(row["elapsed_seconds"])
            for row in serial_rows
        }
        speedup_by_method: dict[str, list[float]] = defaultdict(list)
        for row in rows:
            if row["architecture"] not in {"master_slave", "particle_parallel"}:
                continue
            baseline = serial_by_seed_topology.get((int(row["seed"]), row["topology"]))
            if baseline is not None and float(row["elapsed_seconds"]) > 0:
                speedup_by_method[row["method"]].append(baseline / float(row["elapsed_seconds"]))
        if speedup_by_method:
            labels = sorted(speedup_by_method)
            fig, ax = plt.subplots(figsize=(7.5, 4.8), constrained_layout=True)
            med = [statistics.median(speedup_by_method[label]) for label in labels]
            low = [quantile(speedup_by_method[label], 0.25) for label in labels]
            high = [quantile(speedup_by_method[label], 0.75) for label in labels]
            positions = list(range(len(labels)))
            ax.errorbar(
                positions, med,
                yerr=[[m - lo for m, lo in zip(med, low)], [hi - m for m, hi in zip(med, high)]],
                fmt="o", capsize=4,
            )
            ax.axhline(1.0, linestyle="--", linewidth=1)
            ax.set_xticks(positions, labels, rotation=30, ha="right")
            ax.set_ylabel("Paired speedup versus serial (same synchronous algorithm)")
            ax.set_title(f"Equivalent synchronous backend speedup — {benchmark}, D={dimension}")
            ax.grid(True, axis="y", alpha=0.25)
            stem = f"{benchmark}_d{dimension}_synchronous_speedup"
            for ext in ("png", "pdf", "svg"):
                fig.savefig(figure_dir / f"{stem}.{ext}", dpi=220 if ext == "png" else None)
            plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run and analyze a paired CPSO architecture study")
    parser.add_argument("config", type=Path, help="JSON study configuration")
    parser.add_argument("--output", type=Path, default=None, help="output directory; defaults under results/raw")
    parser.add_argument("--no-plots", action="store_true", default=True, help="write tables/statistics without Matplotlib figures")
    parser.add_argument("--dry-run", action="store_true", help="validate config and report run count without executing")
    parser.add_argument("--with-inline-plots", dest="no_plots", action="store_false",
                        help="Opt in to inline plots under the selected results/ output")
    args = parser.parse_args()

    study = json.loads(args.config.read_text(encoding="utf-8"))
    if args.output is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output = ROOT / "results" / "raw" / "parallel" / f"architecture_study_{stamp}"
    else:
        output = args.output.expanduser().resolve()
    from cpso_serial.output_policy import results_output
    output = results_output(output)
    result_dir = execute_study(study, output, no_plots=args.no_plots, dry_run=args.dry_run)
    if not args.dry_run:
        print(f"Study outputs written to: {result_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
