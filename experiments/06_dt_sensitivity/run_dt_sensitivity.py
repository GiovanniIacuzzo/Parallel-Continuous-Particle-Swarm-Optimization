#!/usr/bin/env python3
"""Run a paired screening study of the CPSO time interval ``dt``.

The study uses the native serial engine so that only the CPSO dynamics change.
For each benchmark/dimension/seed block, the same seed is reused at every dt
(common random numbers). The design is a screening experiment; a smaller,
independent confirmatory sweep should follow selection of a robust dt region.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import random
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
PYTHON_ROOT = ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

BOUNDS: dict[str, tuple[float, float]] = {
    "sphere": (-100.0, 100.0),
    "rastrigin": (-5.12, 5.12),
    "rosenbrock": (-5.0, 10.0),
    "ackley": (-32.768, 32.768),
    "griewank": (-600.0, 600.0),
    "salomon": (-100.0, 100.0),
}


def load_config(path: Path) -> dict[str, Any]:
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid dt-sensitivity JSON: {exc}") from exc
    validate_config(config)
    return config


def validate_config(config: dict[str, Any]) -> None:
    required = {
        "campaign_id", "schedule_seed", "benchmarks", "dimensions", "population_size",
        "iterations", "seeds", "dt_values", "reference_dt", "topology",
    }
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"dt-sensitivity config missing fields: {missing}")
    if not isinstance(config["campaign_id"], str) or not config["campaign_id"].strip():
        raise ValueError("campaign_id must be a non-empty string")
    if config["topology"] != "global":
        raise ValueError("The dt screening protocol is fixed to global-best topology")

    benchmarks = config["benchmarks"]
    dimensions = config["dimensions"]
    seeds = config["seeds"]
    dt_values = config["dt_values"]
    if not isinstance(benchmarks, list) or len(benchmarks) < 2 or len(set(benchmarks)) != len(benchmarks):
        raise ValueError("benchmarks must contain at least two unique benchmark names")
    unknown = sorted(set(benchmarks) - set(BOUNDS))
    if unknown:
        raise ValueError(f"Unsupported native benchmarks: {unknown}")
    for label, values in (("dimensions", dimensions), ("seeds", seeds)):
        if not isinstance(values, list) or len(values) < 2 or len(set(values)) != len(values):
            raise ValueError(f"{label} must contain at least two unique values")
    if any(isinstance(v, bool) or not isinstance(v, int) or v < 1 for v in dimensions):
        raise ValueError("dimensions must be positive integers")
    if any(isinstance(v, bool) or not isinstance(v, int) or not 0 <= v < 2**64 for v in seeds):
        raise ValueError("seeds must be unsigned 64-bit integers")
    if any(isinstance(v, bool) or not isinstance(v, int) for v in (config["population_size"], config["iterations"])):
        raise ValueError("population_size and iterations must be integers")
    if config["population_size"] < 2 or config["iterations"] < 1:
        raise ValueError("population_size must be >=2 and iterations must be >=1")
    if not isinstance(dt_values, list) or len(dt_values) < 3:
        raise ValueError("dt_values must contain at least three values")
    try:
        dts = [float(value) for value in dt_values]
        reference_dt = float(config["reference_dt"])
    except (TypeError, ValueError) as exc:
        raise ValueError("dt_values and reference_dt must be numeric") from exc
    if any(not math.isfinite(value) or value <= 0 for value in dts):
        raise ValueError("Every dt value must be finite and strictly positive")
    if len(set(dts)) != len(dts):
        raise ValueError("dt_values must be unique")
    if dts != sorted(dts):
        raise ValueError("dt_values must be ordered increasingly for a meaningful log-scale profile")
    if not math.isfinite(reference_dt) or reference_dt not in dts:
        raise ValueError("reference_dt must be finite and included in dt_values")
    for key, default in (("mu_max", 0.9), ("mu_min", 0.4), ("cognitive", 2.05),
                         ("social", 2.05), ("initial_velocity_fraction", 0.1),
                         ("boundary_restitution", 0.5)):
        value = float(config.get(key, default))
        if not math.isfinite(value):
            raise ValueError(f"{key} must be finite")


def build_cpso_config(config: dict[str, Any], benchmark: str, dimension: int,
                      seed: int, dt: float):
    # Import lazily so that config validation and --dry-run do not require a
    # compiled native library.
    from cpso_serial import CpsoConfig

    lower, upper = BOUNDS[benchmark]
    result = CpsoConfig(
        lower_bounds=(lower,) * dimension,
        upper_bounds=(upper,) * dimension,
        population_size=int(config["population_size"]),
        iterations=int(config["iterations"]),
        seed=int(seed),
        mu_max=float(config.get("mu_max", 0.9)),
        mu_min=float(config.get("mu_min", 0.4)),
        cognitive=float(config.get("cognitive", 2.05)),
        social=float(config.get("social", 2.05)),
        dt=float(dt),
        initial_velocity_fraction=float(config.get("initial_velocity_fraction", 0.1)),
        boundary_restitution=float(config.get("boundary_restitution", 0.5)),
        topology="global",
    )
    result.validate()
    return result


def planned_runs(config: dict[str, Any]) -> list[tuple[str, int, int, float]]:
    jobs = list(itertools.product(
        config["benchmarks"], config["dimensions"], config["seeds"],
        [float(value) for value in config["dt_values"]],
    ))
    random.Random(int(config["schedule_seed"])).shuffle(jobs)
    return [(str(b), int(d), int(s), float(dt)) for b, d, s, dt in jobs]


def execute(config_path: Path, output: Path, *, dry_run: bool = False) -> int:
    config = load_config(config_path)
    jobs = planned_runs(config)
    population = int(config["population_size"])
    evaluations_per_run = population * (int(config["iterations"]) + 1)
    plan = {
        "campaign_id": config["campaign_id"],
        "title": config.get("title", "Paired CPSO dt sensitivity screening"),
        "benchmarks": config["benchmarks"],
        "dimensions": config["dimensions"],
        "seeds": config["seeds"],
        "dt_values": [float(v) for v in config["dt_values"]],
        "reference_dt": float(config["reference_dt"]),
        "population_size": population,
        "iterations": int(config["iterations"]),
        "evaluations_per_run": evaluations_per_run,
        "expected_run_count": len(jobs),
        "expected_total_function_evaluations": len(jobs) * evaluations_per_run,
        "common_random_numbers": "same seed within benchmark x dimension across all dt values",
        "execution": "native serial CPSO; isolates dt from executor concurrency",
        "output_directory": str(output),
    }
    if dry_run:
        print(json.dumps(plan, indent=2))
        return 0

    output.mkdir(parents=True, exist_ok=True)
    run_path = output / "runs.jsonl"
    if run_path.exists() and run_path.stat().st_size:
        raise FileExistsError(
            f"Refusing to mix a new dt campaign with existing records: {run_path}. "
            "Choose a new output directory or archive the previous campaign."
        )
    from cpso_serial import native_build_info, native_version, run_serial

    manifest = {
        "schema_version": 1,
        "plan": plan,
        "study_config": config,
        "config_source": str(config_path.resolve()),
        "native_version": native_version(),
        "native_build_info": native_build_info(),
    }
    (output / "study_config.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )

    with run_path.open("w", encoding="utf-8") as stream:
        for index, (benchmark, dimension, seed, dt) in enumerate(jobs, start=1):
            cpso_config = build_cpso_config(config, benchmark, dimension, seed, dt)
            result = run_serial(cpso_config, benchmark)
            expected_fe = evaluations_per_run
            if int(result.function_evaluations) != expected_fe:
                raise RuntimeError(
                    f"FE counter mismatch for {benchmark}, D={dimension}, seed={seed}, dt={dt:g}: "
                    f"{result.function_evaluations} != {expected_fe}"
                )
            record = {
                "schema_version": 1,
                "run_index": index,
                "benchmark": benchmark,
                "dimension": dimension,
                "seed": seed,
                "dt": dt,
                "reference_dt": float(config["reference_dt"]),
                "topology": "global",
                "population_size": population,
                "iterations": int(config["iterations"]),
                "function_evaluations": int(result.function_evaluations),
                "best_value": float(result.best_value),
                "best_position": [float(v) for v in result.best_position],
                "history_evaluations": [int(v) for v in result.history_evaluations],
                "best_history": [float(v) for v in result.best_history],
                "elapsed_seconds": float(result.elapsed_seconds),
            }
            if not math.isfinite(record["best_value"]) or not math.isfinite(record["elapsed_seconds"]):
                raise RuntimeError(f"Non-finite run output for {benchmark}, D={dimension}, seed={seed}, dt={dt:g}")
            stream.write(json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n")
            stream.flush()
            print(
                f"[{index:>4}/{len(jobs)}] {benchmark:10s} D={dimension:<4d} "
                f"seed={seed:<12d} dt={dt:<7g} f={record['best_value']:.6g}"
            )
    print(f"Completed {len(jobs)} runs; raw records: {run_path}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path,
        default=ROOT / "configs/campaigns/cpso_dt_sensitivity_screening.json",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    output = args.output or ROOT / "results/raw" / "cpso_dt_sensitivity_screening_v1"
    return execute(args.config, output, dry_run=args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
