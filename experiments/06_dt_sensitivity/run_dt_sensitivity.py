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
import os
import random
import hashlib
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
PYTHON_ROOT = ROOT / "python"
ARCHITECTURE_EXPERIMENT_ROOT = ROOT / "experiments" / "04_architectures"
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))
if str(ARCHITECTURE_EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(ARCHITECTURE_EXPERIMENT_ROOT))

from environment_metadata import collect_environment_metadata  # noqa: E402

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


def config_fingerprint(config: dict[str, Any]) -> str:
    encoded = json.dumps(config, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def run_key(benchmark: str, dimension: int, seed: int, dt: float) -> tuple[str, int, int, float]:
    return benchmark, int(dimension), int(seed), float(dt)


def serialized_run_key(key: tuple[str, int, int, float]) -> str:
    benchmark, dimension, seed, dt = key
    return f"{benchmark}|D{dimension}|S{seed}|dt{dt:.17g}"


def load_completed_records(
    run_path: Path,
    *,
    expected_keys: set[tuple[str, int, int, float]],
    expected_fe: int,
    population: int,
    iterations: int,
    repair_truncated_tail: bool,
) -> dict[tuple[str, int, int, float], dict[str, Any]]:
    """Load validated JSONL records, optionally dropping only a torn final line."""
    if not run_path.exists():
        return {}
    raw = run_path.read_bytes()
    lines = raw.splitlines(keepends=True)
    completed: dict[tuple[str, int, int, float], dict[str, Any]] = {}
    valid_bytes = 0
    for index, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()
        if not line:
            valid_bytes += len(raw_line)
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            is_torn_tail = index == len(lines) and not raw_line.endswith((b"\n", b"\r"))
            if repair_truncated_tail and is_torn_tail:
                with run_path.open("r+b") as stream:
                    stream.truncate(valid_bytes)
                    stream.flush()
                    os.fsync(stream.fileno())
                break
            raise ValueError(f"Invalid JSONL record at {run_path}:{index}; refusing unsafe resume") from exc
        if not isinstance(record, dict):
            raise ValueError(f"Expected a JSON object at {run_path}:{index}")
        try:
            key = run_key(record["benchmark"], record["dimension"], record["seed"], record["dt"])
            function_evaluations = int(record["function_evaluations"])
            best_value = float(record["best_value"])
            elapsed = float(record["elapsed_seconds"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Malformed dt-sensitivity record at {run_path}:{index}") from exc
        if key not in expected_keys:
            raise ValueError(f"Unexpected run key in existing JSONL: {key}")
        if key in completed:
            raise ValueError(f"Duplicate run key in existing JSONL: {key}")
        if record.get("schema_version") != 2 or record.get("run_key") != serialized_run_key(key):
            raise ValueError(f"Record schema/run_key mismatch at {run_path}:{index}")
        if function_evaluations != expected_fe:
            raise ValueError(f"FE mismatch in existing record {key}: {function_evaluations} != {expected_fe}")
        if not math.isfinite(best_value) or not math.isfinite(elapsed) or elapsed <= 0.0:
            raise ValueError(f"Non-finite/invalid result in existing record {key}")
        if int(record.get("population_size", -1)) != population or int(record.get("iterations", -1)) != iterations:
            raise ValueError(f"Population/iteration mismatch in existing record {key}")
        if record.get("topology") != "global":
            raise ValueError(f"Unexpected topology in existing record {key}")
        history = record.get("best_history")
        axis = record.get("history_evaluations")
        position = record.get("best_position")
        if not isinstance(history, list) or not isinstance(axis, list):
            raise ValueError(f"Missing best-so-far history in existing record {key}")
        if len(history) != iterations + 1 or len(axis) != iterations + 1:
            raise ValueError(f"History length mismatch in existing record {key}")
        expected_axis = [population * (step + 1) for step in range(iterations + 1)]
        if [int(value) for value in axis] != expected_axis:
            raise ValueError(f"History FE axis mismatch in existing record {key}")
        if any(not math.isfinite(float(value)) for value in history):
            raise ValueError(f"Non-finite history in existing record {key}")
        if any(float(next_value) > float(value) + 1.0e-12
               for value, next_value in zip(history, history[1:])):
            raise ValueError(f"Best-so-far history increased in existing record {key}")
        if not isinstance(position, list) or len(position) != key[1]:
            raise ValueError(f"Best-position shape mismatch in existing record {key}")
        if any(not math.isfinite(float(value)) for value in position):
            raise ValueError(f"Non-finite best position in existing record {key}")
        if abs(float(history[-1]) - best_value) > 1.0e-12 * max(1.0, abs(best_value)):
            raise ValueError(f"Final objective/history mismatch in existing record {key}")
        completed[key] = record
        valid_bytes += len(raw_line)
    return completed


def execute(
    config_path: Path,
    output: Path,
    *,
    dry_run: bool = False,
    resume: bool = False,
) -> int:
    config = load_config(config_path)
    jobs = planned_runs(config)
    population = int(config["population_size"])
    iterations = int(config["iterations"])
    evaluations_per_run = population * (iterations + 1)
    expected_keys = {run_key(*job) for job in jobs}
    fingerprint = config_fingerprint(config)
    plan = {
        "campaign_id": config["campaign_id"],
        "title": config.get("title", "Paired CPSO dt sensitivity screening"),
        "benchmarks": config["benchmarks"],
        "dimensions": config["dimensions"],
        "seeds": config["seeds"],
        "dt_values": [float(v) for v in config["dt_values"]],
        "reference_dt": float(config["reference_dt"]),
        "population_size": population,
        "iterations": iterations,
        "evaluations_per_run": evaluations_per_run,
        "expected_run_count": len(jobs),
        "expected_total_function_evaluations": len(jobs) * evaluations_per_run,
        "common_random_numbers": "same seed within benchmark x dimension across all dt values",
        "execution": "native serial CPSO; isolates dt from executor concurrency",
        "resume_supported": True,
        "config_sha256": fingerprint,
        "output_directory": str(output),
    }
    if dry_run:
        print(json.dumps(plan, indent=2))
        return 0

    output.mkdir(parents=True, exist_ok=True)
    run_path = output / "runs.jsonl"
    manifest_path = output / "study_config.json"
    if resume:
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Cannot resume without the original manifest: {manifest_path}")
        existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(existing_manifest, dict):
            raise ValueError(f"Invalid dt campaign manifest: {manifest_path}")
        if existing_manifest.get("config_sha256") != fingerprint:
            raise ValueError(
                "The supplied dt config differs from the existing campaign manifest. "
                "Use a new output directory; do not mix experimental designs."
            )
    else:
        if run_path.exists() and run_path.stat().st_size:
            raise FileExistsError(
                f"Refusing to mix a new dt campaign with existing records: {run_path}. "
                "Use --resume to continue the same configuration or choose a new output directory."
            )
        if manifest_path.exists():
            raise FileExistsError(
                f"Existing campaign manifest found at {manifest_path}; use --resume or a new output directory."
            )

    from cpso_serial import native_build_info, native_version, run_serial

    # A config fingerprint does not detect a recompiled/changed native binary.
    # Preserve binary identity in the manifest and reject incompatible resumes.
    def native_binary_sha256() -> str | None:
        import hashlib
        try:
            from cpso_serial.native import _library_candidates
        except ImportError:
            # Mocked Python test modules have no native loader; real native
            # campaigns still require a loadable shared library to run.
            return None
        candidates = _library_candidates()
        library_path = next((candidate for candidate in candidates if candidate.is_file()), None)
        if library_path is None:
            return None
        digest = hashlib.sha256()
        with library_path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
        return digest.hexdigest()

    current_native_hash = native_binary_sha256()
    if resume:
        stored_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        old_hash = stored_manifest.get("native_library_sha256")
        if old_hash is not None and old_hash != current_native_hash:
            raise RuntimeError(
                "Cannot resume with a different native CPSO binary. "
                "Use the original build or start a NEW experimental campaign."
            )
        if stored_manifest.get("native_version") != native_version() or stored_manifest.get("native_build_info") != native_build_info():
            raise RuntimeError("Cannot resume CPSO dt campaign after native version/build metadata changed")

    if not resume:
        manifest = {
            "schema_version": 2,
            "plan": plan,
            "study_config": config,
            "config_sha256": fingerprint,
            "config_source": str(config_path.resolve()),
            "native_version": native_version(),
            "native_build_info": native_build_info(),
            "native_library_sha256": current_native_hash,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "environment": collect_environment_metadata(
                native_version=native_version(),
                native_build_info=native_build_info(),
            ),
        }
        temporary = manifest_path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
        )
        os.replace(temporary, manifest_path)

    completed = load_completed_records(
        run_path,
        expected_keys=expected_keys,
        expected_fe=evaluations_per_run,
        population=population,
        iterations=iterations,
        repair_truncated_tail=resume,
    ) if run_path.exists() else {}
    pending = [job for job in jobs if run_key(*job) not in completed]
    print(
        f"Campaign {config['campaign_id']}: {len(completed)}/{len(jobs)} already complete; "
        f"{len(pending)} run(s) remain."
    )
    if not pending:
        print(f"Campaign already complete; validated records: {run_path}")
        return 0

    mode = "a" if resume or run_path.exists() else "w"
    with run_path.open(mode, encoding="utf-8") as stream:
        for index, (benchmark, dimension, seed, dt) in enumerate(jobs, start=1):
            key = run_key(benchmark, dimension, seed, dt)
            if key in completed:
                continue
            cpso_config = build_cpso_config(config, benchmark, dimension, seed, dt)
            result = run_serial(cpso_config, benchmark)
            if int(result.function_evaluations) != evaluations_per_run:
                raise RuntimeError(
                    f"FE counter mismatch for {benchmark}, D={dimension}, seed={seed}, dt={dt:g}: "
                    f"{result.function_evaluations} != {evaluations_per_run}"
                )
            record = {
                "schema_version": 2,
                "run_index": index,
                "run_key": serialized_run_key(key),
                "benchmark": benchmark,
                "dimension": dimension,
                "seed": seed,
                "dt": dt,
                "reference_dt": float(config["reference_dt"]),
                "topology": "global",
                "population_size": population,
                "iterations": iterations,
                "function_evaluations": int(result.function_evaluations),
                "best_value": float(result.best_value),
                "best_position": [float(v) for v in result.best_position],
                "history_evaluations": [int(v) for v in result.history_evaluations],
                "best_history": [float(v) for v in result.best_history],
                "elapsed_seconds": float(result.elapsed_seconds),
            }
            if not math.isfinite(record["best_value"]) or not math.isfinite(record["elapsed_seconds"]):
                raise RuntimeError(f"Non-finite run output for {benchmark}, D={dimension}, seed={seed}, dt={dt:g}")
            if len(record["best_history"]) != iterations + 1 or len(record["history_evaluations"]) != iterations + 1:
                raise RuntimeError(f"Invalid history length for {benchmark}, D={dimension}, seed={seed}, dt={dt:g}")
            line = json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False)
            stream.write(line + "\n")
            stream.flush()
            os.fsync(stream.fileno())
            completed[key] = record
            print(
                f"[{len(completed):>4}/{len(jobs)}] {benchmark:10s} D={dimension:<4d} "
                f"seed={seed:<12d} dt={dt:<7g} f={record['best_value']:.6g}"
            )
    print(f"Completed {len(completed)} / {len(jobs)} runs; raw records: {run_path}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, required=True,
        help="Explicit campaign configuration; no implicit legacy study",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true",
                        help="resume a campaign only when its manifest matches the supplied config")
    args = parser.parse_args()
    from cpso_serial.output_policy import results_output
    output = args.output or ROOT / "results/raw/dt" / args.config.stem
    output = results_output(output)
    return execute(args.config, output, dry_run=args.dry_run, resume=args.resume)


if __name__ == "__main__":
    raise SystemExit(main())
