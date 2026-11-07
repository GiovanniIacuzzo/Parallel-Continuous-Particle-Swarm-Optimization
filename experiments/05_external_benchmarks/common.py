from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import platform
from pathlib import Path
import re
import sys
from typing import Any, Callable, Sequence

from cpso_serial import CpsoConfig, native_build_info, native_version, run_serial_objective
from cpso_serial.pso_baselines import PSOConfig, minimize_pso

METHODS = {
    "cpso_continuous": ("cpso", None),
    "pso_classic_gbest": ("pso", "classic_gbest"),
    "pso_inertia_gbest": ("pso", "inertia_gbest"),
    "pso_constriction_gbest": ("pso", "constriction_gbest"),
    "pso_inertia_lbest_ring": ("pso", "inertia_lbest_ring"),
    "pso_clpso": ("pso", "clpso"),
    "pso_slpso": ("pso", "slpso"),
    "pso_spso2011": ("pso", "spso2011"),
    "pso_fips": ("pso", "fips_ring"),
    "pso_cso": ("pso", "cso"),
    "pso_dms": ("pso", "dms_pso"),
}


ROOT = Path(__file__).resolve().parents[2]
_THREAD_ENVIRONMENT = (
    "OMP_NUM_THREADS", "OMP_DYNAMIC", "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS",
)
_EXTERNAL_PACKAGES = ("numpy", "coco-experiment", "cocopp", "cec2013lsgo")
_SOURCE_ROOTS = (
    "experiments/05_external_benchmarks",
    "python/cpso_serial",
    "cpp/src",
    "cpp/include",
)
_SOURCE_FILES = ("CMakeLists.txt", "pyproject.toml")



def collect_external_provenance(config_path: Path) -> dict[str, Any]:
    """Capture the code, build, interpreter and optional-package provenance."""
    source_hash = hashlib.sha256()
    files = {ROOT / relative for relative in _SOURCE_FILES}
    for relative_root in _SOURCE_ROOTS:
        root = ROOT / relative_root
        files.update(path for path in root.rglob("*") if path.is_file() and path.suffix in {".py", ".cpp", ".hpp", ".h"})
    for path in sorted(files):
        relative = path.relative_to(ROOT).as_posix()
        source_hash.update(relative.encode("utf-8"))
        source_hash.update(b"\0")
        source_hash.update(path.read_bytes())
        source_hash.update(b"\0")
    packages: dict[str, str | None] = {}
    for distribution in _EXTERNAL_PACKAGES:
        try:
            packages[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            packages[distribution] = None
    return {
        "project_version": "0.7.4",
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "source_code_sha256": source_hash.hexdigest(),
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "processor": platform.processor() or None,
            "logical_cpu_count": os.cpu_count(),
        },
        "python": {
            "version": sys.version,
            "implementation": platform.python_implementation(),
        },
        "native": {
            "version": native_version(),
            "build_info": native_build_info(),
            "c_abi": 7,
        },
        "packages": packages,
        "thread_environment": {key: os.environ.get(key) for key in _THREAD_ENVIRONMENT},
    }


def write_campaign_manifest(path: Path, plan: dict[str, Any], config_path: Path) -> None:
    """Write provenance once; reject incompatible resumes rather than overwrite it."""
    manifest = {**plan, "provenance": collect_external_provenance(config_path)}
    if path.exists():
        try:
            previous = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Cannot resume: campaign manifest {path} is unreadable") from exc
        if previous != manifest:
            raise RuntimeError(
                "Cannot resume external benchmark campaign: configuration, source, "
                "native build, package versions or execution environment differ from the manifest"
            )
        return
    write_json(path, manifest)




def _selector_values(specification: str, *, name: str) -> list[int]:
    values: list[int] = []
    for item in specification.split(","):
        token = item.strip()
        if not token:
            continue
        if "-" in token:
            pieces = token.split("-")
            if len(pieces) != 2:
                raise ValueError(f"Invalid {name} selector token {token!r}")
            start, end = (int(part) for part in pieces)
            if end < start:
                raise ValueError(f"Descending {name} selector range {token!r}")
            values.extend(range(start, end + 1))
        else:
            values.append(int(token))
    if not values or len(set(values)) != len(values):
        raise ValueError(f"{name} selector must contain unique integer values")
    return values


def _option_value(options: str, key: str) -> str | None:
    match = re.search(rf"(?:^|\s){re.escape(key)}\s*:\s*([^\s]+)", options)
    return match.group(1) if match else None


def estimate_bbob_campaign(
    suite_name: str,
    instances: str,
    options: str,
    *,
    budget_multiplier: int,
    population_size: int,
    method_count: int,
) -> dict[str, Any]:
    """Parse the project's COCO selectors and preflight the exact FE/run counts."""
    dimension_defaults = {
        "bbob": [2, 3, 5, 10, 20, 40],
        "bbob-largescale": [20, 40, 80, 160, 320, 640],
    }
    if suite_name not in dimension_defaults:
        raise ValueError(f"Unsupported COCO suite {suite_name!r}")
    dimension_selector = _option_value(options, "dimensions")
    if dimension_selector is None and _option_value(options, "dimension_indices") is not None:
        raise ValueError("Use explicit COCO 'dimensions:' values in project campaign configs")
    dimensions = (
        _selector_values(dimension_selector, name="dimension")
        if dimension_selector else dimension_defaults[suite_name]
    )
    function_selector = _option_value(options, "function_indices")
    function_ids = _selector_values(function_selector, name="function") if function_selector else list(range(1, 25))
    instance_selector = _option_value(instances, "instances")
    instance_ids = _selector_values(instance_selector, name="instance") if instance_selector else list(range(1, 16))
    if any(dimension not in dimension_defaults[suite_name] for dimension in dimensions):
        raise ValueError(f"Invalid dimensions for {suite_name}: {dimensions}")
    if any(function_id < 1 or function_id > 24 for function_id in function_ids):
        raise ValueError("COCO function_indices must lie in [1,24]")
    if any(instance_id < 1 or instance_id > 15 for instance_id in instance_ids):
        raise ValueError("COCO instances must lie in [1,15]")
    incompatible = [
        dimension for dimension in dimensions
        if budget_multiplier * dimension < population_size
        or (budget_multiplier * dimension) % population_size != 0
    ]
    if incompatible:
        raise ValueError(
            f"FE budget is not a positive multiple of population_size for dimensions {incompatible}"
        )
    problem_count = len(dimensions) * len(function_ids) * len(instance_ids)
    run_count = problem_count * method_count
    fe_count = budget_multiplier * sum(dimensions) * len(function_ids) * len(instance_ids) * method_count
    return {
        "dimensions": dimensions,
        "function_ids": function_ids,
        "instances": instance_ids,
        "expected_problem_count": problem_count,
        "expected_method_runs": run_count,
        "expected_total_function_evaluations": fe_count,
    }


def deterministic_problem_seed(problem_id: str, campaign_seed: int = 20261105) -> int:
    payload = f"{campaign_seed}:{problem_id}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little", signed=False)


def run_method(
    method: str,
    objective: Callable[[Sequence[float]], float],
    lower_bounds: Sequence[float],
    upper_bounds: Sequence[float],
    *,
    population_size: int,
    max_evaluations: int,
    seed: int,
    cpso_dt: float = 0.7,
):
    if method not in METHODS:
        raise ValueError(f"Unknown method {method!r}; supported methods: {sorted(METHODS)}")
    if max_evaluations < population_size or max_evaluations % population_size != 0:
        raise ValueError("For strict cross-method FE parity, max_evaluations must be a multiple of population_size")
    kind, variant = METHODS[method]
    if kind == "cpso":
        if not math.isfinite(float(cpso_dt)) or float(cpso_dt) <= 0.0:
            raise ValueError("cpso_dt must be finite and strictly positive")
        config = CpsoConfig(
            lower_bounds=tuple(float(v) for v in lower_bounds),
            upper_bounds=tuple(float(v) for v in upper_bounds),
            population_size=population_size,
            iterations=max_evaluations // population_size - 1,
            seed=seed,
            mu_max=0.9,
            mu_min=0.4,
            cognitive=2.05,
            social=2.05,
            dt=float(cpso_dt),
            initial_velocity_fraction=0.1,
            boundary_restitution=0.5,
            topology="global",
        )
        return run_serial_objective(config, objective)

    pso_config = PSOConfig(
        lower_bounds=tuple(float(v) for v in lower_bounds),
        upper_bounds=tuple(float(v) for v in upper_bounds),
        population_size=population_size,
        max_evaluations=max_evaluations,
        seed=seed,
        cognitive=2.05,
        social=2.05,
        inertia_max=0.9,
        inertia_min=0.4,
        initial_velocity_fraction=0.1,
        velocity_limit_fraction=0.2,
        boundary_restitution=0.5,
        ring_radius=1,
        spso_informants=3,
        fips_phi=4.1,
        cso_phi=0.1,
        dms_subswarm_size=3,
        dms_regroup_interval=5,
    )
    return minimize_pso(objective, pso_config, method=str(variant))


def checkpoint_values(result: Any, milestones: Sequence[int]) -> dict[str, float]:
    values: dict[str, float] = {}
    history_fe = tuple(int(v) for v in result.history_evaluations)
    history = tuple(float(v) for v in result.best_history)
    if len(history_fe) != len(history):
        raise RuntimeError("Optimizer history FE/value arrays are misaligned")
    for milestone in milestones:
        try:
            index = history_fe.index(int(milestone))
        except ValueError as exc:
            raise RuntimeError(
                f"No convergence-history sample exactly at requested milestone FE={milestone}"
            ) from exc
        values[str(int(milestone))] = history[index]
    return values


def build_run_record(
    *,
    suite: str,
    problem_id: str,
    problem_name: str,
    function_id: int | None,
    dimension: int,
    method: str,
    seed: int,
    max_evaluations: int,
    result: Any,
    milestones: Sequence[int] = (),
    cpso_dt: float = 0.7,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "schema_version": 1,
        "suite": suite,
        "problem_id": problem_id,
        "problem_name": problem_name,
        "function_id": function_id,
        "dimension": int(dimension),
        "method": method,
        "seed": int(seed),
        "max_evaluations": int(max_evaluations),
        "function_evaluations": int(result.function_evaluations),
        "best_value": float(result.best_value),
        "elapsed_seconds": float(result.elapsed_seconds),
        "iterations_completed": int(result.iterations_completed),
        "best_position": [float(v) for v in result.best_position],
        "checkpoints": checkpoint_values(result, milestones) if milestones else {},
    }
    if hasattr(result, "parameters"):
        record["algorithm_parameters"] = result.parameters
    else:
        record["algorithm_parameters"] = {
            "model": "continuous-time PSO",
            "mu_max": 0.9,
            "mu_min": 0.4,
            "cognitive": 2.05,
            "social": 2.05,
            "dt": float(cpso_dt),
            "topology": "global-best",
            "boundary_restitution": 0.5,
            "initial_velocity_fraction": 0.1,
        }
    if record["function_evaluations"] != max_evaluations:
        raise RuntimeError(
            f"FE budget mismatch for {method}: {record['function_evaluations']} != {max_evaluations}"
        )
    return record


def append_jsonl_durable(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(payload + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def load_existing_run_keys(path: Path) -> set[str]:
    if not path.exists():
        return set()
    payload = path.read_bytes()
    lines = payload.splitlines(keepends=True)
    keys: set[str] = set()
    valid_end = 0
    for index, raw_line in enumerate(lines):
        if not raw_line.strip():
            valid_end += len(raw_line)
            continue
        try:
            record = json.loads(raw_line)
        except json.JSONDecodeError as exc:
            is_incomplete_tail = index == len(lines) - 1 and not raw_line.endswith(b"\n")
            if is_incomplete_tail:
                break
            raise ValueError(f"Corrupt benchmark checkpoint {path}, line {index + 1}") from exc
        if not isinstance(record, dict) or "run_key" not in record:
            raise ValueError(f"Invalid benchmark checkpoint record in {path}, line {index + 1}")
        keys.add(str(record["run_key"]))
        valid_end += len(raw_line)
    if valid_end != len(payload):
        # Recover only a non-newline-terminated, malformed final record. Any
        # corruption in a complete record is rejected rather than hidden.
        with path.open("r+b") as stream:
            stream.truncate(valid_end)
            stream.flush()
            os.fsync(stream.fileno())
    elif payload and not payload.endswith(b"\n"):
        with path.open("ab") as stream:
            stream.write(b"\n")
            stream.flush()
            os.fsync(stream.fileno())
    return keys


def record_with_key(record: dict[str, Any]) -> dict[str, Any]:
    key = "|".join(str(record.get(field, "")) for field in (
        "suite", "problem_id", "method", "seed"
    ))
    return {"run_key": key, **record}


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


__all__ = [
    "METHODS", "append_jsonl_durable", "build_run_record", "checkpoint_values",
    "deterministic_problem_seed", "estimate_bbob_campaign", "load_existing_run_keys",
    "record_with_key", "run_method", "write_campaign_manifest", "write_json",
]
