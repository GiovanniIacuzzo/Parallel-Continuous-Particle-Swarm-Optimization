from __future__ import annotations

import copy
import tomllib
from pathlib import Path
from typing import Any

BENCHMARK_DEFAULTS: dict[str, tuple[float, float]] = {
    "sphere": (-100.0, 100.0),
    "rastrigin": (-5.12, 5.12),
    "rosenbrock": (-5.0, 5.0),
    "schwefel": (-100.0, 100.0),
    "griewank": (-100.0, 100.0),
    "ackley": (-100.0, 100.0),
    "salomon": (-100.0, 100.0),
}


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a TOML experiment file and return a regular Python dictionary."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with path.open("rb") as stream:
        cfg = tomllib.load(stream)
    return validate_config(cfg)


def _expand_bound(value: Any, dimension: int, name: str) -> list[float]:
    if isinstance(value, (int, float)):
        return [float(value)] * dimension
    if isinstance(value, list) and len(value) == dimension:
        return [float(x) for x in value]
    raise ValueError(f"{name} must be a scalar or a list of length dimension={dimension}.")


def validate_config(config: dict[str, Any]) -> dict[str, Any]:
    """Validate the experiment configuration without silently changing its meaning."""
    cfg = copy.deepcopy(config)
    for section in ("problem", "algorithm", "cpso"):
        if section not in cfg or not isinstance(cfg[section], dict):
            raise ValueError(f"Missing required TOML section [{section}].")

    problem = cfg["problem"]
    algorithm = cfg["algorithm"]
    cpso = cfg["cpso"]

    name = str(problem.get("name", "")).lower()
    if name not in BENCHMARK_DEFAULTS:
        raise ValueError(f"Unknown benchmark {name!r}; choose from {sorted(BENCHMARK_DEFAULTS)}.")
    problem["name"] = name

    dimension = int(problem.get("dimension", 0))
    if dimension < 1:
        raise ValueError("[problem].dimension must be >= 1.")
    if name == "rosenbrock" and dimension < 2:
        raise ValueError("Rosenbrock requires dimension >= 2.")
    problem["dimension"] = dimension

    default_lo, default_hi = BENCHMARK_DEFAULTS[name]
    lower = _expand_bound(problem.get("lower_bound", default_lo), dimension, "lower_bound")
    upper = _expand_bound(problem.get("upper_bound", default_hi), dimension, "upper_bound")
    for i, (lo, hi) in enumerate(zip(lower, upper)):
        if not lo < hi:
            raise ValueError(f"Invalid bounds at dimension {i}: lower_bound must be < upper_bound.")
    problem["lower_bounds"] = lower
    problem["upper_bounds"] = upper

    if "optimum_position" in problem:
        optimum = _expand_bound(problem["optimum_position"], dimension, "optimum_position")
    else:
        optimum = [0.0] * dimension
    for i, (x, lo, hi) in enumerate(zip(optimum, lower, upper)):
        if not lo <= x <= hi:
            raise ValueError(f"optimum_position[{i}] must lie inside the search bounds.")
    problem["optimum_position"] = optimum

    population = int(algorithm.get("population_size", 0))
    workers = int(algorithm.get("workers", 1))
    iterations = int(algorithm.get("iterations", 0))
    sync_interval = int(algorithm.get("sync_interval", 1))
    topology = str(algorithm.get("topology", "star")).lower()
    if population < 1 or workers < 1 or population < workers:
        raise ValueError("Require population_size >= workers >= 1.")
    if iterations < 0:
        raise ValueError("iterations must be >= 0.")
    if sync_interval < 1:
        raise ValueError("sync_interval must be >= 1.")
    if topology not in {"independent", "star", "ring"}:
        raise ValueError("topology must be 'independent', 'star' or 'ring'.")
    algorithm.update({
        "population_size": population,
        "workers": workers,
        "iterations": iterations,
        "sync_interval": sync_interval,
        "topology": topology,
        "seed": int(algorithm.get("seed", 12345)),
    })

    cpso_defaults = {
        "mu_max": 0.9,
        "mu_min": 0.4,
        "cognitive": 2.05,
        "social": 2.05,
        "dt": 1.0,
        "initial_velocity_fraction": 0.1,
        "boundary_restitution": 0.5,
    }
    for key, default in cpso_defaults.items():
        cpso[key] = float(cpso.get(key, default))
    if not (0 <= cpso["mu_min"] <= cpso["mu_max"] <= 1):
        raise ValueError("For a non-negative damping coefficient require 0 <= mu_min <= mu_max <= 1.")
    if cpso["cognitive"] < 0 or cpso["social"] < 0 or cpso["cognitive"] + cpso["social"] <= 0:
        raise ValueError("cognitive and social must be non-negative and not both zero.")
    if cpso["dt"] <= 0:
        raise ValueError("dt must be > 0.")
    if cpso["initial_velocity_fraction"] < 0:
        raise ValueError("initial_velocity_fraction must be >= 0.")
    if not 0 <= cpso["boundary_restitution"] <= 1:
        raise ValueError("boundary_restitution must lie in [0,1].")

    return cfg
