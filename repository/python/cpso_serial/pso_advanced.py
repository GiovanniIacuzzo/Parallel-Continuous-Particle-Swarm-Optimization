"""Advanced, literature-based PSO comparators used by external campaigns.

These implementations are deliberately isolated from the native CPSO engine:
all methods are scalar-objective, serial NumPy reference solvers.  The module
records the implementation choices that are needed for reproducibility and
uses the shared exact-FE result contract.
"""
from __future__ import annotations

import math
import time
from typing import Callable, Sequence

import numpy as np

from .pso_baselines import (
    Objective,
    PSOConfig,
    PSOResult,
    _apply_clamped_restitution,
)


def _evaluate(objective: Objective, point: np.ndarray, *, fe: int) -> float:
    value = float(objective(tuple(float(v) for v in point)))
    if not math.isfinite(value):
        raise ValueError(f"Objective returned a non-finite value at FE={fe}")
    return value


def _result(
    method: str,
    best_value: float,
    best_position: np.ndarray,
    best_history: list[float],
    history_evaluations: list[int],
    function_evaluations: int,
    started: float,
    parameters: dict[str, float | int | str | None],
) -> PSOResult:
    return PSOResult(
        method=method,
        best_value=float(best_value),
        best_position=tuple(float(v) for v in best_position),
        best_history=tuple(float(v) for v in best_history),
        history_evaluations=tuple(int(v) for v in history_evaluations),
        function_evaluations=int(function_evaluations),
        iterations_completed=len(best_history) - 1,
        elapsed_seconds=time.perf_counter() - started,
        parameters=parameters,
    )


def _spso2011_informants(
    population_size: int,
    informant_count: int,
    rng: np.random.Generator,
) -> list[np.ndarray]:
    """Build the SPSO-2011 directed random informant graph.

    Each particle informs K randomly selected particles and itself. Repeated
    targets are allowed, as in the standard description. The returned row i
    contains the particles informing particle i, including i itself.
    """
    links = rng.integers(0, population_size, size=(population_size, informant_count))
    neighborhoods: list[list[int]] = [[i] for i in range(population_size)]
    for source in range(population_size):
        for target in links[source]:
            neighborhoods[int(target)].append(source)
    return [np.asarray(items, dtype=int) for items in neighborhoods]


def _minimize_spso2011(
    objective: Objective,
    config: PSOConfig,
    parameters: dict[str, float | int | str | None],
) -> PSOResult:
    """Run the standard SPSO-2011 geometric update in a normalized box.

    The adaptive random informant graph is directed: each particle informs K
    randomly selected particles and itself. A particle moves by sampling a
    point in the hypersphere centred at the gravity centre of its current
    position, personal best and best informant. The default SPSO-2011 sampler
    uses a uniform direction and a uniform radius (not a uniform-volume draw).
    """
    if config.population_size < 3:
        raise ValueError("spso2011 requires population_size >= 3")
    if config.spso_informants > config.population_size:
        raise ValueError("spso_informants cannot exceed population_size")

    n = config.population_size
    d = config.dimension
    lower = np.asarray(config.lower_bounds, dtype=float)
    width = np.asarray(config.upper_bounds, dtype=float) - lower
    rng = np.random.default_rng(config.seed)
    started = time.perf_counter()

    # The adaptive random information graph is defined at initialization.
    informant_count = int(config.spso_informants)
    informants = _spso2011_informants(n, informant_count, rng)

    # SPSO-2011 recommends hypercube normalization when bounds are anisotropic.
    x = rng.random((n, d))
    v = rng.uniform(-x, 1.0 - x)
    values = np.empty(n, dtype=float)
    for i in range(n):
        values[i] = _evaluate(objective, lower + width * x[i], fe=i + 1)

    pbest_x = x.copy()
    pbest_f = values.copy()
    best_index = int(np.argmin(pbest_f))
    best_value = float(pbest_f[best_index])
    best_position = lower + width * pbest_x[best_index]
    fe = n
    best_history = [best_value]
    history_evaluations = [fe]

    c = 0.5 + math.log(2.0)
    inertia = 1.0 / (2.0 * math.log(2.0))

    while fe < config.max_evaluations:
        active = min(n, config.max_evaluations - fe)
        x0, v0, p0, f0 = x.copy(), v.copy(), pbest_x.copy(), pbest_f.copy()
        proposals = x0[:active].copy()
        proposed_v = v0[:active].copy()

        for i in range(active):
            neighborhood = informants[i]
            local_index = int(neighborhood[int(np.argmin(f0[neighborhood]))])
            personal = p0[i]
            local = p0[local_index]

            if np.array_equal(personal, local):
                # SPSO-2011's special two-point centre when l_i = p_i.
                gravity = x0[i] + c * (personal - x0[i]) / 2.0
            else:
                gravity = x0[i] + c * (personal + local - 2.0 * x0[i]) / 3.0

            radius = float(np.linalg.norm(gravity - x0[i]))
            if radius > 0.0:
                direction = rng.normal(size=d)
                norm = float(np.linalg.norm(direction))
                while norm <= np.finfo(float).tiny:
                    direction = rng.normal(size=d)
                    norm = float(np.linalg.norm(direction))
                # Default SPSO-2011: uniform direction and uniform radius.
                hypersphere_point = gravity + direction * (radius * float(rng.random()) / norm)
            else:
                hypersphere_point = gravity.copy()

            next_velocity = inertia * v0[i] + hypersphere_point - x0[i]
            raw_position = x0[i] + next_velocity
            outside = (raw_position < 0.0) | (raw_position > 1.0)
            next_position = np.clip(raw_position, 0.0, 1.0)
            # SPSO-2011 confinement: reverse half of each offending component.
            next_velocity[outside] *= -0.5
            proposals[i] = next_position
            proposed_v[i] = next_velocity

        next_values = np.empty(active, dtype=float)
        for i in range(active):
            next_values[i] = _evaluate(
                objective, lower + width * proposals[i], fe=fe + i + 1
            )

        x[:active] = proposals
        v[:active] = proposed_v
        improved = next_values < pbest_f[:active]
        idx = np.flatnonzero(improved)
        pbest_f[idx] = next_values[idx]
        pbest_x[idx] = proposals[idx]
        fe += active

        generation_best_index = int(np.argmin(pbest_f))
        improved_global = float(pbest_f[generation_best_index]) < best_value
        if improved_global:
            best_value = float(pbest_f[generation_best_index])
            best_position = lower + width * pbest_x[generation_best_index]
        elif fe < config.max_evaluations:
            # Rebuild the adaptive random informant graph after stagnation.
            informants = _spso2011_informants(n, informant_count, rng)

        best_history.append(best_value)
        history_evaluations.append(fe)

    return _result(
        "spso2011", best_value, best_position, best_history,
        history_evaluations, fe, started, parameters,
    )


def _minimize_fips(
    objective: Objective,
    config: PSOConfig,
    parameters: dict[str, float | int | str | None],
) -> PSOResult:
    """Fully informed PSO with a degree-two periodic ring neighborhood."""
    if config.population_size < 3:
        raise ValueError("fips_ring requires population_size >= 3")

    n = config.population_size
    lower = np.asarray(config.lower_bounds, dtype=float)
    upper = np.asarray(config.upper_bounds, dtype=float)
    width = upper - lower
    rng = np.random.default_rng(config.seed)
    started = time.perf_counter()

    x = rng.uniform(lower, upper, size=(n, config.dimension))
    v = rng.uniform(
        -config.initial_velocity_fraction * width,
        config.initial_velocity_fraction * width,
        size=(n, config.dimension),
    )
    values = np.empty(n, dtype=float)
    for i in range(n):
        values[i] = _evaluate(objective, x[i], fe=i + 1)
    pbest_x = x.copy()
    pbest_f = values.copy()
    best_index = int(np.argmin(pbest_f))
    best_value = float(pbest_f[best_index])
    best_position = pbest_x[best_index].copy()
    fe = n
    best_history = [best_value]
    history_evaluations = [fe]

    phi = float(config.fips_phi)
    chi = 2.0 / (phi - 2.0 + math.sqrt(phi * phi - 4.0 * phi))
    while fe < config.max_evaluations:
        active = min(n, config.max_evaluations - fe)
        x0, v0, p0 = x.copy(), v.copy(), pbest_x.copy()
        next_x = x0[:active].copy()
        next_v = v0[:active].copy()
        for i in range(active):
            neighbors = ((i - 1) % n, (i + 1) % n)
            social = np.zeros(config.dimension, dtype=float)
            for neighbor in neighbors:
                weights = rng.uniform(0.0, phi, size=config.dimension)
                social += weights * (p0[neighbor] - x0[i]) / float(len(neighbors))
            velocity = chi * (v0[i] + social)
            position, velocity = _apply_clamped_restitution(
                x0[i] + velocity, velocity, lower, upper,
                config.boundary_restitution,
            )
            next_x[i] = position
            next_v[i] = velocity

        next_values = np.empty(active, dtype=float)
        for i in range(active):
            next_values[i] = _evaluate(objective, next_x[i], fe=fe + i + 1)
        x[:active] = next_x
        v[:active] = next_v
        improved = next_values < pbest_f[:active]
        idx = np.flatnonzero(improved)
        pbest_f[idx] = next_values[idx]
        pbest_x[idx] = next_x[idx]
        fe += active
        best_index = int(np.argmin(pbest_f))
        if float(pbest_f[best_index]) < best_value:
            best_value = float(pbest_f[best_index])
            best_position = pbest_x[best_index].copy()
        best_history.append(best_value)
        history_evaluations.append(fe)

    return _result(
        "fips_ring", best_value, best_position, best_history,
        history_evaluations, fe, started, parameters,
    )


def _minimize_cso(
    objective: Objective,
    config: PSOConfig,
    parameters: dict[str, float | int | str | None],
) -> PSOResult:
    """Competitive Swarm Optimizer (pairwise winner/loser learning)."""
    n = config.population_size
    if n % 2:
        raise ValueError("cso requires an even population_size for pairwise competition")

    lower = np.asarray(config.lower_bounds, dtype=float)
    upper = np.asarray(config.upper_bounds, dtype=float)
    width = upper - lower
    rng = np.random.default_rng(config.seed)
    started = time.perf_counter()

    x = rng.uniform(lower, upper, size=(n, config.dimension))
    v = np.zeros_like(x)
    values = np.empty(n, dtype=float)
    for i in range(n):
        values[i] = _evaluate(objective, x[i], fe=i + 1)
    best_index = int(np.argmin(values))
    best_value = float(values[best_index])
    best_position = x[best_index].copy()
    fe = n
    best_history = [best_value]
    history_evaluations = [fe]
    phi = float(config.cso_phi)

    while fe < config.max_evaluations:
        pair_order = rng.permutation(n)
        swarm_mean = np.mean(x, axis=0)
        next_x = x.copy()
        next_v = v.copy()
        next_values = values.copy()
        remaining = config.max_evaluations - fe
        active_pairs = min(n // 2, remaining)

        for pair in range(active_pairs):
            first = int(pair_order[2 * pair])
            second = int(pair_order[2 * pair + 1])
            if values[first] <= values[second]:
                winner, loser = first, second
            else:
                winner, loser = second, first

            r1 = rng.random(config.dimension)
            r2 = rng.random(config.dimension)
            r3 = rng.random(config.dimension)
            velocity = (
                r1 * v[loser]
                + r2 * (x[winner] - x[loser])
                + phi * r3 * (swarm_mean - x[loser])
            )
            position, velocity = _apply_clamped_restitution(
                x[loser] + velocity, velocity, lower, upper,
                config.boundary_restitution,
            )
            value = _evaluate(objective, position, fe=fe + pair + 1)
            next_x[loser] = position
            next_v[loser] = velocity
            next_values[loser] = value
            if value < best_value:
                best_value = value
                best_position = position.copy()

        x, v, values = next_x, next_v, next_values
        fe += active_pairs
        best_history.append(best_value)
        history_evaluations.append(fe)

    return _result(
        "cso", best_value, best_position, best_history,
        history_evaluations, fe, started, parameters,
    )


def _dynamic_groups(
    population_size: int,
    target_size: int,
    rng: np.random.Generator,
) -> list[np.ndarray]:
    """Randomly regroup particles; absorb a final remainder into last group."""
    order = rng.permutation(population_size)
    groups = [order[start:start + target_size] for start in range(0, population_size, target_size)]
    if len(groups) > 1 and groups[-1].size < 2:
        groups[-2] = np.concatenate((groups[-2], groups[-1]))
        groups.pop()
    if any(group.size < 2 for group in groups):
        raise RuntimeError("DMS regrouping produced a singleton sub-swarm")
    return groups


def _minimize_dms_pso(
    objective: Objective,
    config: PSOConfig,
    parameters: dict[str, float | int | str | None],
) -> PSOResult:
    """Dynamic multi-swarm PSO with periodic random regrouping (no local search)."""
    if config.population_size < 4:
        raise ValueError("dms_pso requires population_size >= 4")
    if config.dms_subswarm_size >= config.population_size:
        raise ValueError("dms_subswarm_size must be smaller than population_size")

    n = config.population_size
    lower = np.asarray(config.lower_bounds, dtype=float)
    upper = np.asarray(config.upper_bounds, dtype=float)
    width = upper - lower
    rng = np.random.default_rng(config.seed)
    started = time.perf_counter()

    x = rng.uniform(lower, upper, size=(n, config.dimension))
    v = rng.uniform(
        -config.initial_velocity_fraction * width,
        config.initial_velocity_fraction * width,
        size=(n, config.dimension),
    )
    values = np.empty(n, dtype=float)
    for i in range(n):
        values[i] = _evaluate(objective, x[i], fe=i + 1)
    pbest_x = x.copy()
    pbest_f = values.copy()
    best_index = int(np.argmin(pbest_f))
    best_value = float(pbest_f[best_index])
    best_position = pbest_x[best_index].copy()
    fe = n
    best_history = [best_value]
    history_evaluations = [fe]

    groups = _dynamic_groups(n, config.dms_subswarm_size, rng)
    total_generations = max(1, math.ceil((config.max_evaluations - n) / n))
    generation = 0
    while fe < config.max_evaluations:
        active = min(n, config.max_evaluations - fe)
        x0, v0, p0, f0 = x.copy(), v.copy(), pbest_x.copy(), pbest_f.copy()
        social = np.empty_like(x0)
        for group in groups:
            leader = int(group[int(np.argmin(f0[group]))])
            social[group] = p0[leader]

        progress = generation / max(1, total_generations - 1)
        inertia = 0.9 + (0.2 - 0.9) * min(1.0, progress)
        r1 = rng.random((active, config.dimension))
        r2 = rng.random((active, config.dimension))
        velocity = (
            inertia * v0[:active]
            + 2.0 * r1 * (p0[:active] - x0[:active])
            + 2.0 * r2 * (social[:active] - x0[:active])
        )
        if config.velocity_limit_fraction is not None:
            vmax = config.velocity_limit_fraction * width
            velocity = np.clip(velocity, -vmax, vmax)
        positions, velocity = _apply_clamped_restitution(
            x0[:active] + velocity, velocity, lower, upper,
            config.boundary_restitution,
        )
        next_values = np.empty(active, dtype=float)
        for i in range(active):
            next_values[i] = _evaluate(objective, positions[i], fe=fe + i + 1)

        x[:active] = positions
        v[:active] = velocity
        improved = next_values < pbest_f[:active]
        idx = np.flatnonzero(improved)
        pbest_f[idx] = next_values[idx]
        pbest_x[idx] = positions[idx]
        fe += active
        best_index = int(np.argmin(pbest_f))
        if float(pbest_f[best_index]) < best_value:
            best_value = float(pbest_f[best_index])
            best_position = pbest_x[best_index].copy()

        generation += 1
        if generation % config.dms_regroup_interval == 0 and fe < config.max_evaluations:
            groups = _dynamic_groups(n, config.dms_subswarm_size, rng)
        best_history.append(best_value)
        history_evaluations.append(fe)

    return _result(
        "dms_pso", best_value, best_position, best_history,
        history_evaluations, fe, started, parameters,
    )


def minimize_advanced_pso(
    objective: Objective,
    config: PSOConfig,
    *,
    method: str,
    parameters: dict[str, float | int | str | None],
) -> PSOResult:
    """Dispatch advanced PSO variants under the common result contract."""
    dispatch: dict[str, Callable[..., PSOResult]] = {
        "spso2011": _minimize_spso2011,
        "fips_ring": _minimize_fips,
        "cso": _minimize_cso,
        "dms_pso": _minimize_dms_pso,
    }
    try:
        implementation = dispatch[method]
    except KeyError as exc:
        raise ValueError(f"Unknown advanced PSO method {method!r}") from exc
    return implementation(objective, config, parameters)


__all__ = ["minimize_advanced_pso"]
