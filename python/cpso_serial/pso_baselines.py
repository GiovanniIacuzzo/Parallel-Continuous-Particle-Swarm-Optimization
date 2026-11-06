"""Reference PSO baselines for black-box benchmark studies.

The implementations are intentionally independent of the native CPSO engine.
They use NumPy for vectorized particle updates and accept a scalar objective
callback, allowing the same optimizer implementation to run on COCO/CEC test
problems. They are quality-comparison baselines, not performance backends.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Callable, Sequence

import numpy as np


Objective = Callable[[Sequence[float]], float]


@dataclass(frozen=True)
class PSOConfig:
    lower_bounds: Sequence[float]
    upper_bounds: Sequence[float]
    population_size: int = 40
    max_evaluations: int = 10_000
    seed: int = 1
    cognitive: float = 2.05
    social: float = 2.05
    inertia_max: float = 0.9
    inertia_min: float = 0.4
    initial_velocity_fraction: float = 0.1
    velocity_limit_fraction: float | None = 0.2
    boundary_restitution: float = 0.5
    ring_radius: int = 1

    @property
    def dimension(self) -> int:
        return len(self.lower_bounds)

    def validate(self) -> None:
        lower = np.asarray(self.lower_bounds, dtype=float)
        upper = np.asarray(self.upper_bounds, dtype=float)
        if lower.ndim != 1 or upper.ndim != 1 or lower.size == 0:
            raise ValueError("PSO bounds must be non-empty one-dimensional sequences")
        if lower.shape != upper.shape:
            raise ValueError("lower_bounds and upper_bounds must have equal lengths")
        if not np.all(np.isfinite(lower)) or not np.all(np.isfinite(upper)):
            raise ValueError("PSO bounds must be finite")
        if np.any(upper <= lower) or not np.all(np.isfinite(upper - lower)):
            raise ValueError("Every PSO bound must satisfy finite lower < upper")
        for name, value, minimum in (
            ("population_size", self.population_size, 2),
            ("max_evaluations", self.max_evaluations, 2),
            ("ring_radius", self.ring_radius, 1),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        if self.max_evaluations < self.population_size:
            raise ValueError("max_evaluations must be >= population_size")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or not 0 <= self.seed < 2**64:
            raise ValueError("seed must be an unsigned 64-bit integer")
        if not (math.isfinite(self.cognitive) and math.isfinite(self.social)
                and self.cognitive > 0 and self.social > 0
                and math.isfinite(self.cognitive + self.social)):
            raise ValueError("cognitive and social coefficients and their sum must be finite and > 0")
        if not (math.isfinite(self.inertia_max) and math.isfinite(self.inertia_min)
                and 0 <= self.inertia_min <= self.inertia_max <= 1.5):
            raise ValueError("inertia bounds must satisfy 0 <= min <= max <= 1.5")
        if not (math.isfinite(self.initial_velocity_fraction)
                and self.initial_velocity_fraction >= 0):
            raise ValueError("initial_velocity_fraction must be finite and >= 0")
        if self.velocity_limit_fraction is not None and not (
            math.isfinite(self.velocity_limit_fraction) and self.velocity_limit_fraction > 0
        ):
            raise ValueError("velocity_limit_fraction must be None or finite and > 0")
        if not (math.isfinite(self.boundary_restitution)
                and 0 <= self.boundary_restitution <= 1):
            raise ValueError("boundary_restitution must lie in [0,1]")
        width = upper - lower
        with np.errstate(over="ignore", invalid="ignore"):
            initial_velocity_range = self.initial_velocity_fraction * width
            if not np.all(np.isfinite(initial_velocity_range)):
                raise ValueError("initial velocity range overflows for the selected bounds")
            if self.velocity_limit_fraction is not None:
                velocity_limit = self.velocity_limit_fraction * width
                if not np.all(np.isfinite(velocity_limit)):
                    raise ValueError("velocity limit overflows for the selected bounds")


@dataclass(frozen=True)
class PSOResult:
    method: str
    best_value: float
    best_position: tuple[float, ...]
    best_history: tuple[float, ...]
    history_evaluations: tuple[int, ...]
    function_evaluations: int
    iterations_completed: int
    elapsed_seconds: float
    parameters: dict[str, float | int | str | None]


def available_pso_methods() -> tuple[str, ...]:
    return (
        "classic_gbest",
        "inertia_gbest",
        "constriction_gbest",
        "inertia_lbest_ring",
        "clpso",
        "slpso",
    )


def pso_method_metadata(method: str, config: PSOConfig) -> dict[str, float | int | str | None]:
    """Return the fully resolved parameterization used by a named baseline."""
    if method not in available_pso_methods():
        raise ValueError(f"Unknown PSO method {method!r}; choose from {available_pso_methods()}")
    c1 = config.cognitive
    c2 = config.social
    if method == "classic_gbest":
        c1 = c2 = 2.0
        inertia_max = inertia_min = 1.0
        chi = 1.0
        velocity_limit = config.velocity_limit_fraction
        topology = "global-best"
    elif method == "inertia_gbest":
        inertia_max, inertia_min = config.inertia_max, config.inertia_min
        chi = 1.0
        velocity_limit = config.velocity_limit_fraction
        topology = "global-best"
    elif method == "constriction_gbest":
        if c1 + c2 <= 4.0:
            raise ValueError("constriction_gbest requires cognitive + social > 4")
        phi = c1 + c2
        if not math.isfinite(phi):
            raise ValueError("constriction coefficient sum must be finite")
        chi = 2.0 / (phi - 2.0 + math.sqrt(phi * phi - 4.0 * phi))
        inertia_max = inertia_min = 1.0
        velocity_limit = None
        topology = "global-best"
    elif method == "inertia_lbest_ring":
        inertia_max, inertia_min = config.inertia_max, config.inertia_min
        chi = 1.0
        velocity_limit = config.velocity_limit_fraction
        topology = f"periodic-ring(radius={config.ring_radius})"
    elif method == "clpso":
        # Literature parameterization of CLPSO (Liang et al., IEEE TEVC, 2006).
        # The project applies its common clamped-restitution boundary map.
        c1 = 1.49445
        c2 = 0.0
        inertia_max, inertia_min = 0.9, 0.4
        chi = 1.0
        velocity_limit = 0.2
        topology = "comprehensive-learning-exemplars-per-coordinate"
    else:
        # SL-PSO (Cheng & Jin, Information Sciences, 2015). The population is
        # fixed by the shared benchmark protocol rather than dimension-scaled.
        c1 = c2 = 0.0
        inertia_max = inertia_min = 1.0
        chi = 1.0
        velocity_limit = None
        topology = "fitness-sorted-social-demonstrator-learning"
    return {
        "cognitive": c1,
        "social": c2,
        "inertia_max": inertia_max,
        "inertia_min": inertia_min,
        "constriction_factor": chi,
        "velocity_limit_fraction": velocity_limit,
        "initial_velocity_fraction": (
            0.0 if method == "slpso" else config.initial_velocity_fraction
        ),
        "boundary_restitution": config.boundary_restitution,
        "topology": topology,
        "ring_radius": config.ring_radius if method == "inertia_lbest_ring" else None,
        "clpso_learning_probability": (
            "0.05 + 0.45*(exp(10*i/(N-1))-1)/(exp(10)-1), i=0,...,N-1"
            if method == "clpso" else None
        ),
        "clpso_refresh_gap": 7 if method == "clpso" else None,
        "clpso_acceleration": 1.49445 if method == "clpso" else None,
        "slpso_reference_M": 100 if method == "slpso" else None,
        "slpso_alpha": 0.5 if method == "slpso" else None,
        "slpso_beta": 0.01 if method == "slpso" else None,
        "slpso_social_influence_factor": (
            0.01 * config.dimension / 100.0 if method == "slpso" else None
        ),
        "slpso_population_policy": (
            "fixed_population_for_equal_FE_comparison; paper default M+ceil(D/10)"
            if method == "slpso" else None
        ),
        "update_semantics": "frozen-generation-snapshot",
        "population_size": config.population_size,
        "max_evaluations": config.max_evaluations,
        "seed": config.seed,
    }


def _social_attractors(
    personal_best_positions: np.ndarray,
    personal_best_values: np.ndarray,
    method: str,
    ring_radius: int,
) -> np.ndarray:
    population_size = personal_best_positions.shape[0]
    if method != "inertia_lbest_ring":
        index = int(np.argmin(personal_best_values))
        return np.broadcast_to(personal_best_positions[index], personal_best_positions.shape)
    result = np.empty_like(personal_best_positions)
    offsets = range(-ring_radius, ring_radius + 1)
    for particle in range(population_size):
        neighborhood = sorted({(particle + offset) % population_size for offset in offsets})
        local_values = personal_best_values[neighborhood]
        local_index = neighborhood[int(np.argmin(local_values))]
        result[particle] = personal_best_positions[local_index]
    return result


def _apply_clamped_restitution(
    proposed_positions: np.ndarray,
    velocities: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    restitution: float,
) -> tuple[np.ndarray, np.ndarray]:
    if not np.all(np.isfinite(proposed_positions)) or not np.all(np.isfinite(velocities)):
        raise FloatingPointError("PSO propagation produced a non-finite position or velocity")
    positions = np.minimum(np.maximum(proposed_positions, lower), upper)
    outside = (proposed_positions < lower) | (proposed_positions > upper)
    adjusted_velocity = velocities.copy()
    adjusted_velocity[outside] *= -restitution
    return positions, adjusted_velocity


def _select_clpso_exemplars(
    personal_best_values: np.ndarray,
    particle_index: int,
    learning_probability: float,
    dimension: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Choose one pbest exemplar per coordinate using CLPSO tournament selection."""
    population_size = personal_best_values.size
    candidates = np.delete(np.arange(population_size, dtype=int), particle_index)
    exemplars = np.full(dimension, particle_index, dtype=int)
    learn_mask = rng.random(dimension) < learning_probability

    def tournament_winner() -> int:
        pair = rng.choice(candidates, size=2, replace=False)
        return int(pair[0] if personal_best_values[pair[0]] <= personal_best_values[pair[1]] else pair[1])

    for coordinate in np.flatnonzero(learn_mask):
        exemplars[coordinate] = tournament_winner()
    if np.all(exemplars == particle_index):
        coordinate = int(rng.integers(0, dimension))
        exemplars[coordinate] = tournament_winner()
    return exemplars


def _minimize_clpso(
    objective: Objective, config: PSOConfig, parameters: dict[str, float | int | str | None]
) -> PSOResult:
    """Run CLPSO with per-coordinate exemplars and generation snapshots.

    Exemplar selection and all particle proposals use the personal-best and
    current-position state from the start of a generation. Objective calls are
    then evaluated particle by particle, after all proposals have been formed.
    This avoids an accidental asynchronous information flow between particles.
    The project-wide clamp-and-restitution boundary map is retained.
    """
    if config.population_size < 3:
        raise ValueError("CLPSO requires population_size >= 3 for two-way tournament selection")

    lower = np.asarray(config.lower_bounds, dtype=float)
    upper = np.asarray(config.upper_bounds, dtype=float)
    width = upper - lower
    vmax = 0.2 * width
    rng = np.random.default_rng(config.seed)
    started = time.perf_counter()

    positions = rng.uniform(lower, upper, size=(config.population_size, config.dimension))
    initial_velocity_range = config.initial_velocity_fraction * width
    velocities = rng.uniform(
        -initial_velocity_range, initial_velocity_range,
        size=(config.population_size, config.dimension),
    )
    values = np.empty(config.population_size, dtype=float)
    for i in range(config.population_size):
        value = float(objective(tuple(float(v) for v in positions[i])))
        if not math.isfinite(value):
            raise ValueError(f"Objective returned a non-finite value during initialization of particle {i}")
        values[i] = value

    personal_best_positions = positions.copy()
    personal_best_values = values.copy()
    best_index = int(np.argmin(personal_best_values))
    global_best_value = float(personal_best_values[best_index])
    global_best_position = personal_best_positions[best_index].copy()
    function_evaluations = config.population_size
    best_history = [global_best_value]
    history_evaluations = [function_evaluations]
    no_improvement = np.zeros(config.population_size, dtype=int)

    particle_indices = np.arange(config.population_size, dtype=int)
    exponent = np.exp(10.0 * particle_indices / max(1, config.population_size - 1))
    learning_probabilities = 0.05 + 0.45 * (exponent - 1.0) / (math.exp(10.0) - 1.0)
    exemplars = np.vstack([
        _select_clpso_exemplars(
            personal_best_values, i, float(learning_probabilities[i]), config.dimension, rng
        )
        for i in range(config.population_size)
    ])

    total_update_batches = max(1, math.ceil(
        (config.max_evaluations - config.population_size) / config.population_size
    ))
    iterations_completed = 0
    while function_evaluations < config.max_evaluations:
        progress = iterations_completed / max(1, total_update_batches - 1)
        inertia = 0.9 + (0.4 - 0.9) * min(1.0, progress)
        active_count = min(config.population_size, config.max_evaluations - function_evaluations)

        # Refresh exemplars only from the previous generation's improvement
        # counters and pbest values, then freeze them for this generation.
        for i in range(config.population_size):
            if no_improvement[i] >= 7:
                exemplars[i] = _select_clpso_exemplars(
                    personal_best_values, i, float(learning_probabilities[i]), config.dimension, rng
                )
                no_improvement[i] = 0

        positions_snapshot = positions.copy()
        velocities_snapshot = velocities.copy()
        pbest_snapshot = personal_best_positions.copy()
        proposals = positions_snapshot[:active_count].copy()
        proposed_velocities = velocities_snapshot[:active_count].copy()
        for i in range(active_count):
            exemplar_positions = pbest_snapshot[exemplars[i], np.arange(config.dimension)]
            random_vector = rng.random(config.dimension)
            next_velocity = (
                inertia * velocities_snapshot[i]
                + 1.49445 * random_vector * (exemplar_positions - positions_snapshot[i])
            )
            next_velocity = np.clip(next_velocity, -vmax, vmax)
            next_position, next_velocity = _apply_clamped_restitution(
                positions_snapshot[i] + next_velocity,
                next_velocity,
                lower,
                upper,
                config.boundary_restitution,
            )
            proposals[i] = next_position
            proposed_velocities[i] = next_velocity

        next_values = np.empty(active_count, dtype=float)
        for i in range(active_count):
            value = float(objective(tuple(float(v) for v in proposals[i])))
            if not math.isfinite(value):
                raise ValueError(f"Objective returned a non-finite value at FE={function_evaluations + i + 1}")
            next_values[i] = value

        positions[:active_count] = proposals
        velocities[:active_count] = proposed_velocities
        improved = next_values < personal_best_values[:active_count]
        improved_indices = np.flatnonzero(improved)
        personal_best_values[improved_indices] = next_values[improved_indices]
        personal_best_positions[improved_indices] = proposals[improved_indices]
        no_improvement[:active_count] = np.where(
            improved, 0, no_improvement[:active_count] + 1
        )
        if active_count:
            local_best_index = int(np.argmin(personal_best_values))
            if personal_best_values[local_best_index] < global_best_value:
                global_best_value = float(personal_best_values[local_best_index])
                global_best_position = personal_best_positions[local_best_index].copy()

        function_evaluations += active_count
        iterations_completed += 1
        best_history.append(global_best_value)
        history_evaluations.append(function_evaluations)

    elapsed = time.perf_counter() - started
    return PSOResult(
        method="clpso",
        best_value=global_best_value,
        best_position=tuple(float(v) for v in global_best_position),
        best_history=tuple(best_history),
        history_evaluations=tuple(history_evaluations),
        function_evaluations=function_evaluations,
        iterations_completed=iterations_completed,
        elapsed_seconds=elapsed,
        parameters=parameters,
    )

def _minimize_slpso(
    objective: Objective, config: PSOConfig, parameters: dict[str, float | int | str | None]
) -> PSOResult:
    """Run the dimension-aware social-learning PSO of Cheng & Jin (2015).

    The swarm is ordered worst-to-best. Every imitator selects demonstrators
    only from better-ranked particles; all proposals are generated from a
    frozen generation snapshot. A shared fixed population is used to preserve
    exact FE comparability with the other methods in this project.
    """
    lower = np.asarray(config.lower_bounds, dtype=float)
    upper = np.asarray(config.upper_bounds, dtype=float)
    width = upper - lower
    rng = np.random.default_rng(config.seed)
    started = time.perf_counter()

    positions = rng.uniform(lower, upper, size=(config.population_size, config.dimension))
    # SL-PSO has no canonical PSO velocity initialization; Delta X(0)=0.
    velocities = np.zeros_like(positions)
    values = np.empty(config.population_size, dtype=float)
    for i in range(config.population_size):
        value = float(objective(tuple(float(v) for v in positions[i])))
        if not math.isfinite(value):
            raise ValueError(f"Objective returned a non-finite value during initialization of particle {i}")
        values[i] = value

    best_index = int(np.argmin(values))
    best_value = float(values[best_index])
    best_position = positions[best_index].copy()
    function_evaluations = config.population_size
    best_history = [best_value]
    history_evaluations = [function_evaluations]

    reference_M = 100
    alpha = 0.5
    beta = 0.01
    dimension_factor = max(1, math.ceil(config.dimension / reference_M))
    learning_exponent = alpha * math.log(dimension_factor)
    gamma = beta * config.dimension / reference_M
    particle_count = config.population_size
    while function_evaluations < config.max_evaluations:
        active_count = min(particle_count, config.max_evaluations - function_evaluations)
        # Descending fitness: the worst particle is first, the best is last.
        order = np.argsort(values)[::-1]
        positions_snapshot = positions.copy()
        velocities_snapshot = velocities.copy()
        mean_position = np.mean(positions_snapshot, axis=0)
        proposals = positions_snapshot.copy()
        proposed_velocities = velocities_snapshot.copy()

        for rank in range(min(active_count, particle_count - 1)):
            particle = int(order[rank])
            learning_probability = (1.0 - rank / particle_count) ** learning_exponent
            if rng.random() > learning_probability:
                continue

            # The coefficients are scalar random weights for this particle's
            # social-learning update; each coordinate has its own demonstrator.
            r1, r2, r3 = rng.random(3)
            demonstrator_ranks = rng.integers(rank + 1, particle_count, size=config.dimension)
            demonstrators = order[demonstrator_ranks]
            imitation = positions_snapshot[demonstrators, np.arange(config.dimension)] - positions_snapshot[particle]
            social_influence = gamma * (mean_position - positions_snapshot[particle])
            delta = (
                r1 * velocities_snapshot[particle]
                + r2 * imitation
                + r3 * social_influence
            )
            next_velocity = delta
            next_position, next_velocity = _apply_clamped_restitution(
                positions_snapshot[particle] + next_velocity,
                next_velocity,
                lower,
                upper,
                config.boundary_restitution,
            )
            proposals[particle] = next_position
            proposed_velocities[particle] = next_velocity

        # The FE order is deterministic (swarm order), but all coordinates in
        # each proposal came from the same frozen state.
        evaluated_particles = order[:active_count]
        next_values = np.empty(active_count, dtype=float)
        for index, particle_value in enumerate(evaluated_particles):
            particle = int(particle_value)
            value = float(objective(tuple(float(v) for v in proposals[particle])))
            if not math.isfinite(value):
                raise ValueError(f"Objective returned a non-finite value at FE={function_evaluations + index + 1}")
            next_values[index] = value

        for index, particle_value in enumerate(evaluated_particles):
            particle = int(particle_value)
            positions[particle] = proposals[particle]
            velocities[particle] = proposed_velocities[particle]
            values[particle] = next_values[index]
            if next_values[index] < best_value:
                best_value = float(next_values[index])
                best_position = proposals[particle].copy()

        function_evaluations += active_count
        best_history.append(best_value)
        history_evaluations.append(function_evaluations)

    elapsed = time.perf_counter() - started
    return PSOResult(
        method="slpso",
        best_value=best_value,
        best_position=tuple(float(v) for v in best_position),
        best_history=tuple(best_history),
        history_evaluations=tuple(history_evaluations),
        function_evaluations=function_evaluations,
        iterations_completed=len(best_history) - 1,
        elapsed_seconds=elapsed,
        parameters=parameters,
    )


def minimize_pso(
    objective: Objective,
    config: PSOConfig,
    *,
    method: str = "inertia_gbest",
) -> PSOResult:
    """Minimize a deterministic scalar black-box function under an exact FE budget.

    Function evaluations are performed synchronously. The last generation may
    contain fewer than `population_size` particle updates to consume a budget
    that is not divisible by the swarm size. For the standard COCO/CEC campaign
    configurations, budgets are chosen to be exact multiples of the population.
    """
    config.validate()
    if not callable(objective):
        raise TypeError("objective must be callable")
    parameters = pso_method_metadata(method, config)
    if method == "clpso":
        return _minimize_clpso(objective, config, parameters)
    if method == "slpso":
        return _minimize_slpso(objective, config, parameters)
    if method == "inertia_lbest_ring" and 2 * config.ring_radius + 1 > config.population_size:
        raise ValueError("ring_radius is too large for the selected population")

    lower = np.asarray(config.lower_bounds, dtype=float)
    upper = np.asarray(config.upper_bounds, dtype=float)
    width = upper - lower
    vmax = None
    if parameters["velocity_limit_fraction"] is not None:
        vmax = float(parameters["velocity_limit_fraction"]) * width
    rng = np.random.default_rng(config.seed)
    started = time.perf_counter()

    positions = rng.uniform(lower, upper, size=(config.population_size, config.dimension))
    velocities = rng.uniform(
        -config.initial_velocity_fraction * width,
        config.initial_velocity_fraction * width,
        size=(config.population_size, config.dimension),
    )
    values = np.empty(config.population_size, dtype=float)
    for i in range(config.population_size):
        value = float(objective(tuple(float(v) for v in positions[i])))
        if not math.isfinite(value):
            raise ValueError(f"Objective returned a non-finite value during initialization of particle {i}")
        values[i] = value

    personal_best_positions = positions.copy()
    personal_best_values = values.copy()
    best_index = int(np.argmin(personal_best_values))
    global_best_value = float(personal_best_values[best_index])
    global_best_position = personal_best_positions[best_index].copy()
    function_evaluations = config.population_size
    best_history = [global_best_value]
    history_evaluations = [function_evaluations]
    iterations_completed = 0

    c1 = float(parameters["cognitive"])
    c2 = float(parameters["social"])
    chi = float(parameters["constriction_factor"])
    w_max = float(parameters["inertia_max"])
    w_min = float(parameters["inertia_min"])
    total_update_batches = max(1, math.ceil(
        (config.max_evaluations - config.population_size) / config.population_size
    ))

    while function_evaluations < config.max_evaluations:
        active_count = min(
            config.population_size,
            config.max_evaluations - function_evaluations,
        )
        social_positions = _social_attractors(
            personal_best_positions, personal_best_values, method, config.ring_radius
        )
        if total_update_batches <= 1:
            inertia = w_max
        else:
            progress = iterations_completed / max(1, total_update_batches - 1)
            inertia = w_max + (w_min - w_max) * min(1.0, progress)

        cognitive_random = rng.random((active_count, config.dimension))
        social_random = rng.random((active_count, config.dimension))
        attraction = (
            c1 * cognitive_random * (personal_best_positions[:active_count] - positions[:active_count])
            + c2 * social_random * (social_positions[:active_count] - positions[:active_count])
        )
        if method == "constriction_gbest":
            next_velocities = chi * (velocities[:active_count] + attraction)
        else:
            next_velocities = chi * (inertia * velocities[:active_count] + attraction)
        if vmax is not None:
            next_velocities = np.clip(next_velocities, -vmax, vmax)

        next_positions, next_velocities = _apply_clamped_restitution(
            positions[:active_count] + next_velocities,
            next_velocities,
            lower,
            upper,
            config.boundary_restitution,
        )
        positions[:active_count] = next_positions
        velocities[:active_count] = next_velocities

        for i in range(active_count):
            value = float(objective(tuple(float(v) for v in positions[i])))
            if not math.isfinite(value):
                raise ValueError(
                    f"Objective returned a non-finite value at FE={function_evaluations + 1}"
                )
            function_evaluations += 1
            if value < personal_best_values[i]:
                personal_best_values[i] = value
                personal_best_positions[i] = positions[i]
            if value < global_best_value:
                global_best_value = value
                global_best_position = positions[i].copy()

        iterations_completed += 1
        best_history.append(global_best_value)
        history_evaluations.append(function_evaluations)

    elapsed = time.perf_counter() - started
    return PSOResult(
        method=method,
        best_value=global_best_value,
        best_position=tuple(float(v) for v in global_best_position),
        best_history=tuple(best_history),
        history_evaluations=tuple(history_evaluations),
        function_evaluations=function_evaluations,
        iterations_completed=iterations_completed,
        elapsed_seconds=elapsed,
        parameters=parameters,
    )


__all__ = [
    "Objective", "PSOConfig", "PSOResult", "available_pso_methods",
    "minimize_pso", "pso_method_metadata",
]
