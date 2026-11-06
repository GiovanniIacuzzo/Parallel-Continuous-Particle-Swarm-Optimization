from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence


@dataclass(frozen=True)
class CpsoConfig:
    lower_bounds: Sequence[float]
    upper_bounds: Sequence[float]
    population_size: int = 40
    iterations: int = 100
    seed: int = 1
    mu_max: float = 0.9
    mu_min: float = 0.4
    cognitive: float = 2.05
    social: float = 2.05
    dt: float = 0.7
    initial_velocity_fraction: float = 0.1
    boundary_restitution: float = 0.5
    topology: str = "global"
    grid_rows: int | None = None
    grid_columns: int | None = None
    topology_degree: int = 4
    topology_rewiring_probability: float = 0.15

    @property
    def dimension(self) -> int:
        return len(self.lower_bounds)

    def validate(self) -> None:
        lower = tuple(float(v) for v in self.lower_bounds)
        upper = tuple(float(v) for v in self.upper_bounds)
        if not lower:
            raise ValueError("At least one decision variable is required")
        if len(lower) != len(upper):
            raise ValueError("lower_bounds and upper_bounds must have equal lengths")
        for lo, hi in zip(lower, upper):
            if not (math.isfinite(lo) and math.isfinite(hi) and lo < hi
                    and math.isfinite(hi - lo)):
                raise ValueError("Every bound must be finite, have finite width, and satisfy lower < upper")
        if isinstance(self.population_size, bool) or not isinstance(self.population_size, int) or self.population_size < 1:
            raise ValueError("population_size must be a positive integer")
        if isinstance(self.iterations, bool) or not isinstance(self.iterations, int) or self.iterations < 0:
            raise ValueError("iterations must be a non-negative integer")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or not (0 <= self.seed <= (1 << 64) - 1):
            raise ValueError("seed must be an unsigned 64-bit integer")
        if not (math.isfinite(self.mu_min) and math.isfinite(self.mu_max)
                and 0.0 <= self.mu_min <= self.mu_max <= 1.0):
            raise ValueError("Require 0 <= mu_min <= mu_max <= 1")
        if not (math.isfinite(self.cognitive) and math.isfinite(self.social)
                and self.cognitive >= 0.0 and self.social >= 0.0
                and self.cognitive + self.social > 0.0):
            raise ValueError("cognitive/social must be finite, non-negative, and not both zero")
        if not (math.isfinite(self.dt) and self.dt > 0.0):
            raise ValueError("dt must be finite and > 0")
        if not (math.isfinite(self.initial_velocity_fraction)
                and self.initial_velocity_fraction >= 0.0):
            raise ValueError("initial_velocity_fraction must be finite and >= 0")
        if any(not math.isfinite(self.initial_velocity_fraction * (hi - lo))
               for lo, hi in zip(lower, upper)):
            raise ValueError("initial velocity range overflows for at least one coordinate")
        if not (math.isfinite(self.boundary_restitution)
                and 0.0 <= self.boundary_restitution <= 1.0):
            raise ValueError("boundary_restitution must lie in [0,1]")
        if not isinstance(self.topology, str) or self.topology not in {
            "global", "ring", "grid_von_neumann", "wheel", "random",
            "small_world", "scale_free"
        }:
            raise ValueError(
                "topology must be global, ring, grid_von_neumann, wheel, random, small_world, or scale_free"
            )
        if isinstance(self.topology_degree, bool) or not isinstance(self.topology_degree, int):
            raise ValueError("topology_degree must be an integer")
        if not (math.isfinite(self.topology_rewiring_probability)
                and 0.0 <= self.topology_rewiring_probability <= 1.0):
            raise ValueError("topology_rewiring_probability must lie in [0,1]")
        if self.topology == "grid_von_neumann":
            for name, value in (("grid_rows", self.grid_rows), ("grid_columns", self.grid_columns)):
                if isinstance(value, bool) or not isinstance(value, int) or value < 2:
                    raise ValueError(f"{name} must be an integer >= 2 for grid topology")
            if self.grid_rows * self.grid_columns != self.population_size:
                raise ValueError("grid_rows * grid_columns must equal population_size")
        elif self.grid_rows is not None or self.grid_columns is not None:
            raise ValueError("grid_rows/grid_columns may only be set for grid_von_neumann topology")
        if self.topology == "wheel" and self.population_size < 2:
            raise ValueError("wheel topology requires population_size >= 2")
        if self.topology == "random":
            if not 1 <= self.topology_degree < self.population_size:
                raise ValueError("random topology requires 1 <= topology_degree < population_size")
        if self.topology == "small_world":
            if (self.population_size < 4 or self.topology_degree < 2
                    or self.topology_degree >= self.population_size or self.topology_degree % 2 != 0):
                raise ValueError("small_world requires even 2 <= topology_degree < population_size")
        if self.topology == "scale_free":
            if (self.topology_degree < 2 or self.topology_degree % 2 != 0
                    or self.population_size < self.topology_degree // 2 + 1):
                raise ValueError("scale_free requires even topology_degree >= 2 and population_size >= degree/2 + 1")
