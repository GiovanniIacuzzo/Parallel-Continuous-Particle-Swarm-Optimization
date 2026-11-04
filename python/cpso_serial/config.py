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
