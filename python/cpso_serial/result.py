from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class CpsoResult:
    best_value: float
    best_position: Tuple[float, ...]
    best_history: Tuple[float, ...]
    function_evaluations: int
    iterations_completed: int
    elapsed_seconds: float
    evaluation_wall_seconds: float
    propagation_seconds: float
    worker_count: int

