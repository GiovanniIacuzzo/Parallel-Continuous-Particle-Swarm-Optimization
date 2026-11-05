from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class CpsoResult:
    best_value: float
    best_position: Tuple[float, ...]
    best_history: Tuple[float, ...]
    history_evaluations: Tuple[int, ...]
    function_evaluations: int
    iterations_completed: int
    elapsed_seconds: float
    evaluation_wall_seconds: float
    propagation_seconds: float
    topology_setup_seconds: float
    social_selection_seconds: float
    initialization_seconds: float
    propagation_worker_seconds: float
    evaluation_worker_seconds: float
    migration_seconds: float
    async_update_wall_seconds: float
    migrations_performed: int
    worker_count: int
    island_count: int
