"""Adapter for the external CEC 2013 Large-Scale Global Optimization suite.

The benchmark implementation is not vendored. Install a compatible
`cec2013lsgo` distribution separately and review its GPL-3.0 license before
redistributing an environment that includes it.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class CEC2013LSGOProblem:
    function_id: int
    dimension: int
    lower_bounds: tuple[float, ...]
    upper_bounds: tuple[float, ...]
    optimum: float
    objective: Callable[[tuple[float, ...]], float]
    name: str


class CEC2013LSGOAdapter:
    """Load function definitions from the optional `cec2013lsgo` package."""

    def __init__(self) -> None:
        try:
            from cec2013lsgo.cec2013 import Benchmark  # type: ignore[import-not-found]
        except ImportError as exc:
            raise ImportError(
                "CEC 2013 LSGO support requires the external 'cec2013lsgo' package. "
                "Install it separately and review its GPL-3.0 license first."
            ) from exc
        self._benchmark_type = Benchmark
        try:
            self._benchmark = Benchmark()
        except TypeError:
            self._benchmark = None

    def get_problem(self, function_id: int) -> CEC2013LSGOProblem:
        if isinstance(function_id, bool) or not isinstance(function_id, int) or not 1 <= function_id <= 15:
            raise ValueError("CEC 2013 LSGO function_id must be an integer in [1,15]")

        benchmark = self._benchmark
        info = None
        if benchmark is not None and hasattr(benchmark, "get_info"):
            try:
                info = benchmark.get_info(function_id)
            except TypeError:
                info = None
        if info is None:
            # Some compatible wrappers bind one function in the constructor
            # and expose get_info()/eval() without a function-ID argument.
            benchmark = self._benchmark_type(function_id)
            if not hasattr(benchmark, "get_info"):
                raise RuntimeError("Unsupported cec2013lsgo wrapper: missing get_info")
            info = benchmark.get_info()

        dimension = int(info["dimension"])
        lower = float(info["lower"])
        upper = float(info["upper"])
        optimum = float(info.get("best", 0.0))
        if dimension < 1 or not lower < upper:
            raise RuntimeError(f"Invalid metadata returned for CEC function F{function_id}")

        if hasattr(benchmark, "get_function"):
            try:
                fitness = benchmark.get_function(function_id)
            except TypeError:
                fitness = benchmark.get_function()
        elif hasattr(benchmark, "eval"):
            fitness = benchmark.eval
        else:
            raise RuntimeError("Unsupported cec2013lsgo wrapper: missing get_function/eval")

        def objective(point: tuple[float, ...]) -> float:
            if len(point) != dimension:
                raise ValueError(f"CEC F{function_id} expects dimension {dimension}, got {len(point)}")
            # The commonly used Cython wrapper documents NumPy vectors as its
            # input type. NumPy is an optional dependency of the external
            # benchmark layer, not of the core CPSO package.
            try:
                import numpy as np
            except ImportError as exc:
                raise ImportError(
                    "CEC 2013 LSGO objective evaluation requires NumPy; install "
                    "the optional benchmark requirements"
                ) from exc
            vector = np.asarray(point, dtype=np.float64)
            return float(fitness(vector))

        return CEC2013LSGOProblem(
            function_id=function_id,
            dimension=dimension,
            lower_bounds=(lower,) * dimension,
            upper_bounds=(upper,) * dimension,
            optimum=optimum,
            objective=objective,
            name=f"CEC2013-LSGO-F{function_id:02d}",
        )


__all__ = ["CEC2013LSGOAdapter", "CEC2013LSGOProblem"]
