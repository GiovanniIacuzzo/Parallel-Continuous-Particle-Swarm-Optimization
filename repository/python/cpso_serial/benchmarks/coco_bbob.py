"""COCO/BBOB suite helpers. The external `coco-experiment` package is optional."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class COCOProblemView:
    problem_id: str
    name: str
    dimension: int
    lower_bounds: tuple[float, ...]
    upper_bounds: tuple[float, ...]
    objective: Any
    coco_problem: Any


def load_coco_experiment():
    try:
        import cocoex  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ImportError(
            "COCO/BBOB support requires the optional 'coco-experiment' package. "
            "Install it with: python -m pip install 'cpso-parallel[coco]'"
        ) from exc
    return cocoex


def wrap_coco_problem(problem: Any) -> COCOProblemView:
    """Validate and normalize the public fields exposed by a `cocoex` problem."""
    try:
        dimension = int(problem.dimension)
        lower = tuple(float(value) for value in problem.lower_bounds)
        upper = tuple(float(value) for value in problem.upper_bounds)
        problem_id = str(problem.id)
        name = str(problem.name)
    except (AttributeError, TypeError, ValueError) as exc:
        raise RuntimeError("The installed cocoex problem object lacks required public attributes") from exc
    if dimension < 1 or len(lower) != dimension or len(upper) != dimension:
        raise RuntimeError(f"Invalid bounds/dimension for COCO problem {problem_id!r}")
    if any(not lo < hi for lo, hi in zip(lower, upper)):
        raise RuntimeError(f"Invalid search bounds for COCO problem {problem_id!r}")
    return COCOProblemView(
        problem_id=problem_id,
        name=name,
        dimension=dimension,
        lower_bounds=lower,
        upper_bounds=upper,
        objective=problem,
        coco_problem=problem,
    )


def make_bbob_suite(cocoex, *, suite_name: str = "bbob", instances: str, options: str):
    """Create a fresh COCO suite; a fresh suite is needed for every algorithm."""
    supported = {"bbob", "bbob-largescale"}
    if suite_name not in supported:
        raise ValueError(f"Unsupported COCO suite {suite_name!r}; choose from {sorted(supported)}")
    if suite_name not in getattr(cocoex, "known_suite_names", []):
        raise RuntimeError(f"The installed COCO package does not expose the {suite_name!r} suite")
    return cocoex.Suite(suite_name, instances, options)


def make_bbob_observer(
    cocoex, *, suite_name: str = "bbob", output_directory: str, algorithm_name: str
):
    if suite_name not in {"bbob", "bbob-largescale"}:
        raise ValueError(f"Unsupported COCO observer suite {suite_name!r}")
    return cocoex.Observer(
        suite_name,
        f"result_folder: {output_directory} algorithm_name: {algorithm_name}",
    )


__all__ = [
    "COCOProblemView", "load_coco_experiment", "make_bbob_observer",
    "make_bbob_suite", "wrap_coco_problem",
]
