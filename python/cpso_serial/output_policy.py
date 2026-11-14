"""All command-line generated files must remain inside the repository results/ tree.

Only users may copy selected PDF figures into paper/. Never write there from a runner.
"""
from __future__ import annotations
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = REPOSITORY_ROOT / "results"

def results_output(path: str | Path, *, name: str = "output") -> Path:
    """Resolve an output path and reject anything outside repository results/."""
    value = Path(path).expanduser().resolve()
    try:
        relative = value.relative_to(RESULTS_ROOT.resolve())
    except ValueError as exc:
        raise ValueError(
            f"{name} must be inside {RESULTS_ROOT}. Received: {value}. "
            "The paper/ directory is read-only for all project scripts."
        ) from exc
    if not relative.parts:
        raise ValueError(f"{name} must name a subdirectory within results/, not its root")
    return value
