"""Collect non-identifying software and hardware provenance for experiments."""
from __future__ import annotations

import importlib.metadata
import os
import platform
import sys
from typing import Any


_THREAD_ENVIRONMENT = (
    "OMP_NUM_THREADS",
    "OMP_DYNAMIC",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)
_PACKAGES = (
    "numpy",
    "scipy",
    "matplotlib",
    "coco-experiment",
    "pytest",
)


def _cpu_model() -> str | None:
    model = platform.processor().strip()
    if model:
        return model
    # Linux fallback; macOS/Windows deliberately avoid shelling out to sysctl or
    # WMIC so metadata collection remains side-effect free and portable.
    cpuinfo = "/proc/cpuinfo"
    try:
        with open(cpuinfo, "r", encoding="utf-8", errors="replace") as stream:
            for line in stream:
                if line.lower().startswith(("model name", "hardware", "processor")) and ":" in line:
                    candidate = line.split(":", 1)[1].strip()
                    if candidate:
                        return candidate
    except OSError:
        pass
    return None


def _package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for package in _PACKAGES:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def collect_environment_metadata(
    *,
    native_version: str | None = None,
    native_build_info: str | None = None,
) -> dict[str, Any]:
    """Return stable provenance fields without recording hostnames or paths."""
    logical_cpus = os.cpu_count()
    physical_cpus: int | None = None
    try:
        import psutil  # type: ignore[import-not-found]

        physical_cpus = psutil.cpu_count(logical=False)
    except (ImportError, OSError):
        pass

    return {
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": _cpu_model(),
            "platform_string": platform.platform(),
        },
        "python": {
            "version": sys.version,
            "implementation": platform.python_implementation(),
        },
        "cpu": {
            "logical_count": logical_cpus,
            "physical_count": physical_cpus,
        },
        "thread_environment": {
            key: os.environ.get(key) for key in _THREAD_ENVIRONMENT
        },
        "packages": _package_versions(),
        "native": {
            "version": native_version,
            "build_info": native_build_info,
        },
    }
