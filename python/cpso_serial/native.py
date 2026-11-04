from __future__ import annotations

import ctypes
import os
from pathlib import Path
from typing import Optional

from .config import CpsoConfig
from .result import CpsoResult


_ABI_VERSION = 3
_BENCHMARK_IDS = {
    "sphere": 1,
    "rastrigin": 2,
    "rosenbrock": 3,
    "ackley": 4,
    "griewank": 5,
    "salomon": 6,
    "sphere_work": 7,
}


class _CConfig(ctypes.Structure):
    _fields_ = [
        ("abi_version", ctypes.c_uint32),
        ("benchmark_id", ctypes.c_int),
        ("dimension", ctypes.c_int),
        ("population_size", ctypes.c_int),
        ("iterations", ctypes.c_int),
        ("seed", ctypes.c_uint64),
        ("lower_bounds", ctypes.POINTER(ctypes.c_double)),
        ("upper_bounds", ctypes.POINTER(ctypes.c_double)),
        ("mu_max", ctypes.c_double),
        ("mu_min", ctypes.c_double),
        ("cognitive", ctypes.c_double),
        ("social", ctypes.c_double),
        ("dt", ctypes.c_double),
        ("initial_velocity_fraction", ctypes.c_double),
        ("boundary_restitution", ctypes.c_double),
        ("objective_work_iterations", ctypes.c_uint64),
    ]


class _CResult(ctypes.Structure):
    _fields_ = [
        ("best_value", ctypes.c_double),
        ("function_evaluations", ctypes.c_uint64),
        ("iterations_completed", ctypes.c_int),
        ("history_size", ctypes.c_size_t),
        ("elapsed_seconds", ctypes.c_double),
        ("evaluation_wall_seconds", ctypes.c_double),
        ("propagation_seconds", ctypes.c_double),
        ("worker_count", ctypes.c_size_t),
    ]


def _library_candidates() -> list[Path]:
    override = os.environ.get("CPSO_SERIAL_LIBRARY")
    if override:
        return [Path(override).expanduser().resolve()]

    package_root = Path(__file__).resolve().parents[2]
    build_root = package_root / "build"
    names = ("libcpso_serial.dylib", "libcpso_serial.so", "cpso_serial.dll")
    candidates: list[Path] = []
    for name in names:
        candidates.extend((build_root / "lib" / name, build_root / name))
    if (build_root / "lib").is_dir():
        candidates.extend(sorted((build_root / "lib").glob("libcpso_serial.*")))
    return candidates


def _load_library() -> ctypes.CDLL:
    candidates = _library_candidates()
    for candidate in candidates:
        if candidate.is_file():
            library = ctypes.CDLL(str(candidate))
            library.cpso_abi_version.restype = ctypes.c_uint32
            library.cpso_version.restype = ctypes.c_char_p
            library.cpso_build_info.restype = ctypes.c_char_p
            common_argtypes = [
                ctypes.POINTER(_CConfig),
                ctypes.POINTER(_CResult),
                ctypes.POINTER(ctypes.c_double),
                ctypes.POINTER(ctypes.c_double),
                ctypes.c_size_t,
                ctypes.POINTER(ctypes.c_char),
                ctypes.c_size_t,
            ]
            library.cpso_run_serial.argtypes = common_argtypes
            library.cpso_run_serial.restype = ctypes.c_int
            library.cpso_run_master_slave.argtypes = [
                ctypes.POINTER(_CConfig),
                ctypes.c_size_t,
                ctypes.POINTER(_CResult),
                ctypes.POINTER(ctypes.c_double),
                ctypes.POINTER(ctypes.c_double),
                ctypes.c_size_t,
                ctypes.POINTER(ctypes.c_char),
                ctypes.c_size_t,
            ]
            library.cpso_run_master_slave.restype = ctypes.c_int
            actual_abi = int(library.cpso_abi_version())
            if actual_abi != _ABI_VERSION:
                raise RuntimeError(
                    f"Native ABI mismatch: Python expects {_ABI_VERSION}, library provides {actual_abi}"
                )
            return library

    attempted = "\n".join(str(path) for path in candidates)
    raise FileNotFoundError(
        "Could not find the CPSO native library. Build it with CMake first or set "
        "CPSO_SERIAL_LIBRARY. Searched:\n" + attempted
    )


def _run(
    config: CpsoConfig,
    benchmark: str,
    *,
    workers: Optional[int],
    work_iterations: int = 0,
) -> CpsoResult:
    config.validate()
    if benchmark not in _BENCHMARK_IDS:
        raise ValueError(f"Unsupported benchmark {benchmark!r}; choose from {sorted(_BENCHMARK_IDS)}")
    if isinstance(work_iterations, bool) or not isinstance(work_iterations, int):
        raise ValueError("work_iterations must be an integer")
    if not 0 <= work_iterations <= 100_000_000:
        raise ValueError("work_iterations must be in [0,100000000]")
    if benchmark != "sphere_work" and work_iterations != 0:
        raise ValueError("work_iterations is only valid for benchmark='sphere_work'")
    if config.dimension > 2_147_483_647 or config.population_size > 2_147_483_647:
        raise ValueError("dimension and population_size must fit a signed 32-bit integer")
    if config.iterations > 2_147_483_646:
        raise ValueError("iterations exceeds the native API limit")
    if config.seed > (1 << 64) - 1:
        raise ValueError("seed must fit an unsigned 64-bit integer")
    if workers is not None:
        if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
            raise ValueError("workers must be a positive integer")
        if workers > 4_096:
            raise ValueError("workers exceeds the supported safety limit (4096)")

    lower = (ctypes.c_double * config.dimension)(*(float(x) for x in config.lower_bounds))
    upper = (ctypes.c_double * config.dimension)(*(float(x) for x in config.upper_bounds))
    best_position = (ctypes.c_double * config.dimension)()
    history_capacity = config.iterations + 1
    history = (ctypes.c_double * history_capacity)()
    error = ctypes.create_string_buffer(2048)

    c_config = _CConfig(
        abi_version=_ABI_VERSION,
        benchmark_id=_BENCHMARK_IDS[benchmark],
        dimension=config.dimension,
        population_size=config.population_size,
        iterations=config.iterations,
        seed=config.seed,
        lower_bounds=lower,
        upper_bounds=upper,
        mu_max=config.mu_max,
        mu_min=config.mu_min,
        cognitive=config.cognitive,
        social=config.social,
        dt=config.dt,
        initial_velocity_fraction=config.initial_velocity_fraction,
        boundary_restitution=config.boundary_restitution,
        objective_work_iterations=work_iterations,
    )
    c_result = _CResult()
    library = _load_library()
    if workers is None:
        status = library.cpso_run_serial(
            ctypes.byref(c_config), ctypes.byref(c_result), best_position,
            history, history_capacity, error, len(error),
        )
    else:
        status = library.cpso_run_master_slave(
            ctypes.byref(c_config), workers, ctypes.byref(c_result), best_position,
            history, history_capacity, error, len(error),
        )
    if status != 0:
        message = error.value.decode("utf-8", errors="replace") or "Unknown native error"
        raise RuntimeError(f"CPSO native execution failed: {message}")

    history_size = int(c_result.history_size)
    if history_size > history_capacity:
        raise RuntimeError("Native library returned an invalid history_size")
    return CpsoResult(
        best_value=float(c_result.best_value),
        best_position=tuple(float(best_position[i]) for i in range(config.dimension)),
        best_history=tuple(float(history[i]) for i in range(history_size)),
        function_evaluations=int(c_result.function_evaluations),
        iterations_completed=int(c_result.iterations_completed),
        elapsed_seconds=float(c_result.elapsed_seconds),
        evaluation_wall_seconds=float(c_result.evaluation_wall_seconds),
        propagation_seconds=float(c_result.propagation_seconds),
        worker_count=int(c_result.worker_count),
    )


def run_serial(
    config: CpsoConfig,
    benchmark: str = "sphere",
    *,
    work_iterations: int = 0,
) -> CpsoResult:
    """Run the serial synchronous reference implementation."""
    return _run(config, benchmark, workers=None, work_iterations=work_iterations)


def run_master_slave(
    config: CpsoConfig,
    benchmark: str = "sphere",
    workers: int = 2,
    *,
    work_iterations: int = 0,
) -> CpsoResult:
    """Run synchronous master--slave CPSO with parallel objective batches.

    Particle propagation and state commits remain serial and identical to the
    reference. Objective callables in the native C++ API must be thread-safe;
    this Python API currently exposes only built-in pure benchmark functions.
    """
    return _run(config, benchmark, workers=workers, work_iterations=work_iterations)


def native_version() -> str:
    library = _load_library()
    value: Optional[bytes] = library.cpso_version()
    return value.decode("ascii") if value else "unknown"


def native_build_info() -> str:
    library = _load_library()
    value: Optional[bytes] = library.cpso_build_info()
    return value.decode("utf-8") if value else "unknown"
