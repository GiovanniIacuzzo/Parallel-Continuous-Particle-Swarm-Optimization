from __future__ import annotations

import ctypes
import os
from pathlib import Path
from typing import Optional

from .config import CpsoConfig
from .result import CpsoResult


_ABI_VERSION = 5
_TOPOLOGY_IDS = {"global": 0, "ring": 1, "grid_von_neumann": 2}
_BENCHMARK_IDS = {
    "sphere": 1,
    "rastrigin": 2,
    "rosenbrock": 3,
    "ackley": 4,
    "griewank": 5,
    "salomon": 6,
    "sphere_work": 7,
}
_ARCHITECTURE_IDS = {
    "serial": 0,
    "master_slave": 1,
    "particle_parallel": 2,
    "asynchronous": 3,
    "islands": 4,
}
_MIGRATION_TOPOLOGY_IDS = {"ring": 0, "fully_connected": 1}


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
        ("topology_id", ctypes.c_int),
        ("grid_rows", ctypes.c_int),
        ("grid_columns", ctypes.c_int),
        ("objective_work_iterations", ctypes.c_uint64),
    ]


class _CExecutionConfig(ctypes.Structure):
    _fields_ = [
        ("abi_version", ctypes.c_uint32),
        ("algorithm_id", ctypes.c_int),
        ("worker_count", ctypes.c_size_t),
        ("island_count", ctypes.c_size_t),
        ("migration_interval", ctypes.c_size_t),
        ("migration_topology_id", ctypes.c_int),
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
        ("topology_setup_seconds", ctypes.c_double),
        ("social_selection_seconds", ctypes.c_double),
        ("worker_count", ctypes.c_size_t),
        ("initialization_seconds", ctypes.c_double),
        ("propagation_worker_seconds", ctypes.c_double),
        ("evaluation_worker_seconds", ctypes.c_double),
        ("migration_seconds", ctypes.c_double),
        ("async_update_wall_seconds", ctypes.c_double),
        ("migrations_performed", ctypes.c_uint64),
        ("island_count", ctypes.c_size_t),
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
            library.cpso_run_architecture.argtypes = [
                ctypes.POINTER(_CConfig),
                ctypes.POINTER(_CExecutionConfig),
                ctypes.POINTER(_CResult),
                ctypes.POINTER(ctypes.c_double),
                ctypes.POINTER(ctypes.c_double),
                ctypes.POINTER(ctypes.c_uint64),
                ctypes.c_size_t,
                ctypes.POINTER(ctypes.c_char),
                ctypes.c_size_t,
            ]
            library.cpso_run_architecture.restype = ctypes.c_int
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
    architecture: str,
    workers: Optional[int] = None,
    islands: int = 0,
    migration_interval: int = 0,
    migration_topology: str = "ring",
    work_iterations: int = 0,
) -> CpsoResult:
    config.validate()
    if benchmark not in _BENCHMARK_IDS:
        raise ValueError(f"Unsupported benchmark {benchmark!r}; choose from {sorted(_BENCHMARK_IDS)}")
    if architecture not in _ARCHITECTURE_IDS:
        raise ValueError(f"Unsupported architecture {architecture!r}")
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

    if architecture == "serial":
        workers = 1
    elif workers is None:
        raise ValueError(f"workers must be provided for architecture={architecture!r}")
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise ValueError("workers must be a positive integer")
    if workers > 4_096:
        raise ValueError("workers exceeds the supported safety limit (4096)")

    if architecture == "islands":
        if isinstance(islands, bool) or not isinstance(islands, int) or islands < 2:
            raise ValueError("islands must be an integer >= 2 for the island architecture")
        if islands > config.population_size or config.population_size % islands != 0:
            raise ValueError("population_size must be divisible by islands, with at least one particle per island")
        if config.population_size // islands < 2:
            raise ValueError("Each island must contain at least two particles")
        if workers > islands:
            raise ValueError("workers cannot exceed islands in the island architecture")
        if config.topology == "grid_von_neumann":
            raise ValueError("grid_von_neumann is not currently supported as the inner island topology")
        if isinstance(migration_interval, bool) or not isinstance(migration_interval, int) or migration_interval < 0:
            raise ValueError("migration_interval must be a non-negative integer")
        if migration_topology not in _MIGRATION_TOPOLOGY_IDS:
            raise ValueError("migration_topology must be 'ring' or 'fully_connected'")
    else:
        if islands not in (0, 1):
            raise ValueError("islands is only valid for architecture='islands'")
        if migration_interval != 0:
            raise ValueError("migration_interval is only valid for architecture='islands'")

    lower = (ctypes.c_double * config.dimension)(*(float(x) for x in config.lower_bounds))
    upper = (ctypes.c_double * config.dimension)(*(float(x) for x in config.upper_bounds))
    best_position = (ctypes.c_double * config.dimension)()
    if architecture == "asynchronous":
        history_capacity = config.population_size * config.iterations + 1
    else:
        history_capacity = config.iterations + 1
    history = (ctypes.c_double * history_capacity)()
    history_evaluations = (ctypes.c_uint64 * history_capacity)()
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
        topology_id=_TOPOLOGY_IDS[config.topology],
        grid_rows=config.grid_rows or 0,
        grid_columns=config.grid_columns or 0,
        objective_work_iterations=work_iterations,
    )
    execution = _CExecutionConfig(
        abi_version=_ABI_VERSION,
        algorithm_id=_ARCHITECTURE_IDS[architecture],
        worker_count=workers,
        island_count=islands,
        migration_interval=migration_interval,
        migration_topology_id=_MIGRATION_TOPOLOGY_IDS.get(migration_topology, 0),
    )
    c_result = _CResult()
    library = _load_library()
    status = library.cpso_run_architecture(
        ctypes.byref(c_config), ctypes.byref(execution), ctypes.byref(c_result),
        best_position, history, history_evaluations, history_capacity,
        error, len(error),
    )
    if status != 0:
        message = error.value.decode("utf-8", errors="replace") or "Unknown native error"
        raise RuntimeError(f"CPSO native execution failed: {message}")

    history_size = int(c_result.history_size)
    if history_size > history_capacity:
        raise RuntimeError("Native library returned an invalid history_size")
    history_values = tuple(float(history[i]) for i in range(history_size))
    history_fe = tuple(int(history_evaluations[i]) for i in range(history_size))
    if len(history_values) != len(history_fe):
        raise RuntimeError("Native history and evaluation axes have different lengths")

    return CpsoResult(
        best_value=float(c_result.best_value),
        best_position=tuple(float(best_position[i]) for i in range(config.dimension)),
        best_history=history_values,
        history_evaluations=history_fe,
        function_evaluations=int(c_result.function_evaluations),
        iterations_completed=int(c_result.iterations_completed),
        elapsed_seconds=float(c_result.elapsed_seconds),
        evaluation_wall_seconds=float(c_result.evaluation_wall_seconds),
        propagation_seconds=float(c_result.propagation_seconds),
        topology_setup_seconds=float(c_result.topology_setup_seconds),
        social_selection_seconds=float(c_result.social_selection_seconds),
        initialization_seconds=float(c_result.initialization_seconds),
        propagation_worker_seconds=float(c_result.propagation_worker_seconds),
        evaluation_worker_seconds=float(c_result.evaluation_worker_seconds),
        migration_seconds=float(c_result.migration_seconds),
        async_update_wall_seconds=float(c_result.async_update_wall_seconds),
        migrations_performed=int(c_result.migrations_performed),
        worker_count=int(c_result.worker_count),
        island_count=int(c_result.island_count),
    )


def run_serial(
    config: CpsoConfig,
    benchmark: str = "sphere",
    *,
    work_iterations: int = 0,
) -> CpsoResult:
    """Run the serial synchronous reference implementation."""
    return _run(config, benchmark, architecture="serial", work_iterations=work_iterations)


def run_master_slave(
    config: CpsoConfig,
    benchmark: str = "sphere",
    workers: int = 2,
    *,
    work_iterations: int = 0,
) -> CpsoResult:
    """Parallelize only objective evaluation; preserve the synchronous trajectory."""
    return _run(
        config, benchmark, architecture="master_slave", workers=workers,
        work_iterations=work_iterations,
    )


def run_particle_parallel(
    config: CpsoConfig,
    benchmark: str = "sphere",
    workers: int = 2,
    *,
    work_iterations: int = 0,
) -> CpsoResult:
    """Parallelize particle initialization/propagation and evaluation batches."""
    return _run(
        config, benchmark, architecture="particle_parallel", workers=workers,
        work_iterations=work_iterations,
    )


def run_asynchronous(
    config: CpsoConfig,
    benchmark: str = "sphere",
    workers: int = 2,
    *,
    work_iterations: int = 0,
) -> CpsoResult:
    """Run event-driven asynchronous CPSO with immediate social-state publication."""
    return _run(
        config, benchmark, architecture="asynchronous", workers=workers,
        work_iterations=work_iterations,
    )


def run_island_model(
    config: CpsoConfig,
    benchmark: str = "sphere",
    *,
    islands: int = 4,
    workers: int = 4,
    migration_interval: int = 0,
    migration_topology: str = "ring",
    work_iterations: int = 0,
) -> CpsoResult:
    """Run parallel independent islands with synchronous migration barriers."""
    return _run(
        config, benchmark, architecture="islands", workers=workers,
        islands=islands, migration_interval=migration_interval,
        migration_topology=migration_topology, work_iterations=work_iterations,
    )


def native_version() -> str:
    library = _load_library()
    value: Optional[bytes] = library.cpso_version()
    return value.decode("ascii") if value else "unknown"


def native_build_info() -> str:
    library = _load_library()
    value: Optional[bytes] = library.cpso_build_info()
    return value.decode("utf-8") if value else "unknown"
