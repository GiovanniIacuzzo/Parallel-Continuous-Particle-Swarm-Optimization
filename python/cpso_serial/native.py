from __future__ import annotations

from contextlib import contextmanager
import ctypes
import os
import math
from pathlib import Path
from typing import Callable, Optional, Sequence

from .config import CpsoConfig
from .result import CpsoResult


_ABI_VERSION = 7
_TOPOLOGY_IDS = {
    "global": 0,
    "ring": 1,
    "grid_von_neumann": 2,
    "wheel": 3,
    "random": 4,
    "small_world": 5,
    "scale_free": 6,
}
_BENCHMARK_IDS = {
    "sphere": 1,
    "rastrigin": 2,
    "rosenbrock": 3,
    "ackley": 4,
    "griewank": 5,
    "salomon": 6,
    "sphere_work": 7,
    "inverse_diffusion": 8,
}
_ARCHITECTURE_IDS = {
    "serial": 0,
    "master_slave": 1,
    "particle_parallel": 2,
    "asynchronous": 3,
    "islands": 4,
}
_MIGRATION_TOPOLOGY_IDS = {"ring": 0, "fully_connected": 1}


_ObjectiveCallback = ctypes.CFUNCTYPE(
    ctypes.c_double,
    ctypes.POINTER(ctypes.c_double),
    ctypes.c_size_t,
    ctypes.c_void_p,
)


class _CSynchronousStudy(ctypes.Structure):
    _fields_ = [
        ("seconds", ctypes.c_double),
        ("schedule_iterations", ctypes.c_size_t),
        ("adapt_step", ctypes.c_int),
        ("recovery", ctypes.c_int),
        ("discarded_evaluations", ctypes.c_uint64),
        ("triggers", ctypes.c_uint64),
        ("recovered_particles", ctypes.c_uint64),
        ("recovery_improvements", ctypes.c_uint64),
        ("mean_trigger_dispersion", ctypes.c_double),
    ]


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
        ("topology_degree", ctypes.c_int),
        ("topology_rewiring_probability", ctypes.c_double),
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
            library.cpso_run_custom_serial.argtypes = [
                ctypes.POINTER(_CConfig),
                _ObjectiveCallback,
                ctypes.c_void_p,
                ctypes.POINTER(_CResult),
                ctypes.POINTER(ctypes.c_double),
                ctypes.POINTER(ctypes.c_double),
                ctypes.POINTER(ctypes.c_uint64),
                ctypes.c_size_t,
                ctypes.POINTER(ctypes.c_char),
                ctypes.c_size_t,
            ]
            library.cpso_run_custom_serial.restype = ctypes.c_int
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
    custom_objective: Optional[Callable[[Sequence[float]], float]] = None,
    experiment: tuple[int, int] | None = None,
    research: tuple[int, float] | None = None,
    synchronous_study: tuple[_CSynchronousStudy, dict, int, int] | None = None,
) -> CpsoResult:
    config.validate()
    if custom_objective is not None and architecture != "serial" and synchronous_study is None:
        raise ValueError("Custom objectives are supported only by the serial architecture")
    if custom_objective is None and benchmark not in _BENCHMARK_IDS:
        raise ValueError(f"Unsupported benchmark {benchmark!r}; choose from {sorted(_BENCHMARK_IDS)}")
    if architecture not in _ARCHITECTURE_IDS:
        raise ValueError(f"Unsupported architecture {architecture!r}")
    if isinstance(work_iterations, bool) or not isinstance(work_iterations, int):
        raise ValueError("work_iterations must be an integer")
    if not 0 <= work_iterations <= 100_000_000:
        raise ValueError("work_iterations must be in [0,100000000]")
    if custom_objective is not None and work_iterations != 0:
        raise ValueError("work_iterations cannot be combined with a custom objective")
    if custom_objective is None and benchmark not in {"sphere_work", "inverse_diffusion"} and work_iterations != 0:
        raise ValueError("work_iterations is only valid for sphere_work/inverse_diffusion")
    if custom_objective is None and benchmark == "inverse_diffusion" and (
        work_iterations < 16 or work_iterations > 100_000 or work_iterations % 4 != 0
    ):
        raise ValueError("inverse_diffusion requires work_iterations divisible by 4 in [16,100000]")
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
    if research is not None:
        if architecture != "asynchronous" or experiment is not None or custom_objective is not None:
            raise ValueError("Research mode requires native asynchronous objective")
        history_capacity = 1
    elif architecture == "asynchronous":
        history_capacity = config.population_size * config.iterations + 1
    else:
        history_capacity = config.iterations + 1
    history = (ctypes.c_double * history_capacity)()
    history_evaluations = (ctypes.c_uint64 * history_capacity)()
    error = ctypes.create_string_buffer(2048)

    c_config = _CConfig(
        abi_version=_ABI_VERSION,
        benchmark_id=_BENCHMARK_IDS.get(benchmark, 1),
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
        topology_degree=config.topology_degree,
        topology_rewiring_probability=config.topology_rewiring_probability,
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
    callback_errors: list[BaseException] = []
    if custom_objective is not None:
        def evaluate_custom(
            position_pointer: ctypes.POINTER(ctypes.c_double),
            dimension: int,
            _context: ctypes.c_void_p,
        ) -> float:
            try:
                point = tuple(float(position_pointer[i]) for i in range(dimension))
                value = float(custom_objective(point))
                return value
            except BaseException as exc:  # never allow Python exceptions to cross the C ABI
                callback_errors.append(exc)
                return math.nan

        callback = _ObjectiveCallback(evaluate_custom)
    else:
        callback = None
    if synchronous_study is not None:
        options, details, update_batch_size, refinement = synchronous_study
        flags, graph_seed = experiment
        batch_seconds = (ctypes.c_double * history_capacity)()
        function = (library.cpso_run_synchronous_batched_study if update_batch_size
                    else library.cpso_run_synchronous_study)
        counts = None
        if refinement:
            function = library.cpso_run_synchronous_refined_study
            counts = (ctypes.c_uint64 * (6 if refinement & 4 else 4))()
        function.argtypes = [
            ctypes.POINTER(_CConfig), ctypes.POINTER(_CExecutionConfig),
            ctypes.c_int, ctypes.c_uint64, _ObjectiveCallback, ctypes.POINTER(_CSynchronousStudy),
            ctypes.POINTER(_CResult), ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_double), ctypes.c_size_t, ctypes.c_char_p, ctypes.c_size_t,
        ]
        if update_batch_size:
            function.argtypes += [ctypes.c_size_t]
        function.restype = ctypes.c_int
        batch_argument = (update_batch_size,) if update_batch_size else ()
        if refinement:
            function.argtypes += [ctypes.c_uint, ctypes.POINTER(ctypes.c_uint64)]
            batch_argument += (refinement, counts)
        status = function(
            ctypes.byref(c_config), ctypes.byref(execution), flags, graph_seed,
            callback if callback is not None else _ObjectiveCallback(), ctypes.byref(options),
            ctypes.byref(c_result), best_position, history, history_evaluations,
            batch_seconds, history_capacity, error, len(error), *batch_argument,
        )
        if status == 0:
            details["batch_seconds"] = batch_seconds[:c_result.history_size]
            for field, _ in _CSynchronousStudy._fields_[4:]:
                details[field] = getattr(options, field)
            if counts is not None:
                fields = ("local_trials", "local_improvements", "restart_trials", "restart_improvements")
                if refinement & 4:
                    fields += ("quadratic_trials", "quadratic_improvements")
                details.update(zip(fields, counts, strict=True))
    elif research is not None:
        function = library.cpso_run_research
        function.argtypes = [ctypes.POINTER(_CConfig), ctypes.POINTER(_CExecutionConfig),
            ctypes.c_int, ctypes.c_double, ctypes.POINTER(_CResult),
            ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_uint64), ctypes.c_size_t, ctypes.c_char_p,
            ctypes.c_size_t]
        function.restype = ctypes.c_int
        status = function(ctypes.byref(c_config), ctypes.byref(execution),
            research[0], research[1], ctypes.byref(c_result), best_position,
            history, history_evaluations, history_capacity, error, len(error))
    elif experiment is not None:
        function = library.cpso_run_experiment
        function.argtypes = [ctypes.POINTER(_CConfig), ctypes.POINTER(_CExecutionConfig),
            ctypes.c_int, ctypes.c_uint64, _ObjectiveCallback, ctypes.c_void_p,
            ctypes.POINTER(_CResult), ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_uint64),
            ctypes.c_size_t, ctypes.c_char_p, ctypes.c_size_t]
        function.restype = ctypes.c_int
        status = function(ctypes.byref(c_config), ctypes.byref(execution), *experiment,
            callback if callback is not None else _ObjectiveCallback(), None,
            ctypes.byref(c_result), best_position, history, history_evaluations,
            history_capacity, error, len(error))
    elif callback is not None:
        status = library.cpso_run_custom_serial(
            ctypes.byref(c_config), callback, None, ctypes.byref(c_result),
            best_position, history, history_evaluations, history_capacity, error, len(error))
    else:
        status = library.cpso_run_architecture(
            ctypes.byref(c_config), ctypes.byref(execution), ctypes.byref(c_result),
            best_position, history, history_evaluations, history_capacity, error, len(error))
    if status != 0:
        if callback_errors:
            raise callback_errors[0]
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


def run_serial_objective(
    config: CpsoConfig,
    objective: Callable[[Sequence[float]], float],
) -> CpsoResult:
    """Run serial CPSO against a caller-supplied deterministic scalar objective.

    The callback is invoked synchronously on the calling thread. This makes it
    suitable for COCO/CEC benchmark adapters without assuming foreign callback
    thread-safety; it is not a parallel objective-evaluation interface.
    """
    if not callable(objective):
        raise TypeError("objective must be callable")
    return _run(
        config, "sphere", architecture="serial", custom_objective=objective
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


def run_controlled(config: CpsoConfig, objective=None, *, dynamics="analytic",
                   randomness="scalar", graph_seed=None, benchmark="sphere",
                   architecture="serial", workers=1, work_iterations=0, boundary_policy="clamp") -> CpsoResult:
    """Matched native ablation, without the reference baseline's velocity clamp.

    Both dynamics share initialization, graph, RNG, inertia schedule, boundary
    map and FE budget. Discrete PSO uses a unit step; dt applies only to analytic.
    Callback execution remains serial. Legacy entry points are unaffected.
    """
    if dynamics not in {"analytic", "discrete"} or randomness not in {"scalar", "coordinate"}:
        raise ValueError("Invalid dynamics or randomness")
    if architecture not in {"serial", "master_slave", "particle_parallel"}:
        raise ValueError("Controlled experiments require synchronous execution")
    if objective is not None and not callable(objective):
        raise TypeError("objective must be callable")
    if graph_seed is not None and (isinstance(graph_seed, bool) or not isinstance(graph_seed, int)
                                   or not 0 <= graph_seed < 2**64):
        raise ValueError("graph_seed must be an unsigned 64-bit integer")
    if boundary_policy not in {"clamp", "none"}:
        raise ValueError("Invalid boundary policy")
    flags = (int(boundary_policy == "none") << 3) | int(dynamics == "discrete") | (int(randomness == "coordinate") << 1)
    if graph_seed is not None:
        flags |= 4
    return _run(config, benchmark, architecture=architecture, workers=workers,
                work_iterations=work_iterations, custom_objective=objective,
                experiment=(flags, graph_seed or 0))


def controlled_graph(config: CpsoConfig, graph_seed: int):
    """Return the actual native adjacency (self included), not a reimplementation."""
    config.validate()
    if not isinstance(graph_seed, int) or isinstance(graph_seed, bool) or not 0 <= graph_seed < 2**64:
        raise ValueError("Invalid graph seed")
    if config.population_size > 4096:
        raise ValueError("Graph inspection requires population <= 4096")
    import numpy as np
    n = config.population_size
    buffer = (ctypes.c_ubyte * (n*n))()
    error = ctypes.create_string_buffer(2048)
    function = _load_library().cpso_experiment_graph
    function.argtypes = [ctypes.c_int, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t,
        ctypes.c_uint64, ctypes.c_size_t, ctypes.c_double, ctypes.POINTER(ctypes.c_ubyte),
        ctypes.c_char_p, ctypes.c_size_t]
    function.restype = ctypes.c_int
    status = function(_TOPOLOGY_IDS[config.topology], n, config.grid_rows or 0,
        config.grid_columns or 0, graph_seed, config.topology_degree,
        config.topology_rewiring_probability, buffer, error, len(error))
    if status:
        raise RuntimeError(error.value.decode())
    return np.ctypeslib.as_array(buffer).reshape(n,n).astype(bool)


def controlled_initial_state(config: CpsoConfig):
    """Exact native initial positions/velocities for geometric diagnostics."""
    config.validate()
    import numpy as np
    n,d=config.population_size,config.dimension
    if n>4096 or d>100000:
        raise ValueError('Initial-state inspection size exceeded')
    lower=(ctypes.c_double*d)(*config.lower_bounds)
    upper=(ctypes.c_double*d)(*config.upper_bounds)
    x=(ctypes.c_double*(n*d))();v=(ctypes.c_double*(n*d))()
    error=ctypes.create_string_buffer(2048)
    function=_load_library().cpso_experiment_initial_state
    ptr=ctypes.POINTER(ctypes.c_double)
    function.argtypes=[ctypes.c_uint64,ctypes.c_size_t,ctypes.c_size_t,ptr,ptr,
                       ctypes.c_double,ptr,ptr,ctypes.c_char_p,ctypes.c_size_t]
    function.restype=ctypes.c_int
    status=function(config.seed,n,d,lower,upper,config.initial_velocity_fraction,x,v,error,len(error))
    if status: raise RuntimeError(error.value.decode())
    return (np.ctypeslib.as_array(x).reshape(n,d).copy(),
            np.ctypeslib.as_array(v).reshape(n,d).copy())


def run_research(config: CpsoConfig, benchmark: str, *, adaptive: bool = False,
                 workers: int = 4, seconds: float = 0.0,
                 work_iterations: int = 0) -> CpsoResult:
    """Opt-in native asynchronous CPSO. Uses one-value summary to bound memory."""
    if not math.isfinite(seconds) or seconds < 0.0:
        raise ValueError("seconds must be a finite nonnegative duration")
    return _run(config, benchmark, architecture="asynchronous", workers=workers,
                work_iterations=work_iterations,
                research=(16 if adaptive else 0, float(seconds)))



def run_synchronous_study(
    config: CpsoConfig, benchmark: str, *, workers: int, graph_seed: int,
    coordinate: bool = False, discrete: bool = False, adapt_step: bool = False,
    recovery: bool = False, seconds: float = 0.0, schedule_iterations: int = 0,
    work_iterations: int = 0, objective: Callable | None = None,
    update_batch_size: int = 0,
    local_search: bool = False, partial_restart: bool = False,
    quadratic_local: bool = False,
) -> tuple[CpsoResult, dict]:
    """Run the synchronous competition and retain completed-batch timestamps."""
    if not math.isfinite(seconds) or seconds < 0.0 or schedule_iterations < 0:
        raise ValueError("Invalid deadline or schedule")
    if not 0 <= graph_seed < 2**64:
        raise ValueError("Invalid graph seed")
    if update_batch_size < 0 or (update_batch_size and config.population_size % update_batch_size):
        raise ValueError("The update batch must divide the population")
    if 0 < update_batch_size < config.population_size and (adapt_step or recovery):
        raise ValueError("Partial batches require a fixed step and no recovery")
    if quadratic_local and not local_search:
        raise ValueError("Quadratic probes require local search")
    refinement = int(local_search) | (int(partial_restart) << 1) | (int(quadratic_local) << 2)
    if refinement:
        update_batch_size = update_batch_size or config.population_size
        if update_batch_size < 4 or any((adapt_step, recovery, discrete)):
            raise ValueError("Refinement requires continuous fixed-step CPSO and batches of at least four")
    flags = 4 | (int(coordinate) << 1) | int(discrete)
    options = _CSynchronousStudy(seconds, schedule_iterations, adapt_step, recovery)
    details = {}
    result = _run(
        config, benchmark, architecture="particle_parallel",
        workers=workers, work_iterations=work_iterations, custom_objective=objective,
        experiment=(flags, graph_seed), synchronous_study=(options, details, update_batch_size, refinement),
    )
    return result, details


@contextmanager
def native_batch_evaluator(benchmark: str, workers: int, work_iterations: int = 0):
    """Reuse the native benchmark and a persistent evaluation pool."""
    import numpy as np

    library = _load_library()
    create = library.cpso_create_batch_evaluator
    evaluate = library.cpso_evaluate_batch
    destroy = library.cpso_destroy_batch_evaluator
    pointer = ctypes.POINTER(ctypes.c_double)
    create.argtypes = [ctypes.c_int, ctypes.c_uint64, ctypes.c_size_t, ctypes.c_char_p, ctypes.c_size_t]
    create.restype = ctypes.c_void_p
    evaluate.argtypes = [ctypes.c_void_p, pointer, ctypes.c_size_t, ctypes.c_size_t, pointer,
                         ctypes.c_char_p, ctypes.c_size_t]
    evaluate.restype = ctypes.c_int
    destroy.argtypes = [ctypes.c_void_p]
    destroy.restype = None

    error = ctypes.create_string_buffer(2048)
    handle = create(_BENCHMARK_IDS[benchmark], work_iterations, workers, error, len(error))
    if not handle:
        raise RuntimeError(error.value.decode("utf-8", errors="replace"))

    def evaluate_points(points):
        points = np.ascontiguousarray(points, dtype=np.float64)
        rows, dimension = points.shape
        values = np.empty(rows, dtype=np.float64)
        status = evaluate(handle, points.ctypes.data_as(pointer), rows, dimension,
                          values.ctypes.data_as(pointer), error, len(error))
        if status != 0:
            raise RuntimeError(error.value.decode("utf-8", errors="replace"))
        return values

    try:
        yield evaluate_points
    finally:
        destroy(handle)
