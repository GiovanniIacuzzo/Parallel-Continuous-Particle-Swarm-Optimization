from __future__ import annotations

import ctypes as ct
import json
import os
import platform
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


_BENCHMARKS = {
    "sphere": 0,
    "rastrigin": 1,
    "rosenbrock": 2,
    "schwefel": 3,
    "griewank": 4,
    "ackley": 5,
    "salomon": 6,
}
_TOPOLOGIES = {"independent": 0, "star": 1, "ring": 2}


class _CConfig(ct.Structure):
    _fields_ = [
        ("abi_version", ct.c_uint32),
        ("dimension", ct.c_int32),
        ("population_size", ct.c_int32),
        ("iterations", ct.c_int32),
        ("workers", ct.c_int32),
        ("sync_interval", ct.c_int32),
        ("topology", ct.c_int32),
        ("benchmark", ct.c_int32),
        ("seed", ct.c_uint64),
        ("lower_bounds", ct.POINTER(ct.c_double)),
        ("upper_bounds", ct.POINTER(ct.c_double)),
        ("optimum_position", ct.POINTER(ct.c_double)),
        ("mu_max", ct.c_double),
        ("mu_min", ct.c_double),
        ("cognitive", ct.c_double),
        ("social", ct.c_double),
        ("dt", ct.c_double),
        ("initial_velocity_fraction", ct.c_double),
        ("boundary_restitution", ct.c_double),
    ]


class _CResult(ct.Structure):
    _fields_ = [
        ("best_value", ct.c_double),
        ("best_position", ct.POINTER(ct.c_double)),
        ("history", ct.POINTER(ct.c_double)),
        ("history_length", ct.c_size_t),
        ("worker_best_values", ct.POINTER(ct.c_double)),
        ("worker_count", ct.c_size_t),
        ("function_evaluations", ct.c_uint64),
        ("elapsed_seconds", ct.c_double),
        ("compute_seconds", ct.c_double),
        ("communication_seconds", ct.c_double),
    ]


@dataclass
class CpsoResult:
    benchmark: str
    topology: str
    dimension: int
    population_size: int
    iterations: int
    workers: int
    seed: int
    configuration: dict[str, Any]
    optimum_position: list[float]
    best_value: float
    best_position: list[float]
    history: list[float]
    history_fes: list[int]
    worker_best_values: list[float]
    function_evaluations: int
    elapsed_seconds: float
    compute_seconds: float
    communication_seconds: float
    openmp_enabled: bool
    max_openmp_threads: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save_json(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    def save_history_csv(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = np.column_stack([np.asarray(self.history_fes), np.asarray(self.history)])
        np.savetxt(path, data, delimiter=",", header="function_evaluations,best_value", comments="")


class CpsoEngine:
    """Thin Python wrapper around the C++17/OpenMP CPSO shared library."""

    def __init__(self, library_path: str | Path | None = None):
        self.library_path = Path(library_path) if library_path else self._find_library()
        if not self.library_path.is_file():
            raise FileNotFoundError(
                "C++ library not found. Build it with `cmake -S . -B build -DCMAKE_BUILD_TYPE=Release` "
                "and `cmake --build build --parallel`, or set CPSO_LIBRARY."
            )
        self.lib = ct.CDLL(str(self.library_path))
        self.lib.cpso_run.argtypes = [
            ct.POINTER(_CConfig), ct.POINTER(_CResult), ct.POINTER(ct.c_char), ct.c_size_t
        ]
        self.lib.cpso_run.restype = ct.c_int
        self.lib.cpso_free_result.argtypes = [ct.POINTER(_CResult)]
        self.lib.cpso_free_result.restype = None
        self.lib.cpso_version.restype = ct.c_char_p
        self.lib.cpso_openmp_enabled.restype = ct.c_int
        self.lib.cpso_max_threads.restype = ct.c_int

    @staticmethod
    def _find_library() -> Path:
        env = os.environ.get("CPSO_LIBRARY")
        if env:
            return Path(env).expanduser().resolve()
        here = Path(__file__).resolve()
        root = here.parents[2]
        candidates = [
            root / "build" / "libcpso_core.so",
            root / "build" / "libcpso_core.dylib",
            root / "build" / "cpso_core.dll",
            root / "lib" / "libcpso_core.so",
            root / "lib" / "libcpso_core.dylib",
        ]
        for candidate in candidates:
            if candidate.is_file():
                return candidate
        return candidates[0]

    @property
    def version(self) -> str:
        return self.lib.cpso_version().decode("utf-8")

    @property
    def openmp_enabled(self) -> bool:
        return bool(self.lib.cpso_openmp_enabled())

    @property
    def max_openmp_threads(self) -> int:
        return int(self.lib.cpso_max_threads())

    def run(self, config: dict[str, Any]) -> CpsoResult:
        from .config import validate_config

        cfg = validate_config(config)
        problem, algorithm, cpso = cfg["problem"], cfg["algorithm"], cfg["cpso"]
        dim = problem["dimension"]
        lower = np.ascontiguousarray(problem["lower_bounds"], dtype=np.float64)
        upper = np.ascontiguousarray(problem["upper_bounds"], dtype=np.float64)
        optimum = np.ascontiguousarray(problem["optimum_position"], dtype=np.float64)

        c_cfg = _CConfig(
            abi_version=1,
            dimension=dim,
            population_size=algorithm["population_size"],
            iterations=algorithm["iterations"],
            workers=algorithm["workers"],
            sync_interval=algorithm["sync_interval"],
            topology=_TOPOLOGIES[algorithm["topology"]],
            benchmark=_BENCHMARKS[problem["name"]],
            seed=algorithm["seed"] & ((1 << 64) - 1),
            lower_bounds=lower.ctypes.data_as(ct.POINTER(ct.c_double)),
            upper_bounds=upper.ctypes.data_as(ct.POINTER(ct.c_double)),
            optimum_position=optimum.ctypes.data_as(ct.POINTER(ct.c_double)),
            mu_max=cpso["mu_max"],
            mu_min=cpso["mu_min"],
            cognitive=cpso["cognitive"],
            social=cpso["social"],
            dt=cpso["dt"],
            initial_velocity_fraction=cpso["initial_velocity_fraction"],
            boundary_restitution=cpso["boundary_restitution"],
        )

        c_result = _CResult()
        error = ct.create_string_buffer(2048)
        status = self.lib.cpso_run(ct.byref(c_cfg), ct.byref(c_result), error, len(error))
        if status != 0:
            raise RuntimeError(f"C++ CPSO failed (status={status}): {error.value.decode('utf-8', errors='replace')}")

        try:
            best_position = np.ctypeslib.as_array(c_result.best_position, shape=(dim,)).copy().tolist()
            history = np.ctypeslib.as_array(c_result.history, shape=(c_result.history_length,)).copy().tolist()
            worker_values = np.ctypeslib.as_array(
                c_result.worker_best_values, shape=(c_result.worker_count,)
            ).copy().tolist()
            # One objective evaluation per particle at initialization and after each transition.
            history_fes = [algorithm["population_size"] * (i + 1) for i in range(len(history))]
            return CpsoResult(
                benchmark=problem["name"],
                topology=algorithm["topology"],
                dimension=dim,
                population_size=algorithm["population_size"],
                iterations=algorithm["iterations"],
                workers=algorithm["workers"],
                seed=algorithm["seed"],
                configuration=cfg,
                optimum_position=optimum.tolist(),
                best_value=float(c_result.best_value),
                best_position=best_position,
                history=history,
                history_fes=history_fes,
                worker_best_values=worker_values,
                function_evaluations=int(c_result.function_evaluations),
                elapsed_seconds=float(c_result.elapsed_seconds),
                compute_seconds=float(c_result.compute_seconds),
                communication_seconds=float(c_result.communication_seconds),
                openmp_enabled=self.openmp_enabled,
                max_openmp_threads=self.max_openmp_threads,
            )
        finally:
            self.lib.cpso_free_result(ct.byref(c_result))


def environment_report(engine: CpsoEngine) -> dict[str, str | int | bool]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "logical_cpu_count": os.cpu_count(),
        "cpp_library_version": engine.version,
        "openmp_enabled": engine.openmp_enabled,
        "max_openmp_threads": engine.max_openmp_threads,
    }
