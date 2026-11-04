from __future__ import annotations

from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

from cpso_serial import CpsoConfig, native_build_info, native_version, run_master_slave, run_serial  # noqa: E402


def make_config() -> CpsoConfig:
    return CpsoConfig(
        lower_bounds=(-5.12, -5.12, -5.12),
        upper_bounds=(5.12, 5.12, 5.12),
        population_size=20,
        iterations=12,
        seed=123456,
        dt=0.5,
    )


def test_python_config_validation() -> None:
    CpsoConfig(lower_bounds=(-1.0,), upper_bounds=(1.0,)).validate()
    with pytest.raises(ValueError):
        CpsoConfig(lower_bounds=(-1.0, -1.0), upper_bounds=(1.0,)).validate()
    with pytest.raises(ValueError):
        CpsoConfig(lower_bounds=(-1.0,), upper_bounds=(1.0,), mu_min=0.8, mu_max=0.4).validate()
    with pytest.raises(ValueError):
        CpsoConfig(lower_bounds=(-1.0,), upper_bounds=(1.0,), population_size=2.5).validate()
    with pytest.raises(ValueError):
        CpsoConfig(lower_bounds=(-1.0,), upper_bounds=(1.0,), seed=1 << 64).validate()


def test_native_version_and_build_info() -> None:
    assert native_version() == "0.3.0"
    info = native_build_info()
    assert "compiler=" in info
    assert "cxx_standard=20" in info
    assert "abi=3" in info


def test_serial_smoke_contract() -> None:
    config = make_config()
    result = run_serial(config, benchmark="sphere")
    assert result.iterations_completed == config.iterations
    assert result.function_evaluations == config.population_size * (config.iterations + 1)
    assert len(result.best_history) == config.iterations + 1
    assert len(result.best_position) == config.dimension
    assert result.best_value == result.best_history[-1]
    assert all(a >= b for a, b in zip(result.best_history, result.best_history[1:]))
    assert result.elapsed_seconds >= 0.0
    assert result.evaluation_wall_seconds >= 0.0
    assert result.propagation_seconds >= 0.0
    assert result.worker_count == 1
    assert all(lo <= x <= hi for lo, x, hi in zip(config.lower_bounds, result.best_position, config.upper_bounds))


def test_fixed_seed_repeatability() -> None:
    config = make_config()
    first = run_serial(config, "rastrigin")
    second = run_serial(config, "rastrigin")
    assert first.best_value == second.best_value
    assert first.best_position == second.best_position
    assert first.best_history == second.best_history
    assert first.function_evaluations == second.function_evaluations


def test_zero_iteration_and_one_dimensional_problem() -> None:
    config = CpsoConfig(
        lower_bounds=(-2.0,), upper_bounds=(2.0,), population_size=1,
        iterations=0, seed=7,
    )
    result = run_serial(config, "sphere")
    assert result.iterations_completed == 0
    assert result.function_evaluations == 1
    assert len(result.best_history) == 1


def test_master_slave_exactly_matches_serial_reference() -> None:
    config = make_config()
    serial = run_serial(config, "rastrigin")
    for workers in (1, 2, 4, 8):
        parallel = run_master_slave(config, "rastrigin", workers=workers)
        assert parallel.best_value == serial.best_value
        assert parallel.best_position == serial.best_position
        assert parallel.best_history == serial.best_history
        assert parallel.function_evaluations == serial.function_evaluations
        assert parallel.iterations_completed == serial.iterations_completed
        assert parallel.worker_count == workers
        assert parallel.elapsed_seconds >= 0.0
        assert parallel.evaluation_wall_seconds >= 0.0
        assert parallel.propagation_seconds >= 0.0


def test_master_slave_zero_iterations() -> None:
    config = CpsoConfig(
        lower_bounds=(-2.0,), upper_bounds=(2.0,), population_size=5,
        iterations=0, seed=99,
    )
    result = run_master_slave(config, "sphere", workers=3)
    assert result.function_evaluations == 5
    assert result.iterations_completed == 0
    assert len(result.best_history) == 1
    assert result.worker_count == 3


def test_master_slave_worker_validation() -> None:
    config = make_config()
    for invalid in (0, -1, 1.5, True, 4097):
        with pytest.raises(ValueError):
            run_master_slave(config, "sphere", workers=invalid)  # type: ignore[arg-type]


def test_synthetic_work_objective_matches_serial_reference() -> None:
    config = make_config()
    serial = run_serial(config, "sphere_work", work_iterations=5000)
    for workers in (1, 2, 4):
        parallel = run_master_slave(
            config, "sphere_work", workers=workers, work_iterations=5000
        )
        assert parallel.best_value == serial.best_value
        assert parallel.best_position == serial.best_position
        assert parallel.best_history == serial.best_history
        assert parallel.function_evaluations == serial.function_evaluations


def test_synthetic_work_argument_validation() -> None:
    config = make_config()
    with pytest.raises(ValueError):
        run_serial(config, "sphere", work_iterations=10)
    with pytest.raises(ValueError):
        run_serial(config, "sphere_work", work_iterations=-1)
    with pytest.raises(ValueError):
        run_serial(config, "sphere_work", work_iterations=100_000_001)
