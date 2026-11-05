from __future__ import annotations

from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

from cpso_serial import (  # noqa: E402
    CpsoConfig,
    native_build_info,
    native_version,
    run_asynchronous,
    run_island_model,
    run_master_slave,
    run_particle_parallel,
    run_serial,
)


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
    assert native_version() == "0.6.0"
    info = native_build_info()
    assert "compiler=" in info
    assert "cxx_standard=20" in info
    assert "abi=5" in info


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
    assert result.topology_setup_seconds >= 0.0
    assert result.social_selection_seconds >= 0.0
    assert result.worker_count == 1
    assert result.history_evaluations == tuple(config.population_size * (i + 1) for i in range(config.iterations + 1))
    assert all(lo <= x <= hi for lo, x, hi in zip(config.lower_bounds, result.best_position, config.upper_bounds))


def test_fixed_seed_repeatability() -> None:
    config = make_config()
    first = run_serial(config, "rastrigin")
    second = run_serial(config, "rastrigin")
    assert first.best_value == second.best_value
    assert first.best_position == second.best_position
    assert first.best_history == second.best_history
    assert first.function_evaluations == second.function_evaluations


def test_distinct_run_seeds_produce_distinct_initialization_signatures() -> None:
    signatures = set()
    for seed in range(700, 704):
        config = CpsoConfig(
            lower_bounds=(-5.12,) * 3,
            upper_bounds=(5.12,) * 3,
            population_size=20,
            iterations=0,
            seed=seed,
        )
        result = run_serial(config, "sphere")
        signatures.add((result.best_value, result.best_position))
    assert len(signatures) == 4


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
        assert parallel.topology_setup_seconds >= 0.0
        assert parallel.social_selection_seconds >= 0.0


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


def test_topology_config_validation() -> None:
    CpsoConfig(lower_bounds=(-1.0,), upper_bounds=(1.0,), topology="global").validate()
    CpsoConfig(lower_bounds=(-1.0,), upper_bounds=(1.0,), topology="ring").validate()
    CpsoConfig(
        lower_bounds=(-1.0,) * 12,
        upper_bounds=(1.0,) * 12,
        population_size=12,
        topology="grid_von_neumann",
        grid_rows=3,
        grid_columns=4,
    ).validate()
    with pytest.raises(ValueError):
        CpsoConfig(lower_bounds=(-1.0,), upper_bounds=(1.0,), topology="unknown").validate()
    with pytest.raises(ValueError):
        CpsoConfig(
            lower_bounds=(-1.0,) * 12, upper_bounds=(1.0,) * 12,
            population_size=12, topology="grid_von_neumann", grid_rows=2, grid_columns=5,
        ).validate()
    with pytest.raises(ValueError):
        CpsoConfig(
            lower_bounds=(-1.0,) * 12, upper_bounds=(1.0,) * 12,
            population_size=12, topology="ring", grid_rows=3, grid_columns=4,
        ).validate()
    with pytest.raises(ValueError):
        CpsoConfig(
            lower_bounds=(-1.0,) * 10, upper_bounds=(1.0,) * 10,
            population_size=10, topology="grid_von_neumann", grid_rows=1, grid_columns=10,
        ).validate()


def test_topology_serial_master_slave_equivalence() -> None:
    base = make_config()
    topology_cases = (
        {"topology": "global"},
        {"topology": "ring"},
        {"topology": "grid_von_neumann", "grid_rows": 4, "grid_columns": 5},
    )
    for topology_kwargs in topology_cases:
        config = CpsoConfig(
            lower_bounds=base.lower_bounds,
            upper_bounds=base.upper_bounds,
            population_size=base.population_size,
            iterations=base.iterations,
            seed=base.seed,
            dt=base.dt,
            **topology_kwargs,
        )
        serial = run_serial(config, "rastrigin")
        for workers in (1, 2, 4):
            parallel = run_master_slave(config, "rastrigin", workers=workers)
            assert parallel.best_value == serial.best_value
            assert parallel.best_position == serial.best_position
            assert parallel.best_history == serial.best_history
            assert parallel.function_evaluations == serial.function_evaluations


def test_topologies_are_algorithmically_distinct() -> None:
    base = make_config()
    results = {}
    for topology_kwargs in (
        {"topology": "global"},
        {"topology": "ring"},
        {"topology": "grid_von_neumann", "grid_rows": 4, "grid_columns": 5},
    ):
        config = CpsoConfig(
            lower_bounds=base.lower_bounds,
            upper_bounds=base.upper_bounds,
            population_size=base.population_size,
            iterations=base.iterations,
            seed=base.seed,
            dt=base.dt,
            **topology_kwargs,
        )
        results[config.topology] = run_serial(config, "rastrigin")
    signatures = {(r.best_value, r.best_position, r.best_history) for r in results.values()}
    assert len(signatures) == 3


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


def test_particle_parallel_exact_equivalence_to_serial() -> None:
    base = make_config()
    for topology_kwargs in (
        {"topology": "global"},
        {"topology": "ring"},
        {"topology": "grid_von_neumann", "grid_rows": 4, "grid_columns": 5},
    ):
        config = CpsoConfig(
            lower_bounds=base.lower_bounds,
            upper_bounds=base.upper_bounds,
            population_size=base.population_size,
            iterations=base.iterations,
            seed=base.seed,
            dt=base.dt,
            **topology_kwargs,
        )
        serial = run_serial(config, "rastrigin")
        for workers in (1, 2, 4, 8):
            parallel = run_particle_parallel(config, "rastrigin", workers=workers)
            assert parallel.best_value == serial.best_value
            assert parallel.best_position == serial.best_position
            assert parallel.best_history == serial.best_history
            assert parallel.history_evaluations == serial.history_evaluations
            assert parallel.function_evaluations == serial.function_evaluations


def test_asynchronous_history_contract() -> None:
    config = make_config()
    result = run_asynchronous(config, "rastrigin", workers=4)
    expected_fe = config.population_size * (config.iterations + 1)
    assert result.function_evaluations == expected_fe
    assert len(result.best_history) == 1 + config.population_size * config.iterations
    assert len(result.history_evaluations) == len(result.best_history)
    assert result.history_evaluations[0] == config.population_size
    assert result.history_evaluations[-1] == expected_fe
    assert all(b == a + 1 for a, b in zip(result.history_evaluations, result.history_evaluations[1:]))
    assert all(b <= a for a, b in zip(result.best_history, result.best_history[1:]))
    assert result.best_value == result.best_history[-1]
    assert result.async_update_wall_seconds >= 0.0


def test_asynchronous_single_worker_repeatability() -> None:
    config = make_config()
    first = run_asynchronous(config, "sphere", workers=1)
    second = run_asynchronous(config, "sphere", workers=1)
    assert first.best_value == second.best_value
    assert first.best_position == second.best_position
    assert first.best_history == second.best_history
    assert first.history_evaluations == second.history_evaluations


def test_island_model_fe_budget_and_migration() -> None:
    config = CpsoConfig(
        lower_bounds=(-5.12,) * 4,
        upper_bounds=(5.12,) * 4,
        population_size=24,
        iterations=8,
        seed=7654,
        topology="global",
    )
    independent = run_island_model(
        config, "rastrigin", islands=4, workers=4, migration_interval=0
    )
    migrated = run_island_model(
        config, "rastrigin", islands=4, workers=4,
        migration_interval=2, migration_topology="ring",
    )
    all_to_all = run_island_model(
        config, "rastrigin", islands=4, workers=2,
        migration_interval=2, migration_topology="fully_connected",
    )
    expected_fe = config.population_size * (config.iterations + 1)
    for result in (independent, migrated, all_to_all):
        assert result.function_evaluations == expected_fe
        assert result.iterations_completed == config.iterations
        assert len(result.best_history) == config.iterations + 1
        assert result.history_evaluations[-1] == expected_fe
        assert all(b <= a for a, b in zip(result.best_history, result.best_history[1:]))
    assert independent.migrations_performed == 0
    assert migrated.island_count == 4
    assert all_to_all.worker_count == 2


def test_island_validation() -> None:
    config = make_config()
    with pytest.raises(ValueError):
        run_island_model(config, "sphere", islands=3, workers=3)
    with pytest.raises(ValueError):
        run_island_model(config, "sphere", islands=4, workers=5)
    with pytest.raises(ValueError):
        run_island_model(config, "sphere", islands=4, workers=4, migration_topology="star")
    with pytest.raises(ValueError):
        run_island_model(config, "sphere", islands=4, workers=4, migration_interval=-1)
