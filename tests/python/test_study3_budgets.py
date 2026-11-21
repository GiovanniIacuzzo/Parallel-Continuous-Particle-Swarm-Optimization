from dataclasses import replace

import numpy as np
import pytest

from cpso_serial import CpsoConfig
from cpso_serial.native import native_batch_evaluator, run_synchronous_study
from cpso_serial.pso_baselines import PSOConfig, available_pso_methods, minimize_pso


def sphere(point):
    return sum(x * x for x in map(float, point))


@pytest.mark.parametrize("method", available_pso_methods())
def test_batch_pso_keeps_scalar_trajectory_and_exact_partial_budget(method):
    config = PSOConfig((-5.0,) * 5, (5.0,) * 5, population_size=20, max_evaluations=147)
    scalar = minimize_pso(sphere, config, method=method)
    batch = minimize_pso(sphere, config, method=method,
                         batch_objective=lambda points: [sphere(point) for point in points])
    assert scalar.function_evaluations == batch.function_evaluations == 147
    assert scalar.best_position == batch.best_position
    assert scalar.best_history == batch.best_history
    assert scalar.history_evaluations == batch.history_evaluations


def test_native_recovery_is_opt_in_and_deterministic_across_workers():
    config = CpsoConfig((-5.0,) * 5, (5.0,) * 5, topology="random",
                        topology_degree=8, population_size=20, iterations=27)

    def flat(_):
        return 1.0

    fixed = run_synchronous_study(config, "sphere", workers=1, graph_seed=10, objective=flat)
    for adapt_step, recovery in ((True, False), (False, True), (True, True)):
        one, first = run_synchronous_study(config, "sphere", workers=1, graph_seed=10,
                                          objective=flat, coordinate=True,
                                          adapt_step=adapt_step, recovery=recovery)
        many, second = run_synchronous_study(config, "sphere", workers=4, graph_seed=10,
                                             objective=flat, coordinate=True,
                                             adapt_step=adapt_step, recovery=recovery)
        assert first["triggers"] == second["triggers"] == 2
        assert first["recovered_particles"] == second["recovered_particles"] == (8 if recovery else 0)
        assert first["recovery_improvements"] == 0
        assert len(first["batch_seconds"]) == len(one.best_history) == 28
        assert one.best_history == many.best_history
        assert one.best_position == many.best_position
        assert one.function_evaluations == many.function_evaluations == 560
    assert fixed[1]["recovered_particles"] == fixed[1]["triggers"] == 0


def test_fixed_schedule_independent_of_native_iteration_cap():
    config = CpsoConfig((-5.0,) * 5, (5.0,) * 5, population_size=20, iterations=7)
    short, _ = run_synchronous_study(config, "sphere", workers=1, graph_seed=2,
                                    schedule_iterations=11)
    long, _ = run_synchronous_study(replace(config, iterations=16), "sphere", workers=4,
                                   graph_seed=2, schedule_iterations=11)
    assert short.best_history == long.best_history[:len(short.best_history)]

    for method in ("inertia_gbest", "clpso", "dms_pso"):
        pso = PSOConfig(config.lower_bounds, config.upper_bounds, population_size=20,
                        max_evaluations=160, schedule_iterations=11)
        a = minimize_pso(sphere, pso, method=method)
        b = minimize_pso(sphere, replace(pso, max_evaluations=340), method=method)
        assert a.best_history == b.best_history[:len(a.best_history)]


def test_deadline_before_first_batch_has_no_available_native_solution():
    config = CpsoConfig((-2.0,) * 5, (2.0,) * 5, population_size=20, iterations=100)
    result, trace = run_synchronous_study(config, "inverse_diffusion", workers=4,
                                          graph_seed=2, seconds=1e-9,
                                          schedule_iterations=100, work_iterations=512)
    assert result.function_evaluations == 0
    assert result.best_history == ()
    assert trace["batch_seconds"] == []


def test_same_native_evaluator_at_one_and_four_workers():
    points = np.random.default_rng(10).uniform(-1, 1, size=(25, 7))
    for benchmark, work in (("rastrigin", 0), ("inverse_diffusion", 16)):
        with native_batch_evaluator(benchmark, 1, work) as serial:
            expected = serial(points)
        with native_batch_evaluator(benchmark, 4, work) as parallel:
            assert np.array_equal(expected, parallel(points))
