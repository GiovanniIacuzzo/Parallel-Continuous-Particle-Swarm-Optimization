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


def test_partial_updates_preserve_inactive_state_and_consume_exact_fe():
    config = CpsoConfig((-5.0,) * 5, (5.0,) * 5, population_size=100,
                        iterations=3, mu_min=0.5, mu_max=0.5)

    def run(config, workers, batch=0):
        visited = []

        def flat(point):
            visited.append(point)
            return 1.0

        result, _ = run_synchronous_study(config, "sphere", workers=workers, graph_seed=2,
                                          objective=flat, update_batch_size=batch)
        return result, visited

    full, full_points = run(config, 1)
    partial_config = replace(config, iterations=6)
    partial, partial_points = run(partial_config, 4, 50)
    _, serial_partial_points = run(partial_config, 1, 50)
    # With a flat objective and constant inertia the social snapshot never
    # changes. Every particle must follow exactly its full-sweep trajectory.
    assert full_points == partial_points == serial_partial_points
    assert full.function_evaluations == partial.function_evaluations == 400
    assert partial.history_evaluations == (100, 150, 200, 250, 300, 350, 400)


def test_native_transform_shares_the_feasible_minimizer_across_paired_variants():
    from publication_tools.study3_transforms import native_instance

    shifted, optimum = native_instance("rosenbrock", 20, "shifted", 101)
    rotated, rotated_optimum = native_instance("rosenbrock", 20, "rotated", 101)
    assert optimum == rotated_optimum
    assert all(-5.0 < x < 10.0 for x in optimum)
    assert shifted(optimum) == rotated(optimum) == 0.0
    probe = np.asarray(optimum) + 0.3
    assert shifted(probe) != pytest.approx(rotated(probe))
    assert optimum != native_instance("rosenbrock", 20, "shifted", 201)[1]


def test_partial_batch_refinement_accounts_for_trials_and_is_worker_deterministic():
    config = CpsoConfig((-5.0,) * 5, (5.0,) * 5, population_size=100, iterations=160,
                        dt=1.0, cognitive=0.0, social=2.05, mu_min=0.4, mu_max=0.4)

    for local_search, partial_restart in ((True, False), (False, True), (True, True)):
        runs = [run_synchronous_study(config, "sphere", workers=workers, graph_seed=2,
                                    objective=lambda point: 1.0, update_batch_size=50,
                                    local_search=local_search, partial_restart=partial_restart)
                for workers in (1, 4)]
        (one, a), (four, b) = runs
        assert one.best_history == four.best_history
        assert one.best_position == four.best_position
        assert one.history_evaluations == four.history_evaluations == tuple(range(100, 8101, 50))
        assert (a["local_trials"] > 0) == local_search
        assert (a["restart_trials"] > 0) == partial_restart
        for key in ("local_trials", "local_improvements", "restart_trials", "restart_improvements"):
            assert a[key] == b[key]
        assert a["local_improvements"] == a["restart_improvements"] == 0
