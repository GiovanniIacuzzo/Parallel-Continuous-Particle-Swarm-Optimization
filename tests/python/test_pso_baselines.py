from __future__ import annotations

import math

import pytest

from cpso_serial.pso_baselines import (
    PSOConfig,
    available_pso_methods,
    minimize_pso,
    pso_method_metadata,
)


def sphere(point):
    return sum(value * value for value in point)


def make_config(*, budget=800, seed=77):
    return PSOConfig(
        lower_bounds=(-5.0, -5.0, -5.0),
        upper_bounds=(5.0, 5.0, 5.0),
        population_size=40,
        max_evaluations=budget,
        seed=seed,
    )


@pytest.mark.parametrize("method", available_pso_methods())
def test_pso_methods_consume_exact_budget_and_keep_history_monotone(method):
    result = minimize_pso(sphere, make_config(), method=method)
    assert result.function_evaluations == 800
    assert result.history_evaluations[0] == 40
    assert result.history_evaluations[-1] == 800
    assert result.best_history[-1] == result.best_value
    assert all(b <= a for a, b in zip(result.best_history, result.best_history[1:]))
    assert all(b > a for a, b in zip(result.history_evaluations, result.history_evaluations[1:]))
    assert len(result.best_position) == 3
    assert math.isfinite(result.best_value)


@pytest.mark.parametrize("method", available_pso_methods())
def test_pso_methods_are_repeatable_for_fixed_seed(method):
    first = minimize_pso(sphere, make_config(seed=107), method=method)
    second = minimize_pso(sphere, make_config(seed=107), method=method)
    assert first.best_value == second.best_value
    assert first.best_position == second.best_position
    assert first.best_history == second.best_history
    assert first.history_evaluations == second.history_evaluations


def test_classic_and_inertia_variants_have_resolved_parameters():
    config = make_config()
    classic = pso_method_metadata("classic_gbest", config)
    inertia = pso_method_metadata("inertia_gbest", config)
    constriction = pso_method_metadata("constriction_gbest", config)
    ring = pso_method_metadata("inertia_lbest_ring", config)
    slpso = pso_method_metadata("slpso", config)

    assert classic["cognitive"] == 2.0
    assert classic["inertia_max"] == 1.0
    assert inertia["inertia_max"] == 0.9
    assert inertia["inertia_min"] == 0.4
    assert constriction["velocity_limit_fraction"] is None
    assert constriction["constriction_factor"] == pytest.approx(0.7298437881283576)
    assert ring["topology"] == "periodic-ring(radius=1)"
    assert slpso["slpso_reference_M"] == 100
    assert slpso["slpso_alpha"] == pytest.approx(0.5)
    assert slpso["slpso_beta"] == pytest.approx(0.01)
    assert slpso["slpso_social_influence_factor"] == pytest.approx(0.0003)
    assert slpso["initial_velocity_fraction"] == 0.0
    assert slpso["velocity_limit_fraction"] is None


def test_constriction_variant_rejects_non_stable_phi_range():
    config = PSOConfig(
        lower_bounds=(-1.0, -1.0), upper_bounds=(1.0, 1.0),
        population_size=8, max_evaluations=80, cognitive=1.0, social=1.0,
    )
    with pytest.raises(ValueError, match=r"cognitive \+ social > 4"):
        minimize_pso(sphere, config, method="constriction_gbest")


def test_pso_accepts_non_multiple_budget_without_exceeding_it():
    config = PSOConfig(
        lower_bounds=(-2.0,), upper_bounds=(2.0,),
        population_size=10, max_evaluations=27, seed=3,
    )
    result = minimize_pso(sphere, config, method="inertia_gbest")
    assert result.function_evaluations == 27
    assert result.history_evaluations[-1] == 27


def test_reflection_map_keeps_every_objective_evaluation_inside_bounds():
    calls = 0

    def bounded_objective(point):
        nonlocal calls
        calls += 1
        assert all(-1.0 <= value <= 1.0 for value in point)
        return sum(value * value for value in point)

    config = PSOConfig(
        lower_bounds=(-1.0, -1.0), upper_bounds=(1.0, 1.0),
        population_size=10, max_evaluations=200, seed=99,
        initial_velocity_fraction=3.0, velocity_limit_fraction=None,
    )
    result = minimize_pso(bounded_objective, config, method="constriction_gbest")
    assert calls == result.function_evaluations == 200


def test_pso_rejects_invalid_bounds_and_budget():
    with pytest.raises(ValueError):
        PSOConfig(lower_bounds=(1.0,), upper_bounds=(1.0,), population_size=4,
                  max_evaluations=20).validate()
    with pytest.raises(ValueError):
        PSOConfig(lower_bounds=(-1.0,), upper_bounds=(1.0,), population_size=10,
                  max_evaluations=9).validate()


def test_social_topologies_produce_distinct_deterministic_trajectories():
    config = make_config(budget=1600, seed=410)
    global_result = minimize_pso(sphere, config, method="inertia_gbest")
    ring_result = minimize_pso(sphere, config, method="inertia_lbest_ring")
    assert global_result.best_history != ring_result.best_history


def test_gbest_variant_supports_two_particles_but_ring_requires_a_valid_neighborhood():
    config = PSOConfig(
        lower_bounds=(-1.0,), upper_bounds=(1.0,),
        population_size=2, max_evaluations=4, seed=17,
    )
    result = minimize_pso(sphere, config, method="classic_gbest")
    assert result.function_evaluations == 4
    with pytest.raises(ValueError, match="ring_radius is too large"):
        minimize_pso(sphere, config, method="inertia_lbest_ring")


def test_pso_raises_instead_of_propagating_nonfinite_internal_state():
    config = PSOConfig(
        lower_bounds=(-1.0,), upper_bounds=(1.0,),
        population_size=4, max_evaluations=8, seed=4,
        cognitive=1e308, social=1e308, velocity_limit_fraction=None,
    )
    with pytest.raises((ValueError, FloatingPointError)):
        minimize_pso(sphere, config, method="inertia_gbest")


def test_clpso_uses_published_parameters_and_frozen_snapshot_semantics():
    metadata = pso_method_metadata("clpso", make_config())
    assert metadata["clpso_acceleration"] == pytest.approx(1.49445)
    assert metadata["clpso_refresh_gap"] == 7
    assert metadata["inertia_max"] == pytest.approx(0.9)
    assert metadata["inertia_min"] == pytest.approx(0.4)
    assert metadata["velocity_limit_fraction"] == pytest.approx(0.2)
    assert metadata["update_semantics"] == "frozen-generation-snapshot"


def test_clpso_requires_three_particles_for_tournament_selection():
    config = PSOConfig(
        lower_bounds=(-1.0,), upper_bounds=(1.0,),
        population_size=2, max_evaluations=4, seed=11,
    )
    with pytest.raises(ValueError, match="population_size >= 3"):
        minimize_pso(sphere, config, method="clpso")


def test_clpso_evaluates_only_points_inside_search_box():
    calls = 0

    def bounded_objective(point):
        nonlocal calls
        calls += 1
        assert all(-1.0 <= value <= 1.0 for value in point)
        return sum(value * value for value in point)

    config = PSOConfig(
        lower_bounds=(-1.0, -1.0), upper_bounds=(1.0, 1.0),
        population_size=12, max_evaluations=97, seed=91,
        initial_velocity_fraction=2.0, velocity_limit_fraction=None,
    )
    result = minimize_pso(bounded_objective, config, method="clpso")
    assert calls == result.function_evaluations == 97
    assert result.history_evaluations[-1] == 97
    assert all(b <= a for a, b in zip(result.best_history, result.best_history[1:]))


def test_clpso_respects_configured_initial_velocity_fraction():
    calls = []

    def objective(point):
        calls.append(tuple(point))
        return sum(value * value for value in point)

    config = PSOConfig(
        lower_bounds=(-2.0, -2.0), upper_bounds=(2.0, 2.0),
        population_size=8, max_evaluations=24, seed=333,
        initial_velocity_fraction=0.0,
    )
    result = minimize_pso(objective, config, method="clpso")
    assert result.function_evaluations == len(calls) == 24
    assert all(len(point) == 2 for point in calls)


def test_pso_validates_objective_before_dispatching_clpso():
    config = PSOConfig(
        lower_bounds=(-1.0,), upper_bounds=(1.0,),
        population_size=4, max_evaluations=8,
    )
    with pytest.raises(TypeError, match="objective must be callable"):
        minimize_pso(None, config, method="clpso")  # type: ignore[arg-type]


def test_slpso_uses_exact_fe_budget_and_keeps_every_evaluated_point_in_bounds():
    calls = 0

    def bounded_objective(point):
        nonlocal calls
        calls += 1
        assert all(-1.0 <= value <= 1.0 for value in point)
        return sum(value * value for value in point)

    config = PSOConfig(
        lower_bounds=(-1.0, -1.0, -1.0), upper_bounds=(1.0, 1.0, 1.0),
        population_size=12, max_evaluations=97, seed=902,
    )
    result = minimize_pso(bounded_objective, config, method="slpso")
    assert calls == result.function_evaluations == 97
    assert result.history_evaluations[0] == 12
    assert result.history_evaluations[-1] == 97
    assert all(b <= a for a, b in zip(result.best_history, result.best_history[1:]))


def test_slpso_is_repeatable_and_requires_at_least_two_particles():
    config = PSOConfig(
        lower_bounds=(-2.0, -2.0), upper_bounds=(2.0, 2.0),
        population_size=8, max_evaluations=64, seed=1234,
    )
    first = minimize_pso(sphere, config, method="slpso")
    second = minimize_pso(sphere, config, method="slpso")
    assert first.best_history == second.best_history
    assert first.best_position == second.best_position
    tiny = PSOConfig(
        lower_bounds=(-1.0,), upper_bounds=(1.0,),
        population_size=2, max_evaluations=4,
    )
    tiny_result = minimize_pso(sphere, tiny, method="slpso")
    assert tiny_result.function_evaluations == 4
