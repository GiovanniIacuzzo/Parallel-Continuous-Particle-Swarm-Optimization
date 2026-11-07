from __future__ import annotations

import json
import math
import pytest
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS = ROOT / "experiments" / "04_architectures"
if str(EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS))

from analyze_research_campaign import (  # noqa: E402
    _analyze_quality_study,
    _friedman_permutation_p,
    _wilcoxon,
    function_family_bootstrap_median,
    holm_adjust,
    hierarchical_bootstrap_median,
    validate_completed_study,
)
from environment_metadata import collect_environment_metadata  # noqa: E402
from checkpoint_io import atomic_write_json, read_jsonl_checkpoint  # noqa: E402
from run_research_campaign import count_runs, validate_campaign  # noqa: E402
import run_architecture_study as study_module  # noqa: E402
from run_architecture_study import (  # noqa: E402
    evaluations_to_target,
    execute_study,
    validate_study,
)


def test_holm_adjustment_is_monotone_and_preserves_input_order():
    adjusted = holm_adjust([0.04, 0.01, 0.03])
    assert adjusted == [0.06, 0.03, 0.06]
    assert all(0.0 <= value <= 1.0 for value in adjusted)


def test_hierarchical_bootstrap_is_deterministic_and_order_invariant():
    blocks = {
        ("sphere", 30): [-0.2, -0.1, 0.0],
        ("rastrigin", 30): [0.1, 0.2, 0.3],
        ("rosenbrock", 100): [-0.1, 0.0, 0.1],
    }
    first = hierarchical_bootstrap_median(blocks, seed=1234, replicates=500)
    second = hierarchical_bootstrap_median(blocks, seed=1234, replicates=500)
    assert first == second
    assert first[0] <= first[1]


def test_function_family_bootstrap_resamples_nested_dimensions_and_seeds():
    blocks = {
        ("sphere", 30): [-0.2, -0.1, 0.0],
        ("sphere", 100): [-0.1, 0.0, 0.1],
        ("rastrigin", 30): [0.1, 0.2, 0.3],
        ("rastrigin", 100): [0.2, 0.3, 0.4],
        ("ackley", 30): [-0.1, 0.0, 0.1],
        ("ackley", 100): [-0.2, -0.1, 0.0],
    }
    first = function_family_bootstrap_median(blocks, seed=51, replicates=300)
    second = function_family_bootstrap_median(dict(reversed(list(blocks.items()))), seed=51, replicates=300)
    assert first == second
    assert first[0] <= first[1]



def test_exact_signed_rank_uses_family_level_sign_permutation():
    statistic, p_value = _wilcoxon([-1.0] * 6)
    assert statistic == 0.0
    assert p_value == 2.0 / 64.0
    assert _wilcoxon([0.0] * 6) == (0.0, 1.0)


def test_friedman_block_permutation_is_deterministic():
    ranks = [
        [1.0, 2.0, 3.0],
        [1.0, 3.0, 2.0],
        [2.0, 1.0, 3.0],
        [1.0, 2.0, 3.0],
        [2.0, 3.0, 1.0],
        [1.0, 2.0, 3.0],
    ]
    first = _friedman_permutation_p(ranks, seed=77, replicates=1000)
    second = _friedman_permutation_p(ranks, seed=77, replicates=1000)
    assert first == second
    assert 0.0 < first <= 1.0


def test_quality_analysis_uses_function_families_as_inferential_units(tmp_path):
    functions = ["sphere", "rastrigin", "rosenbrock", "ackley", "griewank", "salomon"]
    methods = ["serial", "async", "islands"]
    study_item = {
        "id": "nested_stats_test",
        "study": {"methods": [{"name": name} for name in methods]},
        "comparisons": [{"method": "async", "reference": "serial", "metric": "quality"}],
    }
    records = []
    for family_index, function in enumerate(functions):
        for dimension in (30, 100):
            for seed in (1, 2, 3):
                base = 1.0 + family_index + dimension / 100.0 + seed / 1000.0
                records.extend([
                    {"benchmark": function, "dimension": dimension, "seed": seed,
                     "method": "serial", "best_value": base},
                    {"benchmark": function, "dimension": dimension, "seed": seed,
                     "method": "async", "best_value": base * 0.9},
                    {"benchmark": function, "dimension": dimension, "seed": seed,
                     "method": "islands", "best_value": base * 1.1},
                ])
    pairs, ranks, friedman = _analyze_quality_study(
        {"id": "test_stage"}, study_item, records, tmp_path
    )
    assert len(pairs) == 1
    assert pairs[0]["n_problem_blocks_descriptive"] == 12
    assert pairs[0]["n_independent_function_families"] == 6
    assert pairs[0]["wilcoxon_unit"] == "benchmark_function_family"
    assert pairs[0]["wilcoxon_p_holm"] == pairs[0]["wilcoxon_p_raw"]
    assert len(ranks) == 3
    assert all(row["n_independent_function_families"] == 6 for row in ranks)
    assert len(friedman) == 1
    assert friedman[0]["n_independent_function_families"] == 6
    assert 0.0 < friedman[0]["friedman_p_block_permutation"] <= 1.0


def test_environment_metadata_is_non_identifying_and_contains_build_context():
    metadata = collect_environment_metadata(native_version="0.7.1", native_build_info="test-build")
    assert metadata["native"]["version"] == "0.7.1"
    assert metadata["native"]["build_info"] == "test-build"
    assert "logical_count" in metadata["cpu"]
    assert "packages" in metadata
    assert "thread_environment" in metadata
    assert "hostname" not in metadata
    assert "home" not in str(metadata).lower()

def test_definitive_campaign_configuration_and_run_count():
    path = ROOT / "configs" / "campaigns" / "definitive_architecture_campaign.json"
    campaign = json.loads(path.read_text(encoding="utf-8"))
    validate_campaign(campaign)
    total = sum(
        count_runs(study_item["study"])
        for stage in campaign["stages"]
        for study_item in stage["studies"]
    )
    assert total == 9360


def test_smoke_campaign_configuration():
    path = ROOT / "configs" / "campaigns" / "campaign_smoke.json"
    campaign = json.loads(path.read_text(encoding="utf-8"))
    validate_campaign(campaign)
    assert sum(
        count_runs(item["study"])
        for stage in campaign["stages"]
        for item in stage["studies"]
    ) == 27


def test_validate_completed_study_checks_exact_fe_budget_and_coverage():
    study = {
        "benchmarks": ["sphere"],
        "dimensions": [2],
        "population_size": 4,
        "iterations": 3,
        "seeds": [7],
        "methods": [{"name": "serial_global"}],
    }
    good = [{
        "run_key": "k1",
        "benchmark": "sphere",
        "dimension": 2,
        "seed": 7,
        "method": "serial_global",
        "function_evaluations": 16,
        "history_evaluations": [4, 8, 12, 16],
        "best_history": [5.0, 4.0, 3.0, 2.0],
        "best_value": 2.0,
        "elapsed_seconds": 0.25,
    }]
    validate_completed_study(study, good, "test")

    bad_fe = [dict(good[0], function_evaluations=15)]
    try:
        validate_completed_study(study, bad_fe, "test")
    except RuntimeError as exc:
        assert "FE mismatch" in str(exc)
    else:
        raise AssertionError("FE-budget mismatch was not rejected")

    wrong_run = [dict(good[0], benchmark="rastrigin")]
    try:
        validate_completed_study(study, wrong_run, "test")
    except RuntimeError as exc:
        assert "unexpected experiment" in str(exc)
    else:
        raise AssertionError("Unexpected benchmark record was not rejected")

    bad_history = [dict(good[0], best_history=[5.0, 4.0, 5.0, 2.0])]
    try:
        validate_completed_study(study, bad_history, "test")
    except RuntimeError as exc:
        assert "best-so-far history increased" in str(exc)
    else:
        raise AssertionError("Non-monotone incumbent history was not rejected")


def test_atomic_manifest_write_replaces_complete_json(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text('{"old": true}\n', encoding="utf-8")

    atomic_write_json(path, {"schema": 2, "runs": 27})

    assert json.loads(path.read_text(encoding="utf-8")) == {"schema": 2, "runs": 27}
    assert not list(tmp_path.glob(".manifest.json.*.tmp"))


def test_checkpoint_reader_recovers_only_incomplete_final_record(tmp_path):
    path = tmp_path / "runs.jsonl"
    first = {"run_key": "complete-1", "value": 1}
    payload = (json.dumps(first) + "\n").encode() + b'{"run_key":"partial'
    path.write_bytes(payload)

    records, state = read_jsonl_checkpoint(path, repair=False)
    assert records == [first]
    assert state["discarded_partial_record"] is True
    assert path.read_bytes() == payload  # read-only mode must not mutate the file

    records, state = read_jsonl_checkpoint(path, repair=True)
    assert records == [first]
    assert state["discarded_bytes"] > 0
    assert path.read_bytes() == (json.dumps(first) + "\n").encode()


def test_checkpoint_reader_rejects_corruption_in_complete_records(tmp_path):
    path = tmp_path / "runs.jsonl"
    path.write_text('{"run_key":"ok"}\n{broken}\n', encoding="utf-8")
    try:
        read_jsonl_checkpoint(path, repair=True)
    except ValueError as exc:
        assert "Corrupt JSONL checkpoint" in str(exc)
    else:
        raise AssertionError("Corrupt newline-terminated records must not be ignored")


def test_checkpoint_reader_normalizes_valid_unterminated_final_record(tmp_path):
    path = tmp_path / "runs.jsonl"
    record = {"run_key": "complete-without-newline"}
    path.write_text(json.dumps(record), encoding="utf-8")
    rows, state = read_jsonl_checkpoint(path, repair=True)
    assert rows == [record]
    assert state["added_final_newline"] is True
    assert path.read_bytes().endswith(b"\n")

def test_evaluations_to_target_returns_first_attainment_fe():
    assert evaluations_to_target([8, 16, 24], [2.0, 0.1, 1e-9], 1e-8) == 24
    assert evaluations_to_target([8, 16, 24], [1e-9, 1e-10, 1e-11], 1e-8) == 8
    assert evaluations_to_target([8, 16, 24], [2.0, 1.0, 0.1], 1e-8) is None


def test_evaluations_to_target_rejects_misaligned_histories():
    try:
        evaluations_to_target([8, 16], [1.0], 1e-8)
    except ValueError as exc:
        assert "equal lengths" in str(exc)
    else:
        raise AssertionError("misaligned histories were not rejected")


def _minimal_study():
    return {
        "benchmarks": ["sphere"],
        "dimensions": [2],
        "population_size": 4,
        "iterations": 1,
        "seeds": [7],
        "methods": [{"name": "serial_global", "architecture": "serial", "topology": "global"}],
    }


def test_architecture_study_dry_run_is_read_only_and_needs_no_native_library(tmp_path, monkeypatch, capsys):
    output = tmp_path / "must_not_be_created"

    def unexpected_native_call(*_args, **_kwargs):
        raise AssertionError("dry-run must not query the native library")

    monkeypatch.setattr(study_module, "native_version", unexpected_native_call)
    monkeypatch.setattr(study_module, "native_build_info", unexpected_native_call)

    returned = execute_study(_minimal_study(), output, dry_run=True)

    assert returned == output
    assert not output.exists()
    assert "1 runs" in capsys.readouterr().out


def test_architecture_study_rejects_duplicate_dimensions():
    study = _minimal_study()
    study["dimensions"] = [2, 2]
    try:
        validate_study(study)
    except ValueError as exc:
        assert "must not contain duplicates" in str(exc)
    else:
        raise AssertionError("duplicate dimensions were not rejected")


def test_architecture_study_rejects_non_integer_grid_shape():
    study = _minimal_study()
    study["population_size"] = 6
    study["methods"] = [{
        "name": "serial_grid",
        "architecture": "serial",
        "topology": "grid_von_neumann",
        "grid_shape": [2, 3.0],
    }]
    try:
        validate_study(study)
    except ValueError as exc:
        assert "two integer values" in str(exc)
    else:
        raise AssertionError("non-integer grid shape was not rejected")


def test_quality_analysis_marks_exactly_tied_architectures_as_degenerate(tmp_path):
    functions = ["sphere", "rastrigin", "rosenbrock", "ackley", "griewank", "salomon"]
    methods = ["serial", "master_slave", "particle_parallel"]
    study_item = {
        "id": "exact_equivalence_test",
        "study": {"methods": [{"name": name} for name in methods]},
        "comparisons": [
            {"method": method, "reference": "serial", "metric": "quality"}
            for method in methods[1:]
        ],
    }
    records = []
    for family_index, function in enumerate(functions):
        for dimension in (10, 30):
            for seed in (101, 202, 303):
                value = float(family_index + dimension / 100.0 + seed / 10000.0)
                for method in methods:
                    records.append({
                        "benchmark": function, "dimension": dimension, "seed": seed,
                        "method": method, "best_value": value,
                    })
    _, ranks, friedman = _analyze_quality_study(
        {"id": "exact_equivalence_stage"}, study_item, records, tmp_path
    )
    assert len(ranks) == 3
    assert all(row["average_rank"] == 2.0 for row in ranks)
    assert len(friedman) == 1
    assert friedman[0]["friedman_status"] == "all_methods_tied_no_inferential_test"
    assert math.isnan(friedman[0]["friedman_p_block_permutation"])


def _small_performance_study():
    study = {
        "benchmarks": ["sphere_work"],
        "dimensions": [2],
        "population_size": 4,
        "iterations": 2,
        "seeds": [101, 202, 303],
        "work_iterations": 16,
        "check_equivalence": True,
        "methods": [
            {"name": "serial_global", "architecture": "serial", "topology": "global"},
            {"name": "ms_global_w2", "architecture": "master_slave", "topology": "global", "workers": 2},
        ],
    }
    return {
        "id": "speedup_test",
        "study": study,
        "comparisons": [{"method": "ms_global_w2", "reference": "serial_global", "metric": "runtime"}],
    }


def test_speedup_protocol_accepts_only_same_configuration_synchronous_executor_pairs():
    from run_research_campaign import validate_performance_study
    stage = {"id": "04_strong_scaling", "kind": "performance"}
    item = _small_performance_study()
    validate_performance_study(stage, item)

    bad_equivalence = json.loads(json.dumps(item))
    bad_equivalence["study"]["check_equivalence"] = False
    with pytest.raises(ValueError, match="check_equivalence"):
        validate_performance_study(stage, bad_equivalence)

    bad_topology = json.loads(json.dumps(item))
    bad_topology["study"]["methods"][1]["topology"] = "ring"
    with pytest.raises(ValueError, match="topology"):
        validate_performance_study(stage, bad_topology)

    bad_algorithm = json.loads(json.dumps(item))
    bad_algorithm["study"]["methods"][1]["dt"] = 0.7
    with pytest.raises(ValueError, match="algorithmic settings"):
        validate_performance_study(stage, bad_algorithm)

    bad_architecture = json.loads(json.dumps(item))
    bad_architecture["study"]["methods"][1]["architecture"] = "asynchronous"
    with pytest.raises(ValueError, match="master_slave or particle_parallel"):
        validate_performance_study(stage, bad_architecture)

    with pytest.raises(ValueError, match="performance stage"):
        validate_performance_study({"id": "quality", "kind": "quality"}, item)


def test_speedup_is_median_of_paired_end_to_end_ratios_and_checks_fe(tmp_path):
    from analyze_research_campaign import _analyze_performance_study
    item = _small_performance_study()
    stage = {"id": "04_strong_scaling", "kind": "performance"}
    records = []
    serial_times = [6.0, 8.0, 10.0]
    parallel_times = [2.0, 2.0, 5.0]
    for seed, serial_time, parallel_time in zip(item["study"]["seeds"], serial_times, parallel_times):
        for method, architecture, workers, elapsed in (
            ("serial_global", "serial", 1, serial_time),
            ("ms_global_w2", "master_slave", 2, parallel_time),
        ):
            records.append({
                "benchmark": "sphere_work", "dimension": 2, "seed": seed,
                "method": method, "architecture": architecture, "topology": "global",
                "workers": workers, "population_size": 4, "iterations": 2,
                "function_evaluations": 12, "elapsed_seconds": elapsed,
            })

    result = _analyze_performance_study(stage, item, records)
    assert len(result) == 1
    assert result[0]["n_paired_runs"] == 3
    assert result[0]["median_speedup"] == pytest.approx(3.0)  # median([3, 4, 2])
    assert result[0]["median_serial_runtime_seconds"] == pytest.approx(8.0)
    assert result[0]["median_parallel_runtime_seconds"] == pytest.approx(2.0)
    assert result[0]["parallel_efficiency"] == pytest.approx(1.5)

    broken = [dict(row) for row in records]
    broken[1]["function_evaluations"] = 11
    with pytest.raises(RuntimeError, match="FE budget mismatch"):
        _analyze_performance_study(stage, item, broken)

    missing = records[:-1]
    with pytest.raises(RuntimeError, match="Missing paired speedup record"):
        _analyze_performance_study(stage, item, missing)


def test_topology_scaling_uses_one_matched_serial_reference_per_topology():
    path = ROOT / "configs/campaigns/cpso_topology_expansion_confirmatory.json"
    campaign = json.loads(path.read_text(encoding="utf-8"))
    validate_campaign(campaign)
    performance = next(stage for stage in campaign["stages"] if stage["kind"] == "performance")
    item = performance["studies"][0]
    assert len(item["comparisons"]) == 14
    assert len({comparison["reference"] for comparison in item["comparisons"]}) == 7
