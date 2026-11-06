from __future__ import annotations

import json
from pathlib import Path
import sys
import types

import pytest

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = ROOT / "experiments" / "05_external_benchmarks"
if str(EXPERIMENT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT))

from common import (  # noqa: E402
    METHODS,
    checkpoint_values,
    deterministic_problem_seed,
    estimate_bbob_campaign,
    run_method,
    write_campaign_manifest,
)
from cpso_serial.benchmarks.cec2013_lsgo import CEC2013LSGOAdapter  # noqa: E402
from cpso_serial.benchmarks.coco_bbob import wrap_coco_problem  # noqa: E402
from run_bbob import execute as execute_bbob  # noqa: E402
from run_cec2013_lsgo import execute as execute_cec  # noqa: E402


def test_problem_seed_is_stable_and_problem_specific():
    assert deterministic_problem_seed("bbob_f01_i01_d02", 9) == deterministic_problem_seed(
        "bbob_f01_i01_d02", 9
    )
    assert deterministic_problem_seed("bbob_f01_i01_d02", 9) != deterministic_problem_seed(
        "bbob_f02_i01_d02", 9
    )


def test_run_method_enforces_equal_fe_budget_and_custom_cpsp_api():
    calls = 0

    def objective(point):
        nonlocal calls
        calls += 1
        return sum(value * value for value in point)

    result = run_method(
        "cpso_continuous", objective, (-2.0, -2.0), (2.0, 2.0),
        population_size=8, max_evaluations=48, seed=121,
    )
    assert result.function_evaluations == calls == 48
    with pytest.raises(ValueError, match="multiple of population_size"):
        run_method(
            "cpso_continuous", objective, (-2.0,), (2.0,),
            population_size=8, max_evaluations=49, seed=121,
        )


def test_run_method_exposes_all_pso_reference_variants():
    assert set(METHODS) == {
        "cpso_continuous", "pso_classic_gbest", "pso_inertia_gbest",
        "pso_constriction_gbest", "pso_inertia_lbest_ring", "pso_clpso", "pso_slpso",
    }
    for method in METHODS:
        result = run_method(
            method, lambda x: sum(value * value for value in x),
            (-2.0, -2.0), (2.0, 2.0),
            population_size=8, max_evaluations=48, seed=19,
        )
        assert result.function_evaluations == 48


def test_checkpoint_values_require_exact_fe_milestones():
    result = types.SimpleNamespace(
        history_evaluations=(4, 8, 12), best_history=(3.0, 2.0, 1.0)
    )
    assert checkpoint_values(result, [4, 12]) == {"4": 3.0, "12": 1.0}
    with pytest.raises(RuntimeError, match="exactly at"):
        checkpoint_values(result, [10])


def test_coco_problem_adapter_checks_bounds_and_metadata():
    problem = types.SimpleNamespace(
        id="bbob_f01_i01_d02", name="Sphere", dimension=2,
        lower_bounds=(-5.0, -5.0), upper_bounds=(5.0, 5.0),
    )
    view = wrap_coco_problem(problem)
    assert view.problem_id == "bbob_f01_i01_d02"
    assert view.dimension == 2
    assert view.objective is problem
    broken = types.SimpleNamespace(
        id="broken", name="broken", dimension=2,
        lower_bounds=(-1.0,), upper_bounds=(1.0, 1.0),
    )
    with pytest.raises(RuntimeError):
        wrap_coco_problem(broken)


def test_cec_adapter_supports_legacy_get_info_and_get_function(monkeypatch):
    module = types.ModuleType("cec2013lsgo.cec2013")

    class Benchmark:
        def get_info(self, function_id):
            assert function_id == 1
            return {"dimension": 2, "lower": -100.0, "upper": 100.0, "best": 0.0}

        def get_function(self, function_id):
            assert function_id == 1
            return lambda x: sum(float(v) ** 2 for v in x)

    module.Benchmark = Benchmark
    package = types.ModuleType("cec2013lsgo")
    package.cec2013 = module
    monkeypatch.setitem(sys.modules, "cec2013lsgo", package)
    monkeypatch.setitem(sys.modules, "cec2013lsgo.cec2013", module)
    problem = CEC2013LSGOAdapter().get_problem(1)
    assert problem.dimension == 2
    assert problem.objective((2.0, 3.0)) == 13.0


def test_bbob_and_cec_smoke_dry_runs_are_side_effect_free(tmp_path):
    bbob_config = ROOT / "configs/benchmarks/bbob_smoke.json"
    cec_config = ROOT / "configs/benchmarks/cec2013_smoke.json"
    bbob_output = tmp_path / "bbob"
    cec_output = tmp_path / "cec"
    assert execute_bbob(bbob_config, bbob_output, dry_run=True) == 0
    assert execute_cec(cec_config, cec_output, dry_run=True) == 0
    assert not bbob_output.exists()
    assert not cec_output.exists()


def test_bbob_config_has_exact_standard_budget_and_cec_matches_official_budget():
    bbob = json.loads((ROOT / "configs/benchmarks/bbob_confirmatory.json").read_text())
    assert bbob["budget_multiplier"] == 10_000
    assert bbob["population_size"] == 40
    cec = json.loads((ROOT / "configs/benchmarks/cec2013_confirmatory.json").read_text())
    assert len(cec["function_ids"]) == 15
    assert len(cec["seeds"]) == 25
    assert cec["max_evaluations"] == 3_000_000
    assert cec["milestones"] == [120_000, 600_000, 3_000_000]


def test_cec_adapter_supports_function_bound_wrapper_api(monkeypatch):
    module = types.ModuleType("cec2013lsgo.cec2013")

    class BoundBenchmark:
        def __init__(self, function_id):
            assert function_id == 3

        def get_info(self):
            return {"dimension": 2, "lower": -5.0, "upper": 5.0, "best": 0.0}

        def eval(self, point):
            return sum(float(value) ** 2 for value in point)

    module.Benchmark = BoundBenchmark
    package = types.ModuleType("cec2013lsgo")
    package.cec2013 = module
    monkeypatch.setitem(sys.modules, "cec2013lsgo", package)
    monkeypatch.setitem(sys.modules, "cec2013lsgo.cec2013", module)

    problem = CEC2013LSGOAdapter().get_problem(3)
    assert problem.dimension == 2
    assert problem.objective((3.0, 4.0)) == 25.0


def test_bbob_driver_runs_and_resumes_against_controlled_coco_contract(tmp_path, monkeypatch):
    import run_bbob

    class FakeProblem:
        def __init__(self):
            self.id = "bbob-largescale_f01_i01_d20"
            self.name = "Fake large-scale sphere"
            self.dimension = 2
            self.lower_bounds = (-5.0, -5.0)
            self.upper_bounds = (5.0, 5.0)
            self.evaluations = 0
            self.observers = []

        def observe_with(self, observer):
            self.observers.append(observer)

        def __call__(self, point):
            self.evaluations += 1
            return sum(float(value) ** 2 for value in point)

    class FakeSuite:
        def __init__(self, suite_name, instances, options):
            assert suite_name == "bbob-largescale"
            assert instances == "instances: 1"
            assert options == "dimensions: 20 function_indices: 1"

        def __iter__(self):
            return iter([FakeProblem()])

    class FakeObserver:
        def __init__(self, suite_name, options):
            assert suite_name == "bbob-largescale"
            self.options = options

    fake_cocoex = types.SimpleNamespace(
        known_suite_names=["bbob", "bbob-largescale"],
        Suite=FakeSuite,
        Observer=FakeObserver,
    )
    monkeypatch.setattr(run_bbob, "load_coco_experiment", lambda: fake_cocoex)

    config = tmp_path / "bbob-ls.json"
    config.write_text(json.dumps({
        "campaign_id": "fake-bbob-ls",
        "suite": "bbob-largescale",
        "instances": "instances: 1",
        "suite_options": "dimensions: 20 function_indices: 1",
        "population_size": 8,
        "budget_multiplier": 8,
        "seed": 7,
        "methods": ["cpso_continuous", "pso_inertia_gbest"],
    }), encoding="utf-8")
    output = tmp_path / "bbob-output"

    assert run_bbob.execute(config, output, dry_run=False) == 0
    first_records = {}
    for method in ("cpso_continuous", "pso_inertia_gbest"):
        path = output / "records" / f"{method}.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        assert len(rows) == 1
        assert rows[0]["suite"] == "bbob-largescale"
        assert rows[0]["function_evaluations"] == rows[0]["coco_evaluations"] == 16
        first_records[method] = len(rows)

    # A completed run is skipped on resume; it is not appended a second time.
    assert run_bbob.execute(config, output, dry_run=False) == 0
    for method, count in first_records.items():
        path = output / "records" / f"{method}.jsonl"
        assert len(path.read_text().splitlines()) == count


def test_cec_driver_runs_against_controlled_1000d_adapter_and_resumes(tmp_path, monkeypatch):
    import run_cec2013_lsgo

    class FakeProblem:
        dimension = 1000
        lower_bounds = (-1.0,) * 1000
        upper_bounds = (1.0,) * 1000
        name = "CEC2013-LSGO-F01-FAKE"

        @staticmethod
        def objective(point):
            return sum(float(value) ** 2 for value in point)

    class FakeAdapter:
        def get_problem(self, function_id):
            assert function_id == 1
            return FakeProblem()

    monkeypatch.setattr(run_cec2013_lsgo, "CEC2013LSGOAdapter", FakeAdapter)
    config = tmp_path / "cec-smoke.json"
    config.write_text(json.dumps({
        "campaign_id": "fake-cec",
        "function_ids": [1],
        "seeds": [5],
        "population_size": 8,
        "max_evaluations": 16,
        "milestones": [8, 16],
        "methods": ["cpso_continuous", "pso_inertia_gbest"],
    }), encoding="utf-8")
    output = tmp_path / "cec-output"

    assert run_cec2013_lsgo.execute(config, output, dry_run=False) == 0
    rows = [json.loads(line) for line in (output / "records.jsonl").read_text().splitlines()]
    assert len(rows) == 2
    assert {row["method"] for row in rows} == {"cpso_continuous", "pso_inertia_gbest"}
    assert all(row["dimension"] == 1000 for row in rows)
    assert all(row["function_evaluations"] == 16 for row in rows)
    assert all(set(row["checkpoints"]) == {"8", "16"} for row in rows)

    assert run_cec2013_lsgo.execute(config, output, dry_run=False) == 0
    assert len((output / "records.jsonl").read_text().splitlines()) == 2


def test_bbob_largescale_confirmatory_config_matches_official_dimensions_and_suite():
    config = json.loads(
        (ROOT / "configs/benchmarks/bbob_largescale_confirmatory.json").read_text()
    )
    assert config["suite"] == "bbob-largescale"
    assert config["suite_options"] == (
        "dimensions: 20,40,80,160,320,640 function_indices: 1-24"
    )
    assert config["instances"] == "instances: 1-15"
    assert config["budget_multiplier"] == 10_000


def test_preconfirmatory_configs_are_bounded_stages_before_full_campaign():
    bbob = json.loads(
        (ROOT / "configs/benchmarks/bbob_preconfirmatory.json").read_text()
    )
    bbob_ls = json.loads(
        (ROOT / "configs/benchmarks/bbob_largescale_preconfirmatory.json").read_text()
    )
    cec = json.loads(
        (ROOT / "configs/benchmarks/cec2013_preconfirmatory.json").read_text()
    )
    assert bbob["budget_multiplier"] < 10_000
    assert bbob["instances"] == "instances: 1-5"
    assert bbob_ls["suite"] == "bbob-largescale"
    assert bbob_ls["budget_multiplier"] == 250
    assert bbob_ls["instances"] == "instances: 1-3"
    assert len(cec["function_ids"]) == 5
    assert len(cec["seeds"]) == 5
    assert cec["max_evaluations"] == 120_000
    assert cec["milestones"] == [120_000]



def test_external_campaign_manifest_is_pinned_and_rejects_incompatible_resume(tmp_path):
    config = tmp_path / "config.json"
    config.write_text('{"campaign":"one"}', encoding="utf-8")
    manifest_path = tmp_path / "campaign_manifest.json"
    plan = {"campaign_id": "one", "suite": "bbob", "methods": ["cpso_continuous"]}
    write_campaign_manifest(manifest_path, plan, config)
    first = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert first["provenance"]["project_version"] == "0.7.1"
    assert first["provenance"]["native"]["c_abi"] == 6
    assert len(first["provenance"]["source_code_sha256"]) == 64

    # An unchanged manifest is accepted without rewriting it.
    write_campaign_manifest(manifest_path, plan, config)
    assert json.loads(manifest_path.read_text(encoding="utf-8")) == first

    config.write_text('{"campaign":"changed"}', encoding="utf-8")
    with pytest.raises(RuntimeError, match="Cannot resume external benchmark campaign"):
        write_campaign_manifest(manifest_path, plan, config)



def test_bbob_preflight_counts_runs_and_rejects_incompatible_fe_budgets():
    plan = estimate_bbob_campaign(
        "bbob-largescale",
        "instances: 1-3",
        "dimensions: 20,80 function_indices: 1-2",
        budget_multiplier=250,
        population_size=40,
        method_count=5,
    )
    assert plan["expected_problem_count"] == 12
    assert plan["expected_method_runs"] == 60
    assert plan["expected_total_function_evaluations"] == 750_000
    with pytest.raises(ValueError, match="positive multiple of population_size"):
        estimate_bbob_campaign(
            "bbob-largescale", "instances: 1", "dimensions: 20 function_indices: 1",
            budget_multiplier=1, population_size=40, method_count=5,
        )
