from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis import analyze_dt_sensitivity as analyzer  # noqa: E402

_DRIVER_PATH = ROOT / "experiments/06_dt_sensitivity/run_dt_sensitivity.py"
_SPEC = importlib.util.spec_from_file_location("run_dt_sensitivity", _DRIVER_PATH)
assert _SPEC is not None and _SPEC.loader is not None
DRIVER = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(DRIVER)


def test_predeclared_dt_screening_design_and_budget(tmp_path, capsys):
    config_path = ROOT / "configs/campaigns/cpso_dt_sensitivity_screening.json"
    config = DRIVER.load_config(config_path)
    jobs = DRIVER.planned_runs(config)
    fe_per_run = config["population_size"] * (config["iterations"] + 1)
    assert len(jobs) == 1800
    assert fe_per_run == 32064
    assert len(jobs) * fe_per_run == 57_715_200
    assert len({(b, d, s) for b, d, s, _ in jobs}) == 6 * 2 * 10
    assert 30.0 == config["reference_dt"]

    output = tmp_path / "dry-run-output"
    assert DRIVER.execute(config_path, output, dry_run=True) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["expected_run_count"] == 1800
    assert plan["expected_total_function_evaluations"] == 57_715_200
    assert not output.exists()  # dry-run is read-only


def _synthetic_records(config: dict) -> list[dict]:
    records = []
    population = int(config["population_size"])
    iterations = int(config["iterations"])
    for benchmark_index, benchmark in enumerate(config["benchmarks"]):
        for dimension in config["dimensions"]:
            for seed in config["seeds"]:
                for dt in config["dt_values"]:
                    base = 1.0 + benchmark_index + dimension / 100.0 + seed / 1000.0
                    best = base * float(dt) ** 0.1
                    history = [best + 2.0, best + 1.0, best]
                    records.append({
                        "benchmark": benchmark,
                        "dimension": dimension,
                        "seed": seed,
                        "dt": dt,
                        "topology": "global",
                        "population_size": population,
                        "iterations": iterations,
                        "function_evaluations": population * (iterations + 1),
                        "best_value": best,
                        "best_position": [0.0] * dimension,
                        "history_evaluations": [population * (k + 1) for k in range(iterations + 1)],
                        "best_history": history,
                        "elapsed_seconds": 0.1 + dt / 1000.0,
                    })
    return records


def test_dt_analyzer_validates_factorial_and_emits_result_figures(tmp_path):
    config = {
        "campaign_id": "synthetic_dt_test",
        "benchmarks": ["sphere", "rastrigin", "ackley"],
        "dimensions": [2, 4],
        "seeds": [1, 2, 3],
        "dt_values": [0.5, 1.0, 2.0],
        "reference_dt": 1.0,
        "population_size": 4,
        "iterations": 2,
        "analysis_seed": 19,
    }
    input_dir = tmp_path / "raw"
    input_dir.mkdir()
    (input_dir / "study_config.json").write_text(
        json.dumps({"study_config": config}), encoding="utf-8"
    )
    (input_dir / "runs.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in _synthetic_records(config)),
        encoding="utf-8",
    )

    output = tmp_path / "analysis"
    figures = tmp_path / "figures"
    summary = analyzer.analyze(input_dir, output, figures, bootstrap_replicates=300)
    assert summary["validation"]["n_runs"] == 54
    assert (output / "dt_problem_summary.csv").is_file()
    assert (output / "dt_family_effects.csv").is_file()
    assert (output / "dt_family_profile.csv").is_file()
    assert not (output / "dt_paired_tests.csv").exists()
    generated = sorted(path.name for path in figures.glob("*.pdf"))
    assert generated == [
        "fig_dt_sensitivity_family_profile.pdf",
        "fig_dt_sensitivity_problem_heatmap.pdf",
    ]
    assert all(path.read_bytes().startswith(b"%PDF-") for path in figures.glob("*.pdf"))


def test_dt_analyzer_rejects_missing_factorial_cell(tmp_path):
    config = {
        "campaign_id": "synthetic_dt_incomplete",
        "benchmarks": ["sphere", "rastrigin"],
        "dimensions": [2],
        "seeds": [1, 2],
        "dt_values": [0.5, 1.0, 2.0],
        "reference_dt": 1.0,
        "population_size": 4,
        "iterations": 2,
    }
    input_dir = tmp_path / "raw"
    input_dir.mkdir()
    records = _synthetic_records(config)
    records.pop()
    (input_dir / "study_config.json").write_text(json.dumps({"study_config": config}), encoding="utf-8")
    (input_dir / "runs.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in records), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="Incomplete factorial"):
        analyzer.load_and_validate(input_dir)
