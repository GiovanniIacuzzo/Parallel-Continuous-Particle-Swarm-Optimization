from __future__ import annotations

import json
from pathlib import Path

from analysis.postprocess_external_benchmarks import _average_ranks, _rank_campaign


def test_average_ranks_assigns_average_rank_to_ties() -> None:
    ranks = _average_ranks({"a": 1.0, "b": 1.0, "c": 3.0}, ["a", "b", "c"])
    assert ranks == {"a": 1.5, "b": 1.5, "c": 3.0}


def test_bbob_postprocessor_uses_only_complete_matched_groups(tmp_path: Path) -> None:
    campaign = tmp_path / "bbob_pilot"
    records_dir = campaign / "records"
    records_dir.mkdir(parents=True)
    manifest = {
        "suite": "bbob",
        "methods": ["cpso", "baseline"],
        "expected_method_runs": 4,
    }
    (campaign / "campaign_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    records = [
        {"suite": "bbob", "problem_id": "f1_D5_i1", "function_id": 1, "seed": 7,
         "method": "cpso", "best_value": 1.0},
        {"suite": "bbob", "problem_id": "f1_D5_i1", "function_id": 1, "seed": 7,
         "method": "baseline", "best_value": 2.0},
        # Incomplete second problem must not be treated as a paired comparison.
        {"suite": "bbob", "problem_id": "f2_D5_i1", "function_id": 2, "seed": 8,
         "method": "cpso", "best_value": 0.5},
    ]
    (records_dir / "cpso.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in records if row["method"] == "cpso"),
        encoding="utf-8",
    )
    (records_dir / "baseline.jsonl").write_text(
        json.dumps(records[1]) + "\n", encoding="utf-8"
    )

    summary, completeness = _rank_campaign(campaign, manifest)
    assert completeness["records_loaded"] == 3
    assert completeness["complete_problem_seed_groups"] == 1
    assert completeness["incomplete_problem_seed_groups"] == 1
    assert {row["method"]: row["mean_rank_all_groups"] for row in summary} == {
        "cpso": 1.0,
        "baseline": 2.0,
    }


def test_cec_postprocessor_ranks_each_declared_checkpoint(tmp_path: Path) -> None:
    campaign = tmp_path / "cec_pilot"
    campaign.mkdir()
    manifest = {
        "suite": "CEC2013-LSGO",
        "methods": ["cpso", "baseline"],
        "expected_run_count": 2,
        "milestones": [100, 200],
    }
    (campaign / "campaign_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    rows = [
        {"suite": "CEC2013-LSGO", "problem_id": "F1", "function_id": 1, "seed": 3,
         "method": "cpso", "best_value": 1.0, "checkpoints": {"100": 2.0, "200": 1.0}},
        {"suite": "CEC2013-LSGO", "problem_id": "F1", "function_id": 1, "seed": 3,
         "method": "baseline", "best_value": 2.0, "checkpoints": {"100": 1.0, "200": 2.0}},
    ]
    (campaign / "records.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )

    summary, completeness = _rank_campaign(campaign, manifest)
    ranks = {(row["metric"], row["method"]): row["mean_rank_all_groups"] for row in summary}
    assert completeness["complete_problem_seed_groups"] == 1
    assert ranks[("best_at_100_FE", "cpso")] == 2.0
    assert ranks[("best_at_100_FE", "baseline")] == 1.0
    assert ranks[("best_at_200_FE", "cpso")] == 1.0
    assert ranks[("best_at_200_FE", "baseline")] == 2.0
