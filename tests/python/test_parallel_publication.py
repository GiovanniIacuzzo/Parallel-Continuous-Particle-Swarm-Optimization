"""Standalone integration test: all plotted values come from synthetic fixtures.

Run in a real checkout: python -m pytest -q tests/python/test_parallel_publication.py
The fixture is deliberately synthetic and is never shipped as publication data.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from plot_results.plot_parallel_publication import generate, load_observations


def _fixture(tmp_path: Path, *, break_trace: bool = False, missing: bool = False):
    campaign = tmp_path / "campaign"
    campaign.mkdir()
    topologies = ("global", "ring")
    work_cases = (("01_speedup_light", 128), ("02_speedup_heavy", 4096))
    manifest_stages = []
    for stage, work in work_cases:
        studies = []
        for topo in topologies:
            methods = [{"name": f"serial_{topo}", "architecture": "serial", "topology": topo}]
            methods += [{"name": f"{prefix}_{topo}_w{p}", "architecture": arch,
                         "topology": topo, "workers": p}
                        for arch, prefix in (("master_slave", "ms"), ("particle_parallel", "pp"))
                        for p in (1, 2, 4, 8)]
            studies.append({"id": topo, "study": {
                "benchmarks": ["sphere_work"], "check_equivalence": True,
                "dimensions": [30, 100], "population_size": 64,
                "iterations": 60, "work_iterations": work,
                "seeds": list(range(8)), "methods": methods,
            }})
            path = campaign / stage / topo
            path.mkdir(parents=True)
            with (path / "runs.jsonl").open("w") as f:
                for dim in (30, 100):
                    for seed in range(8):
                        base = 1.0 + seed * .02 + (dim == 100) * .4 + (work == 4096) * 5
                        for method in methods:
                            is_serial = method["architecture"] == "serial"
                            p = method.get("workers", 1)
                            factor = 1 if is_serial else (0.9 + .75 * p / (1 + .03 * seed))
                            record = {
                                "benchmark": "sphere_work", "dimension": dim,
                                "seed": seed, "method": method["name"],
                                "architecture": method["architecture"],
                                "workers": p, "topology": topo,
                                "population_size": 64, "iterations": 60,
                                "function_evaluations": 3904,
                                "elapsed_seconds": base / factor,
                                "best_value": float(seed),
                                "best_position": [float(seed)],
                                "best_history": [float(seed + 1), float(seed)],
                                "history_evaluations": [64, 3904],
                            }
                            if break_trace and stage == "01_speedup_light" and topo == "ring" and seed == 7 and not is_serial and p == 8:
                                record["best_value"] += 1.0
                            if missing and stage == "01_speedup_light" and topo == "ring" and seed == 7 and not is_serial and p == 8:
                                continue
                            f.write(json.dumps(record) + "\n")
        manifest_stages.append({"id": stage, "kind": "performance", "studies": studies})
    (campaign / "campaign_manifest.json").write_text(json.dumps({
        "campaign": {"evidence_class": "exploratory", "stages": manifest_stages}
    }))
    return campaign


def test_generated_figures_have_real_matched_points(tmp_path):
    campaign = _fixture(tmp_path)
    out = tmp_path / "plots"
    obsolete = out / "01_speedup_light_global_scaling_foo.pdf"
    out.mkdir()
    obsolete.write_bytes(b"old figure")
    plots, warnings = generate(campaign, out, repetitions=500)
    assert not warnings
    assert len(plots) == 6
    assert all(p.stat().st_size > 2500 for p in plots)
    assert not obsolete.exists()
    with (out / "publication_scaling_aggregate.csv").open() as f:
        aggregate = list(csv.DictReader(f))
    assert len(aggregate) == 2 * 2 * 2 * 4  # D, workload, executor, p
    row = next(r for r in aggregate if r["dimension"] == "30" and
               r["work_iterations"] == "4096" and
               r["architecture"] == "master_slave" and r["workers"] == "4")
    assert float(row["efficiency"]) == pytest.approx(float(row["speedup"]) / 4)
    assert int(row["n_topologies"]) == 2
    assert int(row["min_pairs_per_topology"]) == 8


def test_rejects_non_equivalence(tmp_path):
    with pytest.raises(ValueError, match="Failed exact serial/parallel equivalence"):
        load_observations(_fixture(tmp_path, break_trace=True))


def test_rejects_incomplete_campaign(tmp_path):
    with pytest.raises(ValueError, match="Incomplete performance campaign"):
        load_observations(_fixture(tmp_path, missing=True))
