from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from analysis.postprocess_publication_figures import (
    aggregate_topology_effects_by_family,
    build_figures,
)

ROOT = Path(__file__).resolve().parents[2]


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _campaign_run_count(path: Path) -> int:
    campaign = json.loads(path.read_text(encoding="utf-8"))
    total = 0
    for stage in campaign["stages"]:
        for item in stage["studies"]:
            study = item["study"]
            total += (
                len(study["benchmarks"])
                * len(study["dimensions"])
                * len(study["seeds"])
                * len(study["methods"])
            )
    return total


def test_topology_expansion_campaign_sizes_are_frozen() -> None:
    pilot = ROOT / "configs/campaigns/cpso_topology_expansion_pilot.json"
    confirmatory = ROOT / "configs/campaigns/cpso_topology_expansion_confirmatory.json"
    assert _campaign_run_count(pilot) == 1_050
    assert _campaign_run_count(confirmatory) == 3_150


def test_topology_expansion_covers_seven_particle_graph_families() -> None:
    campaign = json.loads(
        (ROOT / "configs/campaigns/cpso_topology_expansion_pilot.json").read_text(encoding="utf-8")
    )
    quality = campaign["stages"][0]["studies"][0]["study"]
    topologies = {method["topology"] for method in quality["methods"]}
    assert topologies == {
        "global", "ring", "grid_von_neumann", "wheel", "random", "small_world", "scale_free"
    }


def test_topology_effect_plot_aggregation_uses_function_families_as_units() -> None:
    rows = []
    for benchmark, multiplier in (("sphere", 2.0), ("rastrigin", 4.0)):
        for dimension in (30, 100):
            for seed in (1, 2, 3):
                rows.extend([
                    {"benchmark": benchmark, "dimension": str(dimension), "seed": str(seed),
                     "topology": "global", "best_value": "1.0"},
                    {"benchmark": benchmark, "dimension": str(dimension), "seed": str(seed),
                     "topology": "ring", "best_value": str(multiplier)},
                ])
    aggregated = aggregate_topology_effects_by_family(rows, ["global", "ring"])
    assert set(aggregated["ring"]) == {"sphere", "rastrigin"}
    assert len(aggregated["ring"]) == 2  # not 12 dimension/seed cells
    assert aggregated["ring"]["sphere"] == pytest.approx(0.3010299957)
    assert aggregated["ring"]["rastrigin"] == pytest.approx(0.6020599913)


def test_publication_postprocessor_emits_only_confirmatory_result_figures(tmp_path: Path) -> None:
    results = tmp_path / "results"
    architecture = results / "raw/cpso_architecture_campaign_v1"
    strong_scaling = []
    for dimension in (30, 100):
        for method, workers, speedup in (
            ("ms_global_w1", 1, 0.95), ("ms_global_w2", 2, 1.55),
            ("pp_global_w1", 1, 0.92), ("pp_global_w2", 2, 1.70),
        ):
            strong_scaling.append({
                "stage": "04_strong_scaling", "study": "n64_work4096", "work_iterations": 4096,
                "population_size": 64, "dimension": dimension, "method": method,
                "workers": workers, "median_speedup": speedup + dimension / 1000.0,
                "bootstrap_95_low": speedup - 0.1, "bootstrap_95_high": speedup + 0.1,
                "parallel_efficiency": (speedup + dimension / 1000.0) / workers,
            })
    _write_csv(architecture / "strong_scaling_summary.csv", strong_scaling)

    run_rows = []
    method_timings = {
        "serial_global": (0.001, 0.001, 0.05, 0.01),
        "ms_global_w1": (0.001, 0.001, 0.03, 0.01),
        "ms_global_w2": (0.001, 0.001, 0.02, 0.01),
        "pp_global_w1": (0.001, 0.001, 0.03, 0.01),
        "pp_global_w2": (0.001, 0.001, 0.015, 0.01),
    }
    for method, components in method_timings.items():
        run_rows.append({
            "stage": "04_strong_scaling", "study": "n64_work4096", "benchmark": "sphere_work",
            "work_iterations": 4096, "population_size": 64, "dimension": 100, "method": method,
            "elapsed_seconds": sum(components) + 0.005,
            "initialization_seconds": components[0], "topology_setup_seconds": components[1],
            "evaluation_wall_seconds": components[2], "propagation_wall_seconds": components[3],
        })
    _write_csv(architecture / "campaign_runs.csv", run_rows)

    pair_rows = [
        {"stage": "02_asynchronous_dynamics", "method": "async_global", 
         "median_function_family_log10_delta": -0.1,
         "hierarchical_bootstrap_95_low": -0.2, "hierarchical_bootstrap_95_high": 0.0},
        {"stage": "03_island_dynamics", "method": "islands_ring", 
         "median_function_family_log10_delta": 0.05,
         "hierarchical_bootstrap_95_low": -0.05, "hierarchical_bootstrap_95_high": 0.15},
    ]
    _write_csv(architecture / "pairwise_effects.csv", pair_rows)
    equivalence_rows = [
        {"stage": "01_exact_quality_equivalence", "study": "native", "method": method,
         "architecture": architecture_name, "topology": "global", "workers": workers,
         "verified_pairs": 24, "expected_pairs": 24, "fraction_verified": 1.0,
         "status": "exact_match", "checked_fields": "best_value;best_position;best_history;history_evaluations;function_evaluations"}
        for method, architecture_name, workers in (
            ("ms_global_w1", "master_slave", 1), ("ms_global_w2", "master_slave", 2),
            ("pp_global_w1", "particle_parallel", 1), ("pp_global_w2", "particle_parallel", 2),
        )
    ]
    _write_csv(architecture / "equivalence_audit.csv", equivalence_rows)
    (architecture / "campaign_manifest.json").write_text(
        json.dumps({"campaign": {"evidence_class": "confirmatory"}}), encoding="utf-8"
    )

    topology_rows = []
    for benchmark, scale in (("sphere", 1.0), ("rastrigin", 4.0)):
        for dimension in (30, 100):
            for seed in (1, 2):
                topology_rows.extend([
                    {"benchmark": benchmark, "dimension": dimension, "seed": seed,
                     "topology": "global", "best_value": scale},
                    {"benchmark": benchmark, "dimension": dimension, "seed": seed,
                     "topology": "ring", "best_value": scale * 1.5},
                    {"benchmark": benchmark, "dimension": dimension, "seed": seed,
                     "topology": "small_world", "best_value": scale * 1.25},
                ])
    topology_path = (
        results / "raw/cpso_topology_expansion_confirmatory_073"
        / "01_serial_topology_quality/native_benchmarks_d30_d100/runs.csv"
    )
    _write_csv(topology_path, topology_rows)
    topology_campaign = results / "raw/cpso_topology_expansion_confirmatory_073"
    (topology_campaign / "campaign_manifest.json").write_text(
        json.dumps({"campaign": {"evidence_class": "confirmatory"}}), encoding="utf-8"
    )

    output = tmp_path / "figures"
    output.mkdir()
    stale_schematic = output / "fig_09_particle_topology_schematics.pdf"
    stale_schematic.write_bytes(b"obsolete")
    figures, missing = build_figures(results, output)
    assert not missing
    expected = {
        "fig_01_synchronous_equivalence.pdf", "fig_02_strong_scaling.pdf",
        "fig_03_parallel_efficiency.pdf", "fig_04_runtime_components.pdf",
        "fig_05_asynchronous_quality.pdf", "fig_06_island_quality.pdf",
        "fig_08_current_topology_quality.pdf",
    }
    assert {path.name for path in figures} == expected
    assert all(path.read_bytes().startswith(b"%PDF-") for path in figures)
    assert not stale_schematic.exists()
    assert not any("table" in path.name or "schematic" in path.name for path in output.glob("*.pdf"))
    catalogue = (output / "FIGURE_CATALOGUE.md").read_text(encoding="utf-8")
    assert "Figures are generated only from completed, measured campaign records." in catalogue
    assert "Evidence profile: **confirmatory**" in catalogue
    assert "Synchronous equivalence is plotted only after the run-level exact verifier succeeds." in catalogue



def test_exploratory_profile_generates_only_from_explicit_pilot_inputs(tmp_path: Path) -> None:
    results = tmp_path / "results"
    synchronous = results / "raw/cpso_serial_parallel_m2_pilot"
    dynamics = results / "raw/cpso_dynamics_m2_pilot"
    topology = results / "raw/cpso_topology_expansion_m2_pilot"
    for campaign in (synchronous, dynamics, topology):
        campaign.mkdir(parents=True, exist_ok=True)
        (campaign / "campaign_manifest.json").write_text(
            json.dumps({"campaign": {"evidence_class": "exploratory"}}), encoding="utf-8"
        )

    equivalence = [
        {"stage": "01_exact_quality_equivalence", "study": "native", "method": method,
         "architecture": architecture, "topology": "global", "workers": workers,
         "verified_pairs": 30, "expected_pairs": 30, "fraction_verified": 1.0,
         "status": "exact_match", "checked_fields": "best_value;best_position;best_history"}
        for method, architecture, workers in (
            ("ms_global_w1", "master_slave", 1), ("ms_global_w2", "master_slave", 2),
            ("pp_global_w1", "particle_parallel", 1), ("pp_global_w2", "particle_parallel", 2),
        )
    ]
    _write_csv(synchronous / "equivalence_audit.csv", equivalence)

    scaling = []
    runs = []
    for dimension in (30, 100):
        for method, workers, speedup in (
            ("ms_global_w1", 1, 0.9), ("ms_global_w2", 2, 1.4),
            ("pp_global_w1", 1, 0.95), ("pp_global_w2", 2, 1.6),
        ):
            scaling.append({
                "study": "sphere_work_n64_cost4096", "work_iterations": 4096,
                "population_size": 64, "dimension": dimension, "method": method,
                "workers": workers, "median_speedup": speedup,
                "bootstrap_95_low": speedup - 0.1, "bootstrap_95_high": speedup + 0.1,
                "parallel_efficiency": speedup / workers,
            })
        for method, workers in (("serial_global", 1), ("ms_global_w1", 1),
                                ("ms_global_w2", 2), ("pp_global_w1", 1), ("pp_global_w2", 2)):
            runs.append({
                "stage": "02_synthetic_cost_scaling", "study": "sphere_work_n64_cost4096",
                "benchmark": "sphere_work", "work_iterations": 4096,
                "population_size": 64, "dimension": dimension, "method": method,
                "elapsed_seconds": 0.05 / workers,
                "initialization_seconds": 0.001, "topology_setup_seconds": 0.001,
                "evaluation_wall_seconds": 0.03 / workers,
                "propagation_wall_seconds": 0.005,
            })
    _write_csv(synchronous / "strong_scaling_summary.csv", scaling)
    _write_csv(synchronous / "campaign_runs.csv", runs)

    _write_csv(dynamics / "pairwise_effects.csv", [
        {"stage": "01_asynchronous_quality", "method": "async_global_w2",
         "median_function_family_log10_delta": -0.05,
         "hierarchical_bootstrap_95_low": -0.12, "hierarchical_bootstrap_95_high": 0.02},
        {"stage": "02_island_quality", "method": "islands_ring",
         "median_function_family_log10_delta": 0.03,
         "hierarchical_bootstrap_95_low": -0.04, "hierarchical_bootstrap_95_high": 0.10},
    ])

    topology_rows = []
    for benchmark, base in (("sphere", 1.0), ("rastrigin", 4.0)):
        for dimension in (30, 100):
            for seed in (1, 2, 3):
                topology_rows.extend([
                    {"benchmark": benchmark, "dimension": dimension, "seed": seed,
                     "topology": "global", "best_value": base},
                    {"benchmark": benchmark, "dimension": dimension, "seed": seed,
                     "topology": "ring", "best_value": base * 1.2},
                    {"benchmark": benchmark, "dimension": dimension, "seed": seed,
                     "topology": "wheel", "best_value": base * 1.1},
                ])
    _write_csv(
        topology / "01_serial_topology_quality/native_benchmarks_d30_d100/runs.csv",
        topology_rows,
    )

    output = tmp_path / "figures"
    figures, missing = build_figures(
        results, output, profile="exploratory",
        synchronous_campaign_arg=synchronous,
        dynamics_campaign_arg=dynamics,
        topology_campaign_arg=topology,
    )
    assert not missing
    assert {path.name for path in figures} == {
        "fig_01_synchronous_equivalence.pdf", "fig_02_strong_scaling.pdf",
        "fig_03_parallel_efficiency.pdf", "fig_04_runtime_components.pdf",
        "fig_05_asynchronous_quality.pdf", "fig_06_island_quality.pdf",
        "fig_08_current_topology_quality.pdf",
    }
    catalogue = (output / "FIGURE_CATALOGUE.md").read_text(encoding="utf-8")
    assert "Evidence profile: **exploratory**" in catalogue
    for path in figures:
        assert path.read_bytes().startswith(b"%PDF-")
