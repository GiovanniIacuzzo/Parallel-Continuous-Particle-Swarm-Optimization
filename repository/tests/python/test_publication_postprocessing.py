from __future__ import annotations

import json

import pytest
from pathlib import Path

from analysis.postprocess_publication_figures import (
    aggregate_topology_effects_by_family,
    figure_topology_schematics,
)

ROOT = Path(__file__).resolve().parents[2]


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


def test_topology_schematic_is_vector_pdf(tmp_path: Path) -> None:
    output = figure_topology_schematics(tmp_path)
    data = output.read_bytes()
    assert output.name == "fig_09_particle_topology_schematics.pdf"
    assert data.startswith(b"%PDF-")
    assert len(data) > 8_000


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
                rows.append({
                    "benchmark": benchmark, "dimension": str(dimension),
                    "seed": str(seed), "topology": "global", "best_value": "1.0",
                })
                rows.append({
                    "benchmark": benchmark, "dimension": str(dimension),
                    "seed": str(seed), "topology": "ring",
                    "best_value": str(multiplier),
                })
    aggregated = aggregate_topology_effects_by_family(rows, ["global", "ring"])
    assert set(aggregated["ring"]) == {"sphere", "rastrigin"}
    assert len(aggregated["ring"]) == 2  # not 12 dimension/seed cells
    assert aggregated["ring"]["sphere"] == pytest.approx(0.3010299957)
    assert aggregated["ring"]["rastrigin"] == pytest.approx(0.6020599913)
