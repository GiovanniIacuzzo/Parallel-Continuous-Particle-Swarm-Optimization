#!/usr/bin/env python3
"""Publication-oriented paired strong-scaling plots for the native CPSO backend.

NO optimizer execution. Reads completed, unmodified run records from the campaign.
Generates PDF-only, separate D/workload views and reproducible CSV summaries.

Interpretation
--------------
  * Speedup is the MEDIAN of paired (serial time / parallel time) ratios.
  * Efficiency is speedup / number of workers, including for p=1. Do not
    force the observed one-worker executor to the ideal (1, 1) point.
  * Each topology is a fixed experimental stratum, not an independent random
    sample of 'all possible topologies'. Aggregation across the selected
    topologies uses equal weights; CIs resample SEEDS WITHIN each topology.
  * Serial and parallel trajectories (not only their final objectives) must
    be exactly equal. No speedup claims for async/island search dynamics.
  * Dimensions, cost-control workloads and architectures are never pooled.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plot_results.atlas_common import (
    configure_style, discover_internal_records, read_json, save_figure,
)

ARCHITECTURES = ("master_slave", "particle_parallel")
ARCH_LABELS = {"master_slave": "Master–Slave", "particle_parallel": "Particle-Parallel"}
ARCH_COLORS = {"master_slave": "#17618A", "particle_parallel": "#D07827"}
TOPOLOGY_ORDER = (
    "global", "ring", "grid", "wheel", "random_k4", "random_k8",
    "small_world_k4", "scale_free_k4",
)
TOPOLOGY_LABELS = {
    "global": "Global", "ring": "Ring", "grid": "Von Neumann", "wheel": "Wheel",
    "random_k4": "Random (k=4)", "random_k8": "Random (k=8)",
    "small_world_k4": "Small-world (k=4)", "scale_free_k4": "Scale-free (k=4)",
}
TOPOLOGY_COLORS = (
    "#235789", "#D08031", "#3E8E75", "#8F64A8",
    "#C35561", "#618C99", "#827358", "#434B58",
)
OLD_GLOB_PATTERNS = (
    "01_speedup_*_scaling_*.pdf", "02_speedup_*_scaling_*.pdf",
    "01_speedup_*_runtime_components_*.pdf", "02_speedup_*_runtime_components_*.pdf",
)
REQUIRED_TRACE_FIELDS = (
    "best_value", "best_position", "best_history",
    "history_evaluations", "function_evaluations",
)


@dataclass(frozen=True)
class Observation:
    stage: str
    study: str
    dimension: int
    population: int
    work: int
    graph: str
    architecture: str
    workers: int
    seed: int
    t_serial: float
    t_parallel: float

    @property
    def speedup(self) -> float:
        return self.t_serial / self.t_parallel

    @property
    def efficiency(self) -> float:
        return self.speedup / self.workers


def _int(value: Any, field: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"Unexpected Boolean for {field}")
    try:
        converted = int(value)
        if float(value) != converted:
            raise ValueError
        return converted
    except (ValueError, TypeError) as exc:
        raise ValueError(f"Invalid integer {field}={value!r}") from exc


def _positive_time(row: dict[str, Any]) -> float:
    value = float(row.get("elapsed_seconds", float("nan")))
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"Non-positive/non-finite elapsed_seconds for {row.get('method')}: {value}")
    return value


def _stable_seed(*values: Any) -> int:
    payload = json.dumps(values, sort_keys=True, default=str).encode("utf8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _interval(samples: np.ndarray, *, repetitions: int, seed: int) -> tuple[float, float]:
    """Pointwise paired-seed percentile bootstrap CI of the median."""
    if len(samples) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    draw = rng.integers(0, len(samples), size=(repetitions, len(samples)))
    medians = np.median(samples[draw], axis=1)
    return float(np.quantile(medians, .025)), float(np.quantile(medians, .975))


def _fixed_topology_interval(
    by_graph: dict[str, np.ndarray], *, repetitions: int, seed: int
) -> tuple[float, float]:
    """Bootstrap paired seeds within each fixed graph; median of graph medians.

    The CI is CONDITIONAL on these specifically selected graph configurations.
    Graphs are not resampled as if they had been drawn randomly from a universe.
    """
    if not by_graph or any(len(values) < 2 for values in by_graph.values()):
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    per_graph_boot = []
    for graph in sorted(by_graph):
        values = by_graph[graph]
        draw = rng.integers(0, len(values), size=(repetitions, len(values)))
        per_graph_boot.append(np.median(values[draw], axis=1))
    all_boot = np.median(np.stack(per_graph_boot, axis=0), axis=0)
    return float(np.quantile(all_boot, .025)), float(np.quantile(all_boot, .975))


def _manifest_studies(manifest: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    campaign = manifest.get("campaign", manifest)
    if not isinstance(campaign, dict) or not isinstance(campaign.get("stages"), list):
        raise ValueError("Campaign manifest does not contain campaign.stages")
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for stage in campaign["stages"]:
        if stage.get("kind") != "performance":
            continue
        for item in stage.get("studies", []):
            spec = item.get("study", {})
            if spec.get("benchmarks") != ["sphere_work"] or spec.get("check_equivalence") is not True:
                raise ValueError(f"Performance stage {stage.get('id')} is not verified native sphere_work")
            key = (str(stage["id"]), str(item["id"]))
            if key in out:
                raise ValueError(f"Duplicate study in manifest: {key}")
            out[key] = spec
    if not out:
        raise ValueError("No native synchronous performance stages found in campaign manifest")
    return out


def _validate_record_pair(serial: dict[str, Any], parallel: dict[str, Any], label: str) -> None:
    for field in ("benchmark", "topology", "topology_degree", "topology_rewiring_probability",
                  "dimension", "seed", "population_size", "iterations", "work_iterations"):
        if field in serial and field in parallel and str(serial[field]) != str(parallel[field]):
            raise ValueError(f"Non-equivalent {label}: {field} mismatch")
    missing = [field for field in REQUIRED_TRACE_FIELDS if field not in serial or field not in parallel]
    if missing:
        raise ValueError(
            f"Cannot verify trajectory equivalence for {label}; missing {missing}. "
            "Provide original study runs.jsonl or campaign_runs.csv containing full histories."
        )
    for field in REQUIRED_TRACE_FIELDS:
        if serial[field] != parallel[field]:
            raise ValueError(f"Failed exact serial/parallel equivalence: {label} field={field}")
    if _int(serial["function_evaluations"], "function_evaluations") != _int(
        parallel["function_evaluations"], "function_evaluations"
    ):
        raise ValueError(f"Different evaluation budgets: {label}")


def load_observations(campaign_dir: Path) -> tuple[list[Observation], dict[str, Any]]:
    manifest = read_json(campaign_dir / "campaign_manifest.json")
    studies = _manifest_studies(manifest)
    records = discover_internal_records(campaign_dir)
    if not records:
        raise ValueError(f"No completed runs found in {campaign_dir}")

    # Keyed by exact stage, study, problem dimension, seed and method name.
    index: dict[tuple[str, str, int, int, str], dict[str, Any]] = {}
    for row in records:
        stage_study = (str(row.get("stage")), str(row.get("study")))
        if stage_study not in studies:
            continue  # Asynchronous/island quality experiments are separate.
        spec = studies[stage_study]
        row = dict(row)
        row.setdefault("population_size", spec.get("population_size"))
        row.setdefault("iterations", spec.get("iterations"))
        row.setdefault("work_iterations", spec.get("work_iterations", 0))
        dim = _int(row.get("dimension"), "dimension")
        seed = _int(row.get("seed"), "seed")
        method = str(row.get("method"))
        key = (stage_study[0], stage_study[1], dim, seed, method)
        if key in index:
            raise ValueError(f"Duplicate completed run for {key}")
        index[key] = row

    observations: list[Observation] = []
    for (stage, study), spec in studies.items():
        methods = {str(m["name"]): m for m in spec["methods"]}
        serial_methods = [m for m in spec["methods"] if m["architecture"] == "serial"]
        if len(serial_methods) != 1:
            raise ValueError(f"Expected one serial baseline for {stage}/{study}")
        serial_method = serial_methods[0]
        if any(m["architecture"] not in {"serial", *ARCHITECTURES} for m in spec["methods"]):
            raise ValueError(f"Non-synchronous method in performance stage {stage}/{study}")
        for dim in spec["dimensions"]:
            for seed in spec["seeds"]:
                for method_name in methods:
                    key = (stage, study, int(dim), int(seed), method_name)
                    if key not in index:
                        raise ValueError(
                            f"Incomplete performance campaign: missing {key}. "
                            "Resume the original campaign before producing publication plots."
                        )
                base = index[(stage, study, int(dim), int(seed), serial_method["name"])]
                if base.get("benchmark") != "sphere_work":
                    raise ValueError(f"Unexpected benchmark in {stage}/{study}")
                _positive_time(base)
                for name, method in methods.items():
                    if method["architecture"] == "serial":
                        continue
                    row = index[(stage, study, int(dim), int(seed), name)]
                    if row.get("benchmark") != "sphere_work":
                        raise ValueError(f"Unexpected benchmark in {stage}/{study}/{name}")
                    if method.get("topology", "global") != serial_method.get("topology", "global"):
                        raise ValueError(f"Graph differs from serial baseline for {stage}/{study}/{name}")
                    label = f"{stage}/{study}/D{dim}/seed{seed}/{name}"
                    _validate_record_pair(base, row, label)
                    if str(row.get("architecture")) != str(method["architecture"]):
                        raise ValueError(f"Architecture does not match manifest: {label}")
                    if _int(row.get("workers"), "workers") != _int(method["workers"], "workers"):
                        raise ValueError(f"Worker count does not match manifest: {label}")
                    observations.append(Observation(
                        stage=stage, study=study, dimension=int(dim),
                        population=_int(spec["population_size"], "population_size"),
                        work=_int(spec.get("work_iterations", 0), "work_iterations"),
                        graph=str(method.get("topology", "global")),
                        architecture=str(method["architecture"]),
                        workers=_int(method["workers"], "workers"),
                        seed=int(seed), t_serial=_positive_time(base), t_parallel=_positive_time(row),
                    ))
    if not observations:
        raise ValueError("No paired synchronous parallel execution records were found")
    populations = {o.population for o in observations}
    if len(populations) != 1:
        raise ValueError(f"Mixed population sizes cannot be pooled: {sorted(populations)}")
    work_study_stages: dict[tuple[int, str], set[str]] = defaultdict(set)
    for o in observations:
        work_study_stages[(o.work, o.study)].add(o.stage)
    if any(len(stages) > 1 for stages in work_study_stages.values()):
        raise ValueError("Ambiguous duplicate topology/workload across different stages")
    return observations, manifest


def _summaries(
    obs: list[Observation], *, repetitions: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not obs:
        raise ValueError("Cannot summarize an empty set of paired observations")
    population = obs[0].population  # already verified constant by load_observations
    grouped: dict[tuple[int, int, str, str, int], list[Observation]] = defaultdict(list)
    for o in obs:
        grouped[(o.dimension, o.work, o.study, o.architecture, o.workers)].append(o)
    topo_points: list[dict[str, Any]] = []
    for (dim, work, topo, arch, workers), values in sorted(grouped.items()):
        values.sort(key=lambda x: x.seed)
        ratios = np.asarray([o.speedup for o in values], dtype=float)
        speed = float(np.median(ratios))
        ci = _interval(ratios, repetitions=repetitions,
                       seed=_stable_seed("topology", dim, work, topo, arch, workers))
        topo_points.append(dict(
            dimension=dim, population_size=values[0].population,
            work_iterations=work, topology=topo, graph=values[0].graph,
            architecture=arch, workers=workers, n_pairs=len(values),
            speedup=speed, speedup_ci_low=ci[0], speedup_ci_high=ci[1],
            efficiency=speed / workers, efficiency_ci_low=ci[0] / workers,
            efficiency_ci_high=ci[1] / workers,
            median_serial_seconds=statistics.median(o.t_serial for o in values),
            median_parallel_seconds=statistics.median(o.t_parallel for o in values),
        ))
    agg_groups: dict[tuple[int, int, str, int], dict[str, np.ndarray]] = defaultdict(dict)
    for (dim, work, topo, arch, workers), values in grouped.items():
        agg_groups[(dim, work, arch, workers)][topo] = np.asarray(
            [o.speedup for o in values], dtype=float)
    aggregate_points: list[dict[str, Any]] = []
    for (dim, work, arch, workers), by_graph in sorted(agg_groups.items()):
        # Strictly equal weighting of graph configurations; no pooling runs.
        speed = float(np.median([np.median(a) for a in by_graph.values()]))
        low, high = _fixed_topology_interval(
            by_graph, repetitions=repetitions,
            seed=_stable_seed("aggregate", dim, work, arch, workers),
        )
        aggregate_points.append(dict(
            dimension=dim, population_size=population,
            work_iterations=work, architecture=arch,
            workers=workers, n_topologies=len(by_graph),
            min_pairs_per_topology=min(len(a) for a in by_graph.values()),
            speedup=speed, speedup_ci_low=low, speedup_ci_high=high,
            efficiency=speed / workers, efficiency_ci_low=low / workers,
            efficiency_ci_high=high / workers,
        ))
    return topo_points, aggregate_points


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _configure_axis(ax: Any, metric: str, workers: Iterable[int]) -> None:
    workers = sorted(set(workers))
    if metric == "speedup":
        ax.plot([1, max(workers)], [1, max(workers)], color="#555C67",
                ls="--", lw=1.0, alpha=.78, zorder=0)
        ax.axhline(1., color="#858B93", lw=.75, ls=":", alpha=.7)
        ax.set_ylabel("Paired speedup, $S(p)$")
    else:
        ax.axhline(1., color="#555C67", ls="--", lw=1, alpha=.78)
        ax.set_ylabel("Parallel efficiency, $E(p)$")
    ax.set_xlabel("Worker threads, $p$")
    ax.set_xticks(workers)
    ax.set_xlim(min(workers) - .15, max(workers) + .3)
    ax.set_ylim(bottom=0.)
    ax.grid(axis="y", alpha=.18)


def _plot_aggregate(points: list[dict[str, Any]], dimension: int, output: Path) -> Path:
    data = [r for r in points if r["dimension"] == dimension]
    workloads = sorted({r["work_iterations"] for r in data})
    if not workloads:
        raise ValueError(f"No scaling observations for D={dimension}")
    fig, axes = plt.subplots(2, len(workloads), figsize=(5.3 * len(workloads), 6.8),
                             squeeze=False, sharex="col", layout="constrained")
    for j, work in enumerate(workloads):
        matched = [r for r in data if r["work_iterations"] == work]
        for i, metric in enumerate(("speedup", "efficiency")):
            ax = axes[i, j]
            for arch in ARCHITECTURES:
                curve = sorted((r for r in matched if r["architecture"] == arch),
                               key=lambda r: r["workers"])
                if not curve:
                    continue
                xx = [r["workers"] for r in curve]
                yy = [r[metric] for r in curve]
                lo = [r[f"{metric}_ci_low"] for r in curve]
                hi = [r[f"{metric}_ci_high"] for r in curve]
                color = ARCH_COLORS[arch]
                ax.plot(xx, yy, "-o", color=color, label=ARCH_LABELS[arch],
                        lw=1.9, ms=4.5, zorder=3)
                ax.fill_between(xx, lo, hi, color=color, alpha=.12, lw=0)
            _configure_axis(ax, metric, [r["workers"] for r in matched])
            if i == 0:
                ax.set_title(f"Cost-controlled objective · work={work:,}", pad=12)
            if j != 0:
                ax.set_ylabel("")
    populations = {r["population_size"] for r in data}
    if len(populations) != 1:
        raise ValueError("Inconsistent population in aggregate scaling panel")
    fig.suptitle(f"CPSO synchronous strong scaling · D={dimension}, N={populations.pop()}", fontsize=13)
    fig.legend([plt.Line2D([], [], color=ARCH_COLORS[a], marker="o", lw=1.8)
                for a in ARCHITECTURES], [ARCH_LABELS[a] for a in ARCHITECTURES],
               loc="outside lower center", ncol=2, frameon=False)
    created = save_figure(fig, output, f"fig_parallel_scaling_d{dimension}")
    return created[0]


def _plot_topologies(points: list[dict[str, Any]], dimension: int, work: int, output: Path) -> Path:
    data = [r for r in points if r["dimension"] == dimension and r["work_iterations"] == work]
    topo_ids = sorted({r["topology"] for r in data},
                      key=lambda s: TOPOLOGY_ORDER.index(s) if s in TOPOLOGY_ORDER else len(TOPOLOGY_ORDER))
    colors = {t: TOPOLOGY_COLORS[i % len(TOPOLOGY_COLORS)] for i, t in enumerate(topo_ids)}
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 7.8), sharex="col", layout="constrained")
    for j, arch in enumerate(ARCHITECTURES):
        subset = [r for r in data if r["architecture"] == arch]
        for i, metric in enumerate(("speedup", "efficiency")):
            ax = axes[i, j]
            for topo in topo_ids:
                curve = sorted((r for r in subset if r["topology"] == topo),
                               key=lambda r: r["workers"])
                if not curve:
                    continue
                ax.plot([r["workers"] for r in curve], [r[metric] for r in curve],
                        color=colors[topo], marker="o", markersize=3.4,
                        lw=1.35, label=TOPOLOGY_LABELS.get(topo, topo), alpha=.9)
            _configure_axis(ax, metric, [r["workers"] for r in subset])
            if i == 0:
                ax.set_title(ARCH_LABELS[arch])
            if j == 1:
                ax.set_ylabel("")
    populations = {r["population_size"] for r in data}
    if len(populations) != 1:
        raise ValueError("Inconsistent population in topology scaling panel")
    fig.suptitle(f"CPSO scaling by social topology · work={work:,} · D={dimension}, N={populations.pop()}", fontsize=13)
    handles = [plt.Line2D([], [], marker="o", color=colors[t], lw=1.6, markersize=4)
               for t in topo_ids]
    fig.legend(handles, [TOPOLOGY_LABELS.get(t, t) for t in topo_ids],
               loc="outside lower center", frameon=False, ncol=4, fontsize=8)
    created = save_figure(fig, output, f"fig_parallel_topologies_work{work}_d{dimension}")
    return created[0]


def generate(campaign_dir: Path, output: Path, *, repetitions: int = 4000,
             clean_legacy: bool = True) -> tuple[list[Path], list[str]]:
    campaign_dir = Path(campaign_dir).expanduser().resolve()
    output = Path(output).expanduser().resolve()
    if repetitions < 500:
        raise ValueError("Use at least 500 bootstrap repetitions")
    observations, manifest = load_observations(campaign_dir)
    per_topology, aggregate = _summaries(observations, repetitions=repetitions)
    configure_style()
    output.mkdir(parents=True, exist_ok=True)
    pdfs: list[Path] = []
    dimensions = sorted({o.dimension for o in observations})
    for dimension in dimensions:
        pdfs.append(_plot_aggregate(aggregate, dimension, output))
        for work in sorted({o.work for o in observations if o.dimension == dimension}):
            pdfs.append(_plot_topologies(per_topology, dimension, work, output))
    _write_csv(output / "publication_scaling_per_topology.csv", per_topology)
    _write_csv(output / "publication_scaling_aggregate.csv", aggregate)
    warnings: list[str] = []
    deleted: list[str] = []
    if clean_legacy:
        for glob_pattern in OLD_GLOB_PATTERNS:
            for path in sorted(output.glob(glob_pattern)):
                if path.is_file() and path not in pdfs:
                    path.unlink()
                    deleted.append(path.name)
    unique_topologies = {o.study for o in observations}
    note = [
        "# CPSO parallel scaling — publication figures", "",
        f"Source: `{campaign_dir}`", f"Plotted pairs: **{len(observations)}**.",
        f"Dimensions: {dimensions}. Fixed topology configurations: **{len(unique_topologies)}**.",
        f"Synthetic objective work iterations: {sorted({o.work for o in observations})}.",
        f"Evidence class: `{manifest.get('campaign', manifest).get('evidence_class', 'unspecified')}`.",
        "", "## Figures", "",
    ]
    note += [f"- `{p.name}`" for p in pdfs]
    note += [
        "", "## Statistical interpretation", "",
        "- Each speedup is the median of matched serial/parallel elapsed-time ratios for the same seed, topology, dimension and synthetic work.",
        "- Efficiency = speedup / number of worker threads, including the measured parallel executor at p=1.",
        "- Per-topology 95% percentile bootstrap CIs resample paired seeds. Aggregate 95% CIs resample seeds *within each fixed topology* and take a median of topology medians.",
        "- CIs are conditional on these specific eight (or fewer) topology configurations; they are not uncertainty about a population of randomly sampled topologies.",
        "- The topology curves show medians; numeric bootstrap intervals for each point are in publication_scaling_per_topology.csv.",
        "- The native end-to-end elapsed time includes worker-pool lifecycle and scheduler overhead. Dashed lines indicate mathematical ideal scaling only.",
        "- sphere_work is a controllable synthetic CPU workload, not a standard objective-quality benchmark.",
        "- No ratio between non-equivalent asynchronous or island dynamics is called speedup.",
        "- Benchmarks and dimensions are not pooled. Do not interpret these timing plots as measured on COCO/CEC.",
        "", "## CSV outputs", "",
        "- `publication_scaling_per_topology.csv`: exact condition, n, median paired speedup and efficiency, 95% CIs, median serial/parallel elapsed times.",
        "- `publication_scaling_aggregate.csv`: equal-weight median across topology-specific medians, seed-conditional 95% CIs.",
        "", "## Legacy figure cleanup", "",
        f"Removed {len(deleted)} legacy per-topology scaling/runtime-component PDFs from THIS output directory.",
        "Only filenames matching the known 01_speedup/02_speedup patterns were removed.",
        "Raw experiments, analyses, other quality/DT figures, and paper/ are untouched.",
        "",
    ]
    (output / "PARALLEL_PUBLICATION.md").write_text("\n".join(note), encoding="utf8")
    (output / "FIGURE_ATLAS.md").write_text(
        "# Parallel figure atlas — publication selection\n\n"
        "The new native CPSO scaling figures are listed in `PARALLEL_PUBLICATION.md`.\n"
        "Other pre-existing topology/quality diagnostic PDFs are retained unchanged.\n\n"
        + "\n".join(f"- `{p.name}`" for p in pdfs) + "\n", encoding="utf8",
    )
    (output / "parallel_publication_manifest.json").write_text(json.dumps({
        "source": str(campaign_dir),
        "n_paired_records": len(observations),
        "n_topology_configurations": len(unique_topologies),
        "dimensions": dimensions,
        "work_iterations": sorted({o.work for o in observations}),
        "bootstrap_repetitions": repetitions,
        "figures": [p.name for p in pdfs],
        "legacy_removed": deleted,
    }, indent=2) + "\n", encoding="utf8")
    (output / "figure_atlas_manifest.json").write_text(json.dumps({
        "publication_selection": [p.name for p in pdfs],
        "legacy_scaling_removed": deleted,
        "note": "Other pre-existing diagnostic PDF files are not modified",
    }, indent=2) + "\n", encoding="utf8")
    print(f"[CPSO publication] {len(observations)} matched runs; {len(pdfs)} PDF figures")
    print(f"[CPSO publication] Figures and CSVs: {output}")
    return pdfs, warnings


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--campaign", type=Path, required=True,
                    help="results/raw/parallel/<campaign_id> containing campaign_manifest.json and runs.jsonl")
    ap.add_argument("--output", type=Path, default=None,
                    help="Where the publication PDFs/CSVs are stored (default: atlas/parallel/<campaign>)")
    ap.add_argument("--bootstrap", type=int, default=4000,
                    help="Pointwise seed-bootstrap draws (default 4000)")
    ap.add_argument("--keep-legacy", action="store_true",
                    help="Do not delete previous *_scaling_* and *_runtime_components_* PDFs")
    args = ap.parse_args()
    root = Path(__file__).resolve().parents[1]
    campaign = args.campaign.expanduser().resolve()
    target = args.output or root / "results/figures/atlas/parallel" / campaign.name
    # Enforce the repository output policy also for direct invocation.
    from cpso_serial.output_policy import results_output
    generate(campaign, results_output(target), repetitions=args.bootstrap,
             clean_legacy=not args.keep_legacy)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
