#!/usr/bin/env python3
"""Create the seven internal-study figures referenced by paper Section 4.

Inputs are analysis outputs from the existing CPSO campaign analyzers. This
script never creates or imputes experimental observations.
"""
from __future__ import annotations

import argparse
import sys
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plot_results.common import (
    COLORS, PALETTE, as_float, as_int, configure_style, evidence_note,
    manifest_evidence, read_csv, save_figure,
)


def plot_equivalence(campaign: Path, output: Path) -> list[Path]:
    source = campaign / "equivalence_audit.csv"
    rows = read_csv(source, required=("method", "workers", "verified_pairs", "expected_pairs", "fraction_verified"))
    clean = []
    for row in rows:
        expected = as_int(row, "expected_pairs", -1)
        verified = as_int(row, "verified_pairs", -1)
        if expected <= 0 or verified < 0 or verified > expected:
            raise ValueError(f"Conteggi di equivalenza non validi in {source}: {row}")
        fraction = verified / expected
        recorded_fraction = as_float(row, "fraction_verified")
        if math.isfinite(recorded_fraction) and abs(recorded_fraction - fraction) > 1.0e-9:
            raise ValueError(f"fraction_verified incoerente con verified_pairs/expected_pairs in {source}: {row}")
        architecture = row.get("architecture", "")
        method = row.get("method", "")
        if not architecture:
            architecture = "Master-slave" if method.startswith("ms") else "Particle-parallel" if method.startswith("pp") else method
        clean.append({
            "label": f"{architecture.replace('_', ' ')} · {as_int(row, 'workers')} worker",
            "fraction": fraction,
            "verified": verified,
            "expected": expected,
            "architecture": architecture,
            "workers": as_int(row, "workers"),
        })
    clean.sort(key=lambda x: (x["architecture"], x["workers"], x["label"]))
    labels = [r["label"] for r in clean]
    vals = np.asarray([r["fraction"] for r in clean]) * 100.0
    fig, ax = plt.subplots(figsize=(7.0, max(2.5, 0.35 * len(clean) + 1.0)))
    y = np.arange(len(clean))
    colors = [COLORS["blue"] if "master" in r["architecture"] else COLORS["teal"] for r in clean]
    ax.barh(y, vals, color=colors, height=0.62, edgecolor="white", linewidth=0.5)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 105)
    ax.set_xlabel("Verified paired runs (%)")
    ax.set_title("Exact synchronous trajectory equivalence")
    ax.grid(axis="x")
    for i, row in enumerate(clean):
        ax.text(min(vals[i] + 1.0, 100.7), i, f"{row['verified']}/{row['expected']}", va="center", fontsize=7)
    note = f"All listed pairs must pass the run-level trajectory audit. {evidence_note(manifest_evidence(campaign))}."
    fig.text(0.5, 0.005, note, ha="center", fontsize=6.3, color=COLORS["grey"])
    fig.tight_layout(rect=(0, 0.045, 1, 1))
    return list(save_figure(fig, output, "fig_01_synchronous_equivalence"))


def _scaling_data(campaign: Path) -> list[dict[str, str]]:
    rows = read_csv(campaign / "strong_scaling_summary.csv", required=(
        "method", "workers", "dimension", "population_size", "work_iterations", "median_speedup",
        "bootstrap_95_low", "bootstrap_95_high", "parallel_efficiency",
    ))
    rows = [r for r in rows if r.get("method", "").startswith(("ms_", "pp_"))]
    if not rows:
        raise ValueError("strong_scaling_summary.csv non contiene righe ms_* o pp_*")
    return rows


def _draw_scaling(rows: list[dict[str, str]], output: Path, *, efficiency: bool) -> list[Path]:
    configure_style()
    populations = sorted({as_int(r, "population_size") for r in rows})
    dimensions = sorted({as_int(r, "dimension") for r in rows})
    workloads = sorted({as_int(r, "work_iterations") for r in rows})
    metric = "parallel_efficiency" if efficiency else "median_speedup"
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.35), sharey=True)
    markers = ["o", "s", "^", "D"]
    line_styles = ["-", "--", ":", "-."]
    for ax, prefix, title in zip(axes, ("ms_", "pp_"), ("Master–slave", "Particle-parallel")):
        subset_arch = [r for r in rows if r["method"].startswith(prefix)]
        if not subset_arch:
            ax.set_title(title + " (no data)")
            ax.set_xlabel("Worker threads")
            ax.grid(axis="y")
            continue
        for pidx, population in enumerate(populations):
            subset_pop = [r for r in subset_arch if as_int(r, "population_size") == population]
            for didx, dimension in enumerate(dimensions):
                for widx, workload in enumerate(workloads):
                    values = [r for r in subset_pop if as_int(r, "dimension") == dimension and as_int(r, "work_iterations") == workload]
                    values.sort(key=lambda r: as_int(r, "workers"))
                    if not values:
                        continue
                    raw_x = np.asarray([as_int(r, "workers") for r in values], dtype=float)
                    raw_y = np.asarray([as_float(r, metric) for r in values], dtype=float)
                    valid = np.isfinite(raw_x) & np.isfinite(raw_y)
                    label = f"N={population}, D={dimension}, work={workload}"
                    if efficiency:
                        x, y = raw_x[valid], raw_y[valid]
                        if x.size == 0:
                            continue
                        ax.plot(x, y, marker=markers[didx % len(markers)], linestyle=line_styles[widx % len(line_styles)],
                                color=PALETTE[(didx + pidx) % len(PALETTE)], label=label)
                    else:
                        low_all = np.asarray([as_float(r, "bootstrap_95_low") for r in values], dtype=float)
                        high_all = np.asarray([as_float(r, "bootstrap_95_high") for r in values], dtype=float)
                        valid_ci = valid & np.isfinite(low_all) & np.isfinite(high_all)
                        x, y = raw_x[valid_ci], raw_y[valid_ci]
                        low, high = low_all[valid_ci], high_all[valid_ci]
                        if x.size == 0:
                            continue
                        yerr = np.vstack([np.maximum(0.0, y - low), np.maximum(0.0, high - y)])
                        ax.errorbar(x, y, yerr=yerr, marker=markers[didx % len(markers)],
                                    linestyle=line_styles[widx % len(line_styles)], capsize=2,
                                    color=PALETTE[(didx + pidx) % len(PALETTE)], label=label)
        if not efficiency:
            max_workers = max(as_int(r, "workers") for r in subset_arch)
            ax.plot([1, max_workers], [1, max_workers], "k--", linewidth=0.8, label="Ideal")
            ax.axhline(1, color=COLORS["grey"], linestyle=":", linewidth=0.7)
        else:
            ax.axhline(1, color=COLORS["grey"], linestyle="--", linewidth=0.8, label="Ideal efficiency")
        ax.set_title(title)
        ax.set_xlabel("Worker threads")
        ax.set_xticks(sorted({as_int(r, "workers") for r in subset_arch}))
        ax.grid(axis="y")
    axes[0].set_ylabel("Parallel efficiency" if efficiency else "Paired end-to-end speedup")
    legend_items: dict[str, Any] = {}
    for axis in axes:
        handles, labels = axis.get_legend_handles_labels()
        for handle, label in zip(handles, labels):
            legend_items.setdefault(label, handle)
    if legend_items:
        fig.legend(list(legend_items.values()), list(legend_items.keys()), loc="lower center", ncol=2,
                   frameon=False, bbox_to_anchor=(0.5, -0.16))
    title = "Parallel efficiency" if efficiency else "Strong scaling of synchronous CPSO executors"
    fig.suptitle(title, y=1.02, weight="bold")
    note = "Timing is machine-specific. Speedup is meaningful only for exact synchronous trajectory-equivalent runs."
    fig.text(0.5, -0.19, note, ha="center", fontsize=6.3, color=COLORS["grey"])
    fig.tight_layout(rect=(0, 0.07, 1, 0.96))
    filename = "fig_03_parallel_efficiency" if efficiency else "fig_02_strong_scaling"
    return list(save_figure(fig, output, filename))


def plot_scaling(campaign: Path, output: Path) -> list[Path]:
    rows = _scaling_data(campaign)
    return _draw_scaling(rows, output, efficiency=False) + _draw_scaling(rows, output, efficiency=True)


def plot_runtime_components(campaign: Path, output: Path) -> list[Path]:
    source = campaign / "campaign_runs.csv"
    required = ("benchmark", "method", "elapsed_seconds", "initialization_seconds",
                "topology_setup_seconds", "evaluation_wall_seconds", "propagation_wall_seconds")
    rows = read_csv(source, required=required)
    rows = [r for r in rows if r.get("benchmark") == "sphere_work"]
    if not rows:
        raise ValueError(f"Nessuna esecuzione sphere_work in {source}")
    numeric_fields = ("elapsed_seconds", "initialization_seconds", "topology_setup_seconds",
                      "evaluation_wall_seconds", "propagation_wall_seconds")
    for field in numeric_fields:
        if not any(math.isfinite(as_float(row, field)) for row in rows):
            raise ValueError(f"Nessun valore numerico finito per {field} in {source}")
    combo = max((as_int(r, "population_size"), as_int(r, "dimension"), as_int(r, "work_iterations", 0)) for r in rows)
    rows = [r for r in rows if (as_int(r, "population_size"), as_int(r, "dimension"), as_int(r, "work_iterations", 0)) == combo]
    components = [
        ("initialization_seconds", "Initialization"),
        ("topology_setup_seconds", "Topology setup"),
        ("evaluation_wall_seconds", "Objective evaluation"),
        ("propagation_wall_seconds", "Propagation / social update"),
    ]
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["method"]].append(row)
    methods = sorted(grouped, key=lambda m: (0 if m.startswith("serial") else 1, m))
    if len(methods) < 2:
        raise ValueError("Servono almeno due metodi con misure di componenti per la figura runtime")
    medians: dict[str, list[float]] = {}
    for method in methods:
        selected = grouped[method]
        total = [as_float(r, "elapsed_seconds") for r in selected]
        parts = [[max(0.0, as_float(r, col, 0.0)) for col, _ in components] for r in selected]
        residual = [max(0.0, t - sum(p)) for t, p in zip(total, parts)]
        medians[method] = [float(np.median([p[i] for p in parts])) for i in range(len(components))]
        medians[method].append(float(np.median(residual)))
    labels = [name for _, name in components] + ["Residual / orchestration"]
    colors = [COLORS["grey"], COLORS["gold"], COLORS["blue"], COLORS["purple"], COLORS["light"]]
    fig, ax = plt.subplots(figsize=(7.1, 3.7))
    x = np.arange(len(methods))
    bottom = np.zeros(len(methods))
    for i, (label, color) in enumerate(zip(labels, colors)):
        values = np.asarray([medians[m][i] for m in methods])
        ax.bar(x, values, bottom=bottom, width=0.72, color=color, edgecolor="white", linewidth=0.4, label=label)
        bottom += values
    method_labels = []
    for method in methods:
        if method == "serial_global":
            method_labels.append("Serial")
        elif method.startswith("ms_"):
            method_labels.append(f"MS · {method.rsplit('w', 1)[-1]} workers")
        elif method.startswith("pp_"):
            method_labels.append(f"PP · {method.rsplit('w', 1)[-1]} workers")
        else:
            method_labels.append(method.replace("_", " "))
    ax.set_xticks(x, method_labels, rotation=25, ha="right")
    ax.set_ylabel("Median wall time (s)")
    ax.set_title(f"Measured runtime components · N={combo[0]}, D={combo[1]}, work={combo[2]:,}")
    ax.grid(axis="y")
    ax.legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.30))
    fig.text(0.5, 0.005, "Residual includes elapsed time not assigned to the instrumented components; sphere_work is a cost-control workload.",
             ha="center", fontsize=6.2, color=COLORS["grey"])
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    return list(save_figure(fig, output, "fig_04_runtime_components"))


def plot_pairwise_quality(campaign: Path, output: Path, *, kind: str) -> list[Path]:
    source = campaign / "pairwise_effects.csv"
    rows = read_csv(source, required=("stage", "study", "method", "median_function_family_log10_delta",
                                      "hierarchical_bootstrap_95_low", "hierarchical_bootstrap_95_high"))
    token = "async" if kind == "asynchronous" else "island"
    selected = [r for r in rows if token in " ".join((r.get("stage", ""), r.get("study", ""), r.get("method", ""))).lower()]
    if not selected:
        raise ValueError(f"Nessun confronto {kind} trovato in {source}. Stage/study disponibili: "
                         + ", ".join(sorted({r.get('stage', '') + '/' + r.get('study', '') for r in rows})))
    selected.sort(key=lambda r: (r.get("stage", ""), r.get("study", ""), r.get("method", "")))
    values = np.asarray([as_float(r, "median_function_family_log10_delta") for r in selected])
    lows = np.asarray([as_float(r, "hierarchical_bootstrap_95_low") for r in selected])
    highs = np.asarray([as_float(r, "hierarchical_bootstrap_95_high") for r in selected])
    valid = np.isfinite(values) & np.isfinite(lows) & np.isfinite(highs)
    selected = [r for r, ok in zip(selected, valid) if ok]
    values, lows, highs = values[valid], lows[valid], highs[valid]
    if not selected:
        raise ValueError(f"Nessun intervallo numerico utilizzabile per {kind} in {source}")
    labels = [f"{r['stage']} / {r['method']}".replace("_", " ") for r in selected]
    y = np.arange(len(selected))
    fig, ax = plt.subplots(figsize=(7.1, max(2.8, 0.34 * len(selected) + 1.0)))
    for i, value in enumerate(values):
        ax.errorbar(value, y[i], xerr=[[max(0.0, value - lows[i])], [max(0.0, highs[i] - value)]],
                    fmt="o", ms=4.0, capsize=2.5, color=PALETTE[i % len(PALETTE)])
    ax.axvline(0.0, color=COLORS["grey"], linestyle="--", linewidth=0.9)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel(r"Median function-family paired $\Delta\log_{10}(f_{best})$ vs. serial reference")
    title = "Asynchronous update: paired quality effects" if kind == "asynchronous" else "Island model: paired quality effects"
    ax.set_title(title)
    ax.grid(axis="x")
    note = "Negative values favour the tested variant. Comparisons are FE-matched, not trajectory-equivalent speedups."
    fig.text(0.5, 0.005, note, ha="center", fontsize=6.3, color=COLORS["grey"])
    fig.tight_layout(rect=(0, 0.045, 1, 1))
    filename = "fig_05_asynchronous_quality" if kind == "asynchronous" else "fig_06_island_quality"
    return list(save_figure(fig, output, filename))


def _topology_effects(rows: list[dict[str, str]]) -> dict[str, dict[str, float]]:
    topology_order = ["global", "ring", "grid_von_neumann", "wheel", "random", "small_world", "scale_free"]
    baseline: dict[tuple[str, int, int], float] = {}
    for row in rows:
        if row.get("topology") == "global":
            key = (row.get("benchmark", ""), as_int(row, "dimension", -1), as_int(row, "seed", -1))
            baseline[key] = as_float(row, "best_value")
    paired: dict[str, dict[tuple[str, int], list[float]]] = {t: defaultdict(list) for t in topology_order}
    for row in rows:
        topology = row.get("topology", "")
        if topology not in paired or topology == "global":
            continue
        key = (row.get("benchmark", ""), as_int(row, "dimension", -1), as_int(row, "seed", -1))
        ref = baseline.get(key, math.nan)
        value = as_float(row, "best_value")
        if not (math.isfinite(ref) and math.isfinite(value)):
            continue
        delta = math.log10(max(value, 1e-16)) - math.log10(max(ref, 1e-16))
        paired[topology][(key[0], key[1])].append(delta)
    # First reduce paired seeds in each function x dimension cell, then
    # dimensions within a function. Each plotted point is one function family.
    result: dict[str, dict[str, float]] = {t: {} for t in topology_order}
    for topology, cells in paired.items():
        by_family: dict[str, list[float]] = defaultdict(list)
        for (family, _dimension), values in cells.items():
            if values:
                by_family[family].append(float(np.median(values)))
        for family, dimension_medians in by_family.items():
            result[topology][family] = float(np.median(dimension_medians))
    return result


def plot_topologies(campaign: Path, output: Path) -> list[Path]:
    preferred = sorted(campaign.glob("01_serial_topology_quality/*/runs.csv"))
    candidates = preferred or sorted(campaign.rglob("runs.csv"))
    if not candidates:
        raise FileNotFoundError(f"Nessun runs.csv trovato sotto {campaign}")
    # Choose the first file with the required schema, not simply the first path.
    rows = None
    source = None
    for candidate in candidates:
        try:
            candidate_rows = read_csv(candidate, required=("benchmark", "dimension", "seed", "topology", "best_value"))
        except (FileNotFoundError, ValueError):
            continue
        if candidate_rows:
            rows, source = candidate_rows, candidate
            break
    if rows is None:
        raise ValueError(f"Nessun runs.csv con benchmark/dimension/seed/topology/best_value in {campaign}")
    effects = _topology_effects(rows)
    labels = {
        "global": "Global", "ring": "Ring", "grid_von_neumann": "Von Neumann", "wheel": "Wheel",
        "random": "Random", "small_world": "Small-world", "scale_free": "Scale-free",
    }
    order = ["global", "ring", "grid_von_neumann", "wheel", "random", "small_world", "scale_free"]
    available = [t for t in order if effects[t]]
    if len(available) < 2:
        raise ValueError(f"Meno di due topologie con dati appaiati rispetto a global in {source}")
    common_families = sorted(set.intersection(*(set(effects[t]) for t in available)))
    if not common_families:
        raise ValueError("Nessuna famiglia benchmark comune tra le topologie disponibili")
    data = [[effects[t][family] for family in common_families] for t in available]
    fig, ax = plt.subplots(figsize=(7.1, 3.75))
    box = ax.boxplot(data, patch_artist=True, widths=0.58, showfliers=False,
                     medianprops={"color": "white", "linewidth": 1.3})
    for i, patch in enumerate(box["boxes"]):
        patch.set_facecolor(PALETTE[i % len(PALETTE)])
        patch.set_alpha(0.84)
        patch.set_edgecolor(COLORS["ink"])
    for i, values in enumerate(data, start=1):
        offsets = np.linspace(-0.065, 0.065, len(values)) if len(values) > 1 else [0.0]
        ax.scatter(i + offsets, values, s=20, color=COLORS["ink"], edgecolors="white", linewidths=0.35, zorder=3)
    ax.axhline(0.0, color=COLORS["grey"], linestyle="--", linewidth=0.85)
    ax.set_xticks(np.arange(1, len(available) + 1), [labels[t] for t in available])
    ax.set_ylabel(r"Function-family median paired $\Delta\log_{10}(f_{best})$ vs. global")
    ax.set_title("Particle-social topology quality effects")
    ax.grid(axis="y")
    fig.text(0.5, 0.005, f"Each point is one benchmark function family (n={len(common_families)}); paired by seed and dimension. Negative favours the topology.",
             ha="center", fontsize=6.2, color=COLORS["grey"])
    fig.tight_layout(rect=(0, 0.045, 1, 1))
    return list(save_figure(fig, output, "fig_08_current_topology_quality"))


def generate(sync: Path, dynamics: Path, topology: Path, output: Path) -> dict[str, list[Path] | str]:
    configure_style()
    results: dict[str, list[Path] | str] = {}
    tasks = [
        ("equivalence", lambda: plot_equivalence(sync, output)),
        ("scaling", lambda: plot_scaling(sync, output)),
        ("runtime_components", lambda: plot_runtime_components(sync, output)),
        ("asynchronous", lambda: plot_pairwise_quality(dynamics, output, kind="asynchronous")),
        ("islands", lambda: plot_pairwise_quality(dynamics, output, kind="islands")),
        ("topologies", lambda: plot_topologies(topology, output)),
    ]
    for name, func in tasks:
        try:
            paths = func()
            results[name] = paths
            print(f"[OK] {name}")
            for path in paths:
                print(f"     {path}")
        except (FileNotFoundError, ValueError) as exc:
            results[name] = str(exc)
            print(f"[SKIP] {name}: {exc}")
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sync-campaign", type=Path, default=Path("results/raw/cpso_serial_parallel_m2_pilot"))
    parser.add_argument("--dynamics-campaign", type=Path, default=Path("results/raw/cpso_dynamics_m2_pilot"))
    parser.add_argument("--topology-campaign", type=Path, default=Path("results/raw/cpso_topology_expansion_m2_pilot"))
    parser.add_argument("--output", type=Path, default=Path("paper/figures/internal_pilots"))
    args = parser.parse_args()
    result = generate(args.sync_campaign, args.dynamics_campaign, args.topology_campaign, args.output)
    return 0 if any(isinstance(value, list) for value in result.values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
