#!/usr/bin/env python3
"""Create publication-ready vector PDF figures from completed CPSO campaigns.

This script performs post-processing only. It never launches optimization runs.
It accepts either a results directory containing ``raw/`` and ``external/`` or
an archive extraction directory containing a nested ``results/`` / ``risultati/``.

All figures use one typography and colour system and are exported as vector PDF.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.plot_style import PALETTE, configure_style  # noqa: E402

COLORS = {
    "navy": "#173F5F",
    "blue": "#2F6B9A",
    "teal": "#3A8D8C",
    "ochre": "#B87922",
    "red": "#B65D58",
    "plum": "#76558F",
    "slate": "#66717E",
    "light": "#DCE5EC",
    "grid": "#D7DEE5",
    "ink": "#17212B",
}
METHOD_COLORS = [COLORS["navy"], COLORS["blue"], COLORS["teal"], COLORS["ochre"], COLORS["plum"], COLORS["red"], COLORS["slate"]]


def resolve_results_root(path: Path) -> Path:
    path = path.expanduser().resolve()
    candidates = [path, path / "results", path / "risultati"]
    for candidate in candidates:
        if (candidate / "raw").is_dir():
            return candidate
    raise FileNotFoundError(
        f"Could not find a results root containing raw/: searched {', '.join(str(p) for p in candidates)}"
    )


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def as_float(row: dict[str, str], key: str, default: float = math.nan) -> float:
    try:
        value = float(row.get(key, ""))
        return value if math.isfinite(value) else default
    except (TypeError, ValueError):
        return default


def as_int(row: dict[str, str], key: str, default: int = 0) -> int:
    try:
        return int(float(row.get(key, "")))
    except (TypeError, ValueError):
        return default


def configure_publication_style() -> None:
    configure_style()
    plt.rcParams.update({
        "font.size": 8.4,
        "axes.titlesize": 9.2,
        "axes.labelsize": 8.3,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7.2,
        "figure.titlesize": 11,
        "axes.edgecolor": COLORS["slate"],
        "axes.labelcolor": COLORS["ink"],
        "text.color": COLORS["ink"],
        "xtick.color": COLORS["ink"],
        "ytick.color": COLORS["ink"],
        "grid.color": COLORS["grid"],
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
    })


def save_pdf(fig: plt.Figure, output: Path, name: str) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    target = output / name
    fig.savefig(target, format="pdf", bbox_inches="tight", metadata={
        "Title": name.replace("_", " ").replace(".pdf", ""),
        "Creator": "CPSO Parallel publication post-processing",
        "Subject": "Vector scientific figure generated from campaign records",
    })
    plt.close(fig)
    return target


def _load_json_array(value: str) -> list[float] | None:
    try:
        parsed = json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return None
    if not isinstance(parsed, list):
        return None
    try:
        values = [float(x) for x in parsed]
    except (TypeError, ValueError):
        return None
    return values if all(math.isfinite(x) for x in values) else None


def figure_synchronous_equivalence(results: Path, output: Path) -> Path | None:
    path = results / "raw/cpso_serial_parallel_pilot_mac/campaign_runs.csv"
    rows = [r for r in read_csv(path) if r.get("stage") == "01_exact_quality_equivalence"]
    if not rows:
        return None
    groups: dict[tuple[str, int, int], dict[str, dict[str, str]]] = defaultdict(dict)
    for row in rows:
        key = (row["benchmark"], as_int(row, "dimension"), as_int(row, "seed"))
        groups[key][row["method"]] = row

    methods = [
        "ms_global_w1", "ms_global_w2", "ms_global_w4",
        "pp_global_w1", "pp_global_w2", "pp_global_w4",
    ]
    labels = ["Master-slave (1)", "Master-slave (2)", "Master-slave (4)",
              "Particle-parallel (1)", "Particle-parallel (2)", "Particle-parallel (4)"]
    stats = []
    for method in methods:
        compared = 0
        exact_value = 0
        exact_position = 0
        exact_history = 0
        for block in groups.values():
            baseline = block.get("serial_global")
            candidate = block.get(method)
            if baseline is None or candidate is None:
                continue
            compared += 1
            exact_value += as_float(baseline, "best_value") == as_float(candidate, "best_value")
            exact_position += _load_json_array(baseline.get("best_position", "")) == _load_json_array(candidate.get("best_position", ""))
            exact_history += _load_json_array(baseline.get("best_history", "")) == _load_json_array(candidate.get("best_history", ""))
        if compared:
            stats.append((method, labels[methods.index(method)], compared,
                          exact_value / compared, exact_position / compared, exact_history / compared))

    fig, ax = plt.subplots(figsize=(7.2, 3.55))
    ax.axis("off")
    table_data = [[label, str(n), f"{v:.0%}", f"{p:.0%}", f"{h:.0%}"]
                  for _, label, n, v, p, h in stats]
    table = ax.table(
        cellText=table_data,
        colLabels=["Synchronous executor", "Paired runs", "Best value", "Best position", "Full history"],
        cellLoc="center", colLoc="center", loc="center",
        colWidths=[0.30, 0.13, 0.16, 0.18, 0.17],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.55)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("#C9D2DC")
        cell.set_linewidth(0.55)
        if r == 0:
            cell.set_facecolor(COLORS["navy"])
            cell.get_text().set_color("white")
            cell.get_text().set_weight("bold")
        else:
            cell.set_facecolor("#F2F5F8" if r % 2 == 0 else "white")
            if c > 1:
                cell.get_text().set_color(COLORS["teal"])
                cell.get_text().set_weight("bold")
    ax.set_title("Exact trajectory equivalence of synchronous CPSO executors", pad=18, weight="bold")
    fig.text(0.5, 0.075,
             "Matched by benchmark, dimension and seed. Equality is checked on stored scalar values, positions and complete best-so-far histories.",
             ha="center", va="center", fontsize=7.4, color=COLORS["slate"])
    return save_pdf(fig, output, "fig_01_synchronous_equivalence.pdf")


def figure_strong_scaling(results: Path, output: Path) -> Path | None:
    rows = read_csv(results / "raw/cpso_serial_parallel_pilot_mac/strong_scaling_summary.csv")
    rows = [r for r in rows if r.get("stage") == "02_synthetic_cost_scaling"]
    if not rows:
        return None
    work_values = sorted({as_int(r, "work_iterations") for r in rows})
    dimensions = sorted({as_int(r, "dimension") for r in rows})
    fig, axes = plt.subplots(2, 2, figsize=(7.25, 5.0), sharex=True, sharey=True)
    for row_index, work in enumerate(work_values[:2]):
        for col_index, method_prefix in enumerate(("ms_", "pp_")):
            ax = axes[row_index, col_index]
            subset = [r for r in rows if as_int(r, "work_iterations") == work
                      and r.get("method", "").startswith(method_prefix)]
            for dim_index, dimension in enumerate(dimensions):
                values = sorted((r for r in subset if as_int(r, "dimension") == dimension),
                                key=lambda r: as_int(r, "workers"))
                if not values:
                    continue
                xs = np.array([as_int(r, "workers") for r in values], dtype=float)
                ys = np.array([as_float(r, "median_speedup") for r in values])
                lows = np.array([as_float(r, "bootstrap_95_low") for r in values])
                highs = np.array([as_float(r, "bootstrap_95_high") for r in values])
                color = METHOD_COLORS[dim_index % len(METHOD_COLORS)]
                ax.plot(xs, ys, marker="o", ms=4, lw=1.6, color=color,
                        label=f"D = {dimension}")
                if np.all(np.isfinite(lows)) and np.all(np.isfinite(highs)):
                    ax.fill_between(xs, lows, highs, color=color, alpha=0.13, linewidth=0)
            ax.axhline(1.0, color=COLORS["slate"], lw=0.9, ls="--", zorder=0)
            ax.set_title(("Master-slave" if col_index == 0 else "Particle-parallel")
                         + f" · {work:,} work iterations", pad=5)
            ax.set_xticks([1, 2, 4])
            ax.grid(axis="y")
            if row_index == 1:
                ax.set_xlabel("Worker threads")
            if col_index == 0:
                ax.set_ylabel("Paired speedup")
            ax.legend(frameon=False, loc="best")
    fig.suptitle("Strong scaling of synchronous CPSO executors", y=1.015, weight="bold")
    fig.text(0.5, -0.012,
             "Synthetic objective workload; 10 paired seeds per configuration. Shading denotes bootstrap 95% intervals. Speedup = T(serial) / T(parallel).",
             ha="center", fontsize=7.2, color=COLORS["slate"])
    fig.tight_layout()
    return save_pdf(fig, output, "fig_02_strong_scaling.pdf")


def figure_parallel_efficiency(results: Path, output: Path) -> Path | None:
    rows = read_csv(results / "raw/cpso_serial_parallel_pilot_mac/strong_scaling_summary.csv")
    rows = [r for r in rows if r.get("stage") == "02_synthetic_cost_scaling"]
    if not rows:
        return None
    work_values = sorted({as_int(r, "work_iterations") for r in rows})
    dimensions = sorted({as_int(r, "dimension") for r in rows})
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.35), sharey=True)
    for col_index, method_prefix in enumerate(("ms_", "pp_")):
        ax = axes[col_index]
        subset = [r for r in rows if as_int(r, "work_iterations") == max(work_values)
                  and r.get("method", "").startswith(method_prefix)]
        for dim_index, dimension in enumerate(dimensions):
            values = sorted((r for r in subset if as_int(r, "dimension") == dimension),
                            key=lambda r: as_int(r, "workers"))
            if not values:
                continue
            xs = np.array([as_int(r, "workers") for r in values], dtype=float)
            ys = np.array([as_float(r, "parallel_efficiency") for r in values])
            color = METHOD_COLORS[dim_index % len(METHOD_COLORS)]
            ax.plot(xs, ys, marker="o", ms=4, lw=1.6, color=color, label=f"D = {dimension}")
        ax.axhline(1.0, color=COLORS["slate"], lw=0.9, ls="--")
        ax.set_title("Master-slave" if col_index == 0 else "Particle-parallel")
        ax.set_xticks([1, 2, 4])
        ax.set_xlabel("Worker threads")
        ax.grid(axis="y")
        ax.legend(frameon=False)
    axes[0].set_ylabel("Parallel efficiency")
    fig.suptitle(f"Parallel efficiency ({max(work_values):,} synthetic work iterations)", y=1.04, weight="bold")
    fig.text(0.5, -0.045, "Efficiency = speedup / worker count. Values above 1 may reflect measurement noise and should be interpreted cautiously.",
             ha="center", fontsize=7.1, color=COLORS["slate"])
    fig.tight_layout()
    return save_pdf(fig, output, "fig_03_parallel_efficiency.pdf")


def _boxplot_paired_quality(path: Path, output: Path, filename: str, title: str,
                            subtitle: str, preferred_methods: list[str],
                            display_labels: dict[str, str] | None = None) -> Path | None:
    rows = read_csv(path)
    if not rows:
        return None
    series = []
    labels = []
    counts = []
    for method in preferred_methods:
        values = [as_float(row, "delta_log10_best_vs_serial") for row in rows
                  if row.get("method") == method]
        values = [v for v in values if math.isfinite(v)]
        if values:
            series.append(values)
            labels.append((display_labels or {}).get(method, method.replace("_", "\n")))
            counts.append(len(values))
    if not series:
        return None
    fig, ax = plt.subplots(figsize=(7.1, 3.65))
    artists = ax.boxplot(series, patch_artist=True, widths=0.58, showfliers=False,
                         medianprops={"color": "white", "linewidth": 1.5},
                         whiskerprops={"color": COLORS["slate"], "linewidth": 0.9},
                         capprops={"color": COLORS["slate"], "linewidth": 0.9})
    for idx, patch in enumerate(artists["boxes"]):
        patch.set_facecolor(METHOD_COLORS[idx % len(METHOD_COLORS)])
        patch.set_alpha(0.82)
        patch.set_edgecolor(COLORS["ink"])
        patch.set_linewidth(0.65)
    ax.axhline(0.0, color=COLORS["slate"], lw=0.95, ls="--")
    ax.set_xticks(range(1, len(labels) + 1), labels)
    ax.set_ylabel(r"Paired objective difference, $\Delta \log_{10}(f_{best})$")
    ax.set_title(title, weight="bold", pad=9)
    ax.grid(axis="y")
    ax.text(0.99, 0.98, "Negative values favour the parallel variant",
            transform=ax.transAxes, ha="right", va="top", fontsize=7.2, color=COLORS["slate"])
    ax.set_xlabel("Configuration (number of paired runs: " + ", ".join(map(str, counts)) + ")")
    fig.text(0.5, -0.015, subtitle, ha="center", fontsize=7.1, color=COLORS["slate"])
    fig.tight_layout()
    return save_pdf(fig, output, filename)


def figure_historical_topologies(results: Path, output: Path) -> Path | None:
    path = results / "raw/cpso_architecture_campaign_v1/01_topology_effect/topology_quality/runs.csv"
    rows = read_csv(path)
    if not rows:
        return None
    by_key = {(r["benchmark"], as_int(r, "dimension"), as_int(r, "seed"), r["topology"]): r for r in rows}
    methods = [("ring", "Ring"), ("grid_von_neumann", "Von Neumann")]
    data, labels, n_counts = [], [], []
    for topology, label in methods:
        deltas = []
        for key, row in by_key.items():
            benchmark, dimension, seed, current_topology = key
            if current_topology != topology:
                continue
            baseline = by_key.get((benchmark, dimension, seed, "global"))
            if baseline is None:
                continue
            value = as_float(row, "best_value")
            ref = as_float(baseline, "best_value")
            if value > 0 and ref > 0:
                deltas.append(math.log10(value) - math.log10(ref))
        if deltas:
            data.append(deltas)
            labels.append(label)
            n_counts.append(len(deltas))
    if not data:
        return None
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    artists = ax.boxplot(data, patch_artist=True, widths=0.55, showfliers=False,
                         medianprops={"color": "white", "linewidth": 1.5})
    for idx, patch in enumerate(artists["boxes"]):
        patch.set_facecolor([COLORS["ochre"], COLORS["teal"]][idx])
        patch.set_alpha(0.82)
        patch.set_edgecolor(COLORS["ink"])
    ax.axhline(0, color=COLORS["slate"], lw=0.9, ls="--")
    ax.set_xticks(range(1, len(labels) + 1), labels)
    ax.set_ylabel(r"Paired objective difference vs. global, $\Delta \log_{10}(f_{best})$")
    ax.set_title("Historical CPSO topology pilot", weight="bold")
    ax.grid(axis="y")
    fig.text(0.5, -0.015,
             "Exploratory only: legacy software 0.5.0, 30 seeds per benchmark-dimension block. Not pooled with 0.7.x campaigns.",
             ha="center", fontsize=7.0, color=COLORS["red"])
    fig.tight_layout()
    return save_pdf(fig, output, "fig_07_legacy_topology_quality_exploratory.pdf")


def aggregate_topology_effects_by_family(
    rows: list[dict[str, str]], topology_order: list[str]
) -> dict[str, dict[str, float]]:
    """Reduce paired seed/dimension observations to one descriptive value per function family.

    For each topology, seed-level log differences are first summarized by the
    median within each function-dimension cell. The two dimensions are then
    summarized within their parent function family. This prevents the plot
    from visually treating seeds or dimensions as independent benchmark
    families.
    """
    baseline: dict[tuple[str, int, int], float] = {}
    for row in rows:
        if row.get("topology") == "global":
            baseline[(row["benchmark"], as_int(row, "dimension"), as_int(row, "seed"))] = as_float(row, "best_value")

    cell_differences: dict[str, dict[tuple[str, int], list[float]]] = {
        topology: defaultdict(list) for topology in topology_order
    }
    for row in rows:
        topology = row.get("topology", "")
        if topology not in cell_differences or topology == "global":
            continue
        benchmark = row.get("benchmark", "")
        dimension = as_int(row, "dimension")
        seed = as_int(row, "seed")
        reference = baseline.get((benchmark, dimension, seed), math.nan)
        value = as_float(row, "best_value")
        if value > 0 and reference > 0 and math.isfinite(reference):
            cell_differences[topology][(benchmark, dimension)].append(
                math.log10(value) - math.log10(reference)
            )

    family_effects: dict[str, dict[str, float]] = {topology: {} for topology in topology_order}
    for topology, cells in cell_differences.items():
        by_family: dict[str, list[float]] = defaultdict(list)
        for (benchmark, _dimension), values in cells.items():
            if values:
                by_family[benchmark].append(float(np.median(values)))
        for benchmark, dimension_medians in by_family.items():
            if dimension_medians:
                family_effects[topology][benchmark] = float(np.median(dimension_medians))
    return family_effects


def figure_current_topology_pilot(results: Path, output: Path) -> Path | None:
    # Prefer confirmatory records when both campaigns exist; the pilot is a
    # temporary source only until confirmatory data become available.
    candidates = [
        results / "raw/cpso_topology_expansion_confirmatory_mac/01_serial_topology_quality/native_benchmarks_d30_d100/runs.csv",
        results / "raw/cpso_topology_expansion_pilot_mac/01_serial_topology_quality/native_benchmarks_d30_d100/runs.csv",
    ]
    rows: list[dict[str, str]] = []
    chosen = None
    for candidate in candidates:
        rows = read_csv(candidate)
        if rows:
            chosen = candidate
            break
    if not rows:
        return None

    topology_order = ["global", "ring", "grid_von_neumann", "wheel", "random", "small_world", "scale_free"]
    family_effects = aggregate_topology_effects_by_family(rows, topology_order)
    plot_topologies = [topology for topology in topology_order if family_effects[topology]]
    if len(plot_topologies) < 2:
        return None

    labels = {
        "global": "Global",
        "ring": "Ring",
        "grid_von_neumann": "Von Neumann",
        "wheel": "Wheel",
        "random": "Random",
        "small_world": "Small-world",
        "scale_free": "Scale-free",
    }
    data = [[family_effects[topology][family] for family in sorted(family_effects[topology])] for topology in plot_topologies]
    family_counts = [len(values) for values in data]

    fig, ax = plt.subplots(figsize=(7.4, 3.8))
    artists = ax.boxplot(data, patch_artist=True, widths=0.58, showfliers=False,
                         medianprops={"color": "white", "linewidth": 1.4})
    for i, patch in enumerate(artists["boxes"]):
        patch.set_facecolor(METHOD_COLORS[i % len(METHOD_COLORS)])
        patch.set_alpha(0.82)
        patch.set_edgecolor(COLORS["ink"])
    for index, values in enumerate(data, start=1):
        if len(values) == 1:
            offsets = [0.0]
        else:
            offsets = np.linspace(-0.055, 0.055, len(values))
        ax.scatter(index + offsets, values, s=20, color=COLORS["ink"],
                   edgecolors="white", linewidths=0.45, zorder=3)
    ax.axhline(0, color=COLORS["slate"], lw=0.9, ls="--")
    tick_labels = [f"{labels[topology]}\n(n={count})" for topology, count in zip(plot_topologies, family_counts)]
    ax.set_xticks(range(1, len(plot_topologies) + 1), tick_labels)
    ax.set_ylabel(r"Function-family median paired difference vs. global, $\Delta \log_{10}(f_{best})$")
    ax.set_title("CPSO quality across particle communication topologies", weight="bold")
    ax.grid(axis="y")
    fig.text(0.5, -0.015,
             f"Source: {chosen.parent.parent.name}; seeds summarized within function-dimension cells, then dimensions within function. Each point is one function family (n={max(family_counts)}); descriptive only.",
             ha="center", fontsize=6.8, color=COLORS["slate"])
    fig.tight_layout()
    return save_pdf(fig, output, "fig_08_current_topology_quality.pdf")


def _circle_positions(n: int, phase: float = math.pi / 2) -> dict[int, tuple[float, float]]:
    return {i: (math.cos(phase - 2 * math.pi * i / n), math.sin(phase - 2 * math.pi * i / n)) for i in range(n)}


def _make_random_connected_edges(n: int, degree: int, seed: int) -> set[tuple[int, int]]:
    rng = random.Random(seed)
    p = degree / (n - 1)
    edges = {(i, j) for i in range(n) for j in range(i + 1, n) if rng.random() < p}
    parent = list(range(n))
    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for i, j in list(edges):
        parent[find(i)] = find(j)
    while len({find(i) for i in range(n)}) > 1:
        comps: dict[int, list[int]] = defaultdict(list)
        for i in range(n):
            comps[find(i)].append(i)
        roots = list(comps)
        a, b = rng.sample(roots, 2)
        i, j = rng.choice(comps[a]), rng.choice(comps[b])
        edges.add((min(i, j), max(i, j)))
        parent[find(i)] = find(j)
    return edges


def _make_small_world_edges(n: int, degree: int, beta: float, seed: int) -> set[tuple[int, int]]:
    rng = random.Random(seed)
    edges = {(i, (i + offset) % n) for i in range(n) for offset in range(1, degree // 2 + 1)}
    edges = {(min(i, j), max(i, j)) for i, j in edges}
    # Canonical Watts-Strogatz rewiring: each clockwise ring edge is considered
    # once. As in the model, connectivity is not forcibly repaired after rewiring.
    for i in range(n):
        for offset in range(1, degree // 2 + 1):
            j = (i + offset) % n
            edge = (min(i, j), max(i, j))
            if edge not in edges or rng.random() >= beta:
                continue
            candidates = [k for k in range(n) if k != i and (min(i, k), max(i, k)) not in edges]
            if candidates:
                edges.remove(edge)
                k = rng.choice(candidates)
                edges.add((min(i, k), max(i, k)))
    return edges


def _make_scale_free_edges(n: int, m: int, seed: int) -> set[tuple[int, int]]:
    rng = random.Random(seed)
    edges = {(i, j) for i in range(m + 1) for j in range(i + 1, m + 1)}
    degrees = [0] * n
    for i, j in edges:
        degrees[i] += 1
        degrees[j] += 1
    for node in range(m + 1, n):
        available = list(range(node))
        selected: list[int] = []
        for _ in range(m):
            weights = [degrees[k] if k not in selected else 0 for k in available]
            total = sum(weights)
            target = rng.randrange(total)
            running = 0
            for candidate, weight in zip(available, weights):
                running += weight
                if target < running:
                    selected.append(candidate)
                    break
        for candidate in selected:
            edges.add((min(node, candidate), max(node, candidate)))
            degrees[node] += 1
            degrees[candidate] += 1
    order = list(range(n))
    rng.shuffle(order)
    return {(min(order[i], order[j]), max(order[i], order[j])) for i, j in edges}


def _draw_graph(ax: plt.Axes, n: int, edges: Iterable[tuple[int, int]],
                positions: dict[int, tuple[float, float]], degrees: list[int] | None = None,
                dashed_edges: set[tuple[int, int]] | None = None) -> None:
    edge_set = set(edges)
    dashed_edges = dashed_edges or set()
    for i, j in sorted(edge_set):
        x0, y0 = positions[i]
        x1, y1 = positions[j]
        style = "--" if (i, j) in dashed_edges else "-"
        ax.plot([x0, x1], [y0, y1], style, color="#AAB5C0", lw=0.75, alpha=0.82, zorder=1)
    degree_values = degrees or [sum(1 for edge in edge_set if i in edge) for i in range(n)]
    max_degree = max(degree_values) if degree_values else 1
    for i, (x, y) in positions.items():
        size = 34 + 20 * degree_values[i] / max(1, max_degree)
        ax.scatter([x], [y], s=size, c=COLORS["navy"], edgecolors="white", linewidths=0.6, zorder=2)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_xlim(-1.25, 1.25)
    ax.set_ylim(-1.2, 1.2)


def figure_topology_schematics(output: Path) -> Path:
    fig, axes = plt.subplots(2, 4, figsize=(8.0, 5.05))
    panels = [
        ("Global-best", "complete graph; degree N-1", 10, "complete"),
        ("Ring", "periodic; degree 2", 12, "ring"),
        ("Von Neumann", "toroidal grid; degree 4", 16, "grid"),
        ("Wheel", "one focal hub; mean degree < 2", 12, "wheel"),
        ("Connected random", "ER-like; target mean degree 4", 16, "random"),
        ("Watts-Strogatz", "k = 4; beta = 0.15", 16, "small_world"),
        ("Barabasi-Albert", "preferential attachment; m = 2", 16, "scale_free"),
    ]
    for ax, (label, descriptor, n, kind) in zip(axes.flat, panels):
        if kind == "complete":
            edges = {(i, j) for i in range(n) for j in range(i + 1, n)}
            pos = _circle_positions(n)
        elif kind == "ring":
            edges = {(i, (i + 1) % n) for i in range(n)}
            pos = _circle_positions(n)
        elif kind == "grid":
            rows = cols = 4
            edges = set()
            for r in range(rows):
                for c in range(cols):
                    u = r * cols + c
                    if c + 1 < cols:
                        edges.add((u, u + 1))
                    if r + 1 < rows:
                        edges.add((u, u + cols))
            pos = {r * cols + c: (c - 1.5, 1.5 - r) for r in range(rows) for c in range(cols)}
            # Normalize the grid layout to match the circular diagrams.
            pos = {i: (x / 1.75, y / 1.75) for i, (x, y) in pos.items()}
        elif kind == "wheel":
            pos = _circle_positions(n)
            hub = n - 1
            # Canonical PSO wheel: a focal particle is connected to all
            # peripheral particles; peripheral nodes have no mutual edges.
            edges = {(hub, i) for i in range(n - 1)}
            pos[hub] = (0.0, 0.0)
        elif kind == "random":
            edges = _make_random_connected_edges(n, 4, 20261106)
            pos = _circle_positions(n)
        elif kind == "small_world":
            edges = _make_small_world_edges(n, 4, 0.15, 20261106)
            pos = _circle_positions(n)
        else:
            edges = _make_scale_free_edges(n, 2, 20261106)
            pos = _circle_positions(n)
        degree_values = [sum(1 for edge in edges if i in edge) for i in range(n)]
        _draw_graph(ax, n, edges, pos, degree_values)
        ax.set_title(f"{label}\n{descriptor}", fontsize=7.0, pad=3, linespacing=1.25)
    axes.flat[-1].axis("off")
    fig.suptitle("Particle communication topologies considered in the CPSO framework", y=0.985, weight="bold")
    fig.text(0.5, 0.025,
             "Illustrative seeded undirected graphs (N = 10-16). Self-inclusion in each social neighborhood is omitted; periodic grid wrap-around edges are not drawn.",
             ha="center", fontsize=6.7, color=COLORS["slate"])
    fig.subplots_adjust(left=0.035, right=0.985, top=0.90, bottom=0.12, wspace=0.12, hspace=0.46)
    return save_pdf(fig, output, "fig_09_particle_topology_schematics.pdf")


def figure_runtime_components(results: Path, output: Path) -> Path | None:
    rows = read_csv(results / "raw/cpso_serial_parallel_pilot_mac/campaign_runs.csv")
    rows = [r for r in rows if r.get("stage") == "02_synthetic_cost_scaling"
            and "4096" in r.get("study", "") and as_int(r, "dimension") == 30]
    methods = ["serial_global", "ms_global_w1", "ms_global_w2", "ms_global_w4",
               "pp_global_w1", "pp_global_w2", "pp_global_w4"]
    components = [
        ("initialization_seconds", "Initialization", COLORS["slate"]),
        ("topology_setup_seconds", "Topology setup", COLORS["ochre"]),
        ("evaluation_wall_seconds", "Objective evaluation", COLORS["blue"]),
        ("propagation_wall_seconds", "Propagation", COLORS["plum"]),
    ]
    medians: dict[str, list[float]] = {}
    for method in methods:
        selected = [r for r in rows if r.get("method") == method]
        if not selected:
            continue
        medians[method] = [float(np.median([as_float(r, key, 0.0) for r in selected])) for key, _, _ in components]
    if not medians:
        return None
    # The residual includes orchestration, synchronization and measurement overhead
    # not attributed to the separately timed phases. It is never allowed to go negative.
    for method, values in medians.items():
        elapsed = float(np.median([as_float(r, "elapsed_seconds") for r in rows if r.get("method") == method]))
        values.append(max(0.0, elapsed - sum(values)))
    component_labels = [label for _, label, _ in components] + ["Residual overhead"]
    component_colors = [color for _, _, color in components] + ["#CBD3DC"]
    fig, ax = plt.subplots(figsize=(7.3, 3.35))
    x = np.arange(len(medians))
    bottoms = np.zeros(len(medians))
    for component_index, (label, color) in enumerate(zip(component_labels, component_colors)):
        heights = np.array([medians[m][component_index] for m in medians])
        ax.bar(x, heights, bottom=bottoms, color=color, edgecolor="white", linewidth=0.45, label=label, width=0.72)
        bottoms += heights
    display = {
        "serial_global": "Serial",
        "ms_global_w1": "Master--slave\n(1 worker)",
        "ms_global_w2": "Master--slave\n(2 workers)",
        "ms_global_w4": "Master--slave\n(4 workers)",
        "pp_global_w1": "Particle-parallel\n(1 worker)",
        "pp_global_w2": "Particle-parallel\n(2 workers)",
        "pp_global_w4": "Particle-parallel\n(4 workers)",
    }
    ax.set_xticks(x, [display.get(m, m) for m in medians], fontsize=6.9)
    ax.set_ylabel("Median wall time per run (seconds)")
    ax.set_title("Measured runtime components under a high synthetic objective cost", weight="bold")
    ax.grid(axis="y")
    ax.legend(ncol=3, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.20))
    fig.subplots_adjust(left=0.105, right=0.985, top=0.84, bottom=0.34)
    fig.text(0.5, 0.035,
             "D = 30, population = 64, 4,096 synthetic work iterations; paired-seed medians. Social-selection time is included in propagation.",
             ha="center", fontsize=6.7, color=COLORS["slate"])
    return save_pdf(fig, output, "fig_04_runtime_components.pdf")


def create_catalog(output: Path, figures: list[Path], missing: list[str], results: Path) -> Path:
    lines = [
        "# Publication figure catalogue",
        "",
        "All listed graphics are vector PDF files generated by `analysis/postprocess_publication_figures.py`.",
        "No optimization run is launched by this post-processing command.",
        "",
        "Source results: user-supplied directory containing `raw/` and `external/` records.",
        "",
        "## Generated figures",
        "",
    ]
    for figure in figures:
        lines.append(f"- `{figure.name}`")
    lines.extend(["", "## Data limitations and interpretation", "",
                  "- Synchronous-equivalence and scaling figures use the 0.7.1 pilot records supplied with the results archive.",
        "- The current seven-topology comparison is generated automatically after `cpso_topology_expansion_pilot_mac` or the confirmatory campaign is run.",
                  "- Asynchronous and island plots are exploratory paired comparisons from five seeds; they are not confirmatory evidence.",
                  "- The legacy topology comparison is from software 0.5.0 and is explicitly exploratory. It must not be pooled with 0.7.x runs.",
                  "- BBOB and CEC smoke records are integration checks only. They are not used for publication-level performance claims.",
                  "- The topology schematics are explanatory network diagrams, not measured experimental outcomes.",
                  "- COCO post-processing plots (e.g. ECDFs) require official COCO observer data and `cocopp`; JSONL records alone are insufficient.",
                  ""])
    if missing:
        lines.extend(["## Figures not generated because source data were not present", ""])
        lines.extend(f"- `{name}`" for name in missing)
    target = output / "FIGURE_CATALOGUE.md"
    target.write_text("\n".join(lines), encoding="utf-8")
    return target


def build_figures(results_arg: Path, output: Path) -> tuple[list[Path], list[str]]:
    results = resolve_results_root(results_arg)
    configure_publication_style()
    output.mkdir(parents=True, exist_ok=True)
    figures: list[Path] = []
    missing: list[str] = []

    makers = [
        ("fig_01_synchronous_equivalence.pdf", lambda: figure_synchronous_equivalence(results, output)),
        ("fig_02_strong_scaling.pdf", lambda: figure_strong_scaling(results, output)),
        ("fig_03_parallel_efficiency.pdf", lambda: figure_parallel_efficiency(results, output)),
        ("fig_04_runtime_components.pdf", lambda: figure_runtime_components(results, output)),
        ("fig_05_asynchronous_quality.pdf", lambda: _boxplot_paired_quality(
            results / "raw/cpso_dynamics_pilot_mac/01_asynchronous_quality/async_global_and_ring_d30_d100/paired_comparisons.csv",
            output, "fig_05_asynchronous_quality.pdf", "Asynchronous CPSO: paired quality differences",
            "Exploratory pilot: five seeds per function-dimension block; do not interpret boxplot spread as independent observations.",
            ["async_global_w2", "async_global_w4", "async_ring_w2", "async_ring_w4"],
            {"async_global_w2": "Global\n2 workers", "async_global_w4": "Global\n4 workers",
             "async_ring_w2": "Ring\n2 workers", "async_ring_w4": "Ring\n4 workers"})),
        ("fig_06_island_quality.pdf", lambda: _boxplot_paired_quality(
            results / "raw/cpso_dynamics_pilot_mac/02_island_quality/island_migration_and_topology_d30_d100/paired_comparisons.csv",
            output, "fig_06_island_quality.pdf", "Island-model CPSO: paired quality differences",
            "Exploratory pilot: five seeds per function-dimension block; migration policies and island counts are configuration-specific.",
            ["island_global_m2_no_migration", "island_global_m4_ring_mig20",
             "island_global_m4_full_mig20", "island_ring_m4_ring_mig20"],
            {"island_global_m2_no_migration": "Global; 2 islands\nno migration",
             "island_global_m4_ring_mig20": "Global; 4 islands\nring migration",
             "island_global_m4_full_mig20": "Global; 4 islands\nfully connected migration",
             "island_ring_m4_ring_mig20": "Ring; 4 islands\nring migration"})),
        ("fig_07_legacy_topology_quality_exploratory.pdf", lambda: figure_historical_topologies(results, output)),
        ("fig_08_current_topology_quality.pdf", lambda: figure_current_topology_pilot(results, output)),
        ("fig_09_particle_topology_schematics.pdf", lambda: figure_topology_schematics(output)),
    ]
    for expected_name, maker in makers:
        try:
            result = maker()
        except Exception as exc:  # preserve other figures and report a useful diagnostic
            print(f"WARNING: failed to create {expected_name}: {type(exc).__name__}: {exc}", file=sys.stderr)
            missing.append(f"{expected_name} (generation error: {type(exc).__name__}: {exc})")
            continue
        if result is None:
            missing.append(f"{expected_name} (source data not found or insufficient)")
        else:
            figures.append(result)
    create_catalog(output, figures, missing, results)
    return figures, missing


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=Path("results"),
                        help="Directory containing raw/ and external/ (or its parent extraction directory)")
    parser.add_argument("--output", type=Path, default=Path("paper/figures/publication_stage_A"),
                        help="Output directory for vector PDF figures")
    args = parser.parse_args()
    figures, missing = build_figures(args.results_root, args.output)
    print(f"Generated {len(figures)} vector PDF figure(s) in {args.output}")
    for figure in figures:
        print(f"  {figure}")
    if missing:
        print("Not generated:")
        for item in missing:
            print(f"  - {item}")
    return 0 if figures else 2


if __name__ == "__main__":
    raise SystemExit(main())
