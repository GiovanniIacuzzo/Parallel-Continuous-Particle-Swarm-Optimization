#!/usr/bin/env python3
"""Create only result-bearing figures from completed CPSO campaigns.

The script is deliberately a post-processor: it never launches an optimizer.
It excludes pilot/legacy sources, explanatory topology schematics and
presentation tables. Missing confirmatory datasets are reported, not silently
replaced with exploratory results.
"""
from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
COLORS = {
    "navy": "#173F5F",
    "blue": "#2F6B9A",
    "teal": "#3A8D8C",
    "ochre": "#B87922",
    "red": "#B65D58",
    "plum": "#76558F",
    "slate": "#66717E",
    "grid": "#D7DEE5",
    "ink": "#17212B",
}
PALETTE = [COLORS["navy"], COLORS["blue"], COLORS["teal"], COLORS["ochre"],
           COLORS["plum"], COLORS["red"], COLORS["slate"]]


def resolve_results_root(path: Path) -> Path:
    path = path.expanduser().resolve()
    for candidate in (path, path / "results", path / "risultati"):
        if (candidate / "raw").is_dir():
            return candidate
    raise FileNotFoundError(f"No results root containing raw/ found below {path}")


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def number(row: dict[str, str], key: str, default: float = math.nan) -> float:
    try:
        value = float(row.get(key, ""))
        return value if math.isfinite(value) else default
    except (TypeError, ValueError):
        return default


def integer(row: dict[str, str], key: str, default: int = 0) -> int:
    try:
        return int(float(row.get(key, "")))
    except (TypeError, ValueError):
        return default


def configure_style() -> None:
    plt.rcParams.update({
        "font.size": 8.2,
        "axes.titlesize": 9.2,
        "axes.labelsize": 8.3,
        "xtick.labelsize": 7.4,
        "ytick.labelsize": 7.4,
        "legend.fontsize": 7.2,
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
        "Subject": "Vector figure generated from confirmatory experimental records",
    })
    plt.close(fig)
    return target


def _campaign_root(results: Path, campaign_id: str) -> Path:
    return results / "raw" / campaign_id


def figure_parallel_speedup(results: Path, output: Path) -> Path | None:
    path = _campaign_root(results, "cpso_architecture_campaign_v1") / "strong_scaling_summary.csv"
    rows = [r for r in read_csv(path) if r.get("stage") == "04_strong_scaling"]
    if not rows:
        return None
    work_values = [integer(r, "work_iterations") for r in rows]
    max_work = max(work_values)
    rows = [r for r in rows if integer(r, "work_iterations") == max_work]
    populations = sorted({integer(r, "population_size") for r in rows})
    dimensions = sorted({integer(r, "dimension") for r in rows})
    if not populations or not dimensions:
        return None

    fig, axes = plt.subplots(len(populations), 2, figsize=(7.2, 2.75 * len(populations)),
                             sharex=True, sharey=True, squeeze=False)
    for row_index, population in enumerate(populations):
        for col_index, prefix in enumerate(("ms_", "pp_")):
            ax = axes[row_index, col_index]
            subset = [r for r in rows if integer(r, "population_size") == population
                      and r.get("method", "").startswith(prefix)]
            for dim_index, dimension in enumerate(dimensions):
                values = sorted(
                    (r for r in subset if integer(r, "dimension") == dimension),
                    key=lambda r: integer(r, "workers"),
                )
                if not values:
                    continue
                x = np.asarray([integer(r, "workers") for r in values], dtype=float)
                y = np.asarray([number(r, "median_speedup") for r in values], dtype=float)
                low = np.asarray([number(r, "bootstrap_95_low") for r in values], dtype=float)
                high = np.asarray([number(r, "bootstrap_95_high") for r in values], dtype=float)
                valid = np.isfinite(x) & np.isfinite(y)
                x, y, low, high = x[valid], y[valid], low[valid], high[valid]
                color = PALETTE[dim_index % len(PALETTE)]
                yerr = np.vstack([np.maximum(0.0, y - low), np.maximum(0.0, high - y)])
                ax.errorbar(x, y, yerr=yerr, fmt="o-", capsize=2.5, ms=3.8,
                            lw=1.45, color=color, label=f"D = {dimension}")
            max_worker = max((integer(r, "workers") for r in subset), default=1)
            ax.plot([1, max_worker], [1, max_worker], "--", color=COLORS["slate"],
                    linewidth=0.85, label="Ideal scaling")
            ax.axhline(1.0, color=COLORS["grid"], linewidth=0.65)
            ax.set_title(f"{'Master–slave' if col_index == 0 else 'Particle-parallel'} · N={population}")
            ax.set_xticks(sorted({integer(r, "workers") for r in subset}))
            ax.grid(axis="y", alpha=0.8)
            if row_index == len(populations) - 1:
                ax.set_xlabel("Worker threads")
            if col_index == 0:
                ax.set_ylabel("Paired end-to-end speedup")
            ax.legend(frameon=False, loc="best")
    fig.suptitle(f"Strong scaling of synchronous CPSO executors (synthetic workload = {max_work:,})",
                 y=1.005, weight="bold")
    fig.text(0.5, -0.005,
             "Speedup = T(serial) / T(parallel), paired by dimension and seed; full native wall time includes thread-pool lifecycle. Shading/whiskers: bootstrap 95% CI.",
             ha="center", fontsize=6.8, color=COLORS["slate"])
    fig.tight_layout()
    return save_pdf(fig, output, "fig_parallel_speedup.pdf")


def figure_parallel_efficiency(results: Path, output: Path) -> Path | None:
    path = _campaign_root(results, "cpso_architecture_campaign_v1") / "strong_scaling_summary.csv"
    rows = [r for r in read_csv(path) if r.get("stage") == "04_strong_scaling"]
    if not rows:
        return None
    max_work = max(integer(r, "work_iterations") for r in rows)
    rows = [r for r in rows if integer(r, "work_iterations") == max_work]
    populations = sorted({integer(r, "population_size") for r in rows})
    dimensions = sorted({integer(r, "dimension") for r in rows})
    fig, axes = plt.subplots(len(populations), 2, figsize=(7.2, 2.65 * len(populations)),
                             sharex=True, sharey=True, squeeze=False)
    for row_index, population in enumerate(populations):
        for col_index, prefix in enumerate(("ms_", "pp_")):
            ax = axes[row_index, col_index]
            subset = [r for r in rows if integer(r, "population_size") == population
                      and r.get("method", "").startswith(prefix)]
            for dim_index, dimension in enumerate(dimensions):
                values = sorted((r for r in subset if integer(r, "dimension") == dimension),
                                key=lambda r: integer(r, "workers"))
                if not values:
                    continue
                x = np.asarray([integer(r, "workers") for r in values], dtype=float)
                y = np.asarray([number(r, "parallel_efficiency") for r in values], dtype=float)
                color = PALETTE[dim_index % len(PALETTE)]
                ax.plot(x, y, marker="o", ms=3.8, lw=1.45, color=color, label=f"D = {dimension}")
            ax.axhline(1.0, color=COLORS["slate"], linewidth=0.85, linestyle="--")
            ax.set_title(f"{'Master–slave' if col_index == 0 else 'Particle-parallel'} · N={population}")
            ax.grid(axis="y")
            if row_index == len(populations) - 1:
                ax.set_xlabel("Worker threads")
            if col_index == 0:
                ax.set_ylabel("Parallel efficiency (S / p)")
            ax.legend(frameon=False, loc="best")
    fig.suptitle(f"Parallel efficiency under the high-cost workload ({max_work:,})", y=1.005, weight="bold")
    fig.text(0.5, -0.005,
             "Efficiency is paired speedup divided by the number of worker threads; values above one indicate superlinear measured speedup.",
             ha="center", fontsize=6.8, color=COLORS["slate"])
    fig.tight_layout()
    return save_pdf(fig, output, "fig_parallel_efficiency.pdf")


def figure_runtime_decomposition(results: Path, output: Path) -> Path | None:
    path = _campaign_root(results, "cpso_architecture_campaign_v1") / "campaign_runs.csv"
    rows = [r for r in read_csv(path) if r.get("stage") == "04_strong_scaling"]
    if not rows:
        return None
    work = max(integer(r, "work_iterations") for r in rows) if "work_iterations" in rows[0] else 4096
    # The study id is the authoritative config-level work descriptor in campaign_runs.csv.
    rows = [r for r in rows if f"work{work:04d}" in r.get("study", "")]
    population = max(integer(r, "population_size") for r in rows) if rows else 0
    dimension = max(integer(r, "dimension") for r in rows) if rows else 0
    rows = [r for r in rows if integer(r, "population_size") == population
            and integer(r, "dimension") == dimension]
    if not rows:
        return None

    methods = ["serial_global"] + [f"{prefix}_global_w{worker}" for prefix in ("ms", "pp")
                                    for worker in (1, 2, 4, 8)]
    components = [
        ("initialization_seconds", "Initialization"),
        ("topology_setup_seconds", "Topology setup"),
        ("evaluation_wall_seconds", "Objective evaluation"),
        ("propagation_wall_seconds", "Propagation / social update"),
    ]
    rows_by_method: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        rows_by_method[row.get("method", "")].append(row)
    medians: dict[str, list[float]] = {}
    for method in methods:
        selected = rows_by_method.get(method, [])
        if not selected:
            continue
        elapsed_by_run = [number(row, "elapsed_seconds") for row in selected]
        component_by_run = [
            [max(0.0, number(row, key, 0.0)) for key, _ in components]
            for row in selected
        ]
        residual_by_run = [
            max(0.0, elapsed - sum(parts))
            for elapsed, parts in zip(elapsed_by_run, component_by_run)
        ]
        values = [float(np.median([parts[i] for parts in component_by_run]))
                  for i in range(len(components))]
        values.append(float(np.median(residual_by_run)))
        medians[method] = values
    if len(medians) < 3:
        return None

    fig, ax = plt.subplots(figsize=(7.4, 3.8))
    x = np.arange(len(medians))
    bottom = np.zeros(len(medians))
    component_labels = [name for _, name in components] + ["Residual / orchestration"]
    component_colors = [COLORS["slate"], COLORS["ochre"], COLORS["blue"], COLORS["plum"], "#CBD3DC"]
    for index, (label, color) in enumerate(zip(component_labels, component_colors)):
        heights = np.asarray([medians[method][index] for method in medians])
        ax.bar(x, heights, bottom=bottom, width=0.72, color=color, edgecolor="white",
               linewidth=0.4, label=label)
        bottom += heights
    labels = []
    for method in medians:
        if method == "serial_global":
            labels.append("Serial")
        elif method.startswith("ms_"):
            labels.append(f"MS · {method.rsplit('w', 1)[1]} workers")
        else:
            labels.append(f"PP · {method.rsplit('w', 1)[1]} workers")
    ax.set_xticks(x, labels, rotation=25, ha="right")
    ax.set_ylabel("Component-wise median wall time (s)")
    ax.set_title(f"Measured runtime components · N={population}, D={dimension}, work={work:,}", weight="bold")
    ax.grid(axis="y")
    ax.legend(ncol=3, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.30))
    fig.subplots_adjust(left=0.11, right=0.99, top=0.85, bottom=0.36)
    fig.text(0.5, 0.025,
             "Each segment is the median of that phase across paired runs; residual is computed per run. Social-selection time is contained in propagation.",
             ha="center", fontsize=6.7, color=COLORS["slate"])
    return save_pdf(fig, output, "fig_runtime_decomposition.pdf")


def figure_quality_effects(
    results: Path, output: Path, *, stage: str, filename: str, title: str,
) -> Path | None:
    path = _campaign_root(results, "cpso_architecture_campaign_v1") / "confirmatory_pairwise_tests.csv"
    rows = [r for r in read_csv(path) if r.get("stage") == stage]
    if not rows:
        return None
    rows.sort(key=lambda row: row.get("method", ""))
    labels = [row["method"].replace("_", " ") for row in rows]
    medians = np.asarray([number(row, "median_function_family_log10_delta") for row in rows])
    lows = np.asarray([number(row, "hierarchical_bootstrap_95_low") for row in rows])
    highs = np.asarray([number(row, "hierarchical_bootstrap_95_high") for row in rows])
    if not np.all(np.isfinite(medians)) or not np.all(np.isfinite(lows)) or not np.all(np.isfinite(highs)):
        return None
    y = np.arange(len(rows))
    colors = [PALETTE[index % len(PALETTE)] for index in range(len(rows))]
    fig_height = max(3.2, 0.34 * len(rows) + 1.2)
    fig, ax = plt.subplots(figsize=(7.1, fig_height))
    for index in range(len(rows)):
        ax.errorbar(medians[index], y[index],
                    xerr=[[max(0.0, medians[index] - lows[index])],
                          [max(0.0, highs[index] - medians[index])]],
                    fmt="o", ms=4.2, capsize=2.5, color=colors[index])
    ax.axvline(0.0, color=COLORS["slate"], linewidth=0.9, linestyle="--")
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel(r"Median function-family $\Delta\log_{10}(f_{best})$ vs. matched serial topology")
    ax.set_title(title, weight="bold")
    ax.grid(axis="x")
    fig.text(0.5, 0.015,
             "Negative values favour the variant. Points and intervals use function-family summaries; seeds and dimensions are nested, not independent.",
             ha="center", fontsize=6.7, color=COLORS["slate"])
    fig.tight_layout(rect=(0, 0.045, 1, 1))
    return save_pdf(fig, output, filename)


def aggregate_topology_effects_by_family(
    rows: list[dict[str, str]], topology_order: list[str],
) -> dict[str, dict[str, float]]:
    """One median paired log-objective effect per function family and topology."""
    baseline: dict[tuple[str, int, int], float] = {}
    for row in rows:
        if row.get("topology") == "global":
            baseline[(row["benchmark"], integer(row, "dimension"), integer(row, "seed"))] = number(row, "best_value")
    cells: dict[str, dict[tuple[str, int], list[float]]] = {
        topology: defaultdict(list) for topology in topology_order
    }
    for row in rows:
        topology = row.get("topology", "")
        if topology not in cells or topology == "global":
            continue
        benchmark = row.get("benchmark", "")
        dimension = integer(row, "dimension")
        seed = integer(row, "seed")
        ref = baseline.get((benchmark, dimension, seed), math.nan)
        value = number(row, "best_value")
        if not math.isfinite(ref) or not math.isfinite(value) or value < 0 or ref < 0:
            continue
        delta = math.log10(max(value, 1.0e-16)) - math.log10(max(ref, 1.0e-16))
        cells[topology][(benchmark, dimension)].append(delta)
    family: dict[str, dict[str, float]] = {topology: {} for topology in topology_order}
    for topology, blocks in cells.items():
        grouped: dict[str, list[float]] = defaultdict(list)
        for (benchmark, _dimension), values in blocks.items():
            if values:
                grouped[benchmark].append(float(np.median(values)))
        for benchmark, dimension_medians in grouped.items():
            family[topology][benchmark] = float(np.median(dimension_medians))
    return family


def figure_topology_quality(results: Path, output: Path) -> Path | None:
    path = _campaign_root(results, "cpso_topology_expansion_confirmatory_073") / (
        "01_serial_topology_quality/native_benchmarks_d30_d100/runs.csv"
    )
    rows = read_csv(path)
    if not rows:
        return None
    topology_order = ["global", "ring", "grid_von_neumann", "wheel", "random", "small_world", "scale_free"]
    effects = aggregate_topology_effects_by_family(rows, topology_order)
    labels = {
        "global": "Global", "ring": "Ring", "grid_von_neumann": "Von Neumann",
        "wheel": "Wheel", "random": "Random", "small_world": "Small-world", "scale_free": "Scale-free",
    }
    available = [topology for topology in topology_order if effects[topology]]
    if len(available) < 2:
        return None
    families = sorted(set.intersection(*(set(effects[topology]) for topology in available)))
    if not families:
        return None
    data = [[effects[topology][family] for family in families] for topology in available]
    fig, ax = plt.subplots(figsize=(7.3, 3.8))
    box = ax.boxplot(data, patch_artist=True, widths=0.58, showfliers=False,
                     medianprops={"color": "white", "linewidth": 1.4})
    for index, patch in enumerate(box["boxes"]):
        patch.set_facecolor(PALETTE[index % len(PALETTE)])
        patch.set_alpha(0.82)
        patch.set_edgecolor(COLORS["ink"])
    for index, values in enumerate(data, start=1):
        offsets = np.linspace(-0.06, 0.06, len(values)) if len(values) > 1 else [0.0]
        ax.scatter(index + offsets, values, s=22, color=COLORS["ink"],
                   edgecolors="white", linewidths=0.4, zorder=3)
    ax.axhline(0.0, color=COLORS["slate"], linewidth=0.9, linestyle="--")
    ax.set_xticks(np.arange(1, len(available) + 1), [labels[name] for name in available])
    ax.set_ylabel(r"Function-family median paired $\Delta\log_{10}(f_{best})$ vs. global")
    ax.set_title("Confirmatory quality effects of particle-social topology", weight="bold")
    ax.grid(axis="y")
    fig.text(0.5, 0.015,
             f"Confirmatory 0.7.3 campaign; each point is one benchmark family (n={len(families)}). Negative values favour the topology.",
             ha="center", fontsize=6.8, color=COLORS["slate"])
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    return save_pdf(fig, output, "fig_topology_quality_effects.pdf")


def create_catalog(output: Path, figures: list[Path], missing: list[str], results: Path) -> Path:
    lines = [
        "# Publication figure catalogue", "",
        "This post-processing pipeline emits only figures derived from measured campaign results.",
        "It does not generate tables, topology illustrations, or substitute pilot/legacy data when confirmatory records are absent.",
        "", f"Results root: `{results}`", "", "## Generated figures", "",
    ]
    lines.extend(f"- `{path.name}`" for path in figures)
    lines.extend(["", "## Not generated", ""])
    lines.extend(f"- `{item}`" for item in missing)
    lines.extend([
        "", "## Interpretation", "",
        "- Parallel speedup and efficiency are computed from the native `sphere_work` strong-scaling campaign.",
        "- Runtime decomposition is descriptive and uses the highest population/dimension/workload combination available.",
        "- Asynchronous and island plots show quality effects only; their changed search dynamics are not labelled speedup.",
        "- Topology effects use the dedicated confirmatory topology records only; no pilot fallback is performed.",
        "- BBOB/CEC quality and official COCO ECDF analyses remain in `postprocess_external_benchmarks.py`.",
        "- The separate dt analyzer generates the two time-interval sensitivity figures from its own validated campaign.",
        "",
    ])
    target = output / "FIGURE_CATALOGUE.md"
    target.write_text("\n".join(lines), encoding="utf-8")
    return target


def build_figures(results_arg: Path, output: Path) -> tuple[list[Path], list[str]]:
    results = resolve_results_root(results_arg)
    configure_style()
    output.mkdir(parents=True, exist_ok=True)
    # Remove only the obsolete artefacts produced by the old postprocessor.
    obsolete = {
        "fig_01_synchronous_equivalence.pdf", "fig_02_strong_scaling.pdf",
        "fig_03_parallel_efficiency.pdf", "fig_04_runtime_components.pdf",
        "fig_05_asynchronous_quality.pdf", "fig_06_island_quality.pdf",
        "fig_07_legacy_topology_quality_exploratory.pdf", "fig_08_current_topology_quality.pdf",
        "fig_09_particle_topology_schematics.pdf",
    }
    for filename in obsolete:
        stale = output / filename
        if stale.exists():
            stale.unlink()

    makers = [
        ("fig_parallel_speedup.pdf", lambda: figure_parallel_speedup(results, output)),
        ("fig_parallel_efficiency.pdf", lambda: figure_parallel_efficiency(results, output)),
        ("fig_runtime_decomposition.pdf", lambda: figure_runtime_decomposition(results, output)),
        ("fig_asynchronous_quality_effects.pdf", lambda: figure_quality_effects(
            results, output, stage="02_asynchronous_dynamics",
            filename="fig_asynchronous_quality_effects.pdf",
            title="Confirmatory quality effects of asynchronous CPSO",
        )),
        ("fig_island_quality_effects.pdf", lambda: figure_quality_effects(
            results, output, stage="03_island_dynamics",
            filename="fig_island_quality_effects.pdf",
            title="Confirmatory quality effects of island-model CPSO",
        )),
        ("fig_topology_quality_effects.pdf", lambda: figure_topology_quality(results, output)),
    ]
    figures: list[Path] = []
    missing: list[str] = []
    for expected, maker in makers:
        try:
            made = maker()
        except Exception as exc:  # make remaining figures even if one dataset is malformed
            missing.append(f"{expected}: generation failed ({type(exc).__name__}: {exc})")
            continue
        if made is None:
            missing.append(f"{expected}: confirmatory source data not present or incomplete")
        else:
            figures.append(made)
    create_catalog(output, figures, missing, results)
    return figures, missing


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument("--output", type=Path, default=Path("paper/figures/publication_stage_A"))
    args = parser.parse_args()
    figures, missing = build_figures(args.results_root, args.output)
    print(f"Generated {len(figures)} publication-result figure(s) in {args.output}")
    for figure in figures:
        print(f"  {figure}")
    if missing:
        print("Not generated:")
        for item in missing:
            print(f"  - {item}")
    return 0 if figures else 2


if __name__ == "__main__":
    raise SystemExit(main())
