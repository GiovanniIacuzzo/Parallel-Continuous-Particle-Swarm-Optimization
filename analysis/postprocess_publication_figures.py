#!/usr/bin/env python3
"""Generate publication figures from explicitly selected CPSO campaign outputs.

The default profile is confirmatory and reads only the predeclared definitive
campaigns. The exploratory profile requires three explicit pilot campaign
folders and labels every figure as exploratory. No profile fabricates missing
data or silently falls back from one evidence class to another.
"""
from __future__ import annotations

import argparse
import csv
import json
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
    "navy": "#173F5F", "blue": "#2F6B9A", "teal": "#3A8D8C",
    "ochre": "#B87922", "red": "#B65D58", "plum": "#76558F",
    "slate": "#66717E", "grid": "#D7DEE5", "ink": "#17212B",
}
PALETTE = [COLORS["navy"], COLORS["blue"], COLORS["teal"], COLORS["ochre"],
           COLORS["plum"], COLORS["red"], COLORS["slate"]]


def resolve_results_root(path: Path) -> Path:
    path = path.expanduser().resolve()
    for candidate in (path, path / "results", path / "risultati"):
        if (candidate / "raw").is_dir():
            return candidate
    raise FileNotFoundError(f"No results root containing raw/ found below {path}")


def resolve_campaign_dir(results: Path, supplied: Path | None, default_name: str) -> Path:
    if supplied is None:
        candidate = results / "raw" / default_name
    else:
        candidate = supplied.expanduser()
        if not candidate.is_absolute():
            candidate = (ROOT / candidate).resolve()
    if not candidate.is_dir():
        raise FileNotFoundError(f"Campaign output directory not found: {candidate}")
    return candidate.resolve()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def read_manifest(campaign_dir: Path) -> dict[str, Any]:
    path = campaign_dir / "campaign_manifest.json"
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid campaign manifest {path}: {exc}") from exc
    return payload if isinstance(payload, dict) else {}


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
        "font.size": 8.2, "axes.titlesize": 9.2, "axes.labelsize": 8.3,
        "xtick.labelsize": 7.4, "ytick.labelsize": 7.4, "legend.fontsize": 7.2,
        "axes.edgecolor": COLORS["slate"], "axes.labelcolor": COLORS["ink"],
        "text.color": COLORS["ink"], "xtick.color": COLORS["ink"],
        "ytick.color": COLORS["ink"], "grid.color": COLORS["grid"],
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "savefig.facecolor": "white", "figure.facecolor": "white",
    })


def save_pdf(fig: plt.Figure, output: Path, name: str, profile: str) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    target = output / name
    label = "exploratory pilot" if profile == "exploratory" else "confirmatory campaign"
    fig.savefig(target, format="pdf", bbox_inches="tight", metadata={
        "Title": name.replace("_", " ").replace(".pdf", ""),
        "Creator": "CPSO publication post-processing",
        "Subject": f"Measured CPSO experimental records ({label})",
    })
    plt.close(fig)
    return target


def evidence_suffix(profile: str) -> str:
    return " — exploratory M2 pilot" if profile == "exploratory" else ""


def figure_synchronous_equivalence(
    campaign: Path, output: Path, profile: str,
) -> Path | None:
    rows = read_csv(campaign / "equivalence_audit.csv")
    rows = [row for row in rows if row.get("architecture") in {"master_slave", "particle_parallel"}]
    if not rows:
        return None
    rows.sort(key=lambda row: (row.get("architecture", ""), integer(row, "workers"), row.get("method", "")))
    labels = [
        f"{row.get('study', 'study')} · {'MS' if row['architecture'] == 'master_slave' else 'PP'} · {integer(row, 'workers')} workers"
        for row in rows
    ]
    verified = np.asarray([integer(row, "verified_pairs") for row in rows], dtype=float)
    expected = np.asarray([integer(row, "expected_pairs") for row in rows], dtype=float)
    if np.any(expected <= 0) or np.any(verified > expected):
        raise ValueError("Invalid counts in equivalence_audit.csv")
    fractions = verified / expected
    fig, ax = plt.subplots(figsize=(7.0, max(2.8, 0.38 * len(rows) + 1.1)))
    y = np.arange(len(rows))
    ax.barh(y, fractions, color=[PALETTE[i % len(PALETTE)] for i in range(len(rows))], height=0.62)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlim(0.0, 1.08)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0], ["0", "25", "50", "75", "100%"])
    ax.set_xlabel("Exact trajectory pairs verified")
    ax.set_title("Synchronous executor equivalence" + evidence_suffix(profile), weight="bold")
    ax.grid(axis="x")
    for index, (v, e) in enumerate(zip(verified, expected)):
        ax.text(min(1.04, fractions[index] + 0.015), index, f"{int(v)}/{int(e)}", va="center", fontsize=7)
    fig.text(0.5, 0.015,
             "Each pair is matched by benchmark, dimension, seed and topology; values, position, best-so-far trace, FE axis and FE count were compared exactly.",
             ha="center", fontsize=6.6, color=COLORS["slate"])
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    return save_pdf(fig, output, "fig_01_synchronous_equivalence.pdf", profile)


def _performance_rows(campaign: Path) -> list[dict[str, str]]:
    rows = read_csv(campaign / "strong_scaling_summary.csv")
    return [row for row in rows if row.get("method", "").startswith(("ms_", "pp_"))]


def figure_parallel_speedup(campaign: Path, output: Path, profile: str) -> Path | None:
    rows = _performance_rows(campaign)
    if not rows:
        return None
    work_values = [integer(row, "work_iterations") for row in rows]
    max_work = max(work_values)
    rows = [row for row in rows if integer(row, "work_iterations") == max_work]
    populations = sorted({integer(row, "population_size") for row in rows})
    dimensions = sorted({integer(row, "dimension") for row in rows})
    if not populations or not dimensions:
        return None

    fig, axes = plt.subplots(len(populations), 2, figsize=(7.2, 2.75 * len(populations)),
                             sharex=True, sharey=True, squeeze=False)
    for row_index, population in enumerate(populations):
        for col_index, prefix in enumerate(("ms_", "pp_")):
            ax = axes[row_index, col_index]
            subset = [row for row in rows if integer(row, "population_size") == population
                      and row.get("method", "").startswith(prefix)]
            for dim_index, dimension in enumerate(dimensions):
                values = sorted((row for row in subset if integer(row, "dimension") == dimension),
                                key=lambda row: integer(row, "workers"))
                if not values:
                    continue
                x = np.asarray([integer(row, "workers") for row in values], dtype=float)
                y = np.asarray([number(row, "median_speedup") for row in values], dtype=float)
                low = np.asarray([number(row, "bootstrap_95_low") for row in values], dtype=float)
                high = np.asarray([number(row, "bootstrap_95_high") for row in values], dtype=float)
                valid = np.isfinite(x) & np.isfinite(y)
                x, y, low, high = x[valid], y[valid], low[valid], high[valid]
                yerr = np.vstack([np.maximum(0.0, y - low), np.maximum(0.0, high - y)])
                ax.errorbar(x, y, yerr=yerr, fmt="o-", capsize=2.5, ms=3.8,
                            lw=1.45, color=PALETTE[dim_index % len(PALETTE)], label=f"D = {dimension}")
            max_worker = max((integer(row, "workers") for row in subset), default=1)
            ax.plot([1, max_worker], [1, max_worker], "--", color=COLORS["slate"],
                    linewidth=0.85, label="Ideal scaling")
            ax.axhline(1.0, color=COLORS["grid"], linewidth=0.65)
            ax.set_title(f"{'Master–slave' if col_index == 0 else 'Particle-parallel'} · N={population}")
            ax.set_xticks(sorted({integer(row, "workers") for row in subset}))
            ax.grid(axis="y", alpha=0.8)
            if row_index == len(populations) - 1:
                ax.set_xlabel("Worker threads")
            if col_index == 0:
                ax.set_ylabel("Paired end-to-end speedup")
            ax.legend(frameon=False, loc="best")
    fig.suptitle(f"Strong scaling · synthetic workload = {max_work:,}{evidence_suffix(profile)}",
                 y=1.005, weight="bold")
    note = "Timing is machine-specific; sphere_work is a controlled execution-cost instrument, not an optimization benchmark."
    if profile == "exploratory":
        note += " Local M2 pilot: do not compare absolute seconds directly with HPC runs."
    fig.text(0.5, -0.005, note, ha="center", fontsize=6.6, color=COLORS["slate"])
    fig.tight_layout()
    return save_pdf(fig, output, "fig_02_strong_scaling.pdf", profile)


def figure_parallel_efficiency(campaign: Path, output: Path, profile: str) -> Path | None:
    rows = _performance_rows(campaign)
    if not rows:
        return None
    max_work = max(integer(row, "work_iterations") for row in rows)
    rows = [row for row in rows if integer(row, "work_iterations") == max_work]
    populations = sorted({integer(row, "population_size") for row in rows})
    dimensions = sorted({integer(row, "dimension") for row in rows})
    if not populations or not dimensions:
        return None
    fig, axes = plt.subplots(len(populations), 2, figsize=(7.2, 2.65 * len(populations)),
                             sharex=True, sharey=True, squeeze=False)
    for row_index, population in enumerate(populations):
        for col_index, prefix in enumerate(("ms_", "pp_")):
            ax = axes[row_index, col_index]
            subset = [row for row in rows if integer(row, "population_size") == population
                      and row.get("method", "").startswith(prefix)]
            for dim_index, dimension in enumerate(dimensions):
                values = sorted((row for row in subset if integer(row, "dimension") == dimension),
                                key=lambda row: integer(row, "workers"))
                if not values:
                    continue
                x = np.asarray([integer(row, "workers") for row in values], dtype=float)
                y = np.asarray([number(row, "parallel_efficiency") for row in values], dtype=float)
                ax.plot(x, y, marker="o", ms=3.8, lw=1.45,
                        color=PALETTE[dim_index % len(PALETTE)], label=f"D = {dimension}")
            ax.axhline(1.0, color=COLORS["slate"], linewidth=0.85, linestyle="--")
            ax.set_title(f"{'Master–slave' if col_index == 0 else 'Particle-parallel'} · N={population}")
            ax.grid(axis="y")
            if row_index == len(populations) - 1:
                ax.set_xlabel("Worker threads")
            if col_index == 0:
                ax.set_ylabel("Parallel efficiency (S / p)")
            ax.legend(frameon=False, loc="best")
    fig.suptitle(f"Parallel efficiency · high-cost workload{evidence_suffix(profile)}", y=1.005, weight="bold")
    fig.text(0.5, -0.005,
             "Efficiency is paired speedup divided by worker count. Values above one can arise from measurement variability or cache effects; they are not assumed a priori.",
             ha="center", fontsize=6.5, color=COLORS["slate"])
    fig.tight_layout()
    return save_pdf(fig, output, "fig_03_parallel_efficiency.pdf", profile)


def figure_runtime_decomposition(campaign: Path, output: Path, profile: str) -> Path | None:
    all_rows = read_csv(campaign / "campaign_runs.csv")
    scaling_rows = _performance_rows(campaign)
    if not all_rows or not scaling_rows:
        return None
    max_work = max(integer(row, "work_iterations") for row in scaling_rows)
    selected_studies = {row.get("study", "") for row in scaling_rows
                        if integer(row, "work_iterations") == max_work and row.get("study")}
    rows = [row for row in all_rows if row.get("study") in selected_studies
            and row.get("benchmark") == "sphere_work"]
    if not rows:
        return None
    population = max(integer(row, "population_size") for row in rows)
    dimension = max(integer(row, "dimension") for row in rows)
    rows = [row for row in rows if integer(row, "population_size") == population
            and integer(row, "dimension") == dimension]
    components = [
        ("initialization_seconds", "Initialization"),
        ("topology_setup_seconds", "Topology setup"),
        ("evaluation_wall_seconds", "Objective evaluation"),
        ("propagation_wall_seconds", "Propagation / social update"),
    ]
    rows_by_method: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        rows_by_method[row.get("method", "")].append(row)
    methods = ["serial_global"] + [f"{prefix}_global_w{worker}"
                                    for prefix in ("ms", "pp") for worker in (1, 2, 4, 8)]
    medians: dict[str, list[float]] = {}
    for method in methods:
        selected = rows_by_method.get(method, [])
        if not selected:
            continue
        elapsed = [number(row, "elapsed_seconds") for row in selected]
        components_by_run = [[max(0.0, number(row, key, 0.0)) for key, _ in components]
                             for row in selected]
        residual = [max(0.0, total - sum(parts)) for total, parts in zip(elapsed, components_by_run)]
        medians[method] = [float(np.median([parts[i] for parts in components_by_run]))
                           for i in range(len(components))]
        medians[method].append(float(np.median(residual)))
    if len(medians) < 3:
        return None
    fig, ax = plt.subplots(figsize=(7.4, 3.8))
    x = np.arange(len(medians))
    bottom = np.zeros(len(medians))
    labels = [name for _, name in components] + ["Residual / orchestration"]
    colors = [COLORS["slate"], COLORS["ochre"], COLORS["blue"], COLORS["plum"], "#CBD3DC"]
    for index, (label, color) in enumerate(zip(labels, colors)):
        heights = np.asarray([medians[method][index] for method in medians])
        ax.bar(x, heights, bottom=bottom, width=0.72, color=color, edgecolor="white",
               linewidth=0.4, label=label)
        bottom += heights
    method_labels = []
    for method in medians:
        if method == "serial_global":
            method_labels.append("Serial")
        elif method.startswith("ms_"):
            method_labels.append(f"MS · {method.rsplit('w', 1)[1]} workers")
        else:
            method_labels.append(f"PP · {method.rsplit('w', 1)[1]} workers")
    ax.set_xticks(x, method_labels, rotation=25, ha="right")
    ax.set_ylabel("Component-wise median wall time (s)")
    ax.set_title(f"Measured runtime components · N={population}, D={dimension}, work={max_work:,}{evidence_suffix(profile)}", weight="bold")
    ax.grid(axis="y")
    ax.legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.28))
    fig.tight_layout()
    return save_pdf(fig, output, "fig_04_runtime_components.pdf", profile)


def figure_quality_effects(
    campaign: Path, output: Path, *, stage_kind: str, filename: str, title: str,
    profile: str,
) -> Path | None:
    rows = read_csv(campaign / "pairwise_effects.csv")
    if stage_kind == "asynchronous":
        rows = [row for row in rows if "async" in row.get("stage", "").lower()]
    elif stage_kind == "islands":
        rows = [row for row in rows if "island" in row.get("stage", "").lower()]
    else:
        raise ValueError(f"Unknown stage_kind {stage_kind!r}")
    if not rows:
        return None
    rows.sort(key=lambda row: (row.get("stage", ""), row.get("method", "")))
    labels = [f"{row.get('stage', '').replace('_', ' ')} · {row.get('method', '').replace('_', ' ')}" for row in rows]
    medians = np.asarray([number(row, "median_function_family_log10_delta") for row in rows])
    lows = np.asarray([number(row, "hierarchical_bootstrap_95_low") for row in rows])
    highs = np.asarray([number(row, "hierarchical_bootstrap_95_high") for row in rows])
    valid = np.isfinite(medians) & np.isfinite(lows) & np.isfinite(highs)
    if not np.any(valid):
        return None
    rows = [row for row, keep in zip(rows, valid) if keep]
    labels = [label for label, keep in zip(labels, valid) if keep]
    medians, lows, highs = medians[valid], lows[valid], highs[valid]
    y = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(7.1, max(3.2, 0.34 * len(rows) + 1.2)))
    for index in range(len(rows)):
        ax.errorbar(medians[index], y[index],
                    xerr=[[max(0.0, medians[index] - lows[index])],
                          [max(0.0, highs[index] - medians[index])]],
                    fmt="o", ms=4.2, capsize=2.5, color=PALETTE[index % len(PALETTE)])
    ax.axvline(0.0, color=COLORS["slate"], linewidth=0.9, linestyle="--")
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel(r"Median function-family $\Delta\log_{10}(f_{best})$ vs. matched serial reference")
    ax.set_title(title + evidence_suffix(profile), weight="bold")
    ax.grid(axis="x")
    note = "Negative values favour the asynchronous/island variant; comparisons are at matched FE budget, not trajectory-equivalent speedups."
    if profile == "exploratory":
        note += " Pilot intervals are descriptive and do not establish confirmatory effects."
    fig.text(0.5, 0.015, note, ha="center", fontsize=6.4, color=COLORS["slate"])
    fig.tight_layout(rect=(0, 0.055, 1, 1))
    return save_pdf(fig, output, filename, profile)


def aggregate_topology_effects_by_family(
    rows: list[dict[str, str]], topology_order: list[str],
) -> dict[str, dict[str, float]]:
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
        key = (row.get("benchmark", ""), integer(row, "dimension"), integer(row, "seed"))
        ref = baseline.get(key, math.nan)
        value = number(row, "best_value")
        if not math.isfinite(ref) or not math.isfinite(value) or value < 0 or ref < 0:
            continue
        delta = math.log10(max(value, 1.0e-16)) - math.log10(max(ref, 1.0e-16))
        cells[topology][(key[0], key[1])].append(delta)
    family: dict[str, dict[str, float]] = {topology: {} for topology in topology_order}
    for topology, blocks in cells.items():
        grouped: dict[str, list[float]] = defaultdict(list)
        for (benchmark, _dimension), values in blocks.items():
            if values:
                grouped[benchmark].append(float(np.median(values)))
        for benchmark, dimension_medians in grouped.items():
            family[topology][benchmark] = float(np.median(dimension_medians))
    return family


def figure_topology_quality(campaign: Path, output: Path, profile: str) -> Path | None:
    candidates = sorted(campaign.glob("01_serial_topology_quality/*/runs.csv"))
    if not candidates:
        candidates = sorted(campaign.glob("*/*/runs.csv"))
    rows = read_csv(candidates[0]) if candidates else []
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
    ax.set_title("Particle-social topology quality effects" + evidence_suffix(profile), weight="bold")
    ax.grid(axis="y")
    foot = f"Each point is one benchmark family (n={len(families)}); paired by seed and dimension. Negative values favour the topology."
    if profile == "exploratory":
        foot += " Exploratory M2 pilot; not confirmatory."
    fig.text(0.5, 0.015, foot, ha="center", fontsize=6.5, color=COLORS["slate"])
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    return save_pdf(fig, output, "fig_08_current_topology_quality.pdf", profile)


def create_catalog(
    output: Path, figures: list[Path], missing: list[str], *, profile: str,
    campaign_dirs: dict[str, Path],
) -> Path:
    evidence = "exploratory pilot; no confirmatory claims" if profile == "exploratory" else "confirmatory inputs only"
    lines = [
        "# Publication figure catalogue", "",
        f"Evidence profile: **{profile}** ({evidence}).", "",
        "Figures are generated only from completed, measured campaign records.",
        "No missing result is synthesized or silently replaced by another evidence class.", "",
        "## Campaign inputs", "",
    ]
    lines.extend(f"- `{name}`: `{path}`" for name, path in campaign_dirs.items())
    lines.extend(["", "## Generated figures", ""])
    lines.extend(f"- `{path.name}`" for path in figures)
    lines.extend(["", "## Not generated", ""])
    lines.extend(f"- `{item}`" for item in missing)
    lines.extend([
        "", "## Interpretation", "",
        "- Synchronous equivalence is plotted only after the run-level exact verifier succeeds.",
        "- Speedup and efficiency use the measured synthetic `sphere_work` execution-cost instrument; they are hardware-specific.",
        "- Asynchronous and island modes are compared by quality at matched FE budgets, not as trajectory-preserving speedups.",
        "- Topology effects are paired against global-best by benchmark, dimension and seed.",
        "- External COCO/CEC figures and the separate `dt` sensitivity figures are generated by their dedicated analysis scripts.",
        "",
    ])
    target = output / "FIGURE_CATALOGUE.md"
    target.write_text("\n".join(lines), encoding="utf-8")
    return target


def _check_evidence_class(campaign_dir: Path, expected: str) -> None:
    manifest = read_manifest(campaign_dir)
    campaign = manifest.get("campaign", {}) if manifest else {}
    actual = campaign.get("evidence_class") if isinstance(campaign, dict) else None
    if actual != expected:
        raise ValueError(
            f"Campaign {campaign_dir} has evidence_class={actual!r}; expected {expected!r}. "
            "Re-run analysis with the declared campaign config before making figures."
        )


def build_figures(
    results_arg: Path,
    output: Path,
    *,
    profile: str = "confirmatory",
    synchronous_campaign_arg: Path | None = None,
    dynamics_campaign_arg: Path | None = None,
    topology_campaign_arg: Path | None = None,
) -> tuple[list[Path], list[str]]:
    if profile not in {"confirmatory", "exploratory"}:
        raise ValueError("profile must be confirmatory or exploratory")
    results = resolve_results_root(results_arg)
    if profile == "confirmatory":
        if any(value is not None for value in (synchronous_campaign_arg, dynamics_campaign_arg, topology_campaign_arg)):
            raise ValueError("Explicit campaign paths are reserved for --profile exploratory")
        synchronous = resolve_campaign_dir(results, None, "cpso_architecture_campaign_v1")
        dynamics = synchronous
        topology = resolve_campaign_dir(results, None, "cpso_topology_expansion_confirmatory_073")
        _check_evidence_class(synchronous, "confirmatory")
        _check_evidence_class(topology, "confirmatory")
    else:
        if any(value is None for value in (synchronous_campaign_arg, dynamics_campaign_arg, topology_campaign_arg)):
            raise ValueError(
                "Exploratory profile requires --synchronous-campaign, --dynamics-campaign and --topology-campaign"
            )
        synchronous = resolve_campaign_dir(results, synchronous_campaign_arg, "")
        dynamics = resolve_campaign_dir(results, dynamics_campaign_arg, "")
        topology = resolve_campaign_dir(results, topology_campaign_arg, "")
        for campaign_dir in (synchronous, dynamics, topology):
            _check_evidence_class(campaign_dir, "exploratory")

    configure_style()
    output.mkdir(parents=True, exist_ok=True)
    # Remove only the exact output set owned by this script to prevent stale
    # plots being mistaken for newly generated figures.
    expected_names = {
        "fig_01_synchronous_equivalence.pdf", "fig_02_strong_scaling.pdf",
        "fig_03_parallel_efficiency.pdf", "fig_04_runtime_components.pdf",
        "fig_05_asynchronous_quality.pdf", "fig_06_island_quality.pdf",
        "fig_08_current_topology_quality.pdf", "fig_parallel_speedup.pdf",
        "fig_parallel_efficiency.pdf", "fig_runtime_decomposition.pdf",
        "fig_asynchronous_quality_effects.pdf", "fig_island_quality_effects.pdf",
        "fig_topology_quality_effects.pdf", "fig_09_particle_topology_schematics.pdf",
    }
    for filename in expected_names:
        path = output / filename
        if path.exists():
            path.unlink()

    makers = [
        ("fig_01_synchronous_equivalence.pdf", lambda: figure_synchronous_equivalence(synchronous, output, profile)),
        ("fig_02_strong_scaling.pdf", lambda: figure_parallel_speedup(synchronous, output, profile)),
        ("fig_03_parallel_efficiency.pdf", lambda: figure_parallel_efficiency(synchronous, output, profile)),
        ("fig_04_runtime_components.pdf", lambda: figure_runtime_decomposition(synchronous, output, profile)),
        ("fig_05_asynchronous_quality.pdf", lambda: figure_quality_effects(
            dynamics, output, stage_kind="asynchronous", filename="fig_05_asynchronous_quality.pdf",
            title="Asynchronous CPSO quality effects", profile=profile)),
        ("fig_06_island_quality.pdf", lambda: figure_quality_effects(
            dynamics, output, stage_kind="islands", filename="fig_06_island_quality.pdf",
            title="Island-model CPSO quality effects", profile=profile)),
        ("fig_08_current_topology_quality.pdf", lambda: figure_topology_quality(topology, output, profile)),
    ]
    figures: list[Path] = []
    missing: list[str] = []
    for expected, maker in makers:
        try:
            made = maker()
        except Exception as exc:
            missing.append(f"{expected}: generation failed ({type(exc).__name__}: {exc})")
            continue
        if made is None:
            missing.append(f"{expected}: required source data missing or incomplete")
        else:
            figures.append(made)
    create_catalog(output, figures, missing, profile=profile, campaign_dirs={
        "synchronous": synchronous, "dynamics": dynamics, "topology": topology,
    })
    return figures, missing


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument("--output", type=Path, default=Path("paper/figures/internal_pilots"))
    parser.add_argument("--profile", choices=("confirmatory", "exploratory"), default="confirmatory")
    parser.add_argument("--synchronous-campaign", type=Path, default=None)
    parser.add_argument("--dynamics-campaign", type=Path, default=None)
    parser.add_argument("--topology-campaign", type=Path, default=None)
    args = parser.parse_args()
    figures, missing = build_figures(
        args.results_root, args.output, profile=args.profile,
        synchronous_campaign_arg=args.synchronous_campaign,
        dynamics_campaign_arg=args.dynamics_campaign,
        topology_campaign_arg=args.topology_campaign,
    )
    print(f"Generated {len(figures)} measured-result figure(s) in {args.output}")
    for figure in figures:
        print(f"  {figure}")
    if missing:
        print("Not generated:")
        for item in missing:
            print(f"  - {item}")
    # A partial figure set is useful diagnostically, but must not look like a
    # successful publication build to shell scripts or CI.
    return 0 if figures and not missing else 2


if __name__ == "__main__":
    raise SystemExit(main())
