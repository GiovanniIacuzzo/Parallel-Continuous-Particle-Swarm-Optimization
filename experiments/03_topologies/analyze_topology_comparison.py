from __future__ import annotations

import argparse
import csv
from pathlib import Path
import json
import math
import random
import statistics

import numpy as np

TOPOLOGY_ORDER = ("global", "ring", "grid_von_neumann")
LOG_FLOOR = 1.0e-16



def _boxplot_with_labels(ax, values, labels, **kwargs):
    """Use the boxplot label keyword supported by the installed Matplotlib version."""
    import matplotlib

    # The keyword was renamed in Matplotlib 3.9; it is accepted through
    # **kwargs in some releases, so signature introspection is insufficient.
    version = tuple(int(part.split(".")[0]) for part in matplotlib.__version__.split(".")[:2])
    label_keyword = "tick_labels" if version >= (3, 9) else "labels"
    return ax.boxplot(values, **{label_keyword: labels}, **kwargs)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def quantile(values: list[float], q: float) -> float:
    return float(np.quantile(np.asarray(values, dtype=float), q))


def bootstrap_median_interval(values: list[float], seed: int, samples: int = 10000) -> tuple[float, float]:
    if len(values) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    data = np.asarray(values, dtype=float)
    draws = rng.choice(data, size=(samples, data.size), replace=True)
    medians = np.median(draws, axis=1)
    return float(np.quantile(medians, 0.025)), float(np.quantile(medians, 0.975))


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)



def make_topology_schematic(figure_dir: Path, plt) -> None:
    from matplotlib.patches import FancyArrowPatch

    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.8), constrained_layout=True)

    # Global-best: complete information graph among a representative set of particles.
    ax = axes[0]
    n = 8
    angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    xy = np.column_stack((np.cos(angles), np.sin(angles)))
    for i in range(n):
        for j in range(i + 1, n):
            ax.plot([xy[i, 0], xy[j, 0]], [xy[i, 1], xy[j, 1]], alpha=0.18, linewidth=0.8)
    ax.scatter(xy[:, 0], xy[:, 1], s=75, zorder=3)
    for i, (x, y) in enumerate(xy):
        ax.text(1.16 * x, 1.16 * y, str(i), ha="center", va="center", fontsize=8)
    ax.set_title("Global-best\ncomplete neighborhood")
    ax.set_aspect("equal")
    ax.axis("off")

    # Periodic radius-one ring.
    ax = axes[1]
    angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    xy = np.column_stack((np.cos(angles), np.sin(angles)))
    for i in range(n):
        j = (i + 1) % n
        ax.plot([xy[i, 0], xy[j, 0]], [xy[i, 1], xy[j, 1]], linewidth=1.5)
    ax.scatter(xy[:, 0], xy[:, 1], s=75, zorder=3)
    for i, (x, y) in enumerate(xy):
        ax.text(1.16 * x, 1.16 * y, str(i), ha="center", va="center", fontsize=8)
    ax.set_title("Periodic ring\nradius one + self")
    ax.set_aspect("equal")
    ax.axis("off")

    # 3x3 toroidal von-Neumann grid. Dashed outer arcs denote periodic wrap.
    ax = axes[2]
    grid_xy = {(r, c): (float(c), -float(r)) for r in range(3) for c in range(3)}
    for r in range(3):
        for c in range(3):
            if c < 2:
                x1, y1 = grid_xy[(r, c)]
                x2, y2 = grid_xy[(r, c + 1)]
                ax.plot([x1, x2], [y1, y2], linewidth=1.4)
            if r < 2:
                x1, y1 = grid_xy[(r, c)]
                x2, y2 = grid_xy[(r + 1, c)]
                ax.plot([x1, x2], [y1, y2], linewidth=1.4)
    points = np.asarray([grid_xy[(r, c)] for r in range(3) for c in range(3)])
    ax.scatter(points[:, 0], points[:, 1], s=75, zorder=3)
    for r in range(3):
        for c in range(3):
            x, y = grid_xy[(r, c)]
            ax.text(x, y, str(r * 3 + c), ha="center", va="center", fontsize=7, zorder=4)
    wrap_style = "arc3,rad=0.32"
    ax.add_patch(FancyArrowPatch((-0.08, 0.0), (-0.08, -2.0),
                                 connectionstyle=wrap_style, arrowstyle="<->",
                                 mutation_scale=9, linestyle="--", linewidth=1.0,
                                 clip_on=False))
    ax.add_patch(FancyArrowPatch((2.08, 0.0), (2.08, -2.0),
                                 connectionstyle=wrap_style, arrowstyle="<->",
                                 mutation_scale=9, linestyle="--", linewidth=1.0,
                                 clip_on=False))
    ax.add_patch(FancyArrowPatch((0.0, 0.08), (2.0, 0.08),
                                 connectionstyle=wrap_style, arrowstyle="<->",
                                 mutation_scale=9, linestyle="--", linewidth=1.0,
                                 clip_on=False))
    ax.add_patch(FancyArrowPatch((0.0, -2.08), (2.0, -2.08),
                                 connectionstyle=wrap_style, arrowstyle="<->",
                                 mutation_scale=9, linestyle="--", linewidth=1.0,
                                 clip_on=False))
    ax.set_title("Toroidal grid\nvon-Neumann + self")
    ax.set_aspect("equal")
    ax.set_xlim(-0.65, 2.65)
    ax.set_ylim(-2.65, 0.65)
    ax.axis("off")

    fig.suptitle("Particle-social communication topologies", fontsize=12)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(figure_dir / f"topology_schematic.{ext}", dpi=220 if ext == "png" else None)
    plt.close(fig)

def make_analysis(input_dir: Path, no_plots: bool = False) -> Path:
    runs = read_csv(input_dir / "runs.csv")
    convergence = read_csv(input_dir / "convergence.csv")
    if not runs or not convergence:
        raise ValueError("runs.csv and convergence.csv must both contain observations")

    by_group: dict[tuple[str, int, str], list[dict[str, str]]] = {}
    paired: dict[tuple[str, int, int], dict[str, float]] = {}
    for row in runs:
        key = (row["benchmark"], int(row["dimension"]), row["topology"])
        by_group.setdefault(key, []).append(row)
        pair_key = (row["benchmark"], int(row["dimension"]), int(row["seed"]))
        topology_map = paired.setdefault(pair_key, {})
        if row["topology"] in topology_map:
            raise ValueError(f"Duplicate run for {pair_key} / {row['topology']}")
        topology_map[row["topology"]] = float(row["best_value"])

    expected_topologies = set(TOPOLOGY_ORDER)
    for pair_key, topology_map in paired.items():
        if set(topology_map) != expected_topologies:
            raise ValueError(
                f"Incomplete paired topology set for {pair_key}: found {sorted(topology_map)}"
            )
    summary_rows: list[dict] = []
    for (benchmark, dimension, topology), rows in sorted(by_group.items()):
        values = [float(row["best_value"]) for row in rows]
        median = statistics.median(values)
        mad = statistics.median(abs(value - median) for value in values)
        summary_rows.append({
            "benchmark": benchmark,
            "dimension": dimension,
            "topology": topology,
            "n_seeds": len(values),
            "best_median": format(median, ".17g"),
            "best_q25": format(quantile(values, 0.25), ".17g"),
            "best_q75": format(quantile(values, 0.75), ".17g"),
            "best_mad": format(mad, ".17g"),
            "success_rate": format(sum(int(row["success"]) for row in rows) / len(rows), ".17g"),
            "median_elapsed_seconds": format(statistics.median(float(row["elapsed_seconds"]) for row in rows), ".17g"),
            "median_topology_setup_seconds": format(statistics.median(float(row["topology_setup_seconds"]) for row in rows), ".17g"),
            "median_social_selection_seconds": format(statistics.median(float(row["social_selection_seconds"]) for row in rows), ".17g"),
        })
    write_csv(input_dir / "topology_summary.csv", list(summary_rows[0].keys()), summary_rows)

    effect_rows: list[dict] = []
    comparison_pairs = (("ring", "global"), ("grid_von_neumann", "global"), ("grid_von_neumann", "ring"))
    for benchmark, dimension in sorted({(b, d) for b, d, _ in by_group}):
        seed_map = {
            seed: topo_values
            for (b, d, seed), topo_values in paired.items()
            if b == benchmark and d == dimension
        }
        for first, second in comparison_pairs:
            differences = []
            for seed in sorted(seed_map):
                values = seed_map[seed]
                if first not in values or second not in values:
                    continue
                # Paired difference in log10 objective value; the fixed floor
                # keeps exact-zero solutions finite and is stated in metadata.
                differences.append(
                    math.log10(max(values[first], LOG_FLOOR))
                    - math.log10(max(values[second], LOG_FLOOR))
                )
            ci_low, ci_high = bootstrap_median_interval(
                differences, seed=20261104 + dimension + sum(ord(c) for c in benchmark + first + second)
            )
            effect_rows.append({
                "benchmark": benchmark,
                "dimension": dimension,
                "comparison": f"{first}_minus_{second}",
                "n_paired_seeds": len(differences),
                "median_delta_log10_best": format(statistics.median(differences), ".17g") if differences else "",
                "bootstrap_ci95_low": format(ci_low, ".17g") if math.isfinite(ci_low) else "",
                "bootstrap_ci95_high": format(ci_high, ".17g") if math.isfinite(ci_high) else "",
                "interpretation": "negative favors the first topology; descriptive paired bootstrap, not a hypothesis test",
            })
    if effect_rows:
        write_csv(input_dir / "paired_topology_effects.csv", list(effect_rows[0].keys()), effect_rows)

    convergence_groups: dict[tuple[str, int, str, int], list[float]] = {}
    max_iteration: dict[tuple[str, int], int] = {}
    for row in convergence:
        benchmark = row["benchmark"]
        dimension = int(row["dimension"])
        topology = row["topology"]
        iteration = int(row["iteration"])
        key = (benchmark, dimension, topology, iteration)
        convergence_groups.setdefault(key, []).append(float(row["best_value"]))
        max_iteration[(benchmark, dimension)] = max(max_iteration.get((benchmark, dimension), 0), iteration)

    convergence_summary: list[dict] = []
    for (benchmark, dimension, topology, iteration), values in sorted(convergence_groups.items()):
        convergence_summary.append({
            "benchmark": benchmark,
            "dimension": dimension,
            "topology": topology,
            "iteration": iteration,
            "n_seeds": len(values),
            "median_best": format(statistics.median(values), ".17g"),
            "q25_best": format(quantile(values, 0.25), ".17g"),
            "q75_best": format(quantile(values, 0.75), ".17g"),
        })
    write_csv(input_dir / "convergence_summary.csv", list(convergence_summary[0].keys()), convergence_summary)

    if not no_plots:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        figure_dir = input_dir / "figures"
        figure_dir.mkdir(exist_ok=True)
        combos = sorted(max_iteration)
        for benchmark, dimension in combos:
            fig, ax = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
            for topology in TOPOLOGY_ORDER:
                rows = [
                    row for row in convergence_summary
                    if row["benchmark"] == benchmark
                    and int(row["dimension"]) == dimension
                    and row["topology"] == topology
                ]
                if not rows:
                    continue
                rows.sort(key=lambda row: int(row["iteration"]))
                x = np.asarray([int(row["iteration"]) for row in rows])
                median = np.asarray([max(float(row["median_best"]), LOG_FLOOR) for row in rows])
                low = np.asarray([max(float(row["q25_best"]), LOG_FLOOR) for row in rows])
                high = np.asarray([max(float(row["q75_best"]), LOG_FLOOR) for row in rows])
                ax.plot(x, median, label=topology)
                ax.fill_between(x, low, high, alpha=0.16)
            ax.set_yscale("log")
            ax.set_xlabel("Synchronous propagation/evaluation rounds")
            ax.set_ylabel("Median best-so-far objective (log scale)")
            ax.set_title(f"CPSO social-topology comparison — {benchmark}, D={dimension}")
            ax.grid(True, which="both", alpha=0.25)
            ax.legend(title="Topology")
            stem = f"{benchmark}_d{dimension}_convergence"
            for ext in ("png", "pdf", "svg"):
                fig.savefig(figure_dir / f"{stem}.{ext}", dpi=220 if ext == "png" else None)
            plt.close(fig)

            final_by_topology = {
                topology: [
                    math.log10(max(float(row["best_value"]), LOG_FLOOR))
                    for row in runs
                    if row["benchmark"] == benchmark
                    and int(row["dimension"]) == dimension
                    and row["topology"] == topology
                ]
                for topology in TOPOLOGY_ORDER
            }
            fig, ax = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
            labels = [topology for topology in TOPOLOGY_ORDER if final_by_topology[topology]]
            data = [final_by_topology[topology] for topology in labels]
            _boxplot_with_labels(ax, data, labels, showmeans=True)
            ax.set_ylabel("log10(max(final objective, 1e-16))")
            ax.set_title(f"Final objective distribution — {benchmark}, D={dimension}")
            ax.grid(True, axis="y", alpha=0.25)
            stem = f"{benchmark}_d{dimension}_final_distribution"
            for ext in ("png", "pdf", "svg"):
                fig.savefig(figure_dir / f"{stem}.{ext}", dpi=220 if ext == "png" else None)
            plt.close(fig)

        make_topology_schematic(figure_dir, plt)

    report_lines = [
        "# Topology comparison analysis",
        "",
        f"Runs analyzed: {len(runs)}",
        f"Convergence records: {len(convergence)}",
        f"Log floor for plots/effects: {LOG_FLOOR:g}",
        "",
        "The per-topology summaries report the median, interquartile range, MAD, and success rate.",
        "Paired effects are median differences in log10(max(f_best, 1e-16)); negative values favor the first topology.",
        "The bootstrap intervals are descriptive uncertainty intervals, not p-values or formal significance tests.",
        "All topology variants use the same FE budget and are paired by seed. Topology changes the social attractor and therefore defines a different search algorithm.",
        "",
        "Do not treat elapsed-time differences from this serial topology study as parallel speedup.",
    ]
    if not no_plots:
        report_lines.append("Figures are saved in `figures/` in PNG, PDF, and SVG formats.")
    (input_dir / "TOPOLOGY_ANALYSIS.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    return input_dir


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze paired CPSO particle-topology experiments")
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    output = make_analysis(args.input_dir.expanduser().resolve(), no_plots=args.no_plots)
    print(f"Topology analysis written to: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
