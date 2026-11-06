#!/usr/bin/env python3
"""Create summary plots and a concise report from a scaling-sweep directory."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import statistics


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def median(values: list[float]) -> float:
    return statistics.median(values)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path)
    parser.add_argument("--format", choices=("pdf", "svg", "both"), default="pdf")
    args = parser.parse_args()

    run_dir = args.run_directory.resolve()
    raw_path = run_dir / "scaling_runs.csv"
    summary_path = run_dir / "scaling_summary.csv"
    if not raw_path.is_file() or not summary_path.is_file():
        parser.error("run_directory must contain scaling_runs.csv and scaling_summary.csv")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit("Matplotlib is required: python -m pip install matplotlib") from exc

    raw_rows = read_csv(raw_path)
    summary_rows = read_csv(summary_path)
    figures_dir = run_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    scenarios = sorted({(int(row["dimension"]), int(row["population_size"])) for row in summary_rows})
    report_lines = [
        "# Master--slave scaling analysis",
        "",
        f"Source data: `{raw_path.name}` and `{summary_path.name}`.",
        "All plotted speedups are paired against the serial run with the same seed and configuration.",
        "The synthetic `sphere_work` objective is a controlled performance instrument, not a standard benchmark.",
        "",
    ]

    for dimension, population in scenarios:
        subset = [
            row for row in summary_rows
            if int(row["dimension"]) == dimension and int(row["population_size"]) == population
        ]
        work_levels = sorted({int(row["work_iterations"]) for row in subset})
        worker_levels = sorted({int(row["workers"]) for row in subset})
        slug = f"d{dimension}_n{population}"

        fig, ax = plt.subplots(figsize=(7.2, 4.6), constrained_layout=True)
        for work in work_levels:
            work_rows = sorted(
                [row for row in subset if int(row["work_iterations"]) == work],
                key=lambda row: int(row["workers"]),
            )
            x = [int(row["workers"]) for row in work_rows]
            y = [float(row["median_total_speedup"]) for row in work_rows]
            y_low = [float(row["total_speedup_bootstrap_ci95_low"]) for row in work_rows]
            y_high = [float(row["total_speedup_bootstrap_ci95_high"]) for row in work_rows]
            yerr = [[max(0.0, value - low) for value, low in zip(y, y_low)],
                    [max(0.0, high - value) for value, high in zip(y, y_high)]]
            ax.errorbar(x, y, yerr=yerr, marker="o", capsize=3, label=f"work={work}")
        ax.axhline(1.0, linestyle="--", linewidth=1.0, label="serial break-even")
        ax.set_xlabel("Worker threads")
        ax.set_ylabel("Paired total-time speedup")
        ax.set_title(f"Master--slave speedup (D={dimension}, N={population})")
        ax.set_xticks(worker_levels)
        ax.grid(True, alpha=0.25)
        ax.legend(title="Work iterations", frameon=False)
        for suffix in (("pdf", "svg") if args.format == "both" else (args.format,)):
            fig.savefig(figures_dir / f"speedup_{slug}.{suffix}")
        fig.savefig(figures_dir / f"speedup_{slug}.png", dpi=300)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(7.2, 4.6), constrained_layout=True)
        for work in work_levels:
            work_rows = sorted(
                [row for row in subset if int(row["work_iterations"]) == work],
                key=lambda row: int(row["workers"]),
            )
            x = [int(row["workers"]) for row in work_rows]
            y = [float(row["median_parallel_efficiency"]) for row in work_rows]
            y_low = [float(row["total_speedup_bootstrap_ci95_low"]) / int(row["workers"])
                     for row in work_rows]
            y_high = [float(row["total_speedup_bootstrap_ci95_high"]) / int(row["workers"])
                      for row in work_rows]
            yerr = [[max(0.0, value - low) for value, low in zip(y, y_low)],
                    [max(0.0, high - value) for value, high in zip(y, y_high)]]
            ax.errorbar(x, y, yerr=yerr, marker="o", capsize=3, label=f"work={work}")
        ax.axhline(1.0, linestyle="--", linewidth=1.0, label="ideal efficiency")
        ax.set_xlabel("Worker threads")
        ax.set_ylabel("Parallel efficiency (speedup / workers)")
        ax.set_title(f"Master--slave efficiency (D={dimension}, N={population})")
        ax.set_xticks(worker_levels)
        ax.grid(True, alpha=0.25)
        ax.legend(title="Work iterations", frameon=False)
        for suffix in (("pdf", "svg") if args.format == "both" else (args.format,)):
            fig.savefig(figures_dir / f"efficiency_{slug}.{suffix}")
        fig.savefig(figures_dir / f"efficiency_{slug}.png", dpi=300)
        plt.close(fig)

        # Use raw paired timings to show actual end-to-end wall time, not only ratios.
        raw_scenario = [
            row for row in raw_rows
            if int(row["dimension"]) == dimension and int(row["population_size"]) == population
        ]
        mode_groups: dict[tuple[str, int, int], list[float]] = {}
        for row in raw_scenario:
            key = (row["mode"], int(row["workers"]), int(row["work_iterations"]))
            mode_groups.setdefault(key, []).append(float(row["elapsed_seconds"]))
        fig, ax = plt.subplots(figsize=(7.2, 4.6), constrained_layout=True)
        for mode, worker in [("serial", 1)] + [("master_slave", w) for w in worker_levels]:
            x_values: list[int] = []
            y_values: list[float] = []
            lower_errors: list[float] = []
            upper_errors: list[float] = []
            for work in work_levels:
                values = mode_groups.get((mode, worker, work), [])
                if values:
                    center = median(values)
                    x_values.append(work)
                    y_values.append(center)
                    lower_errors.append(max(0.0, center - sorted(values)[max(0, (len(values) - 1) // 4)]))
                    upper_errors.append(max(0.0, sorted(values)[min(len(values) - 1, (3 * (len(values) - 1)) // 4)] - center))
            label = "serial" if mode == "serial" else (
                f"master--slave ({worker} worker)" if worker == 1 else f"master--slave ({worker} workers)"
            )
            ax.errorbar(x_values, y_values, yerr=[lower_errors, upper_errors],
                        marker="o", capsize=3, label=label)
        ax.set_xlabel("Synthetic arithmetic work iterations per evaluation")
        ax.set_ylabel("Median end-to-end wall time (s)")
        ax.set_title(f"End-to-end runtime (D={dimension}, N={population})")
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False)
        for suffix in (("pdf", "svg") if args.format == "both" else (args.format,)):
            fig.savefig(figures_dir / f"runtime_{slug}.{suffix}")
        fig.savefig(figures_dir / f"runtime_{slug}.png", dpi=300)
        plt.close(fig)

        report_lines.extend([
            f"## Dimension {dimension}, population {population}",
            "",
            "| Work iterations | Workers | Median speedup | Bootstrap 95% interval | Median efficiency | Median serial (s) | Median parallel (s) |",
            "|---:|---:|---:|---:|---:|---:|---:|",
        ])
        for row in sorted(subset, key=lambda item: (int(item["work_iterations"]), int(item["workers"]))):
            report_lines.append(
                f"| {int(row['work_iterations'])} | {int(row['workers'])} | "
                f"{float(row['median_total_speedup']):.4f}x | "
                f"[{float(row['total_speedup_bootstrap_ci95_low']):.4f}, "
                f"{float(row['total_speedup_bootstrap_ci95_high']):.4f}] | "
                f"{float(row['median_parallel_efficiency']):.4f} | "
                f"{float(row['median_serial_seconds']):.6f} | "
                f"{float(row['median_parallel_seconds']):.6f} |"
            )
        report_lines.append("")

    report_lines.extend([
        "## Interpretation cautions",
        "",
        "- A speedup greater than one indicates lower end-to-end wall time for that tested configuration only.",
        "- The worker-pool startup and shutdown costs are included in master--slave end-to-end time.",
        "- The synthetic workload is useful for finding a controlled break-even region but does not establish performance on COCO/CEC objectives.",
        "- Results are specific to the hardware, operating system, compiler, build flags, and system load recorded in `scaling_metadata.json`.",
        "- With few seeds, bootstrap intervals are exploratory and should not be interpreted as strong inferential evidence.",
        "",
    ])
    (run_dir / "SCALING_ANALYSIS.md").write_text("\n".join(report_lines), encoding="utf-8")
    print(f"Figures: {figures_dir}")
    print(f"Report: {run_dir / 'SCALING_ANALYSIS.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
