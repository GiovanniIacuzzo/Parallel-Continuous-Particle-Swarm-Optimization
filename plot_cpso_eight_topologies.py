#!/usr/bin/env python3
"""Studio II: eight communication topologies, native CPSO synchronous scaling.

Reads the *existing* publication pipeline output:
  results/publication_v2/paper/02_parallel/selection/analysis/scaling_summary.csv

The summary format is: one row per benchmark/dimension/method/metric, with
'metric', 'estimate', 'lo', 'hi', 'topology', 'architecture', and 'workers'.
It is NOT the confirmation campaign schema and must not be conflated with it.

Writes two vector PDFs. No changes to the repository or run results.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

TOPOS = (
    ("global", "Global"),
    ("ring", "Ring"),
    ("grid_von_neumann", "Von Neumann"),
    ("wheel", "Wheel"),
    ("random_k4", "Random (k=4)"),
    ("random_k8", "Random (k=8)"),
    ("small_world_k4", "Small-world (k=4)"),
    ("scale_free_k4", "Scale-free (k=4)"),
)
WORKERS = (1, 2, 4, 8)
DIMENSIONS = (30, 100)
PALETTE = {30: "#225A81", 100: "#BD5937"}
MARKER = {30: "o", 100: "s"}
LINE = {30: "-", 100: "-"}
REQUIRED = {
    "benchmark", "dimension", "metric", "estimate", "lo", "hi",
    "workers", "architecture", "topology", "seed_clusters", "ratio_kind",
}


def setup_style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9.2,
        "axes.labelsize": 9.5,
        "axes.titlesize": 9.4,
        "axes.titleweight": "normal",
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": .75,
        "xtick.major.width": .75,
        "ytick.major.width": .75,
        "lines.linewidth": 1.6,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.transparent": False,
    })


def load_and_validate(csv_path: Path, benchmark: str, architecture: str, reference: str):
    if not csv_path.is_file():
        raise SystemExit(
            f"CSV non trovato: {csv_path}\n"
            "Usa 02_parallel/selection/analysis/scaling_summary.csv, "
            "NON confirmation/analysis/crossover_scaling.csv."
        )
    df = pd.read_csv(csv_path)
    missing = sorted(REQUIRED.difference(df.columns))
    if missing:
        raise SystemExit(f"Schema inatteso in {csv_path.name}: colonne mancanti {missing}")

    metric = "strong_speedup" if reference == "strong" else "serial_speedup"
    subset = df.loc[
        df["benchmark"].eq(benchmark)
        & df["architecture"].eq(architecture)
        & df["metric"].eq(metric)
    ].copy()
    if subset.empty:
        raise SystemExit(f"Nessuna riga per benchmark={benchmark}, architecture={architecture}, metric={metric}")
    if not subset["ratio_kind"].str.contains("synchronous", case=False, na=False).all():
        raise SystemExit("Il CSV contiene confronti non sincroni: speedup non attribuibile a traiettorie equivalenti.")

    for col in ("dimension", "workers", "seed_clusters", "estimate", "lo", "hi"):
        subset[col] = pd.to_numeric(subset[col], errors="raise")
    if not np.isfinite(subset[["dimension", "workers", "seed_clusters", "estimate", "lo", "hi"]]).all().all():
        raise SystemExit("Valori non finiti nel riepilogo: impossibile tracciare dati affidabili.")
    if (subset[["estimate", "lo", "hi"]] <= 0).any().any():
        raise SystemExit("Tempi/rapporti non positivi nel CSV.")
    if ((subset["lo"] > subset["estimate"]) | (subset["estimate"] > subset["hi"])).any():
        raise SystemExit("Intervalli di confidenza incoerenti nel CSV.")
    if (subset["seed_clusters"] < 2).any():
        raise SystemExit("Numero insufficiente di semi per gli intervalli riportati.")

    expected = {(t, d, p) for t, _ in TOPOS for d in DIMENSIONS for p in WORKERS}
    provided = set(zip(subset["topology"], subset["dimension"], subset["workers"]))
    dup = subset.duplicated(["topology", "dimension", "workers"], keep=False)
    if dup.any():
        raise SystemExit("Righe duplicate nello scaling summary per (topology, dimension, workers).")
    missing_cases = sorted(expected - provided)
    if missing_cases:
        raise SystemExit(
            "Campagna di selezione incompleta per la figura a 8 topologie. "
            "Primi casi mancanti: " + repr(missing_cases[:12])
        )
    # Exclude unexpected extra dimensions/topologies; never silently replace or relabel cases.
    subset = subset.loc[
        subset["topology"].isin([x for x, _ in TOPOS])
        & subset["dimension"].isin(DIMENSIONS)
        & subset["workers"].isin(WORKERS)
    ].copy()
    return subset, metric


def plot_grid(df: pd.DataFrame, metric: str, output: Path, columns: int, ci: bool):
    rows = math.ceil(len(TOPOS) / columns)
    # Large, full-width figures: readable after embedding at ~18 cm width.
    width = 7.3 if columns == 2 else 9.25
    height = 7.55 if columns == 2 else 4.95
    fig, axes = plt.subplots(rows, columns, figsize=(width, height), sharex=True, sharey=True)
    axes = np.asarray(axes).reshape(rows, columns)
    is_eff = metric == "efficiency"

    if is_eff:
        ylim = (0, max(1.07, float((df["hi"] / df["workers"]).max()) * 1.08))
    else:
        ylim = (0.75, max(8.35, float(df["hi"].max()) * 1.08))

    for idx, (topo, label) in enumerate(TOPOS):
        r, c = divmod(idx, columns)
        ax = axes[r, c]
        for d in DIMENSIONS:
            g = df.loc[df["topology"].eq(topo) & df["dimension"].eq(d)].sort_values("workers")
            x = g["workers"].to_numpy(dtype=float)
            scale = x if is_eff else np.ones_like(x)
            y = g["estimate"].to_numpy(dtype=float) / scale
            low = g["lo"].to_numpy(dtype=float) / scale
            high = g["hi"].to_numpy(dtype=float) / scale
            color = PALETTE[d]
            ax.plot(x, y, LINE[d], marker=MARKER[d], markersize=3.5,
                    markerfacecolor="white", markeredgewidth=1.15,
                    color=color, linewidth=1.55, label=f"D={d}", zorder=3)
            if ci:
                # Whiskers avoid nearly invisible, overlapping transparency bands.
                ax.errorbar(x, y, yerr=np.stack([y-low, high-y]),
                            fmt="none", ecolor=color, elinewidth=.75,
                            capsize=1.7, capthick=.75, alpha=.72, zorder=2)

        # The ideal reference is plotted in every panel on the same scale.
        # For strong/serial speedup: S_ideal(p) = p. For efficiency: E_ideal(p) = 1.
        x_ideal = np.asarray(WORKERS, dtype=float)
        y_ideal = np.ones_like(x_ideal) if is_eff else x_ideal
        ax.plot(x_ideal, y_ideal, color="#888888", lw=1.0,
                linestyle=(0, (4, 3)), zorder=1)
        ax.set_title(f"({chr(97+idx)})  {label}", pad=7, fontsize=9.5)
        ax.set_xticks(WORKERS)
        ax.set_xlim(.68, 8.30)
        ax.set_ylim(*ylim)
        ax.grid(axis="y", color="#A9ADB3", alpha=.25, lw=.55)
        ax.set_axisbelow(True)
        ax.tick_params(axis="both", pad=2)
        if c == 0:
            ax.set_ylabel("Efficiency" if is_eff else "Strong speedup" if metric == "strong_speedup" else "Speedup vs serial")
        if r == rows-1:
            ax.set_xlabel("Workers")

    for ax in axes.flat[len(TOPOS):]:
        ax.set_visible(False)
    legend = [Line2D([0], [0], color=PALETTE[d], marker=MARKER[d],
                     markerfacecolor="white", linestyle=LINE[d], lw=1.5,
                     label=f"D = {d}") for d in DIMENSIONS]
    legend.append(Line2D([0], [0], color="#888888", lw=1.0,
                         linestyle=(0, (4, 3)), label="Ideal"))
    fig.legend(handles=legend, loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(.5, .012), borderaxespad=0, fontsize=9.3)
    if columns == 2:
        fig.subplots_adjust(left=.115, right=.985, top=.978, bottom=.096,
                            wspace=.175, hspace=.36)
    else:
        fig.subplots_adjust(left=.068, right=.993, top=.960, bottom=.158,
                            wspace=.285, hspace=.38)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, format="pdf", bbox_inches="tight", pad_inches=.07)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=Path(
        "results/publication_v2/paper/02_parallel/selection/analysis/scaling_summary.csv"))
    parser.add_argument("--outdir", type=Path, default=Path("figures_studio2_topologies"))
    parser.add_argument("--benchmark", default="inverse_diffusion",
                        help="Default predeclared high-cost objective: inverse_diffusion")
    parser.add_argument("--architecture", default="particle_parallel", choices=("particle_parallel", "master_slave"))
    parser.add_argument("--reference", default="strong", choices=("strong", "serial"),
                        help="strong: same backend p=1; serial: matched serial trajectory")
    parser.add_argument("--columns", type=int, default=2, choices=(2, 4),
                        help="2 columns (print legibility) or 4 columns (compact 2x4)")
    parser.add_argument("--no-ci", action="store_true", help="Hide confidence-interval whiskers")
    args = parser.parse_args()

    setup_style()
    df, metric = load_and_validate(args.csv, args.benchmark, args.architecture, args.reference)
    prefix = f"fig_8topologies_{args.benchmark}_{args.architecture}_{args.reference}"
    # The source CSV stores strong_speedup/serial_speedup, NOT a metric named
    # "efficiency". Both figures use these same rows; efficiency is S(p)/p.
    for name in ("speedup", "efficiency"):
        output = args.outdir / f"{prefix}_{name}.pdf"
        plot_grid(df, "efficiency" if name == "efficiency" else metric, output, args.columns, ci=not args.no_ci)
        print(f"Generated: {output}")
    print("Data source: SELECTION, exploratory topology study; not independent hold-out.")
    print("Strong speedup compares each topology/backend against the same backend with 1 worker.")
    print("Efficiency is exactly speedup / workers; same ratio/reference as the speedup figure.")


if __name__ == "__main__":
    main()
