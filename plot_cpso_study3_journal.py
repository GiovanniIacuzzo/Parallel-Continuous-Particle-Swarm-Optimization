#!/usr/bin/env python3
"""Three publication-oriented figures from existing Study III native checkpoints.

Usage:
  python plot_cpso_study3_journal.py \
      --source results/study3_pp_paper/03_comparison/native \
      --dimension 20 --budget 32768

No simulations are run, no repository modules are imported, and no source data
are modified. The script uses ONLY histories of the chosen completed-run FE
budget to draw convergence (never connects independent shorter-budget runs).
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd

PRIMARY = "CPSO selected framework"
CONTROL = "CPSO serial control"
BASELINES = (
    "classic_gbest", "inertia_gbest", "constriction_gbest",
    "inertia_lbest_ring", "clpso", "slpso", "spso2011",
    "fips_ring", "cso", "dms_pso",
)
METHODS = (PRIMARY, *BASELINES)
ALL_METHODS = (PRIMARY, CONTROL, *BASELINES)
FUNCTIONS = ("sphere", "rastrigin", "rosenbrock", "ackley", "griewank", "salomon")
F_LABEL = {f: f.capitalize() for f in FUNCTIONS}
F_LABEL["rosenbrock"] = "Rosenbrock"
M_LABEL = {
    PRIMARY: "CPSO selected",
    CONTROL: "CPSO serial",
    "classic_gbest": "Classic gbest",
    "inertia_gbest": "Inertia gbest",
    "constriction_gbest": "Constriction",
    "inertia_lbest_ring": "Inertia ring",
    "clpso": "CLPSO",
    "slpso": "SL-PSO",
    "spso2011": "SPSO-2011",
    "fips_ring": "FIPS ring",
    "cso": "CSO",
    "dms_pso": "DMS-PSO",
}

# Subdued, legible on white and in small journal two-column figures.
PALETTE = {
    PRIMARY: "#B33E44",      # primary method, burgundy
    "classic_gbest": "#507D99",
    "inertia_gbest": "#799EBA",
    "constriction_gbest": "#176D79",
    "inertia_lbest_ring": "#518B81",
    "clpso": "#AA793C",
    "slpso": "#986E94",
    "spso2011": "#536575",
    "fips_ring": "#6D8C57",
    "cso": "#AE6671",
    "dms_pso": "#8D8B50",
    CONTROL: "#4F4F4F",
}
LINE_STYLES = {
    "classic_gbest": "-", "inertia_gbest": "--", "constriction_gbest": "-",
    "inertia_lbest_ring": "--", "clpso": "-", "slpso": "--",
    "spso2011": "-", "fips_ring": "--", "cso": "-", "dms_pso": "--",
}
FLOOR = 1e-12


def style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 8.5,
        "axes.titlesize": 9.5,
        "axes.titleweight": "normal",
        "axes.labelsize": 9,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.7,
        "axes.grid": True,
        "grid.alpha": 0.15,
        "grid.linewidth": 0.5,
        "grid.color": "#7D8894",
        "lines.linewidth": 1.2,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
    })


def fingerprint(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def load_protocol(folder: Path) -> dict:
    for p in (folder / "protocol.json", folder.parent.parent / "protocol.json"):
        if p.exists():
            obj = json.loads(p.read_text(encoding="utf-8"))
            if "comparison" in obj and "population" in obj:
                return obj
    raise FileNotFoundError(
        "Protocol not found. Pass --source the .../03_comparison/native directory "
        "containing protocol.json and runs/."
    )


def verify_history(result: dict, *, budget: int, name: str) -> tuple[np.ndarray, np.ndarray]:
    if "best_history" not in result or "history_evaluations" not in result:
        raise ValueError(f"{name}: missing real per-run convergence history")
    fe = np.asarray(result["history_evaluations"], dtype=np.int64)
    best = np.asarray(result["best_history"], dtype=float)
    if not len(fe) or len(fe) != len(best):
        raise ValueError(f"{name}: invalid history lengths")
    if int(result["function_evaluations"]) != budget or int(fe[-1]) != budget:
        raise ValueError(f"{name}: FE budget mismatch")
    if fe[0] < 1 or np.any(np.diff(fe) <= 0):
        raise ValueError(f"{name}: history FE are not increasing")
    if not np.isfinite(best).all() or np.any(np.diff(best) > 1e-7 * np.maximum(1.0, np.abs(best[:-1]))):
        raise ValueError(f"{name}: invalid best-so-far history")
    if not np.isclose(best[-1], float(result["best_value"]), atol=1e-7, rtol=1e-7):
        raise ValueError(f"{name}: history endpoint differs from the saved final value")
    return fe, best


def read_runs(folder: Path, cfg: dict, dimension: int, budget: int) -> tuple[pd.DataFrame, dict]:
    population = int(cfg["population"])
    if budget % population:
        raise ValueError("Budget must be divisible by population size")
    iterations = budget // population - 1
    if iterations not in cfg["comparison"]["iterations"]:
        raise ValueError(
            f"Requested budget {budget} FE is not in the original protocol. "
            f"Available: {[population * (x + 1) for x in cfg['comparison']['iterations']]}"
        )
    if dimension not in cfg["comparison"]["dimensions"]:
        raise ValueError(f"Dimension {dimension} is not in the study design")
    seeds = cfg["comparison"]["seeds"]
    names = [f for f in FUNCTIONS if f in cfg["comparison"]["functions"]]
    missing = []
    rows = []
    histories = {}
    for b in names:
        for seed in seeds:
            for method in ALL_METHODS:
                key = dict(benchmark=b, dimension=dimension, seed=seed,
                           iterations=iterations, method=method)
                run_path = folder / "runs" / f"{fingerprint(key)}.json"
                if not run_path.is_file():
                    missing.append((b, seed, method))
                    continue
                record = json.loads(run_path.read_text(encoding="utf-8"))
                if record.get("key") != key:
                    raise ValueError(f"Checkpoint key mismatch in {run_path}")
                result = record["result"]
                if record.get("result_sha256") != fingerprint(result):
                    raise ValueError(f"Modified or corrupted checkpoint: {run_path.name}")
                tag = f"{b}, D={dimension}, seed={seed}, {method}"
                fe, best = verify_history(result, budget=budget, name=tag)
                wall = float(result["wall_seconds"])
                if not np.isfinite(wall) or wall <= 0:
                    raise ValueError(f"{tag}: invalid wall time")
                if best.min() < -1e-5:
                    raise ValueError(f"{tag}: negative native function gap, expected f* = 0")
                rows.append({
                    "benchmark": b, "dimension": dimension, "seed": seed,
                    "method": method, "value": float(result["best_value"]),
                    "wall_seconds": wall,
                    "implementation": result.get("implementation", "unknown"),
                    "architecture": result.get("execution", {}).get("architecture", "unknown"),
                    "workers": result.get("execution", {}).get("workers", "unknown"),
                    "topology": result.get("execution", {}).get("topology", "unknown"),
                })
                histories[(b, seed, method)] = (fe, best)
    if missing:
        short = ", ".join(map(str, missing[:5]))
        raise RuntimeError(
            f"Incomplete Study III at D={dimension}, {budget} FE: "
            f"{len(missing)} / {len(names)*len(seeds)*len(ALL_METHODS)} checkpoints missing. "
            f"First missing: {short}. Do not publish partially pooled curves."
        )
    if len(names) != 6:
        raise ValueError(f"Expected 6 native functions; got {names}")
    return pd.DataFrame(rows), histories


def convergence_figure(frame: pd.DataFrame, histories: dict,
                       budget: int, dimension: int, population: int,
                       out: Path, control: bool = False) -> None:
    methods = list(METHODS) + ([CONTROL] if control else [])
    seeds = sorted(frame["seed"].unique())
    # Checkpoints chosen before examining objective values; log spacing reveals
    # early and late search at the same exact FE budgets across all algorithms.
    grid = np.unique(np.rint(np.geomspace(population, budget, 100)).astype(int))
    grid[0] = population
    grid[-1] = budget
    fig, axs = plt.subplots(2, 3, figsize=(9.0, 5.65), sharex=True)
    for j, b in enumerate(FUNCTIONS):
        ax = axs.flat[j]
        for m in methods:
            curves = []
            for seed in seeds:
                fe, best = histories[(b, seed, m)]
                if fe[0] > grid[0]:
                    raise ValueError(f"History starts after common checkpoint, {b}, {m}")
                # Left-continuous best-so-far hold: no interpolation, no imputed FE.
                ix = np.searchsorted(fe, grid, side="right") - 1
                curves.append(np.log10(np.maximum(best[ix], FLOOR)))
            array = np.asarray(curves)
            median = np.median(array, axis=0)
            if m == PRIMARY:
                lo, hi = np.quantile(array, [0.25, 0.75], axis=0)
                ax.fill_between(grid, lo, hi, color=PALETTE[m], alpha=0.11, lw=0, zorder=1)
            ax.plot(grid, median, color=PALETTE[m],
                    linestyle=":" if m == CONTROL else LINE_STYLES.get(m, "-"),
                    linewidth=2.2 if m == PRIMARY else 0.95 if m != CONTROL else 1.25,
                    alpha=1.0 if m == PRIMARY else 0.84,
                    zorder=4 if m == PRIMARY else 2)
        ax.set_title(f"({chr(97+j)}) {F_LABEL[b]}", pad=5)
        ax.set_xscale("log", base=2)
        ax.set_xlim(population, budget)
        ticks = sorted(set(t for t in [population, 512, 4096, budget] if t <= budget))
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{t // 1024}k" if t >= 4096 and t % 1024 == 0 else f"{t:,}" for t in ticks])
        ax.grid(axis="x", visible=False)
        ax.tick_params(axis="x", rotation=0)
    for ax in axs[1, :]:
        ax.set_xlabel("Function evaluations")
    for ax in axs[:, 0]:
        ax.set_ylabel(r"Median $\log_{10}\,f_{\mathrm{best}}$")
    labels = methods
    handles = [Line2D([0], [0], lw=2.2 if m == PRIMARY else 1.25,
                      color=PALETTE[m], linestyle=":" if m == CONTROL else LINE_STYLES.get(m, "-"))
               for m in labels]
    fig.legend(handles, [M_LABEL[m] for m in labels],
               loc="lower center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, 0.007), columnspacing=1.6,
               handlelength=2.3, handletextpad=0.5)
    fig.subplots_adjust(left=0.075, right=0.985, top=0.96,
                        bottom=0.22, hspace=0.34, wspace=0.26)
    fig.savefig(out, bbox_inches="tight", pad_inches=0.045)
    plt.close(fig)


def score_heatmap(frame: pd.DataFrame, dimension: int, budget: int, out: Path) -> pd.DataFrame:
    ranked = frame[frame["method"].isin(METHODS)].copy()
    ranked["rank"] = ranked.groupby(["benchmark", "seed"])["value"].rank(method="average")
    per = ranked.groupby(["method", "benchmark"])["rank"].mean().unstack()[list(FUNCTIONS)]
    per["Mean"] = per.mean(axis=1)
    per = per.sort_values("Mean", kind="stable")
    values = per.to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(7.3, 4.0))
    cmap = plt.get_cmap("YlGnBu_r")  # lower (better) ranks darker
    im = ax.imshow(values, aspect="auto", cmap=cmap, vmin=1, vmax=len(METHODS), interpolation="nearest")
    ax.set_xticks(range(per.shape[1]), [F_LABEL[c] if c in F_LABEL else c for c in per.columns], rotation=25, ha="right")
    ax.set_yticks(range(len(per)), [M_LABEL[m] for m in per.index])
    ax.tick_params(axis="both", length=0)
    ax.grid(False)
    ax.axvline(5.5, color="white", lw=3.0)
    for i in range(per.shape[0]):
        for j in range(per.shape[1]):
            val = values[i, j]
            rgba = cmap((val - 1) / (len(METHODS) - 1))
            luminance = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
            ax.text(j, i, f"{val:.1f}", ha="center", va="center",
                    fontsize=7.5, color="white" if luminance < 0.49 else "#192D3C")
    if PRIMARY in per.index:
        pos = int(per.index.get_loc(PRIMARY))
        ax.add_patch(Rectangle((-.5, pos - .5), per.shape[1], 1,
                               fill=False, lw=1.6, edgecolor="#B33E44"))
    ax.set_xlim(-.5, per.shape[1] - .5)
    ax.set_ylim(len(per) - .5, -.5)
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.018, shrink=0.86)
    cb.set_label("Mean rank (lower is better)", labelpad=7)
    cb.ax.tick_params(labelsize=7.5)
    fig.subplots_adjust(left=0.20, right=0.91, bottom=0.15, top=0.97)
    fig.savefig(out, bbox_inches="tight", pad_inches=0.045)
    plt.close(fig)
    return per


def runtime_figure(frame: pd.DataFrame, budget: int, dimension: int, out: Path) -> pd.DataFrame:
    # First collapse seeds within function, then summarize across the six fixed
    # benchmark functions; bars are heterogeneity, NOT inferential CI.
    per_fn = frame.groupby(["method", "benchmark"], observed=True)["wall_seconds"].median().unstack()[list(FUNCTIONS)]
    total = pd.DataFrame(index=per_fn.index)
    total["median"] = per_fn.median(axis=1)
    total["q25"] = per_fn.quantile(0.25, axis=1)
    total["q75"] = per_fn.quantile(0.75, axis=1)
    total = total.sort_values("median", kind="stable")
    y = np.arange(len(total))
    fig, ax = plt.subplots(figsize=(7.0, 4.05))
    for i, (m, r) in enumerate(total.iterrows()):
        color = PALETTE[PRIMARY] if m == PRIMARY else PALETTE[CONTROL] if m == CONTROL else "#517B8F"
        ax.plot([r.q25, r.q75], [i, i], lw=2.25, color=color, solid_capstyle="round", alpha=0.8)
        ax.plot(r["median"], i, marker="o", ms=5.4 if m == PRIMARY else 4.5,
                color=color, zorder=3, mec="white", mew=0.45)
    ax.set_yticks(y, [M_LABEL[m] for m in total.index])
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_xlabel("Completed-run wall time (s, logarithmic scale)")
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", alpha=0.18)
    ax.set_ylim(len(total)-0.45,-0.55)
    ax.set_title("Observed implementation runtime", fontweight="normal", pad=10)
    fig.text(0.62, 0.018,
             "CPSO: C++ native  |  PSO baselines: Python/NumPy  |  bars: IQR across six functions",
             ha="center", fontsize=7.3, color="#555555")
    fig.subplots_adjust(left=0.25, right=0.98, top=0.91, bottom=0.18)
    fig.savefig(out, bbox_inches="tight", pad_inches=0.045)
    plt.close(fig)
    return total


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, default=Path("results/study3_pp_paper/03_comparison/native"),
                   help="Native Study III output directory containing runs/ and protocol.json")
    p.add_argument("--dimension", type=int, default=20, choices=[5,20,100])
    p.add_argument("--budget", type=int, default=32768)
    p.add_argument("--outdir", type=Path, default=Path("results/study3_pp_paper/study3_publication_figures"))
    p.add_argument("--include-serial-control", action="store_true", help="Overlay CPSO serial in convergence")
    args = p.parse_args()
    folder = args.source
    if not folder.is_dir() or not (folder / "runs").is_dir():
        p.error(f"Not a comparison result directory: {folder}")
    config = load_protocol(folder)
    style()
    print(f"Reading original native Study III checkpoints: {folder}", flush=True)
    frame, histories = read_runs(folder, config, args.dimension, args.budget)
    # All methods must be present on the same fixed function x seed blocks.
    expected = len(FUNCTIONS) * len(config['comparison']['seeds']) * len(ALL_METHODS)
    assert len(frame) == expected
    actual = frame[frame["method"] == PRIMARY][["benchmark", "architecture", "workers", "topology"]].drop_duplicates()
    print("\nSaved CPSO main execution recipes (verify BEFORE interpreting a parallel claim):")
    print(actual.to_string(index=False))
    if not actual["architecture"].eq("particle_parallel").all():
        print("\nWARNING: some CPSO runs are NOT synchronous particle-parallel. "
              "The plots use the truthful label 'CPSO selected'; check your frozen recipes.", file=sys.stderr)
    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    convergence_figure(frame, histories, args.budget, args.dimension, config["population"],
                       outdir / f"fig_convergence_d{args.dimension}_fe{args.budget}.pdf",
                       control=args.include_serial_control)
    ranks = score_heatmap(frame, args.dimension, args.budget,
                          outdir / f"fig_quality_heatmap_d{args.dimension}_fe{args.budget}.pdf")
    runtime = runtime_figure(frame, args.budget, args.dimension,
                             outdir / f"fig_runtime_d{args.dimension}_fe{args.budget}.pdf")
    print(f"\nAll {len(frame)} saved runs verified. Budget={args.budget} FE, D={args.dimension}, "
          f"functions={len(FUNCTIONS)}, seeds={len(config['comparison']['seeds'])}.")
    print("\nMean quality ranks across six fixed functions (lower is better):")
    for m, r in ranks.iterrows():
        print(f"  {M_LABEL[m]:20s} {r['Mean']:5.2f}")
    print("\nSaved 3 vector PDF figures in:", outdir.resolve())
    print("NOTE: runtime ranks C++ CPSO and Python/NumPy PSO implementations; "
          "do not present that figure as a hardware-normalized algorithm speed test.")
    print("NOTE: quality comparisons use non-translated native benchmarks; BBOB is required "
          "for shift/rotation generalization.")


if __name__ == "__main__":
    main()
