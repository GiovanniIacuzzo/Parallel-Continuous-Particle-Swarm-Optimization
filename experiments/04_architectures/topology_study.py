"""Paired analysis of the M2 social-topology experiment."""
import argparse
import itertools
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binomtest, rankdata

from run_publication_scaling import TOPOS, check_block, write_csv

OBJECTIVES = {"rastrigin": ("Rastrigin", 4), "inverse_diffusion": ("Inverse diffusion", 8)}
LABELS = ["Global", "Ring", "Von Neumann", "Wheel (star)", "Random 4", "Random 8",
          "Small-world", "Scale-free"]


def read_cells(folder, benchmark):
    manifest = json.loads((folder / "publication_scaling_manifest.json").read_text())
    cfg = manifest["execution_fingerprint"]["settings"]
    if cfg["benchmark"] != benchmark or tuple(cfg["topologies"]) != TOPOS:
        raise ValueError(f"Unexpected design in {folder}")
    methods = [("serial", 1)] + list(itertools.product(cfg["architectures"], cfg["workers"]))
    groups = {}
    with (folder / "publication_scaling_runs.jsonl").open() as stream:
        for line in stream:
            r = json.loads(line)
            key = (r["dimension"], r["seed"], r["topology"], r["repetition"])
            method = (r["architecture"], r["workers"])
            group = groups.setdefault(key, {})
            if method in group or r["benchmark"] != benchmark:
                raise ValueError(f"Duplicate or mismatched run {key}")
            group[method] = r
    cells = []
    for d in cfg["dimensions"]:
        quality, times, speed = [], [], []
        for seed in cfg["seeds"]:
            q, t, s = [], [], []
            initial = None
            for topo in TOPOS:
                timings, ratios, history = [], [], None
                for rep in range(1, cfg["repetitions"] + 1):
                    group = groups.pop((d, seed, topo, rep))
                    check_block(group, methods, cfg["population"], cfg["iterations"])
                    serial = group["serial", 1]
                    h = serial["best_history"]
                    if initial is None:
                        initial = h[0]
                    if h[0] != initial or (history is not None and history != h):
                        raise ValueError("Unpaired initialization or inconsistent repeat")
                    history = h
                    elapsed = [group["particle_parallel", w]["elapsed_seconds"] for w in cfg["workers"]]
                    timings.append(elapsed[cfg["workers"].index(OBJECTIVES[benchmark][1])])
                    ratios.append(np.array(elapsed)[cfg["workers"].index(1)] / np.array(elapsed))
                q.append(history[-1])
                t.append(np.median(timings))
                s.append(np.median(ratios, axis=0))
            quality.append(q)
            times.append(t)
            speed.append(s)
        cells.append((benchmark, d, cfg, np.array(quality), np.array(times), np.array(speed)))
    if groups:
        raise ValueError("Extra blocks in results")
    return cells, manifest["execution_fingerprint"]


def holm(rows):
    order = sorted(rows, key=lambda r: r["p_sign"])
    maximum = 0.
    for i, r in enumerate(order):
        maximum = max(maximum, min(1., r["p_sign"] * (len(order) - i)))
        r["p_holm"] = maximum


def analyze(base, out):
    cells, fingerprints = [], []
    for obj in OBJECTIVES:
        data, fingerprint = read_cells(base / obj, obj)
        cells.extend(data)
        fingerprints.append(fingerprint)
    a, b = fingerprints
    for field in ("dimensions", "seeds", "workers", "population", "iterations", "dt",
                  "repetitions", "architectures", "chunk_size"):
        if a["settings"][field] != b["settings"][field]:
            raise ValueError(f"Incompatible campaigns: {field}")
    if a["native_binary_sha256"] != b["native_binary_sha256"]:
        raise ValueError("Different binaries")
    out.mkdir(parents=True, exist_ok=True)
    summary, tests, scaling, tradeoffs = [], [], [], []
    plt.rcParams.update({"pdf.fonttype":42, "font.size":8, "axes.spines.top":False,
                         "axes.spines.right":False})
    fig, axes = plt.subplots(len(cells), 2, figsize=(7.4, 2.25 * len(cells)), squeeze=False,
                             layout="constrained")
    curves, panels = plt.subplots(2, 4, figsize=(8, 4.3), sharex=True, sharey=True,
                                  layout="constrained")
    eff_fig, eff_panels = plt.subplots(2, 4, figsize=(8, 4.3), sharex=True, sharey=True,
                                       layout="constrained")
    for cell, (obj, d, cfg, quality, times, speed) in enumerate(cells):
        n = len(quality)
        ix = np.random.default_rng(51509).integers(n, size=(4000, n))
        observed = np.column_stack([np.median(quality, axis=0), np.median(times, axis=0)])
        for j, topo in enumerate(TOPOS):
            dominated = np.all(observed <= observed[j], axis=1) & np.any(observed < observed[j], axis=1)
            tradeoffs.append(dict(objective=obj, dimension=d, topology=topo,
                                  final_objective_median=observed[j, 0], seconds_median=observed[j, 1],
                                  paired_time_ratio_to_global=np.median(times[:,j] / times[:,0]),
                                  on_observed_pareto_front=not np.any(dominated)))
        for col, (metric, matrix) in enumerate([("quality", quality), ("time", times)]):
            if np.any(~np.isfinite(matrix)) or np.any(matrix < 0):
                raise ValueError("Invalid objective/time")
            ranks = rankdata(matrix, axis=1, method="average")
            mean = ranks.mean(axis=0)
            bounds = np.quantile(ranks[ix].mean(axis=1), [.025, .975], axis=0)
            ax = axes[cell,col]
            for j, topo in enumerate(TOPOS):
                summary.append(dict(objective=obj, dimension=d, metric=metric, topology=topo,
                                    workers=OBJECTIVES[obj][1], seeds=n, median=float(np.median(matrix[:,j])),
                                    mean_rank=mean[j], ci_low=bounds[0,j], ci_high=bounds[1,j]))
                color = "#a84c29" if topo == "global" else "#265d78"
                ax.plot(bounds[:,j], [j,j], color=color)
                ax.plot(mean[j], j, "o", ms=4, color=color)
            ax.set_yticks(range(len(TOPOS)), LABELS)
            ax.invert_yaxis()
            ax.set_xlim(.75,8.25)
            ax.set_xticks(range(1,9))
            ax.grid(axis="x", alpha=.2)
            ax.set_title(f"{OBJECTIVES[obj][0]} · D={d} · " +
                         ("final objective" if col == 0 else f"time, p={OBJECTIVES[obj][1]}"))
            if cell == len(cells)-1:
                ax.set_xlabel("Mean paired rank (lower is better)")
            for j,k in itertools.combinations(range(len(TOPOS)),2):
                delta = matrix[:,j] - matrix[:,k]
                wins, losses = int(sum(delta<0)), int(sum(delta>0))
                tests.append(dict(objective=obj, dimension=d, metric=metric, a=TOPOS[j], b=TOPOS[k],
                                  a_better=wins, b_better=losses, ties=n-wins-losses,
                                  median_difference_a_minus_b=np.median(delta),
                                  p_sign=binomtest(wins,wins+losses).pvalue if wins+losses else 1.))
        for j, topo in enumerate(TOPOS):
            mid = np.median(speed[:,j], axis=0)
            lower, upper = np.quantile(np.median(speed[:,j][ix], axis=1), [.025,.975], axis=0)
            workers = np.asarray(cfg["workers"])
            color = "#265d78" if obj=="rastrigin" else "#a84c29"
            for axs, divisor, ylabel in [(panels, 1, "Speedup $T_1/T_p$"),
                                         (eff_panels, workers, "Efficiency (%)")]:
                ax = axs.flat[j]
                factor = 100 if axs is eff_panels else 1
                ax.plot(workers, factor*mid/divisor, color=color,
                        ls="-" if d==30 else "--", marker="o" if d==30 else "s", ms=3,
                        label=f"{OBJECTIVES[obj][0]}, D={d}")
                ax.fill_between(workers, factor*lower/divisor, factor*upper/divisor,
                                color=color, alpha=.09, linewidth=0)
                ax.set_title(LABELS[j])
                ax.set_xticks(workers)
                ax.grid(alpha=.2)
                if j>=4:
                    ax.set_xlabel("Workers")
                if j%4==0:
                    ax.set_ylabel(ylabel)
            for w,v,lo,hi in zip(workers,mid,lower,upper):
                scaling.append(dict(objective=obj,dimension=d,topology=topo,workers=w,
                                    strong_speedup=v,strong_efficiency=v/w,
                                    speedup_ci_low=lo,speedup_ci_high=hi,
                                    efficiency_ci_low=lo/w,efficiency_ci_high=hi/w))
    holm(tests)
    handles, labels = panels.flat[0].get_legend_handles_labels()
    for plot, axs in [(curves, panels), (eff_fig, eff_panels)]:
        axs.flat[0].set_ylim(bottom=0)
        plot.legend(handles, labels, loc="outside upper center", ncols=2, frameon=False)
    for name, obj in [("fig_topology_ranks",fig),("fig_topology_scaling",curves),("fig_topology_efficiency",eff_fig)]:
        obj.savefig(out / (name+".pdf"), bbox_inches="tight")
        plt.close(obj)
    for name, rows in [("topology_ranks",summary),("paired_contrasts",tests),("scaling",scaling),("quality_time",tradeoffs)]:
        write_csv(out / (name+".csv"),rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    from cpso_serial.output_policy import results_output
    analyze(args.base, results_output(args.out))
