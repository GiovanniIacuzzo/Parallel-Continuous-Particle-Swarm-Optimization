#!/usr/bin/env python3
"""Publication-ready, data-faithful plots from COCO Observer and CEC 2013 records.

This script NEVER modifies experimental outputs, and does not generate HTML or JS.
BBOB plot: empirical logged target-attainment profile (not the official cocopp ECDF).
CEC: matched seed/function ranks; no cross-function objective-value pooling.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import ticker
import numpy as np

METHOD_LABELS = {
    "cpso_continuous": "CPSO",
    "pso_classic_gbest": "Classic gbest",
    "pso_inertia_gbest": "Inertia gbest",
    "pso_constriction_gbest": "Constriction gbest",
    "pso_inertia_lbest_ring": "Inertia ring",
    "pso_clpso": "CLPSO",
    "pso_slpso": "SLPSO",
    "pso_spso2011": "SPSO2011",
    "pso_fips": "FIPS",
    "pso_cso": "CSO",
    "pso_dms": "DMS-PSO",
}
COLORS = {
    "cpso_continuous": "#BF4A46",
    "pso_spso2011": "#1B6477",
    "pso_cso": "#96723C",
    "pso_constriction_gbest": "#36619C",
    "pso_inertia_gbest": "#548765",
    "pso_classic_gbest": "#78619E",
    "pso_dms": "#AD7EAA",
}
HIGHLIGHTS = [
    "cpso_continuous", "pso_spso2011", "pso_cso",
    "pso_constriction_gbest", "pso_inertia_gbest", "pso_classic_gbest",
]
CEC_METHODS = list(METHOD_LABELS)


def style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9.2, "axes.titlesize": 10.5,
        "axes.labelsize": 9.5, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
        "legend.fontsize": 8.4, "pdf.fonttype": 42, "ps.fonttype": 42,
        "axes.spines.right": False, "axes.spines.top": False,
        "axes.linewidth": 0.65, "grid.alpha": 0.17,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.10,
    })


def save(fig: plt.Figure, dest: Path, stem: str) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest / f"{stem}.pdf", metadata={"Creator": "CPSO publication plotter"})
    fig.savefig(dest / f"{stem}.png", dpi=260)
    plt.close(fig)
    print(f"[OK] {stem}.pdf / .png")


def write_csv(dest: Path, filename: str, rows: list[dict]) -> None:
    if not rows:
        return
    with (dest / filename).open("w", newline="", encoding="utf-8") as out:
        w = csv.DictWriter(out, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def ranks(values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(values.items(), key=lambda kv: kv[1])
    result = {}
    start = 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and math.isclose(ordered[start][1], ordered[end][1], rel_tol=1e-12, abs_tol=1e-14):
            end += 1
        rank = ((start + 1) + end) / 2
        for i in range(start, end):
            result[ordered[i][0]] = rank
        start = end
    return result


def read_cec(src: Path) -> tuple[list[dict], list[int], list[int], list[int]]:
    data = []
    for no, line in enumerate(src.open(encoding="utf-8"), 1):
        if not line.strip():
            continue
        r = json.loads(line)
        required = ("function_id", "seed", "method", "dimension", "checkpoints", "best_value")
        if any(k not in r for k in required):
            raise ValueError(f"CEC missing fields at line {no}")
        if not isinstance(r["checkpoints"], dict):
            raise ValueError(f"CEC checkpoints not a dict at line {no}")
        data.append(r)
    if not data:
        raise ValueError(f"Empty CEC records: {src}")
    functions = sorted(set(int(r["function_id"]) for r in data))
    seeds = sorted(set(int(r["seed"]) for r in data))
    methods = sorted(set(str(r["method"]) for r in data))
    checkpoints = sorted({int(k) for r in data for k in r["checkpoints"]})
    if len(methods) != 11:
        raise ValueError(f"CEC expected 11 methods, found {len(methods)}")
    required_cells = {(f, s, m) for f in functions for s in seeds for m in methods}
    seen = set()
    for r in data:
        k = (int(r["function_id"]), int(r["seed"]), str(r["method"]))
        if k in seen:
            raise ValueError(f"Duplicate CEC run: {k}")
        seen.add(k)
        if set(map(int, r["checkpoints"])) != set(checkpoints):
            raise ValueError(f"CEC incomplete checkpoint coverage: {k}")
        dim = int(r["dimension"])
        if dim != (905 if k[0] in (13, 14) else 1000):
            raise ValueError(f"CEC dimension incorrect for {k}: D={dim}")
        if int(r["function_evaluations"]) != max(checkpoints):
            raise ValueError(f"CEC FE mismatch for {k}")
        v = [float(r["checkpoints"][str(fe)]) for fe in checkpoints]
        if any(not math.isfinite(x) for x in v):
            raise ValueError(f"CEC nonfinite value for {k}")
        if any(v[i + 1] > v[i] * (1 + 1e-12) + 1e-9 for i in range(len(v)-1)):
            raise ValueError(f"CEC best-so-far increased for {k}")
    if seen != required_cells:
        raise ValueError(f"Incomplete CEC grid: {len(seen)} versus {len(required_cells)}")
    rows = []
    per_cell = defaultdict(dict)
    for r in data:
        for cp, value in r["checkpoints"].items():
            key = (int(r["function_id"]), int(r["seed"]), int(cp))
            per_cell[key][str(r["method"])] = float(value)
    for (function, seed, cp), vals in sorted(per_cell.items()):
        if set(vals) != set(methods):
            raise ValueError(f"Incomplete CEC methods for F{function}/seed{seed}/{cp}")
        ranked = ranks(vals)
        for m in methods:
            rows.append({"function": function, "dimension": 905 if function in (13,14) else 1000,
                         "seed": seed, "FE": cp, "method": m, "rank": ranked[m], "best_value": vals[m]})
    return rows, functions, seeds, checkpoints


def plot_cec_checkpoints(records: list[dict], functions: list[int], checkpoints: list[int], dest: Path) -> None:
    means = defaultdict(list)
    for r in records:
        means[(r["dimension"], r["FE"], r["method"])].append(r["rank"])
    methods = sorted({r["method"] for r in records})
    ordered = sorted(methods, key=lambda m: np.mean([r["rank"] for r in records if r["FE"] == checkpoints[-1] and r["method"] == m]))
    selected = list(dict.fromkeys(["cpso_continuous"] + [m for m in ordered if m != "cpso_continuous"][:5]))
    dims = sorted(set(r["dimension"] for r in records), reverse=True)
    fig, axes = plt.subplots(1, len(dims), figsize=(10.4, 4.2), sharey=True, layout="constrained")
    for ax, dim in zip(np.ravel(axes), dims):
        for m in selected:
            y = [np.mean(means[(dim, cp, m)]) for cp in checkpoints]
            ax.plot(np.array(checkpoints)/1000, y, label=METHOD_LABELS[m], color=COLORS.get(m, "#555"),
                    lw=2.7 if m=="cpso_continuous" else 1.65, marker="o" if m=="cpso_continuous" else "s", ms=4)
        ax.set_title(f"D={dim} · {sum((905 if f in (13, 14) else 1000) == dim for f in functions)} functions")
        ax.set_xlabel("Function evaluations (thousands)")
        ax.set_xticks(np.array(checkpoints)/1000)
        ax.set_ylim(11.45, 0.55)
        ax.yaxis.set_major_locator(ticker.MultipleLocator(2))
        ax.grid(axis="y")
    np.ravel(axes)[0].set_ylabel("Mean within-(function, seed) rank · lower is better")
    handles, labels = np.ravel(axes)[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=3, loc="outside lower center", frameon=False)
    fig.suptitle("CEC 2013 LSGO · relative optimization quality across checkpoints", weight="semibold")
    save(fig, dest, "fig_cec2013_checkpoints")
    print("[INFO] Selection rule: CPSO + five baselines with best mean final rank; remaining methods in full breakdown.")


def plot_cec_breakdown(records: list[dict], functions: list[int], checkpoints: list[int], dest: Path) -> None:
    cp = checkpoints[-1]
    by_method_fn = defaultdict(list)
    for r in records:
        if r["FE"] == cp:
            by_method_fn[(r["method"], r["function"])].append(r["rank"])
    methods = sorted({r["method"] for r in records}, key=lambda m: (np.mean([r["rank"] for r in records if r["method"]==m and r["FE"]==cp]),m))
    mat = np.asarray([[np.mean(by_method_fn[(m,f)]) for f in functions] for m in methods])
    fig, ax = plt.subplots(figsize=(9.1, 5.8), layout="constrained")
    cmap = plt.get_cmap("viridis_r")
    im = ax.imshow(mat, cmap=cmap, aspect="auto", vmin=1, vmax=len(methods))
    ax.set_xticks(range(len(functions)), [f"F{f}\n({905 if f in (13,14) else 1000}D)" for f in functions])
    ax.set_yticks(range(len(methods)), [METHOD_LABELS[m] for m in methods]); ax.tick_params(axis="y", length=0)
    ax.set_xlabel("CEC 2013 LSGO function")
    for i in range(len(methods)):
        for j in range(len(functions)):
            v = mat[i, j]
            col = cmap((v-1)/(len(methods)-1))
            lum=0.2126*col[0]+0.7152*col[1]+0.0722*col[2]
            ax.text(j,i,f"{v:.1f}",ha="center",va="center",fontsize=8.3,color="#17212B" if lum>0.46 else "white")
    ax.set_xticks(np.arange(-.5, len(functions), 1),minor=True)
    ax.set_yticks(np.arange(-.5, len(methods), 1),minor=True)
    ax.grid(which="minor",color="white",linewidth=0.75); ax.tick_params(which="minor",left=False,bottom=False)
    cbar=fig.colorbar(im,ax=ax,shrink=.8,pad=.02);cbar.set_label("Mean rank across three matched seeds · 1 = best")
    fig.suptitle("CEC 2013 LSGO · per-function relative performance at 120,000 FE",weight="semibold")
    save(fig,dest,"fig_cec_function_breakdown")


def parse_coco_trace(path: Path) -> list[tuple[np.ndarray,np.ndarray]]:
    """Read COCO logger time/target trace, first column FE, third column f-Fopt.

    COCO writes instance blocks headed by '%' and 1-based FE within block.
    Returns a nonincreasing best-so-far gap per block; no extrapolation.
    """
    blocks=[]; xx=[];yy=[]
    def flush():
        if xx:
            xs=np.array(xx,dtype=float);ys=np.minimum.accumulate(np.array(yy,dtype=float))
            if np.any(np.diff(xs)<0) or xs[0]<1:
                raise ValueError(f"Invalid FE order in {path}")
            blocks.append((xs,ys))
    for line in path.open(encoding="utf-8"):
        s=line.strip()
        if not s:continue
        if s.startswith("%"):
            flush(); xx=[];yy=[];continue
        parts=s.split()
        if len(parts)<3:raise ValueError(f"Invalid COCO line in {path}: {s[:90]}")
        xx.append(float(parts[0]));yy.append(float(parts[2]))
    flush()
    return blocks


def read_bbob(root: Path, targets: np.ndarray) -> tuple[dict, list[int], list[str], list[int]]:
    if not (root / "exdata").exists():
        raise FileNotFoundError(f"No COCO exdata in {root}")
    problem_values=defaultdict(dict)
    methods=[]
    for method_dir in sorted((root / "exdata").iterdir()):
        if not method_dir.is_dir():continue
        method=method_dir.name
        if method not in METHOD_LABELS:continue
        methods.append(method)
        files=sorted(method_dir.glob("data_f*/*_DIM*.tdat"))
        if not files:raise ValueError(f"COCO .tdat not found for method {method}")
        for file in files:
            m=re.search(r"data_f(\d+).*DIM(\d+)\.tdat$",str(file))
            if not m:raise ValueError(f"Unparseable COCO file {file}")
            fn,dim=map(int,m.groups())
            blocks=parse_coco_trace(file)
            if len(blocks) != (5 if "largescale" not in root.name else 2):
                # Rather than assume 15 official instances, enforce equal counts later.
                if not blocks:raise ValueError(f"No COCO trace blocks in {file}")
            for instance,(fe,quality) in enumerate(blocks,start=1):
                problem=(fn,dim,instance)
                if method in problem_values[problem]:raise ValueError(f"Duplicate trace {problem}/{method}")
                threshold_hits=[]
                for t in targets:
                    idx=np.flatnonzero(quality<=t)
                    threshold_hits.append(float(fe[int(idx[0])]) if idx.size else math.inf)
                problem_values[problem][method]=(threshold_hits,float(fe[-1]))
    if len(methods)!=11:
        raise ValueError(f"Expected 11 COCO methods, found {len(methods)}")
    for problem,records in problem_values.items():
        if set(records)!=set(methods):
            raise ValueError(f"Missing COCO method traces for {problem}: {sorted(set(methods)-set(records))}")
        budgets={x[1] for x in records.values()}
        if len(budgets)!=1:
            raise ValueError(f"Unequal FE budget at {problem}: {budgets}")
    dims=sorted({p[1] for p in problem_values})
    functions=sorted({p[0] for p in problem_values})
    return problem_values, dims, methods, functions


def plot_bbob(root: Path, dest: Path, targets: np.ndarray) -> None:
    problem_values,dims,methods,functions=read_bbob(root,targets)
    all_methods=[m for m in HIGHLIGHTS if m in methods]
    n_instances=len(problem_values)
    print(f"[OK] BBOB traces: {len(methods)} methods, {n_instances} function/instance/dimension problems; targets={list(targets)}")
    fig, axes = plt.subplots(1,len(dims),figsize=(10.2,4.7),sharey=True,layout="constrained")
    grid=np.geomspace(1, 1000, 180)
    rows=[]
    for ax,dim in zip(np.ravel(axes),dims):
        problems={key:values for key,values in problem_values.items() if key[1]==dim}
        n_trials=len(problems)*len(targets)
        for method in methods:
            hit_ratios=[]
            for (fn,d,inst),mvals in problems.items():
                hits,fe_max=mvals[method]
                for hit in hits:
                    hit_ratios.append(hit/dim)
            hits=np.asarray(hit_ratios)
            yy=np.array([np.count_nonzero(hits<=x)/n_trials for x in grid])
            highlight=method in all_methods
            if highlight:
                line,=ax.step(grid,yy,where="post",label=METHOD_LABELS[method],
                              color=COLORS.get(method,"#555"),lw=2.6 if method=="cpso_continuous" else 1.55,
                              ls="-" if method=="cpso_continuous" else "--")
                if method=="cpso_continuous":line.set_zorder(10)
            else:
                ax.step(grid,yy,where="post",color="#A0AAB4",lw=.9,alpha=.5)
            rows.append({"dimension":dim,"method":method,"functions":len(functions),
                         "instances":len(problems),"targets":len(targets),
                         "success_fraction_at_budget":float(yy[-1]),
                         "total_target_trials":n_trials})
        ax.set_xscale("log")
        ax.set_xlim(1,1000);ax.set_ylim(0,1.02)
        ax.set_title(f"D={dim} · {len(problems)} instances × {len(targets)} targets")
        ax.set_xlabel("Function evaluations / dimension")
        ax.grid(axis="both",which="major")
    np.ravel(axes)[0].set_ylabel("Fraction of target–problem pairs reached")
    h,l=np.ravel(axes)[0].get_legend_handles_labels()
    fig.legend(h,l,loc="outside lower center",ncol=3,frameon=False)
    fig.suptitle("BBOB · empirical target-attainment curves from COCO logs",weight="semibold")
    save(fig,dest,"fig_coco_bbob_ecdf")
    write_csv(dest,"bbob_target_attainment_summary.csv",rows)
    print("[NOTE] Curves are computed from logged .tdat hit times; they are not bit-identical to cocopp post-processing ECDFs.")


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--external-root",type=Path,default=Path("results/external_benchmarks"))
    parser.add_argument("--output",type=Path,default=Path("paper/figures/results"))
    args=parser.parse_args()
    style();root=args.external_root;dest=args.output
    dest.mkdir(parents=True,exist_ok=True)
    cec=root/"cec2013_stageC_pilot_corrected_m2"/"records.jsonl"
    rows,functions,seeds,cps=read_cec(cec)
    print(f"[OK] CEC complete: {len(rows)} checkpoint observations, {len(functions)} functions, {len(seeds)} seeds, {len(CEC_METHODS)} methods")
    write_csv(dest,"cec_checkpoint_ranks.csv",rows)
    plot_cec_checkpoints(rows,functions,cps,dest)
    plot_cec_breakdown(rows,functions,cps,dest)
    bbob=root/"bbob_stageC_clean_m2"
    # A fixed prespecified target subset covering six orders of magnitude.
    targets=np.asarray([1e2,1e1,1,1e-1,1e-2,1e-3,1e-4,1e-5,1e-6,1e-7,1e-8],float)
    plot_bbob(bbob,dest,targets)
    print("[OK] Done. No HTML/JS created.")


if __name__=="__main__":main()
