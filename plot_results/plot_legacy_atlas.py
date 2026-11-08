#!/usr/bin/env python3
"""Plot legacy CPSO sweep outputs (master-slave scaling and topology studies).

Reads existing CSV/JSON files only. No optimizer is invoked and no observations
are created. Timing plots are descriptive and hardware-specific.
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

from plot_results.atlas_common import (
    COLORS, LOG_FLOOR, configure_style, finite_float, save_figure, safe_slug,
)


def read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open("r", newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def _groups(rows, keys):
    out = defaultdict(list)
    for row in rows:
        out[tuple(str(row.get(k, "")) for k in keys)].append(row)
    return out


def _summary(values):
    arr = np.asarray([finite_float(x) for x in values], dtype=float)
    arr = arr[np.isfinite(arr)]
    if not arr.size:
        return math.nan, math.nan, math.nan
    return float(np.median(arr)), float(np.quantile(arr, .25)), float(np.quantile(arr, .75))


def _box(ax, groups, labels, title, ylabel, *, log=False):
    if not any(len(g) for g in groups):
        return
    data = [g if g else [np.nan] for g in groups]
    bp = ax.boxplot(data, patch_artist=True, showfliers=False, widths=.62,
                    medianprops={"color": "black", "linewidth": 1})
    for i, patch in enumerate(bp["boxes"]):
        patch.set_facecolor(COLORS[i % len(COLORS)]); patch.set_alpha(.52)
    ax.set_xticks(np.arange(1, len(labels)+1), labels, fontsize=6)
    ax.set_title(title); ax.set_ylabel(ylabel); ax.grid(axis="y")
    if log:
        ax.set_yscale("log")


def plot_scaling(directory: Path, output: Path) -> tuple[list[Path], list[str]]:
    raw = read_csv(directory / "scaling_runs.csv")
    summary = read_csv(directory / "scaling_summary.csv")
    if not raw and not summary:
        return [], [f"No scaling_runs.csv/scaling_summary.csv under {directory}"]
    generated=[]; warnings=[]
    stem=safe_slug(directory.name)

    # A. Total runtime and speedup by worker count, stratified by every D/N/work cell.
    if raw:
        combos=sorted({(int(finite_float(r.get("dimension"),-1)), int(finite_float(r.get("population_size"),-1)),
                        int(finite_float(r.get("work_iterations"),0))) for r in raw})
        for d,n,w in combos:
            sub=[r for r in raw if int(finite_float(r.get("dimension"),-1))==d and
                 int(finite_float(r.get("population_size"),-1))==n and int(finite_float(r.get("work_iterations"),0))==w]
            modes=sorted({str(r.get("mode")) for r in sub})
            workers=sorted({int(finite_float(r.get("workers"),1)) for r in sub if str(r.get("mode"))!="serial"})
            fig,axes=plt.subplots(1,2,figsize=(9.2,3.45))
            # paired seed lookups; only matched serial / worker records contribute.
            lookup={(int(finite_float(r.get("seed"),-1)),str(r.get("mode")),int(finite_float(r.get("workers"),1))):r for r in sub}
            runtime_by=[[],[]]; speedup_by=[]; eff_by=[]
            for worker in workers:
                serial=[]; parallel=[]; sp=[]
                for seed in sorted({int(finite_float(r.get("seed"),-1)) for r in sub}):
                    s=lookup.get((seed,"serial",1)); p=lookup.get((seed,"master_slave",worker))
                    if not s or not p: continue
                    st=finite_float(s.get("elapsed_seconds")); pt=finite_float(p.get("elapsed_seconds"))
                    if st>0 and pt>0:
                        serial.append(st);parallel.append(pt);sp.append(st/pt)
                runtime_by[0].append(_summary(serial)[0]);runtime_by[1].append(_summary(parallel)[0])
                speedup_by.append(_summary(sp)[0]);eff_by.append(_summary(sp)[0]/worker if sp else math.nan)
            x=np.arange(len(workers)); width=.36
            axes[0].bar(x-width/2,runtime_by[0],width,label="Serial",color=COLORS[0])
            axes[0].bar(x+width/2,runtime_by[1],width,label="Master–slave",color=COLORS[1])
            axes[0].set_xticks(x,[str(v) for v in workers]);axes[0].set_xlabel("Worker count")
            axes[0].set_ylabel("Median elapsed time (s)");axes[0].set_title("Matched runtime");axes[0].legend(frameon=False);axes[0].grid(axis="y")
            axes[1].plot(workers,speedup_by,"o-",label="Observed speedup",color=COLORS[2])
            axes[1].plot(workers,workers,"--",label="Ideal",color="#777777",lw=.9)
            axes[1].set_xlabel("Worker count");axes[1].set_ylabel("Median paired speedup");axes[1].set_title("Strong scaling")
            axes[1].set_xticks(workers);axes[1].grid();axes[1].legend(frameon=False)
            fig.suptitle(f"{directory.name}: D={d}, N={n}, synthetic work={w:,}")
            fig.text(.5,.005,"Paired by seed; elapsed times are machine-specific. sphere_work is a synthetic execution-cost instrument.",ha="center",fontsize=6,color="#687481")
            fig.tight_layout(rect=(0,.04,1,.94));generated+=save_figure(fig,output,f"{stem}_runtime_scaling_d{d}_n{n}_work{w}")

            fig,ax=plt.subplots(figsize=(5.8,3.5))
            ax.plot(workers,speedup_by,"o-",label="Speedup",color=COLORS[2])
            ax.plot(workers,eff_by,"s-",label="Parallel efficiency",color=COLORS[3])
            ax.axhline(1,color="#777777",lw=.7,ls=":");ax.set_xticks(workers);ax.set_xlabel("Worker count")
            ax.set_ylabel("Ratio");ax.set_title(f"Scaling metrics · D={d}, N={n}, work={w:,}");ax.grid();ax.legend(frameon=False)
            fig.tight_layout();generated+=save_figure(fig,output,f"{stem}_speedup_efficiency_d{d}_n{n}_work{w}")

        # B. Throughput and instrumented components across all workload cells.
        fields=[("evaluation_batch_wall_seconds","Objective-evaluation batch"),("propagation_seconds","Propagation"),("residual_seconds","Residual")]
        for d,n,w in combos:
            sub=[r for r in raw if int(finite_float(r.get("dimension"),-1))==d and int(finite_float(r.get("population_size"),-1))==n and int(finite_float(r.get("work_iterations"),0))==w]
            workers=sorted({int(finite_float(r.get("workers"),1)) for r in sub if str(r.get("mode"))!="serial"})
            if not workers: continue
            fig,axes=plt.subplots(1,2,figsize=(9.2,3.4))
            for j,(mode,label) in enumerate((("serial","Serial"),("master_slave","Master–slave"))):
                ws=[1] if mode=="serial" else workers
                med=[]
                for worker in ws:
                    vals=[finite_float(r.get("evaluations_per_second")) for r in sub if str(r.get("mode"))==mode and int(finite_float(r.get("workers"),1))==worker and finite_float(r.get("evaluations_per_second"))>0]
                    med.append(_summary(vals)[0])
                axes[0].plot(ws,med,"o-",label=label,color=COLORS[j])
            axes[0].set_xlabel("Workers");axes[0].set_ylabel("Function evaluations / second");axes[0].set_title("Throughput");axes[0].grid();axes[0].legend(frameon=False)
            x=np.arange(len(workers));width=.82/len(fields)
            for k,(field,label) in enumerate(fields):
                vals=[]
                for worker in workers:
                    samples=[finite_float(r.get(field)) for r in sub if str(r.get("mode"))=="master_slave" and int(finite_float(r.get("workers"),1))==worker and finite_float(r.get(field))>=0]
                    vals.append(_summary(samples)[0])
                axes[1].bar(x-.41+width*(k+.5),vals,width,label=label,color=COLORS[k])
            axes[1].set_xticks(x,[str(v) for v in workers]);axes[1].set_xlabel("Master–slave workers");axes[1].set_ylabel("Median seconds");axes[1].set_title("Measured time components");axes[1].legend(frameon=False,fontsize=6);axes[1].grid(axis="y")
            fig.suptitle(f"{directory.name}: throughput and phase timing · D={d}, N={n}, work={w:,}")
            fig.text(.5,.005,"Phase components are reported independently and are not assumed additive.",ha="center",fontsize=6,color="#687481")
            fig.tight_layout(rect=(0,.04,1,.94));generated+=save_figure(fig,output,f"{stem}_throughput_components_d{d}_n{n}_work{w}")

    # C. The saved summary table may carry the original bootstrap interval.
    if summary:
        dgroups=sorted({(int(finite_float(r.get("dimension"),-1)),int(finite_float(r.get("population_size"),-1)),int(finite_float(r.get("work_iterations"),0))) for r in summary})
        for d,n,w in dgroups:
            sub=sorted([r for r in summary if int(finite_float(r.get("dimension"),-1))==d and int(finite_float(r.get("population_size"),-1))==n and int(finite_float(r.get("work_iterations"),0))==w],key=lambda r:finite_float(r.get("workers")))
            if not sub:continue
            x=[finite_float(r.get("workers")) for r in sub];y=[finite_float(r.get("median_total_speedup")) for r in sub]
            lo=[finite_float(r.get("total_speedup_bootstrap_ci95_low")) for r in sub];hi=[finite_float(r.get("total_speedup_bootstrap_ci95_high")) for r in sub]
            fig,ax=plt.subplots(figsize=(5.8,3.5));ax.plot(x,y,"o-",color=COLORS[2],label="Median speedup")
            if all(math.isfinite(v) for v in lo+hi):ax.fill_between(x,lo,hi,color=COLORS[2],alpha=.18,label="95% bootstrap interval")
            ax.plot(x,x,"--",color="#777777",lw=.8,label="Ideal");ax.set_xticks(x);ax.set_xlabel("Workers");ax.set_ylabel("Total speedup")
            ax.set_title(f"Stored bootstrap scaling summary · D={d}, N={n}, work={w:,}");ax.grid();ax.legend(frameon=False)
            fig.tight_layout();generated+=save_figure(fig,output,f"{stem}_bootstrap_speedup_d{d}_n{n}_work{w}")
    return generated,warnings


def plot_topology(directory: Path, output: Path) -> tuple[list[Path], list[str]]:
    runs=read_csv(directory/"runs.csv"); convergence=read_csv(directory/"convergence.csv")
    if not runs:return [],[f"No topology runs.csv under {directory}"]
    generated=[];warnings=[];stem=safe_slug(directory.name)
    topologies=sorted({str(r.get("topology","")) for r in runs})
    benchmarks=sorted({str(r.get("benchmark","")) for r in runs})
    dims=sorted({int(finite_float(r.get("dimension"),-1)) for r in runs})
    methods=topologies
    # 1. Final quality distributions per benchmark and dimension.
    combos=sorted({(str(r.get("benchmark")),int(finite_float(r.get("dimension"),-1))) for r in runs})
    ncols=min(2,len(combos));nrows=math.ceil(len(combos)/ncols)
    fig,axes=plt.subplots(nrows,ncols,figsize=(3.5*ncols,2.7*nrows),squeeze=False)
    for ax,(bench,d) in zip(axes.flat,combos):
        part=[r for r in runs if str(r.get("benchmark"))==bench and int(finite_float(r.get("dimension"),-1))==d]
        vals=[[math.log10(max(finite_float(r.get("best_value")),LOG_FLOOR)) for r in part if str(r.get("topology"))==t and math.isfinite(finite_float(r.get("best_value")))] for t in topologies]
        _box(ax,vals,methods,f"{bench} · D={d}",r"$\log_{10}(f_{best})$")
    for ax in list(axes.flat)[len(combos):]:ax.axis("off")
    fig.suptitle("Topology comparison: final objective distribution across paired seeds",y=1.005)
    fig.tight_layout();generated+=save_figure(fig,output,f"{stem}_final_quality_by_problem")

    # 2. Paired topology deltas relative to global best, stratified by function family.
    lookup={(str(r.get("benchmark")),int(finite_float(r.get("dimension"),-1)),int(finite_float(r.get("seed"),-1)),str(r.get("topology"))):r for r in runs}
    deltas=defaultdict(list)
    for r in runs:
        t=str(r.get("topology"));key=(str(r.get("benchmark")),int(finite_float(r.get("dimension"),-1)),int(finite_float(r.get("seed"),-1)))
        ref=lookup.get((*key,"global"))
        if ref and t!="global":
            a=finite_float(r.get("best_value"));b=finite_float(ref.get("best_value"))
            if math.isfinite(a) and math.isfinite(b):deltas[t].append(math.log10(max(a,LOG_FLOOR))-math.log10(max(b,LOG_FLOOR)))
    if deltas:
        labels=sorted(deltas);fig,ax=plt.subplots(figsize=(6.5,3.6));_box(ax,[deltas[k] for k in labels],labels,"Paired topology effects vs global",r"$\Delta\log_{10}(f_{best})$ (negative favours topology)")
        ax.axhline(0,color="#555555",ls="--",lw=.8);fig.tight_layout();generated+=save_figure(fig,output,f"{stem}_paired_effects_vs_global")

    # 3. Quality heatmap by problem family and topology.
    matrix=np.full((len(benchmarks),len(topologies)),np.nan)
    for i,b in enumerate(benchmarks):
        for j,t in enumerate(topologies):
            v=[math.log10(max(finite_float(r.get("best_value")),LOG_FLOOR)) for r in runs if str(r.get("benchmark"))==b and str(r.get("topology"))==t and math.isfinite(finite_float(r.get("best_value")))]
            if v:matrix[i,j]=np.median(v)
    fig,ax=plt.subplots(figsize=(max(5.5,.9*len(topologies)),max(3,.45*len(benchmarks))))
    im=ax.imshow(matrix,aspect="auto",cmap="viridis");ax.set_xticks(range(len(topologies)),topologies,rotation=25,ha="right");ax.set_yticks(range(len(benchmarks)),benchmarks)
    ax.set_title("Median final quality by function and topology");cb=fig.colorbar(im,ax=ax);cb.set_label(r"Median $\log_{10}(f_{best})$")
    fig.tight_layout();generated+=save_figure(fig,output,f"{stem}_quality_heatmap")

    # 4. Convergence trajectories with median and IQR.
    if convergence:
        by_run=defaultdict(list)
        for r in convergence:by_run[str(r.get("run_id"))].append(r)
        for bench,d in combos:
            fig,ax=plt.subplots(figsize=(6.3,3.8));found=False
            for i,t in enumerate(topologies):
                series=[]
                matching=[r for r in runs if str(r.get("benchmark"))==bench and int(finite_float(r.get("dimension"),-1))==d and str(r.get("topology"))==t]
                for run in matching:
                    trace=sorted(by_run.get(str(run.get("run_id")),[]),key=lambda x:finite_float(x.get("iteration")))
                    if trace:series.append([math.log10(max(finite_float(x.get("best_value")),LOG_FLOOR)) for x in trace])
                if not series:continue
                mat=np.asarray(series,dtype=float);m=np.nanmedian(mat,axis=0);q1=np.nanquantile(mat,.25,axis=0);q3=np.nanquantile(mat,.75,axis=0)
                x=np.arange(len(m));ax.plot(x,m,label=t,color=COLORS[i%len(COLORS)]);ax.fill_between(x,q1,q3,color=COLORS[i%len(COLORS)],alpha=.14)
                found=True
            if found:
                ax.set_xlabel("Iteration");ax.set_ylabel(r"Median $\log_{10}(f_{best})$ [IQR]");ax.set_title(f"Convergence by topology · {bench}, D={d}");ax.grid();ax.legend(frameon=False)
                fig.tight_layout();generated+=save_figure(fig,output,f"{stem}_convergence_{safe_slug(bench)}_d{d}")
    else:warnings.append("convergence.csv absent; no topology convergence figures created")

    # 5. Runtime, objective evaluation and setup/selection diagnostics.
    for field,label in (("elapsed_seconds","elapsed"),("evaluation_wall_seconds","evaluation"),("propagation_seconds","propagation"),("topology_setup_seconds","topology_setup"),("social_selection_seconds","social_selection")):
        if not any(math.isfinite(finite_float(r.get(field))) for r in runs):continue
        data=[[finite_float(r.get(field)) for r in runs if str(r.get("topology"))==t and finite_float(r.get(field))>0] for t in topologies]
        if not any(data):
            continue
        fig,ax=plt.subplots(figsize=(6.5,3.4));_box(ax,data,topologies,f"Topology timing: {label}","Seconds",log=True)
        fig.text(.5,.005,"Elapsed times are descriptive and hardware-specific; this experiment is not a speedup comparison.",ha="center",fontsize=6,color="#687481")
        fig.tight_layout(rect=(0,.04,1,1));generated+=save_figure(fig,output,f"{stem}_timing_{label}")
    # 6. Success rate, if a declared threshold was recorded.
    if any(r.get("success") not in (None,"") for r in runs):
        matrix=np.full((len(benchmarks),len(topologies)),np.nan)
        for i,b in enumerate(benchmarks):
            for j,t in enumerate(topologies):
                g=[r for r in runs if str(r.get("benchmark"))==b and str(r.get("topology"))==t]
                if g:matrix[i,j]=np.mean([1 if str(r.get("success")).lower() in {"1","true"} else 0 for r in g])
        fig,ax=plt.subplots(figsize=(max(5.5,.9*len(topologies)),max(3,.45*len(benchmarks))));im=ax.imshow(matrix,vmin=0,vmax=1,aspect="auto",cmap="viridis")
        ax.set_xticks(range(len(topologies)),topologies,rotation=25,ha="right");ax.set_yticks(range(len(benchmarks)),benchmarks);ax.set_title("Success rate by function and topology")
        cb=fig.colorbar(im,ax=ax);cb.set_label("Success fraction");fig.tight_layout();generated+=save_figure(fig,output,f"{stem}_success_rate")
    return generated,warnings


def generate(directory: Path, output: Path):
    configure_style();output.mkdir(parents=True,exist_ok=True)
    files=[];warnings=[]
    if (directory/"scaling_runs.csv").exists() or (directory/"scaling_summary.csv").exists():
        f,w=plot_scaling(directory,output);files+=f;warnings+=w
    if (directory/"runs.csv").exists() and (directory/"metadata.json").exists():
        f,w=plot_topology(directory,output);files+=f;warnings+=w
    if not files and not warnings:warnings.append("No recognized legacy result files were found")
    (output/"FIGURE_ATLAS.md").write_text("# Legacy experiment figure atlas\n\n"+"\n".join(f"- `{p.name}`" for p in files if p.suffix==".pdf")+"\n\n## Warnings\n\n"+"\n".join(f"- {w}" for w in warnings)+"\n",encoding="utf-8")
    return files,warnings


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument("--input",type=Path,required=True);ap.add_argument("--output",type=Path,default=None);args=ap.parse_args()
    inp=args.input.expanduser().resolve();out=args.output.expanduser().resolve() if args.output else inp/"figure_atlas"
    files,warnings=generate(inp,out);pdfs=[p for p in files if p.suffix==".pdf"]
    print(f"Generated {len(pdfs)} PDF and {len(files)-len(pdfs)} PNG figures from {inp}")
    for warning in warnings:print("WARNING:",warning)
    return 0 if pdfs else 2

if __name__=="__main__":raise SystemExit(main())
