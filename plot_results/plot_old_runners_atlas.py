#!/usr/bin/env python3
"""Plots for early master_slave_runs.csv and topology_runs.csv experiments."""
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

from plot_results.atlas_common import COLORS, LOG_FLOOR, configure_style, finite_float, save_figure, safe_slug


def read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open("r", newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def median_iqr(ax, groups, labels, title, ylabel):
    data=[g if g else [np.nan] for g in groups]
    bp=ax.boxplot(data,patch_artist=True,showfliers=False,medianprops={"color":"black"})
    for i,b in enumerate(bp["boxes"]):b.set_facecolor(COLORS[i%len(COLORS)]);b.set_alpha(.5)
    ax.set_xticks(np.arange(1,len(labels)+1),labels,fontsize=6)
    ax.set_title(title);ax.set_ylabel(ylabel);ax.grid(axis="y")


def _boot_mean(values, seed):
    x=np.asarray([v for v in values if math.isfinite(v)],dtype=float)
    if not x.size:return math.nan,math.nan,math.nan
    m=float(np.mean(x))
    if x.size==1:return m,m,m
    rng=np.random.default_rng(seed);b=rng.choice(x,size=(2500,x.size),replace=True).mean(axis=1)
    return m,float(np.quantile(b,.025)),float(np.quantile(b,.975))


def plot_master_slave(directory: Path, output: Path):
    rows=read_csv(directory/"master_slave_runs.csv")
    if not rows:return [],[]
    meta_path=directory/"master_slave_metadata.json"
    metadata=json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
    workers=sorted({int(finite_float(r.get("workers"),1)) for r in rows if r.get("mode")=="master_slave"})
    seeds=sorted({int(finite_float(r.get("seed"),-1)) for r in rows})
    lookup={(int(finite_float(r.get("seed"),-1)),str(r.get("mode")),int(finite_float(r.get("workers"),1))):r for r in rows}
    stem=safe_slug(directory.name);out=[];warnings=[]
    total={};batch={}
    for w in workers:
        total[w]=[];batch[w]=[]
        for seed in seeds:
            s=lookup.get((seed,"serial",1));p=lookup.get((seed,"master_slave",w))
            if not s or not p:continue
            st=finite_float(s.get("elapsed_seconds"));pt=finite_float(p.get("elapsed_seconds"))
            sb=finite_float(s.get("evaluation_wall_seconds"));pb=finite_float(p.get("evaluation_wall_seconds"))
            if st>0 and pt>0:total[w].append(st/pt)
            if sb>0 and pb>0:batch[w].append(sb/pb)
    # 1. Speedup distributions by worker count.
    fig,axes=plt.subplots(1,2,figsize=(8.2,3.4))
    median_iqr(axes[0],[total[w] for w in workers],[str(w) for w in workers],"End-to-end speedup","Paired serial / parallel time")
    median_iqr(axes[1],[batch[w] for w in workers],[str(w) for w in workers],"Evaluation-batch speedup","Paired serial / parallel time")
    for ax in axes:ax.axhline(1,color="#555555",ls="--",lw=.8);ax.set_xlabel("Worker count")
    fig.suptitle(f"{directory.name}: serial vs master–slave")
    fig.text(.5,.004,"Each distribution is paired by seed. Timings are specific to the machine recorded in the metadata.",ha="center",fontsize=6,color="#687481")
    fig.tight_layout(rect=(0,.04,1,.95));out+=save_figure(fig,output,f"{stem}_speedup_distributions")

    # 2. Runtime and throughput, with serial baseline.
    modes=[("serial",1,"Serial")]+[("master_slave",w,f"MS · {w}") for w in workers]
    fig,axes=plt.subplots(1,2,figsize=(8.2,3.4));x=np.arange(len(modes))
    rtimes=[];through=[]
    for mode,w,_ in modes:
        rs=[r for r in rows if r.get("mode")==mode and int(finite_float(r.get("workers"),1))==w]
        rtimes.append([finite_float(r.get("elapsed_seconds")) for r in rs if finite_float(r.get("elapsed_seconds"))>0])
        through.append([finite_float(r.get("evaluations_per_second")) for r in rs if finite_float(r.get("evaluations_per_second"))>0])
    for ax,data,title,ylabel in ((axes[0],rtimes,"Runtime by executor","Elapsed time (s)"),(axes[1],through,"Evaluation throughput","Function evaluations / second")):
        med=[float(np.median(g)) if g else math.nan for g in data]
        lo=[float(np.quantile(g,.25)) if g else math.nan for g in data];hi=[float(np.quantile(g,.75)) if g else math.nan for g in data]
        ax.errorbar(x,med,yerr=[np.maximum(0,np.asarray(med)-lo),np.maximum(0,np.asarray(hi)-med)],fmt="o",capsize=3,color=COLORS[1])
        ax.set_xticks(x,[m[2] for m in modes]);ax.set_ylabel(ylabel);ax.set_title(title);ax.grid(axis="y")
    fig.tight_layout();out+=save_figure(fig,output,f"{stem}_runtime_throughput")

    # 3. Final-value differences, with the limit of the stored CSV made explicit.
    fig,ax=plt.subplots(figsize=(5.8,3.3));deltas=[]
    for w in workers:
        vals=[]
        for seed in seeds:
            s=lookup.get((seed,"serial",1));p=lookup.get((seed,"master_slave",w))
            if s and p:
                a=finite_float(s.get("best_value"));b=finite_float(p.get("best_value"))
                if math.isfinite(a) and math.isfinite(b):vals.append(b-a)
        deltas.append(vals)
    median_iqr(ax,deltas,[str(w) for w in workers],"Final objective difference","Master–slave − serial")
    ax.axhline(0,color="#555555",ls="--",lw=.8);ax.set_xlabel("Worker count")
    fig.text(.5,.004,"The runner metadata records full-state equivalence; this CSV exposes only the final scalar objective.",ha="center",fontsize=6,color="#687481")
    fig.tight_layout(rect=(0,.04,1,1));out+=save_figure(fig,output,f"{stem}_final_objective_difference")
    if metadata.get("exact_output_equivalence_verified") is not True:
        warnings.append("Exact full-state output equivalence is not affirmed by master_slave_metadata.json.")
    return out,warnings


def plot_topology_runs(directory: Path, output: Path):
    rows=read_csv(directory/"topology_runs.csv")
    if not rows:return [],[]
    stem=safe_slug(directory.name);out=[];warnings=[]
    topologies=sorted({str(r.get("topology")) for r in rows})
    intervals=sorted({int(finite_float(r.get("sync_interval"),-1)) for r in rows})
    workers=sorted({int(finite_float(r.get("workers_requested"),-1)) for r in rows})
    # Histories are stored one JSON object per run using the run ID convention
    # validated by analysis/analyze_topology_results.py.
    histories=defaultdict(list)
    for row in rows:
        t=str(row.get("topology"));p=int(finite_float(row.get("workers_requested"),-1));tau=int(finite_float(row.get("sync_interval"),-1));rep=int(finite_float(row.get("repetition"),-1))
        path=directory/"runs"/f"{t}_p{p}_sync{tau}_run{rep:02d}.json"
        if path.is_file():
            try:
                hist=json.loads(path.read_text(encoding="utf-8")).get("history",[])
                if hist and all(math.isfinite(float(v)) for v in hist):histories[(t,p,tau)].append(np.asarray(hist,dtype=float))
            except (OSError,ValueError,json.JSONDecodeError):warnings.append(f"Could not read history {path}")
    # 1. Convergence at sync interval one.
    if 1 in intervals:
        for p in workers:
            fig,ax=plt.subplots(figsize=(6.4,3.7));found=False
            for i,t in enumerate(topologies):
                group=histories.get((t,p,1),[])
                if not group:continue
                lengths={len(v) for v in group}
                if len(lengths)>1:
                    warnings.append(f"Convergence histories for {t}, P={p}, sync=1 have unequal lengths; plotted on their common prefix.")
                length=min(lengths);mat=np.log10(np.maximum(np.vstack([v[:length] for v in group]),LOG_FLOOR))
                med=np.median(mat,axis=0);q1=np.quantile(mat,.25,axis=0);q3=np.quantile(mat,.75,axis=0)
                pop=int(finite_float(rows[0].get("population_size"),1));fe=np.arange(length)*pop+pop
                ax.plot(fe,med,color=COLORS[i%len(COLORS)],label=t);ax.fill_between(fe,q1,q3,color=COLORS[i%len(COLORS)],alpha=.14);found=True
            if found:
                ax.set_xlabel("Function evaluations");ax.set_ylabel(r"Median $\log_{10}(f_{best})$ [IQR]")
                ax.set_title(f"Island topology convergence · P={p}, sync interval=1");ax.grid();ax.legend(frameon=False)
                fig.tight_layout();out+=save_figure(fig,output,f"{stem}_convergence_p{p}_sync1")
            else:warnings.append(f"No convergence histories found for worker count {p}")
    # 2. Final quality distribution by topology at tau=1.
    selected_tau=1 if 1 in intervals else (intervals[0] if intervals else None)
    for p in workers:
        groups=[[finite_float(r.get("best_value")) for r in rows if str(r.get("topology"))==t and int(finite_float(r.get("workers_requested"),-1))==p and int(finite_float(r.get("sync_interval"),-1))==selected_tau] for t in topologies]
        fig,ax=plt.subplots(figsize=(6.1,3.6));median_iqr(ax,groups,topologies,f"Final quality by topology · P={p}, sync={selected_tau}","Final objective")
        fig.tight_layout();out+=save_figure(fig,output,f"{stem}_final_quality_p{p}_sync{selected_tau}")
    # 3. Final objective, elapsed time and coordination versus exchange interval.
    for field,label in (("best_value","Final objective"),("elapsed_seconds","Elapsed wall time (s)"),("coordination_seconds","Coordination time (s)")):
        fig,ax=plt.subplots(figsize=(6.5,3.8));found=False
        for i,t in enumerate(topologies):
            for p in workers:
                xs=[];means=[];lo=[];hi=[]
                for tau in intervals:
                    vals=[finite_float(r.get(field)) for r in rows if str(r.get("topology"))==t and int(finite_float(r.get("workers_requested"),-1))==p and int(finite_float(r.get("sync_interval"),-1))==tau and math.isfinite(finite_float(r.get(field)))]
                    if not vals:continue
                    m,l,h=_boot_mean(vals,1000+i+p+tau);xs.append(tau);means.append(m);lo.append(l);hi.append(h)
                if xs:
                    ax.plot(xs,means,marker="o",color=COLORS[i%len(COLORS)],linestyle="-" if p==workers[0] else "--",label=f"{t} · P={p}")
                    ax.fill_between(xs,lo,hi,color=COLORS[i%len(COLORS)],alpha=.1);found=True
        if found:
            if intervals and min(intervals)>0:ax.set_xscale("log")
            ax.set_xticks(intervals,[str(v) for v in intervals]);ax.set_xlabel("Exchange interval (generations)");ax.set_ylabel(label);ax.set_title(f"{label} versus exchange interval")
            ax.grid();ax.legend(frameon=False,ncol=2,fontsize=6)
            if field!="best_value":fig.text(.5,.004,"Timing is hardware-specific; coordination is in-process work, not network latency.",ha="center",fontsize=6,color="#687481")
            fig.tight_layout(rect=(0,.035,1,1));out+=save_figure(fig,output,f"{stem}_{field}_vs_exchange_interval")
    # 4. Coordination fraction and Holm-adjusted paired test matrix.
    fig,ax=plt.subplots(figsize=(6.4,3.6));found=False
    for i,t in enumerate(topologies):
        for p in workers:
            xs=[];means=[];lo=[];hi=[]
            for tau in intervals:
                vals=[finite_float(r.get("coordination_seconds"))/finite_float(r.get("elapsed_seconds")) for r in rows if str(r.get("topology"))==t and int(finite_float(r.get("workers_requested"),-1))==p and int(finite_float(r.get("sync_interval"),-1))==tau and finite_float(r.get("elapsed_seconds"))>0]
                if vals:
                    m,l,h=_boot_mean(vals,2000+i+p+tau);xs.append(tau);means.append(m);lo.append(l);hi.append(h)
            if xs:
                ax.plot(xs,means,marker="o",color=COLORS[i%len(COLORS)],linestyle="-" if p==workers[0] else "--",label=f"{t} · P={p}");ax.fill_between(xs,lo,hi,color=COLORS[i%len(COLORS)],alpha=.1);found=True
    if found:
        ax.set_xticks(intervals,[str(v) for v in intervals]);ax.set_xlabel("Exchange interval");ax.set_ylabel("Coordination / elapsed time");ax.set_title("Coordination fraction");ax.grid();ax.legend(frameon=False,ncol=2,fontsize=6)
        fig.tight_layout();out+=save_figure(fig,output,f"{stem}_coordination_fraction")
    tests=read_csv(directory/"paired_topology_tests.csv")
    if tests:
        for p in workers:
            ptests=[r for r in tests if int(finite_float(r.get("workers"),-1))==p]
            if not ptests:
                continue
            pairs=sorted({f"{r.get('topology_A')} vs {r.get('topology_B')}" for r in ptests})
            taus=sorted({int(finite_float(r.get("sync_interval"),-1)) for r in ptests})
            matrix=np.full((len(pairs),len(taus)),np.nan)
            for i,pair in enumerate(pairs):
                for j,tau in enumerate(taus):
                    vals=[finite_float(r.get("p_value_holm_global")) for r in ptests
                          if f"{r.get('topology_A')} vs {r.get('topology_B')}"==pair
                          and int(finite_float(r.get("sync_interval"),-1))==tau]
                    if len(vals)>1:
                        warnings.append(f"Multiple Holm p-values for {pair}, P={p}, sync={tau}; omitted ambiguous cell.")
                    elif vals:
                        matrix[i,j]=vals[0]
            fig,ax=plt.subplots(figsize=(max(5.5,.7*len(taus)),3.2));im=ax.imshow(-np.log10(np.maximum(matrix,1e-16)),aspect="auto",cmap="viridis")
            ax.set_xticks(range(len(taus)),[str(t) for t in taus]);ax.set_yticks(range(len(pairs)),pairs);ax.set_xlabel("Exchange interval");ax.set_title(f"Holm-adjusted paired-test p-values · P={p}")
            fig.colorbar(im,ax=ax,label=r"$-\log_{10}(p_{Holm})$");fig.tight_layout();out+=save_figure(fig,output,f"{stem}_holm_pvalue_heatmap_p{p}")
    return out,warnings


def generate(directory: Path, output: Path):
    configure_style();output.mkdir(parents=True,exist_ok=True);files=[];warnings=[]
    f,w=plot_master_slave(directory,output);files+=f;warnings+=w
    f,w=plot_topology_runs(directory,output);files+=f;warnings+=w
    if not files and not warnings:warnings.append("No old-runner result files found")
    return files,warnings


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument("--input",type=Path,required=True);ap.add_argument("--output",type=Path,default=None);a=ap.parse_args()
    src=a.input.expanduser().resolve();dst=a.output.expanduser().resolve() if a.output else src/"figure_atlas"
    files,warnings=generate(src,dst);pdfs=[x for x in files if x.suffix==".pdf"]
    print(f"Generated {len(pdfs)} PDF figures from {src}")
    for warning in warnings:print("WARNING:",warning)
    return 0 if pdfs else 2

if __name__=="__main__":raise SystemExit(main())
