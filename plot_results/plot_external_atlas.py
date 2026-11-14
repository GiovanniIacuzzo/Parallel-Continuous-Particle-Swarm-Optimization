#!/usr/bin/env python3
"""Create comprehensive descriptive figures for COCO BBOB and CEC 2013 LSGO.

For BBOB, official ECDF/runtime-to-target plots remain the responsibility of
COCO's `cocopp`; this module creates supplementary paired-rank diagnostics.
For CEC, checkpoint trajectories are available directly in recorded JSONL.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plot_results.atlas_common import (
    COLORS, configure_style, finite_float, load_external_records, read_json,
    safe_slug, save_figure,
)


def _function_id(row: dict[str, Any]) -> str:
    value = row.get("function_id")
    if value not in (None, ""):
        return str(value)
    text = str(row.get("problem_id", ""))
    match = re.search(r"(?:f|function)[_-]?0*(\d+)", text, re.I)
    return match.group(1) if match else text.split("_")[0]


def _average_ranks(values: list[float]) -> list[float]:
    arr = np.asarray(values, dtype=float)
    out = np.full(arr.shape, np.nan)
    valid = np.isfinite(arr)
    inds = np.where(valid)[0]
    if not len(inds): return out.tolist()
    order = inds[np.argsort(arr[inds], kind="stable")]
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and math.isclose(arr[order[j]], arr[order[i]], rel_tol=1e-12, abs_tol=1e-15): j += 1
        rank = ((i + 1) + j) / 2
        out[order[i:j]] = rank
        i = j
    return out.tolist()


def _paired_ranks(records: list[dict[str, Any]], methods: list[str], metric_getter):
    groups: dict[tuple[str, int], dict[str, float]] = defaultdict(dict)
    for row in records:
        problem = str(row.get("problem_id", "")); seed = int(finite_float(row.get("seed"), -1)); method = str(row.get("method", ""))
        if not problem or not method: continue
        value = metric_getter(row)
        if math.isfinite(value): groups[(problem, seed)][method] = value
    rank_records=[]; incomplete=0
    for (problem,seed), values in sorted(groups.items()):
        if any(m not in values for m in methods):
            incomplete+=1;continue
        ranks=_average_ranks([values[m] for m in methods])
        rank_records.append({"problem_id":problem,"seed":seed,"function_id":_function_id({"problem_id":problem}),
                             **{m:r for m,r in zip(methods,ranks)}})
    return rank_records,incomplete


def _rank_matrix(rank_records, methods, group_key):
    families=sorted({str(r.get(group_key,"")) for r in rank_records})
    matrix=np.full((len(families),len(methods)),np.nan)
    for i,fam in enumerate(families):
        for j,m in enumerate(methods):
            vals=[finite_float(r.get(m)) for r in rank_records if str(r.get(group_key,""))==fam and math.isfinite(finite_float(r.get(m)))]
            if vals: matrix[i,j]=float(np.mean(vals))
    return families,matrix


def _plot_rank_atlas(rank_records, methods, output, prefix, title_prefix):
    if not rank_records:return []
    generated=[]
    families,matrix=_rank_matrix(rank_records,methods,"function_id")
    fig,ax=plt.subplots(figsize=(max(6.5,.72*len(methods)),max(3.3,.38*len(families))))
    image=ax.imshow(matrix,aspect="auto",cmap="viridis_r")
    ax.set_xticks(np.arange(len(methods)),[m.replace("_","\n") for m in methods],fontsize=6)
    ax.set_yticks(np.arange(len(families)),[f"F{f}" for f in families])
    ax.set_title(f"{title_prefix}: mean paired rank by function (lower is better)")
    cb=fig.colorbar(image,ax=ax,pad=.015);cb.set_label("Mean rank")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if math.isfinite(matrix[i,j]): ax.text(j,i,f"{matrix[i,j]:.1f}",ha="center",va="center",fontsize=5.7,color="white")
    fig.tight_layout();generated+=save_figure(fig,output,f"{prefix}_rank_heatmap_by_function")
    # Rank distributions per method across paired problem-seed blocks.
    groups=[[finite_float(r.get(m)) for r in rank_records if math.isfinite(finite_float(r.get(m)))] for m in methods]
    fig,ax=plt.subplots(figsize=(max(6.5,.66*len(methods)),3.5))
    bp=ax.boxplot(groups,patch_artist=True,showfliers=False,medianprops={"color":"black","linewidth":1})
    for i,p in enumerate(bp["boxes"]):p.set_facecolor(COLORS[i%len(COLORS)]);p.set_alpha(.5)
    ax.set_xticks(np.arange(1,len(methods)+1),[m.replace("_","\n") for m in methods],fontsize=5.7)
    ax.set_ylabel("Within-problem rank");ax.set_title(f"{title_prefix}: distribution of paired problem ranks")
    ax.grid(axis="y");fig.tight_layout();generated+=save_figure(fig,output,f"{prefix}_rank_distribution")
    # Win/tie/loss against CPSO where the exact method label is present.
    cpso=next((m for m in methods if m.lower() in {"cpso","cpso_serial","cpso_native"} or "cpso" in m.lower()),None)
    if cpso:
        others=[m for m in methods if m!=cpso]
        wins=[];ties=[];losses=[]
        for m in others:
            w=t=l=0
            for r in rank_records:
                a=finite_float(r.get(cpso));b=finite_float(r.get(m))
                if not(math.isfinite(a) and math.isfinite(b)):continue
                if a<b:w+=1
                elif a>b:l+=1
                else:t+=1
            wins.append(w);ties.append(t);losses.append(l)
        fig,ax=plt.subplots(figsize=(7,max(2.8,.34*len(others)+.8)));y=np.arange(len(others));left=np.zeros(len(others))
        for vals,label,color in ((wins,"CPSO wins","#3A8D8C"),(ties,"Ties","#AAB3BC"),(losses,"Opponent wins","#B65D58")):
            ax.barh(y,vals,left=left,label=label,color=color,edgecolor="white",linewidth=.3);left+=np.asarray(vals)
        ax.set_yticks(y,others);ax.invert_yaxis();ax.set_xlabel("Matched problem-seed blocks");ax.set_title(f"{title_prefix}: CPSO paired rank win/tie/loss");ax.legend(frameon=False,ncol=3);ax.grid(axis="x")
        fig.tight_layout();generated+=save_figure(fig,output,f"{prefix}_cpso_win_tie_loss")
    return generated


def plot_cec(records, manifest, output):
    methods=manifest.get("methods",sorted({str(r.get("method")) for r in records}))
    methods=[str(m) for m in methods]
    milestones=manifest.get("milestones",[])
    if not milestones:
        milestones=sorted({int(k) for r in records for k in (r.get("checkpoints",{}) or {}).keys() if str(k).isdigit()})
    if not milestones: raise ValueError("CEC records/manifest do not expose FE checkpoints")
    generated=[]
    all_rank_records_by_fe={}
    for fe in milestones:
        def getvalue(row):
            checkpoints=row.get("checkpoints",{}) or {}
            return finite_float(checkpoints.get(str(fe)))
        rank_records,incomplete=_paired_ranks(records,methods,getvalue)
        if rank_records:
            all_rank_records_by_fe[int(fe)]=rank_records
            generated+=_plot_rank_atlas(rank_records,methods,output,f"cec_rank_fe{int(fe)}",f"CEC 2013 LSGO at {int(fe):,} FE")
    # Mean paired rank as a function of FE checkpoint.
    if all_rank_records_by_fe:
        fig,ax=plt.subplots(figsize=(6.8,3.8))
        for i,m in enumerate(methods):
            xs=[];ys=[]
            for fe,rr in sorted(all_rank_records_by_fe.items()):
                vals=[finite_float(r.get(m)) for r in rr if math.isfinite(finite_float(r.get(m)))]
                if vals:xs.append(fe);ys.append(np.mean(vals))
            if xs:ax.plot(xs,ys,marker="o",color=COLORS[i%len(COLORS)],label=m.replace("_"," "))
        ax.set_xscale("log");ax.set_xticks(sorted(all_rank_records_by_fe),[f"{x:,}" for x in sorted(all_rank_records_by_fe)])
        ax.set_xlabel("Function evaluations");ax.set_ylabel("Mean within-problem rank (lower is better)")
        ax.set_title("CEC anytime ranking across official FE checkpoints");ax.grid(axis="y");ax.legend(frameon=False,ncol=2)
        fig.tight_layout();generated+=save_figure(fig,output,"cec_anytime_mean_rank")
    # Final endpoint and runtime distributions: use ranks for cross-function quality, seconds only descriptive.
    if records:
        runs=[[finite_float(r.get("elapsed_seconds")) for r in records if str(r.get("method"))==m and finite_float(r.get("elapsed_seconds"))>0] for m in methods]
        if any(runs):
            fig,ax=plt.subplots(figsize=(max(6,.7*len(methods)),3.3));bp=ax.boxplot(runs,patch_artist=True,showfliers=False)
            for i,p in enumerate(bp["boxes"]):p.set_facecolor(COLORS[i%len(COLORS)]);p.set_alpha(.5)
            ax.set_xticks(np.arange(1,len(methods)+1),[m.replace("_","\n") for m in methods],fontsize=6)
            ax.set_yscale("log");ax.set_ylabel("Elapsed time (s; log scale)");ax.set_title("CEC execution time distributions (machine-specific)");ax.grid(axis="y")
            fig.tight_layout();generated+=save_figure(fig,output,"cec_runtime_distribution")
    return generated


def plot_bbob(records, manifest, output):
    methods=manifest.get("methods",sorted({str(r.get("method")) for r in records}))
    methods=[str(m) for m in methods]
    generated=[]
    rank_records,incomplete=_paired_ranks(records,methods,lambda row:finite_float(row.get("best_value")))
    if rank_records:
        generated+=_plot_rank_atlas(rank_records,methods,output,"bbob_final","COCO BBOB final objective")
        # Dimension-specific function ranks: separate rank blocks by dimension for interpretation.
        dims=sorted({int(finite_float(r.get("dimension"),-1)) for r in records})
        functions=sorted({_function_id(r) for r in records})
        matrix=np.full((len(functions),len(methods)),np.nan)
        for i,f in enumerate(functions):
            for j,m in enumerate(methods):
                vals=[finite_float(r.get(m)) for r in rank_records if str(r.get("function_id"))==f and math.isfinite(finite_float(r.get(m)))]
                if vals:matrix[i,j]=np.mean(vals)
        fig,ax=plt.subplots(figsize=(max(6.5,.72*len(methods)),max(3.3,.38*len(functions))))
        image=ax.imshow(matrix,aspect="auto",cmap="viridis_r");ax.set_xticks(np.arange(len(methods)),[m.replace("_","\n") for m in methods],fontsize=6)
        ax.set_yticks(np.arange(len(functions)),[f"F{f}" for f in functions]);ax.set_title("BBOB average rank by function across selected instances and dimensions")
        cb=fig.colorbar(image,ax=ax,pad=.015);cb.set_label("Mean rank");fig.tight_layout();generated+=save_figure(fig,output,"bbob_rank_by_function")
    # Dimension-specific paired rank panels.
    for dim in sorted({int(finite_float(r.get("dimension"),-1)) for r in records}):
        sub=[r for r in records if int(finite_float(r.get("dimension"),-1))==dim]
        rr,_=_paired_ranks(sub,methods,lambda row:finite_float(row.get("best_value")))
        if rr: generated+=_plot_rank_atlas(rr,methods,output,f"bbob_d{dim}",f"BBOB D={dim}")
    # Runtime is reported descriptively: external objective callbacks are serial in this protocol.
    runtimes=[[finite_float(r.get("elapsed_seconds")) for r in records if str(r.get("method"))==m and finite_float(r.get("elapsed_seconds"))>0] for m in methods]
    if any(runtimes):
        fig,ax=plt.subplots(figsize=(max(6,.7*len(methods)),3.4));bp=ax.boxplot(runtimes,patch_artist=True,showfliers=False)
        for i,p in enumerate(bp["boxes"]):p.set_facecolor(COLORS[i%len(COLORS)]);p.set_alpha(.5)
        ax.set_xticks(np.arange(1,len(methods)+1),[m.replace("_","\n") for m in methods],fontsize=6)
        ax.set_yscale("log");ax.set_ylabel("Elapsed time (s; log scale)");ax.set_title("BBOB run-time distributions (machine-specific)");ax.grid(axis="y")
        fig.tight_layout();generated+=save_figure(fig,output,"bbob_runtime_distribution")
    return generated




def generate(campaign: Path, output: Path):
    manifest=read_json(campaign/"campaign_manifest.json");records=load_external_records(campaign)
    if not records:raise FileNotFoundError(f"No external records found under {campaign}")
    suite=str(manifest.get("suite",records[0].get("suite",""))).lower()
    configure_style();output.mkdir(parents=True,exist_ok=True)
    if "cec" in suite:
        generated=plot_cec(records,manifest,output);title="CEC 2013 LSGO"
    elif "bbob" in suite:
        generated=plot_bbob(records,manifest,output);title=suite.upper()
        coco_warnings=["Official cocopp processing is manual; the atlas does not generate HTML/JS outputs."]
    else:
        raise ValueError(f"Unsupported external suite {suite!r}")
    expected=finite_float(manifest.get("expected_run_count",manifest.get("expected_method_runs")),math.nan)
    warnings=[]
    if "bbob" in suite:
        warnings.extend(coco_warnings)
    if math.isfinite(expected) and len(records)!=int(expected):warnings.append(f"Records loaded {len(records)} / expected {int(expected)}; incomplete campaign")
    notes=[f"# Figure atlas — {title}","",f"Campaign: `{campaign.name}`",f"Records loaded: **{len(records)}**.",
           "All rank comparisons use complete matched problem × seed groups; incomplete groups are excluded and counted.",
           "For COCO BBOB, Observer-based ECDF/runtime-to-target figures are generated with cocopp when genuine .info files and cocopp are available. JSONL-based ranks remain supplementary descriptive diagnostics.",
           "","## Generated figures",""]+[f"- `{p.name}`" for p in generated if p.suffix==".pdf"]+["","## Warnings",""]+[f"- {w}" for w in warnings]
    (output/"FIGURE_ATLAS.md").write_text("\n".join(notes)+"\n",encoding="utf-8")
    (output/"figure_atlas_manifest.json").write_text(json.dumps({"suite":suite,"n_records":len(records),"expected_run_count":expected,"pdf_files":[p.name for p in generated if p.suffix==".pdf"],"warnings":warnings},indent=2),encoding="utf-8")
    return generated,warnings


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument("--campaign",type=Path,required=True);ap.add_argument("--output",type=Path,default=None);args=ap.parse_args()
    campaign=args.campaign.expanduser().resolve();out=args.output.expanduser().resolve() if args.output else Path(__file__).resolve().parents[1]/"results/figures/external"/campaign.name
    from cpso_serial.output_policy import results_output
    out = results_output(out)
    files,warnings=generate(campaign,out);pdfs=[p for p in files if p.suffix==".pdf"]
    print(f"Generated {len(pdfs)} PDF and {len(files)-len(pdfs)} PNG figures in {out}")
    for w in warnings:print("WARNING:",w)
    return 0 if pdfs else 2

if __name__=="__main__":raise SystemExit(main())
