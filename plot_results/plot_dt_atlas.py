#!/usr/bin/env python3
"""Generate the full figure set for a paired Delta-t sensitivity study."""
from __future__ import annotations

import argparse
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
    COLORS, LOG_FLOOR, add_evidence_note, bootstrap_median, configure_style,
    finite_float, interpolate_step_history, read_json, read_jsonl, save_figure,
)


def load_data(campaign: Path):
    config_path = campaign / "study_config.json"
    records_path = campaign / "runs.jsonl"
    config_obj = read_json(config_path)
    config = config_obj.get("study_config", config_obj.get("plan", {}))
    if not config:
        config = config_obj
    records = read_jsonl(records_path, tolerate_partial_tail=False)
    if not records:
        raise FileNotFoundError(f"No complete dt records in {records_path}")
    dt_values = sorted({finite_float(r.get("dt")) for r in records if math.isfinite(finite_float(r.get("dt")))})
    benchmarks = sorted({str(r.get("benchmark")) for r in records})
    dimensions = sorted({int(finite_float(r.get("dimension"), -1)) for r in records})
    seeds = sorted({int(finite_float(r.get("seed"), -1)) for r in records})
    ref = finite_float(records[0].get("reference_dt"), finite_float(config.get("reference_dt")))
    return config, records, dt_values, benchmarks, dimensions, seeds, ref


def build_lookup(records: list[dict[str, Any]]):
    lookup = {}
    for r in records:
        key = (str(r["benchmark"]), int(r["dimension"]), int(r["seed"]), float(r["dt"]))
        if key in lookup and lookup[key] != r:
            raise ValueError(f"Conflicting duplicate dt record for {key}")
        lookup[key] = r
    return lookup


def compute_effects(records, dt_values, benchmarks, dimensions, seeds, reference_dt):
    lookup = build_lookup(records)
    effects = {}
    for benchmark in benchmarks:
        for dim in dimensions:
            for seed in seeds:
                ref = lookup.get((benchmark, dim, seed, reference_dt))
                if ref is None:
                    continue
                refv = math.log10(max(finite_float(ref.get("best_value")), LOG_FLOOR))
                for dt in dt_values:
                    row = lookup.get((benchmark, dim, seed, dt))
                    if row is None:
                        continue
                    val = math.log10(max(finite_float(row.get("best_value")), LOG_FLOOR))
                    effects[(benchmark, dim, seed, dt)] = val - refv
    return lookup, effects


def _make_heatmap(matrix, xlabels, ylabels, title, cbar, out, name, *, cmap="coolwarm", center0=False):
    finite = np.asarray(matrix, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return []
    vmax = max(0.05, float(np.quantile(np.abs(finite), 0.95))) if center0 else None
    fig, ax = plt.subplots(figsize=(max(7, 0.48*len(xlabels)+2), max(3.4, 0.35*len(ylabels)+1.3)))
    image = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=-vmax if center0 else None, vmax=vmax)
    ax.set_xticks(np.arange(len(xlabels)), [f"{x:g}" for x in xlabels], rotation=35, ha="right")
    ax.set_yticks(np.arange(len(ylabels)), ylabels)
    ax.set_title(title)
    cb = fig.colorbar(image, ax=ax, pad=0.015); cb.set_label(cbar)
    fig.tight_layout()
    return save_figure(fig, out, name)


def plot_problem_heatmaps(records, effects, dt_values, benchmarks, dimensions, reference_dt, out):
    generated = []
    for dim in dimensions:
        matrix = np.full((len(benchmarks), len(dt_values)), np.nan)
        for i, benchmark in enumerate(benchmarks):
            for j, dt in enumerate(dt_values):
                vals = [effects[(benchmark, dim, seed, dt)] for seed in sorted({k[2] for k in effects if k[0] == benchmark and k[1] == dim and k[3] == dt})]
                if vals: matrix[i, j] = np.median(vals)
        generated += _make_heatmap(matrix, dt_values, [b.capitalize() for b in benchmarks],
                                   fr"Paired final-quality effect relative to reference $\Delta t={reference_dt:g}$ · D={dim}",
                                   r"Median paired $\Delta\log_{10}(f_{best})$ (negative favours tested $\Delta t$)",
                                   out, f"dt_paired_effect_heatmap_d{dim}", center0=True)
    return generated


def plot_convergence_surfaces(records, dt_values, benchmarks, dimensions, out):
    generated = []
    for dim in dimensions:
        for benchmark in benchmarks:
            panel = [r for r in records if r.get("benchmark") == benchmark and int(r.get("dimension", -1)) == dim]
            all_fe = [float(x) for r in panel for x in r.get("history_evaluations", [])]
            if not all_fe: continue
            lo, hi = min(all_fe), max(all_fe)
            grid = np.linspace(lo, hi, 140)
            mat = np.full((len(dt_values), len(grid)), np.nan)
            for i, dt in enumerate(dt_values):
                curves = []
                for r in panel:
                    if float(r.get("dt")) != dt: continue
                    interp = interpolate_step_history(r, grid)
                    if interp is not None: curves.append(np.log10(np.maximum(interp, LOG_FLOOR)))
                if curves: mat[i] = np.nanmedian(np.vstack(curves), axis=0)
            if not np.isfinite(mat).any(): continue
            fig, ax = plt.subplots(figsize=(6.8, 4.2))
            extent = [lo, hi, len(dt_values)-0.5, -0.5]
            image = ax.imshow(mat, aspect="auto", origin="upper", extent=extent, cmap="viridis")
            ax.set_yticks(np.arange(len(dt_values)), [f"{v:g}" for v in dt_values])
            ax.set_ylabel(r"$\Delta t$"); ax.set_xlabel("Function evaluations")
            ax.set_title(f"Convergence surface across time steps · {benchmark.capitalize()}, D={dim}")
            cb = fig.colorbar(image, ax=ax, pad=0.015); cb.set_label(r"Median $\log_{10}(f_{best})$ across paired seeds")
            fig.tight_layout(); generated += save_figure(fig, out, f"dt_convergence_surface_{benchmark}_d{dim}")
    return generated


def plot_dt_profiles(effects, dt_values, benchmarks, dimensions, reference_dt, out):
    # fixed benchmark set; resample paired seeds inside each family x dimension cell.
    family_effects = defaultdict(dict)
    seed_effects = defaultdict(list)
    for (benchmark, dim, seed, dt), val in effects.items():
        seed_effects[(benchmark, dim, dt)].append(val)
    for (benchmark, dim, dt), vals in seed_effects.items():
        family_effects[(dt, benchmark)][dim] = float(np.median(vals))
    medians, lows, highs = [], [], []
    rng = np.random.default_rng(20261107)
    for dt in dt_values:
        family_values = []
        for benchmark in benchmarks:
            dims = [family_effects[(dt, benchmark)][d] for d in dimensions if d in family_effects[(dt, benchmark)]]
            if dims: family_values.append(float(np.median(dims)))
        if not family_values:
            medians.append(np.nan); lows.append(np.nan); highs.append(np.nan); continue
        medians.append(float(np.median(family_values)))
        if len(family_values) > 1:
            # benchmark functions/dimensions are fixed design factors; paired seeds are the random component.
            boot = []
            for _ in range(3000):
                picked = []
                for benchmark in benchmarks:
                    dimension_values = []
                    for d in dimensions:
                        vals = [v for (b, dd, s, dtt), v in effects.items() if b == benchmark and dd == d and dtt == dt]
                        if vals:
                            sampled = rng.choice(np.asarray(vals), size=len(vals), replace=True)
                            dimension_values.append(float(np.median(sampled)))
                    if dimension_values: picked.append(float(np.median(dimension_values)))
                if picked: boot.append(float(np.median(picked)))
            lows.append(float(np.quantile(boot, .025))); highs.append(float(np.quantile(boot, .975)))
        else:
            lows.append(family_values[0]); highs.append(family_values[0])
    x = np.asarray(dt_values); y = np.asarray(medians); lo = np.asarray(lows); hi = np.asarray(highs)
    fig, ax = plt.subplots(figsize=(6.8, 3.8))
    ax.fill_between(x, lo, hi, color=COLORS[1], alpha=.15, linewidth=0, label="Seed bootstrap 95% interval")
    ax.plot(x, y, marker="o", ms=3.8, color=COLORS[0], label="Median across fixed benchmark set")
    ax.axhline(0, color="#666666", ls="--", lw=.8); ax.axvline(reference_dt, color=COLORS[3], ls=":", lw=1, label=f"Reference {reference_dt:g}")
    ax.set_xscale("log"); ax.set_xticks(x, [f"{v:g}" for v in x]); ax.set_xlabel(r"$\Delta t$ (log scale)")
    ax.set_ylabel(r"Paired $\Delta\log_{10}(f_{best})$"); ax.set_title("Aggregate sensitivity profile")
    ax.grid(axis="y"); ax.legend(frameon=False)
    fig.text(.5,.006,"Selected functions and dimensions are fixed design factors; interval reflects paired-seed variability only.",ha="center",fontsize=6,color="#687481")
    fig.tight_layout(rect=(0,.04,1,1)); return save_figure(fig,out,"dt_aggregate_paired_profile")


def plot_dt_distribution_and_runtime(records, dt_values, benchmarks, dimensions, out):
    generated=[]
    # final objective distributions, faceted by function and dimension
    for dim in dimensions:
        ncols=2; nrows=math.ceil(len(benchmarks)/ncols)
        fig,axes=plt.subplots(nrows,ncols,figsize=(7.0,2.25*nrows),squeeze=False)
        for ax,benchmark in zip(axes.flat,benchmarks):
            groups=[]
            for dt in dt_values:
                vals=[math.log10(max(finite_float(r.get("best_value")),LOG_FLOOR)) for r in records if r.get("benchmark")==benchmark and int(r.get("dimension",-1))==dim and float(r.get("dt"))==dt]
                groups.append(vals if vals else [np.nan])
            bp=ax.boxplot(groups,showfliers=False,widths=.6,patch_artist=True)
            for i,p in enumerate(bp["boxes"]): p.set_facecolor(COLORS[i%len(COLORS)]);p.set_alpha(.45)
            ax.set_xticks(np.arange(1,len(dt_values)+1),[f"{d:g}" for d in dt_values],rotation=45,ha="right",fontsize=5.6)
            ax.set_title(benchmark.capitalize());ax.set_ylabel(r"$\log_{10}(f_{best})$");ax.grid(axis="y")
        for ax in list(axes.flat)[len(benchmarks):]:ax.axis("off")
        fig.suptitle(fr"Final objective distribution across $\Delta t$ · D={dim}",y=1.005)
        fig.tight_layout();generated+=save_figure(fig,out,f"dt_final_distribution_d{dim}")
    # elapsed time vs dt; this is an implementation/runtime diagnostic, not objective quality
    fig,ax=plt.subplots(figsize=(6.8,3.7))
    for i,dim in enumerate(dimensions):
        med=[];q1=[];q3=[]
        for dt in dt_values:
            vals=[finite_float(r.get("elapsed_seconds")) for r in records if int(r.get("dimension",-1))==dim and float(r.get("dt"))==dt and math.isfinite(finite_float(r.get("elapsed_seconds")))]
            med.append(np.median(vals) if vals else np.nan);q1.append(np.quantile(vals,.25) if vals else np.nan);q3.append(np.quantile(vals,.75) if vals else np.nan)
        ax.plot(dt_values,med,marker="o",ms=3,label=f"D={dim}",color=COLORS[i])
        ax.fill_between(dt_values,q1,q3,color=COLORS[i],alpha=.12)
    ax.set_xscale("log");ax.set_xticks(dt_values,[f"{d:g}" for d in dt_values]);ax.set_xlabel(r"$\Delta t$")
    ax.set_ylabel("Median elapsed time (s; IQR)");ax.set_title("Observed runtime as a function of $\\Delta t$");ax.grid(axis="y");ax.legend(frameon=False)
    fig.tight_layout();generated+=save_figure(fig,out,"dt_runtime_profile")
    return generated


def plot_dt_win_loss(effects, dt_values, benchmarks, dimensions, reference_dt, out):
    rows=[]
    for dt in dt_values:
        if dt==reference_dt: continue
        values=[v for (b,d,s,t),v in effects.items() if t==dt]
        wins=sum(v < -1e-14 for v in values);losses=sum(v > 1e-14 for v in values);ties=len(values)-wins-losses
        if values: rows.append((dt,wins,ties,losses,len(values)))
    if not rows:return []
    fig,ax=plt.subplots(figsize=(7.0,3.6));left=np.zeros(len(rows));y=np.arange(len(rows))
    for j,(label,color) in enumerate((("Better", "#3A8D8C"),("Tie","#AAB3BC"),("Worse","#B65D58"))):
        vals=np.asarray([r[j+1] for r in rows],dtype=float)
        ax.barh(y,vals,left=left,color=color,label=label,edgecolor="white",linewidth=.35);left+=vals
    ax.set_yticks(y,[f"{r[0]:g}" for r in rows]);ax.set_ylabel(r"Tested $\Delta t$");ax.set_xlabel("Paired seed × function × dimension comparisons")
    ax.set_title(fr"Paired win/tie/loss counts against reference $\Delta t={reference_dt:g}$");ax.legend(frameon=False,ncol=3);ax.grid(axis="x")
    fig.text(.5,.006,"Counts are descriptive and nested; they are not independent inferential replicates.",ha="center",fontsize=6,color="#687481")
    fig.tight_layout(rect=(0,.04,1,1));return save_figure(fig,out,"dt_paired_win_tie_loss")


def generate(campaign: Path, output: Path):
    config,records,dt_values,benchmarks,dimensions,seeds,ref=load_data(campaign)
    _,effects=compute_effects(records,dt_values,benchmarks,dimensions,seeds,ref)
    configure_style();output.mkdir(parents=True,exist_ok=True)
    generated=[]
    generated+=plot_problem_heatmaps(records,effects,dt_values,benchmarks,dimensions,ref,output)
    generated+=plot_convergence_surfaces(records,dt_values,benchmarks,dimensions,output)
    generated+=plot_dt_profiles(effects,dt_values,benchmarks,dimensions,ref,output)
    generated+=plot_dt_distribution_and_runtime(records,dt_values,benchmarks,dimensions,output)
    generated+=plot_dt_win_loss(effects,dt_values,benchmarks,dimensions,ref,output)
    evidence=config.get("evidence_class",config.get("plan",{}).get("evidence_class","not specified"))
    warnings=[]
    expected=len(benchmarks)*len(dimensions)*len(seeds)*len(dt_values)
    if len(records)!=expected:warnings.append(f"Loaded {len(records)} records; config implies {expected}. Partial campaign: interpret plots accordingly.")
    files=[p.name for p in generated if p.suffix==".pdf"]
    note=[f"# Delta-t figure atlas","",f"Evidence class: **{evidence}**.",f"Records: **{len(records)} / {expected} expected**.",f"Reference dt: **{ref:g}**.","Effects are paired within benchmark × dimension × seed. Bootstrap intervals condition on the selected functions/dimensions and resample seeds only.","","## Generated figures",""]+[f"- `{x}`" for x in files]+["","## Warnings",""]+[f"- {x}" for x in warnings]
    (output/"FIGURE_ATLAS.md").write_text("\n".join(note)+"\n",encoding="utf-8")
    (output/"figure_atlas_manifest.json").write_text(json.dumps({"evidence_class":evidence,"n_records":len(records),"n_expected":expected,"dt_values":dt_values,"reference_dt":ref,"pdf_files":files,"warnings":warnings},indent=2),encoding="utf-8")
    return generated,warnings


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument("--campaign",type=Path,required=True);ap.add_argument("--output",type=Path,default=None);args=ap.parse_args()
    campaign=args.campaign.expanduser().resolve();out=args.output.expanduser().resolve() if args.output else Path(__file__).resolve().parents[1]/"results/figures/dt"/campaign.name
    from cpso_serial.output_policy import results_output
    out = results_output(out)
    generated,warnings=generate(campaign,out);pdfs=[p for p in generated if p.suffix==".pdf"]
    print(f"Generated {len(pdfs)} PDF and {len(generated)-len(pdfs)} PNG figures in {out}")
    for w in warnings:print("WARNING:",w)
    return 0 if pdfs else 2

if __name__=="__main__":raise SystemExit(main())
