#!/usr/bin/env python3
"""Paired statistical analysis of CPSO dt campaigns, with fixed-design checks.

The summary is conditional on the chosen benchmark families, D, population and
FE budget. A selected best dt is not an established continuous-domain optimum.

Requires existing analysis/analyze_dt_sensitivity.py from the repository.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from analysis.analyze_dt_sensitivity import load_and_validate  # noqa: E402

BENCHMARKS = {"sphere", "rastrigin", "rosenbrock", "ackley", "griewank", "salomon"}
BOUNDS = {
    "sphere": (-100,100), "rastrigin": (-5.12,5.12), "rosenbrock": (-5,10),
    "ackley": (-32.768,32.768), "griewank": (-600,600), "salomon": (-100,100),
}
FLOOR = 1e-14


def objective(name: str, x: np.ndarray) -> float:
    """Independent scalar benchmark evaluation (used to audit native outputs)."""
    if name == "sphere":
        return math.fsum(float(v)**2 for v in x)
    if name == "rastrigin":
        return 10*len(x) + math.fsum(float(v)**2-10*math.cos(2*math.pi*float(v)) for v in x)
    if name == "rosenbrock":
        return math.fsum(100*(float(b)-float(a)**2)**2+(1-float(a))**2 for a,b in zip(x[:-1],x[1:]))
    if name == "ackley":
        d=len(x)
        return -20*math.exp(-.2*math.sqrt(math.fsum(float(v)**2 for v in x)/d))-math.exp(math.fsum(math.cos(2*math.pi*float(v)) for v in x)/d)+20+math.e
    if name == "griewank":
        prod=1.0
        for i,v in enumerate(x):
            prod*=math.cos(float(v)/math.sqrt(i+1))
        return math.fsum(float(v)**2 for v in x)/4000.0-prod+1.0
    if name == "salomon":
        r=math.sqrt(math.fsum(float(v)**2 for v in x))
        return 1.0-math.cos(2*math.pi*r)+0.1*r
    raise ValueError(name)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError("Empty rows")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer=csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def assemble(input_dir: Path):
    config, records, validation=load_and_validate(input_dir)
    bs=validation["benchmarks"]
    dims=validation["dimensions"]
    seeds=validation["seeds"]
    dts=validation["dt_values"]
    if set(bs) - BENCHMARKS:
        raise ValueError("Unknown native objective cannot be audited independently")
    if len(seeds)<5:
        raise ValueError("Insufficient seeds for this analysis")
    if config.get("evidence_class")=="confirmatory" and len(seeds)<20:
        raise ValueError("Held-out confirmation needs at least 20 independent seeds")
    cells=[(b,d) for b in bs for d in dims]
    # log10 residual fractions: cell x paired seed x dt
    log_rho=np.empty((len(cells),len(seeds),len(dts)),dtype=float)
    progress_auc=np.empty_like(log_rho)
    attain_90=np.empty_like(log_rho)
    attain_99=np.empty_like(log_rho)
    lookup={(r["benchmark"],r["dimension"],r["seed"],float(r["dt"])): r for r in records}
    initial_by_seed={}
    for ci,(b,d) in enumerate(cells):
        lo,hi=BOUNDS[b]
        for si,seed in enumerate(seeds):
            for ti,dt in enumerate(dts):
                r=lookup[(b,d,seed,dt)]
                x=np.asarray(r["best_position"],dtype=float)
                if x.shape!=(d,) or not np.all(np.isfinite(x)) or np.any(x<lo-1e-12) or np.any(x>hi+1e-12):
                    raise ValueError(f"Invalid final position: {b},D={d},seed={seed},dt={dt}")
                recalculated=objective(b,x)
                saved=float(r["best_value"])
                if not math.isclose(recalculated,saved,rel_tol=3e-10,abs_tol=3e-9):
                    raise ValueError(f"Independent objective mismatch: {b}, D={d}, seed={seed}, dt={dt}: {saved} vs {recalculated}")
                history=np.asarray(r["best_history"],dtype=float)
                initial=float(history[0])
                # Paired experimental design must start from the *same* initial
                # best across dt for every fixed benchmark/dimension/seed.
                init_key=(b,d,seed)
                if init_key in initial_by_seed:
                    if not math.isclose(initial,initial_by_seed[init_key],rel_tol=1e-13,abs_tol=1e-11):
                        raise ValueError(f"Initial state not paired across dt: {init_key}")
                else:
                    initial_by_seed[init_key]=initial
                if initial<=FLOOR:
                    raise ValueError(f"Degenerate initial gap: {init_key}")
                if saved < -1e-10:
                    raise ValueError(f"Unexpected objective below known optimum: {b} {saved}")
                # No value may exceed 1 because best-so-far <= initial.
                fractions=np.clip(np.maximum(history,0.0)/initial, 1e-16,1.0)
                log_rho[ci,si,ti]=math.log10(float(fractions[-1]))
                progress_auc[ci,si,ti]=float(np.mean(np.log10(fractions)))
                attain_90[ci,si,ti]=float(fractions[-1]<=0.1)
                attain_99[ci,si,ti]=float(fractions[-1]<=0.01)
    return config, validation, cells, log_rho, progress_auc, attain_90, attain_99


def summarize(input_dir: Path, output_dir: Path, *, reps: int=5000, bootstrap_seed: int=28743):
    config, val, cells, z, auc, a90, a99=assemble(input_dir)
    dts=np.asarray(val["dt_values"],dtype=float)
    seeds=val["seeds"]
    nseed=len(seeds)
    # Every function-by-dimension cell has exactly the same weight; the same
    # paired seed identifiers are jointly resampled across all 12 cells.
    by_seed=z.mean(axis=0)  # n_seeds x n_dt
    score=by_seed.mean(axis=0)
    rng=np.random.default_rng(bootstrap_seed)
    ix=rng.integers(0,nseed,size=(reps,nseed))
    sampled=by_seed[ix].mean(axis=1)  # reps x n_dt
    lower=np.quantile(sampled,.025,axis=0)
    upper=np.quantile(sampled,.975,axis=0)
    # Both selection frequencies and differences are exploratory for the
    # selection campaign: not a corrected post-selection p-value.
    winners=np.argmin(sampled,axis=1)
    selection_frequency=np.bincount(winners,minlength=len(dts))/reps
    best_i=int(np.argmin(score))
    ref_i=int(np.where(dts==float(config["reference_dt"]))[0][0])
    paired_delta=by_seed-by_seed[:,[ref_i]]
    ref_boot=(by_seed[ix]-by_seed[ix][:,:,[ref_i]]).mean(axis=1)
    cell_mean=z.mean(axis=1)  # n_cells x n_dt
    # Mean within-cell rank (smaller = better), used as a robustness diagnostic.
    ranks=np.empty_like(cell_mean)
    for ci in range(len(cells)):
        order=np.argsort(cell_mean[ci],kind="stable")
        ranks[ci,order]=np.arange(1,len(dts)+1)
    mean_ranks=ranks.mean(axis=0)
    frozen=config.get("frozen_dt")
    if frozen is not None:
        frozen=float(frozen)
        if frozen not in dts:
            raise ValueError("frozen_dt not present in holdout candidate set")
        champion_i=int(np.where(dts==frozen)[0][0])
        report_status="HOLDOUT: frozen candidate, never retuned on holdout scores"
    else:
        champion_i=best_i
        report_status="SELECTION: provisional winner, independent holdout still required"
    out=output_dir
    out.mkdir(parents=True,exist_ok=True)
    rows=[]
    for i,dt in enumerate(dts):
        rows.append({
            "dt":f"{dt:.12g}",
            "equal_cell_mean_log10_residual":float(score[i]),
            "bootstrap_95_low":float(lower[i]),
            "bootstrap_95_high":float(upper[i]),
            "mean_problem_rank":float(mean_ranks[i]),
            "selection_bootstrap_frequency":float(selection_frequency[i]),
            "paired_mean_delta_log10_vs_reference":float(paired_delta[:,i].mean()),
            "paired_delta_vs_reference_95_low":float(np.quantile(ref_boot[:,i],.025)),
            "paired_delta_vs_reference_95_high":float(np.quantile(ref_boot[:,i],.975)),
            "mean_fraction_90pct_improvement":float(a90[:,:,i].mean()),
            "mean_fraction_99pct_improvement":float(a99[:,:,i].mean()),
            "mean_anytime_log10_residual":float(auc[:,:,i].mean()),
        })
    write_csv(out/"dt_statistical_ranking.csv",rows)
    cell_rows=[]
    for ci,(b,d) in enumerate(cells):
        for i,dt in enumerate(dts):
            rr=z[ci,:,i]
            cell_rows.append({
                "benchmark":b,"dimension":d,"dt":f"{dt:.12g}",
                "mean_log10_residual":float(rr.mean()),
                "median_log10_residual":float(np.median(rr)),
                "q25_log10_residual":float(np.quantile(rr,.25)),
                "q75_log10_residual":float(np.quantile(rr,.75)),
                "fraction_90pct_improvement":float(a90[ci,:,i].mean()),
                "fraction_99pct_improvement":float(a99[ci,:,i].mean()),
            })
    write_csv(out/"dt_statistical_by_problem.csv",cell_rows)
    if frozen is not None:
        # Valid only for a precommitted candidate on an independent holdout.
        # All comparisons are paired across the same fresh seed identifiers.
        from scipy.stats import wilcoxon
        comparisons=[]
        for j,dt in enumerate(dts):
            if j==champion_i:
                continue
            differences=by_seed[:,champion_i]-by_seed[:,j]
            pvalue=1.0 if np.allclose(differences,0.0,atol=1e-14,rtol=0) else float(wilcoxon(differences,alternative="two-sided",method="auto").pvalue)
            boot_differences=(by_seed[ix][:,:,champion_i]-by_seed[ix][:,:,j]).mean(axis=1)
            comparisons.append({
                "frozen_dt":frozen,"competitor_dt":float(dt),
                "mean_delta_frozen_minus_competitor":float(differences.mean()),
                "bootstrap_95_low":float(np.quantile(boot_differences,.025)),
                "bootstrap_95_high":float(np.quantile(boot_differences,.975)),
                "wilcoxon_two_sided_p_unadjusted":pvalue,
                "holm_adjusted_p":None,
            })
        # Holm step-down: family-wise error control over all holdout contrasts.
        indexed=sorted(enumerate(comparisons),key=lambda t:t[1]["wilcoxon_two_sided_p_unadjusted"])
        previous=0.0
        m=len(indexed)
        for rank,(original,row) in enumerate(indexed):
            previous=max(previous,min(1.0,(m-rank)*row["wilcoxon_two_sided_p_unadjusted"]))
            comparisons[original]["holm_adjusted_p"]=previous
        write_csv(out/"dt_holdout_paired_comparisons.csv",comparisons)
    ranked=sorted(range(len(dts)),key=lambda i:(float(score[i]),float(mean_ranks[i]),float(dts[i])))
    summary={
        "campaign_id":val["campaign_id"],"status":report_status,
        "n_runs":int(z.size),"n_seed_clusters":nseed,
        "n_problem_dimension_cells":len(cells),"n_dt_candidates":len(dts),
        "primary_metric":"equal-weight mean of per-run log10((f_B-f*)/(f_0-f*)) across fixed problem/dimension cells and paired seeds; min preferred",
        "bootstrap":"resample complete paired seed identifiers jointly across all function/dimension cells, 95 percentile intervals; fixed benchmark cells",
        "bootstrap_replicates":reps,
        "empirical_best_dt":float(dts[best_i]),
        "frozen_dt_if_holdout":frozen,
        "candidate_for_validation_dt":float(dts[champion_i]),
        "ranked_dt_candidates":[float(dts[i]) for i in ranked],
        "relative_comparison_reference_dt":float(dts[ref_i]),
        "caveats":[
            "Selection among candidate dt values is not a proof of a global continuous-domain optimum.",
            "Bootstrap frequencies are selection-stability diagnostics, not posterior probabilities or multiplicity-corrected p-values.",
            "Inference is conditional on this fixed set of functions, dimensions, budget and particle initialization policy.",
            "Independent holdout must be used for final precommitted dt validation; do not retune on holdout results.",
        ],
    }
    (out/"selection_summary.json").write_text(json.dumps(summary,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    make_figures(out,dts,cells,z,score,lower,upper,champion_i,nseed,report_status)
    return summary


def make_figures(out,dts,cells,z,score,lower,upper,champion_i,nseed,label):
    figdir=out/"figures"
    figdir.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({"pdf.fonttype":42,"ps.fonttype":42,"font.size":9,
                         "axes.spines.right":False,"axes.spines.top":False})
    best=float(dts[champion_i])
    # Main figure: combined score and uncertainty. No 100 crowded tick labels.
    fig,ax=plt.subplots(figsize=(7.4,4.2))
    ax.fill_between(dts,lower,upper,alpha=.18,color="#1864ab",label="95% paired-seed bootstrap CI")
    ax.plot(dts,score,"o-",lw=1.6,ms=4,color="#1864ab",label="Equal-weight mean log residual")
    ax.axvline(best,color="#b54833",lw=1.1,linestyle="--",label=f"Candidate Δt = {best:g}")
    ax.set_xscale("log")
    ax.set_xlim(dts.min(),dts.max())
    ax.set_xlabel(r"CPSO interval $\Delta t$ (log scale)")
    ax.set_ylabel(r"Mean $\log_{10}[(f_B-f^*)/(f_0-f^*)]$ (lower is better)")
    ax.set_title(f"CPSO interval sensitivity · {nseed} independent paired seeds")
    ax.grid(axis="y",alpha=.2)
    ax.legend(loc="best",frameon=False,fontsize=8)
    fig.tight_layout()
    fig.savefig(figdir/"fig_dt_statistical_overall.pdf",bbox_inches="tight")
    plt.close(fig)
    # Function panels, D30 and D100 overlaid. Each panel has a meaningful y label.
    benchmarks=list(dict.fromkeys(b for b,_ in cells))
    dims=sorted(set(d for _,d in cells))
    fig,axes=plt.subplots(2,3,figsize=(11.5,6.7),sharex=True)
    colors=["#226bb0","#c16a38","#277f58"]
    for ax,b in zip(axes.flat,benchmarks):
        for col,d in enumerate(dims):
            ci=cells.index((b,d))
            med=np.median(z[ci],axis=0)
            q25=np.quantile(z[ci],.25,axis=0)
            q75=np.quantile(z[ci],.75,axis=0)
            color=colors[col%len(colors)]
            ax.plot(dts,med,"o-",lw=1.3,ms=2.8,label=f"D={d}",color=color)
            ax.fill_between(dts,q25,q75,alpha=.12,color=color)
        ax.axvline(best,color="#b54833",lw=.9,linestyle="--")
        ax.set_xscale("log")
        ax.set_title(b.capitalize())
        ax.set_ylabel("Median log residual (lower = better)",fontsize=8)
        ax.grid(axis="y",alpha=.18)
    for ax in axes[-1]: ax.set_xlabel(r"$\Delta t$ (log scale)")
    axes.flat[0].legend(loc="best",frameon=False,fontsize=8)
    fig.suptitle("CPSO sensitivity by objective and dimension · interquartile bands",y=.995,fontsize=12)
    fig.tight_layout(rect=(0,0,1,.97))
    fig.savefig(figdir/"fig_dt_statistical_by_function.pdf",bbox_inches="tight")
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input",type=Path,required=True)
    parser.add_argument("--output-dir",type=Path,required=True)
    parser.add_argument("--bootstrap-reps",type=int,default=5000)
    args=parser.parse_args()
    if args.bootstrap_reps < 500:
        parser.error("--bootstrap-reps must be >= 500")
    from cpso_serial.output_policy import results_output
    output = results_output(args.output_dir)
    result=summarize(args.input,output,reps=args.bootstrap_reps)
    print(f"VALIDATED: {result['n_runs']} runs; {result['n_seed_clusters']} independent seeds; {result['n_dt_candidates']} dt candidates")
    print(result['status'])
    print(f"Best within candidate set: dt={result['empirical_best_dt']:.8g}")
    if result['frozen_dt_if_holdout'] is not None:
        print(f"PRECOMMITTED frozen candidate: dt={result['frozen_dt_if_holdout']:.8g} (DO NOT RETUNE)")
    print(f"Tables/figures: {args.output_dir}")

if __name__=="__main__":
    main()
