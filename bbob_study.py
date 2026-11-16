"""Fixed-budget BBOB comparison on matched COCO instances."""
import argparse
import hashlib
import json
import math
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest

from study import campaign_lock, pin
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "experiments/05_external_benchmarks"))
from run_bbob import execute
from common import METHODS, collect_external_provenance, deterministic_problem_seed
from cpso_serial.native import _load_library

PRIMARY = "cpso_random_k8"
GLOBAL = "cpso_continuous"
NAMES = [PRIMARY, *METHODS]
TARGETS = (10., 1., .1, .01)
SEED = 20261115


def optimum(folder, method, f, d):
    file = folder / "exdata" / method / f"data_f{f}" / f"bbobexp_f{f}_DIM{d}.dat"
    with file.open() as stream:
        header = stream.readline()
    match = re.search(r"Fopt\s*\(([^)]+)\)", header)
    if not match:
        raise ValueError(f"Missing COCO Fopt in {file}")
    return float(match.group(1))


def read_shard(folder, instance, d, functions, checkpoints):
    rows = []
    for method in NAMES:
        path = folder / "records" / (method+".jsonl")
        seen = set()
        for line in path.read_text().splitlines():
            r = json.loads(line)
            pid = r["problem_id"]
            match = re.fullmatch(r"bbob_f(\d+)_i(\d+)_d(\d+)",pid)
            if not match:
                raise ValueError(pid)
            f, i, dimension = map(int,match.groups())
            budget = checkpoints[-1]*d
            if (f in seen or i != instance or dimension != d or r["method"] != method or
                r["dimension"] != d or r["seed"] != deterministic_problem_seed(pid,SEED) or
                any(r[k]!=budget for k in ("coco_evaluations","max_evaluations","function_evaluations"))):
                raise ValueError(f"Invalid record: {method}, {pid}")
            seen.add(f)
            if method in (GLOBAL,PRIMARY):
                params=r["algorithm_parameters"]
                expected=("random",8,r["seed"]) if method==PRIMARY else ("global-best",None,None)
                if (params["topology"],params["target_degree"],params["graph_seed"])!=expected or params["dt"]!=1 or params["population_size"]!=64:
                    raise ValueError("Incorrect CPSO configuration")
            fopt=optimum(folder,method,f,d)
            previous = r["initial_best"]
            for c in checkpoints:
                value=r["checkpoints"][str(c*d)]
                if not math.isfinite(value) or value>previous or value<fopt-1e-7:
                    raise ValueError(f"Invalid best-so-far: {pid}")
                rows.append(dict(function=f,instance=i,dimension=d,method=method,
                                 budget_per_d=c,fe=c*d,seed=r["seed"],value=value,
                                 fopt=fopt,gap=max(0.,value-fopt),initial=r["initial_best"]))
                previous=value
            if previous!=r["best_value"]:
                raise ValueError("Final checkpoint mismatch")
        if seen != set(functions):
            raise ValueError(f"Incomplete {folder}, {method}")
    frame=pd.DataFrame(rows)
    paired=frame[frame.method.isin([GLOBAL,PRIMARY])].groupby("function")
    if not paired.initial.nunique().eq(1).all() or not frame.groupby("function").fopt.nunique().eq(1).all():
        raise ValueError("Mismatched initial state or target")
    return frame


def summarize(a, out):
    out.mkdir(parents=True,exist_ok=True)
    keys=["function","dimension","instance","budget_per_d"]
    a["rank"]=a.groupby(keys).value.rank(method="average")
    a.to_csv(out/"paired_results.csv",index=False)
    per=a.groupby(["function","dimension","budget_per_d","method"]).agg(
        mean_rank=("rank","mean"),median_gap=("gap","median")).reset_index()
    per.to_csv(out/"by_function.csv",index=False)
    overall=per.groupby(["dimension","budget_per_d","method"]).mean_rank.mean().reset_index()
    overall.to_csv(out/"mean_ranks.csv",index=False)
    targets=[]
    for target in TARGETS:
        hits=(a.gap<=target).groupby([a.dimension,a.budget_per_d,a.method]).mean().rename("fraction").reset_index()
        targets.append(hits.assign(target=target))
    pd.concat(targets).to_csv(out/"target_attainment.csv",index=False)
    # Paired within problem. The sign test uses functions as units per dimension.
    final=a[a.budget_per_d==a.budget_per_d.max()]
    pairs=[]
    detail=[]
    for d,g in final.groupby("dimension"):
        values=g.pivot(index=["function","instance"],columns="method",values="value")
        for method in NAMES[1:]:
            delta=values[PRIMARY]-values[method]
            byfunc=delta.groupby(level="function").median()
            for f,v in byfunc.items():
                obs=delta.xs(f)
                detail.append(dict(dimension=d,function=f,opponent=method,median_delta=v,
                                   wins=int((obs<0).sum()),ties=int((obs==0).sum()),losses=int((obs>0).sum())))
            w=int((byfunc<0).sum()); l=int((byfunc>0).sum())
            pairs.append(dict(dimension=d,opponent=method,functions_won=w,functions_tied=len(byfunc)-w-l,
                              functions_lost=l,p=binomtest(w,w+l).pvalue if w+l else 1.))
    contrasts=pd.DataFrame(pairs).sort_values("p")
    contrasts["p_holm"]=np.minimum(1,np.maximum.accumulate(contrasts.p.to_numpy()*np.arange(len(contrasts),0,-1)))
    contrasts.to_csv(out/"primary_contrasts.csv",index=False)
    pd.DataFrame(detail).to_csv(out/"primary_contrasts_by_function.csv",index=False)
    # Count outright best, shared best, and losses versus the entire panel.
    best=per[per.budget_per_d==per.budget_per_d.max()].pivot(index=["dimension","function"],columns="method",values="mean_rank")
    rivals=best.drop(columns=PRIMARY).min(axis=1)
    pd.DataFrame({"primary_mean_rank":best[PRIMARY],"best_rival_mean_rank":rivals,
                  "outcome":np.select([best[PRIMARY]<rivals,best[PRIMARY]==rivals],["first","shared_first"],default="behind")}).to_csv(out/"primary_first_places.csv")
    dims=sorted(a.dimension.unique())
    plt.rcParams.update({"font.size":8,"pdf.fonttype":42,"axes.spines.top":False,"axes.spines.right":False})
    fig,axes=plt.subplots(1,len(dims),squeeze=False,figsize=(4*len(dims)+.8,4.4),layout="constrained")
    for ax,d in zip(axes[0],dims):
        p=overall[overall.dimension==d].pivot(index="method",columns="budget_per_d",values="mean_rank")
        p=p.sort_values(p.columns[-1])
        y=np.arange(len(p)); names=[m.replace("pso_","",1) if m.startswith("pso_") else ("CPSO Random 8" if m==PRIMARY else "CPSO Global") for m in p.index]
        for j,c in enumerate(p.columns):
            ax.scatter(p[c],y+(j-(len(p.columns)-1)/2)*.18,s=18,marker=["o","s","^"][j],label=f"{c}D FE")
        ax.set(yticks=y,yticklabels=names,xlabel="Mean rank (lower is better)",title=f"BBOB · D={d}",xlim=(.6,12.4))
        ax.invert_yaxis();ax.grid(axis="x",alpha=.2);ax.legend(fontsize=7)
    fig.savefig(out/"fig_bbob_ranks.pdf");plt.close(fig)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("profile",choices=["smoke","paper"])
    ap.add_argument("--analyze-only",action="store_true")
    args=ap.parse_args()
    smoke=args.profile=="smoke"
    dims=[5] if smoke else [5,20]
    instances=[1] if smoke else list(range(6,11))
    functions=[1,8] if smoke else list(range(1,25))
    checkpoints=[64,128] if smoke else [64,192,384]
    root=ROOT/"results/bbob_transfer_v1"/args.profile
    settings=dict(methods=NAMES,dt=1,population=64,dimensions=dims,instances=instances,functions=functions,
                  checkpoints=checkpoints,seed=SEED,targets=list(TARGETS))
    with campaign_lock(ROOT/"results"):
        config=root/"design.json"
        pin(config,settings)
        if not args.analyze_only:
            prov=collect_external_provenance(config)
            pin(root/"provenance.json",dict(provenance=prov,
                native_sha256=hashlib.sha256(Path(_load_library()._name).read_bytes()).hexdigest(),
                launcher_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
        frames=[]
        for d in dims:
            for inst in instances:
                for start in range(0,len(functions),6):
                    group=functions[start:start+6]
                    tag=f"d{d}_i{inst}_f{group[0]}-{group[-1]}"
                    dst=root/"blocks"/tag
                    if not dst.exists():
                        if args.analyze_only: raise SystemExit(f"Incomplete block: {tag}")
                        temp=dst.with_name(tag+".pending")
                        if temp.exists():
                            archive=root/"interrupted"
                            archive.mkdir(parents=True,exist_ok=True)
                            temp.rename(archive/(tag+"_"+datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S%f")))
                        cfg=dict(campaign_id=tag,suite="bbob",instances=f"instances: {inst}",
                                 suite_options=f"dimensions: {d} function_indices: "+",".join(map(str,group)),
                                 population_size=64,methods=NAMES,cpso_dt=1.,seed=SEED,
                                 checkpoint_multipliers=checkpoints,budget_multiplier=checkpoints[-1])
                        cfgfile=pin(root/"configs"/(tag+".json"),cfg)
                        execute(cfgfile,temp)
                        frame=read_shard(temp,inst,d,group,checkpoints)
                        temp.rename(dst)
                    else:
                        frame=read_shard(dst,inst,d,group,checkpoints)
                    frames.append(frame)
                    print("Verified",tag,flush=True)
        a=pd.concat(frames,ignore_index=True)
        summarize(a,root/"analysis")
        print(f"COMPLETE: {len(a)//len(checkpoints)} runs, {len(a)} checkpoints")
        if smoke:
            finals=a[a.budget_per_d==checkpoints[-1]].pivot(index="function",columns="method",values="value")
            print("Smoke, final CPSO values:\n",finals[[PRIMARY,GLOBAL]].to_string())


if __name__=="__main__":
    main()
