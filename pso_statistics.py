"""Statistical comparison of one frozen CPSO recipe against ten PSO baselines."""
import argparse
import hashlib
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, friedmanchisquare, rankdata

ROOT = Path(__file__).resolve().parent
PREFIX = "cpso_random_k8"
METHODS = [PREFIX, "pso_classic_gbest", "pso_inertia_gbest", "pso_inertia_lbest_ring",
           "pso_constriction_gbest", "pso_clpso", "pso_fips", "pso_spso2011", "pso_slpso",
           "pso_cso", "pso_dms"]
LABELS = ["CPSO Random k=8", "Classic gbest", "Inertia gbest", "Inertia lbest ring", "Constriction",
          "CLPSO", "FIPS", "SPSO-2011", "SL-PSO", "CSO", "DMS-PSO"]
OUT = ROOT / "results/pso_statistics_v1"
SOURCE = ROOT / "results/transfer_v2/study"


def matrix(frame, d, replicate=False, smoke=False):
    cec = d is None
    budget = 4096 if cec else (128 if smoke else 384*d)
    primary = PREFIX if cec or replicate else PREFIX + "_by_dimension"
    methods = [primary, *METHODS[1:]]
    a = frame[frame.fe.eq(budget) & frame.method.isin(methods)].copy()
    if not cec:
        a = a[a.dimension.eq(d)]
    else:
        if not a.dimension.eq(np.where(a.function.isin([13, 14]), 905, 1000)).all():
            raise ValueError("Incorrect CEC dimensions")
    if not replicate:
        actual = set(a.loc[a.method.eq(primary), "configuration"])
        expected = f"{PREFIX}@{0.35 if d == 20 else 1.0}"
        if actual != {expected}:
            raise ValueError(f"Unexpected primary configuration: {actual}")
    pivot = a.pivot(index=["function","replicate"], columns="method", values="value")
    funcs = [1, 8] if smoke else range(1, 16 if cec else 25)
    instances = [11] if smoke else range(1,6) if cec else range(11,16)
    keys = pd.MultiIndex.from_product([funcs, instances], names=pivot.index.names)
    if not pivot.index.equals(keys) or set(pivot.columns)!=set(methods):
        raise ValueError("Incomplete or unexpected matched design")
    arr = pivot[methods].to_numpy().reshape(len(funcs), len(instances), len(methods))
    if not np.isfinite(arr).all():
        raise ValueError("Nonfinite objective")
    return arr


def holm(p):
    order = np.argsort(p)
    result = np.empty(len(p))
    result[order] = np.minimum(1, np.maximum.accumulate(np.asarray(p)[order] * np.arange(len(p),0,-1)))
    return result


def statistics(arr, folder):
    folder.mkdir(parents=True, exist_ok=True)
    nf, nr, k = arr.shape
    # Ranks within matched instances/seeds. Every function receives weight 1/nf.
    byfun = rankdata(arr, axis=2).mean(axis=1)
    means = byfun.mean(axis=0)
    rng = np.random.default_rng(20261117)
    boot = byfun[rng.integers(nf, size=(10000,nf))].mean(axis=1)
    low, high = np.quantile(boot, [.025,.975], axis=0)
    first = np.isclose(boot, boot.min(axis=1, keepdims=True), rtol=0, atol=1e-12)
    share = (first / first.sum(axis=1,keepdims=True)).mean(axis=0)
    ranks = pd.DataFrame(dict(method=METHODS,label=LABELS,mean_rank=means,lo=low,hi=high,
                              bootstrap_first_share=share)).sort_values("mean_rank")
    ranks.to_csv(folder/"ranking.csv",index=False)
    # Friedman re-ranks the function-level summaries within each function.
    if np.all(np.ptp(byfun,axis=1)==0):
        chi, omnibus_p = 0., 1.
    else:
        chi, omnibus_p = friedmanchisquare(*byfun.T)
    rows = []
    for i, j in combinations(range(k), 2):
        median = np.median(arr[:,:,i]-arr[:,:,j], axis=1)
        w, l = int((median<0).sum()), int((median>0).sum())
        p = binomtest(w, w+l).pvalue if w+l else 1.
        rows.append(dict(left=METHODS[i],right=METHODS[j],wins=w,ties=nf-w-l,losses=l,
                         mean_rank_difference=means[i]-means[j],p=p))
    pairs = pd.DataFrame(rows)
    pairs["p_holm"] = holm(pairs.p)
    pairs.to_csv(folder/"all_pairs.csv",index=False)
    (folder/"omnibus.json").write_text(json.dumps(dict(functions=nf,replicates=nr,methods=k,
        friedman_chi2=chi, df=k-1,p=omnibus_p,kendall_w=chi/(nf*(k-1)),holm_family=len(pairs)),indent=2))
    top = ranks.iloc[0]
    print(f"{folder.name}: {top.label}, mean rank {top.mean_rank:.3f}; Friedman p={omnibus_p:.3g}")


def analyze():
    b = pd.read_csv(SOURCE/"validate/checkpoints.csv")
    c = pd.read_csv(SOURCE/"cec/checkpoints.csv")
    for name,data,d in [("bbob_d5",b,5),("bbob_d20",b,20),("cec",c,None)]:
        statistics(matrix(data,d), OUT/name)


def replicate(smoke):
    from transfer_study import run_one, problem_stream, campaign_lock, pin, deterministic_problem_seed
    from common import collect_external_provenance
    from cpso_serial.native import _load_library
    folder = OUT / ("replication_smoke" if smoke else "replication_d5")
    design = dict(functions=([1,8] if smoke else list(range(1,25))), dimensions=[5],
                  replicates=([11] if smoke else list(range(11,16))), methods=METHODS, swarm_seed=20261117,
                  dt=1., population=64, budget=(128 if smoke else 1920))
    with campaign_lock(ROOT/"results"):
        proto = pin(folder/"design.json", design)
        provenance = collect_external_provenance(proto)
        for key, path in [("statistics_driver",Path(__file__)), ("transfer_driver",ROOT/"transfer_study.py"),
                          ("native_binary",Path(_load_library()._name))]:
            provenance[key] = hashlib.sha256(path.read_bytes()).hexdigest()
        pin(folder/"provenance.json",provenance)
        rows = []
        for f,i,d,obj,lo,hi in problem_stream("validate",design):
            seed = deterministic_problem_seed(f"validate-{f}-{i}-{d}",design["swarm_seed"])
            fe = design["budget"]
            for m in METHODS:
                key = dict(function=f, replicate=i, dimension=d, method=m, seed=seed, fe=fe)
                path = folder/"runs"/f"f{f}_i{i}_{m}.json"
                if not path.exists():
                    checkpoints = run_one(obj, lo, hi, m, 1., fe, seed, [fe])
                    pin(path,{**key,"value":checkpoints[str(fe)]})
                row=json.loads(path.read_text())
                if any(row[k]!=v for k,v in key.items()) or not np.isfinite(row["value"]):
                    raise ValueError(f"Incompatible run: {path}")
                rows.append(row)
            print(f"Completed F{f} instance {i}",flush=True)
        a=pd.DataFrame(rows)
        arr=matrix(a,5,replicate=True,smoke=smoke)
        a.to_csv(folder/"final_values.csv",index=False)
        if not smoke:
            statistics(arr,folder)


def plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    groups=[("bbob_d5","BBOB · D = 5"),("bbob_d20","BBOB · D = 20"),("cec","CEC 2013 LSGO")]
    if (OUT/"replication_d5/ranking.csv").exists():
        groups.append(("replication_d5","BBOB · D = 5 · new seed"))
    plt.rcParams.update({"pdf.fonttype":42,"font.size":9})
    fig,axes=plt.subplots(1,len(groups),sharey=True,figsize=(2.45*len(groups)+1.4,4),layout="constrained")
    for ax,(name,title) in zip(axes,groups):
        data=pd.read_csv(OUT/name/"ranking.csv").set_index("method").loc[METHODS]
        for y,(_,row) in enumerate(data.iterrows()):
            color="#a13d27" if y==0 else "#25566d"
            ax.plot([row.lo,row.hi],[y,y],color=color)
            ax.scatter(row.mean_rank,y,s=24,color=color)
        ax.set(title=title,xlim=(.6,11.4),xticks=[1,3,5,7,9,11],yticks=range(11),yticklabels=LABELS,
               xlabel="Mean rank")
        ax.grid(axis="x",alpha=.2)
    axes[0].invert_yaxis()
    fig.savefig(OUT/"fig_pso_rank_intervals.pdf")
    plt.close(fig)


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command",choices=["analyze","smoke","replicate","plot"])
    cmd=parser.parse_args().command
    if cmd=="analyze": analyze()
    elif cmd=="plot": plot()
    else: replicate(cmd=="smoke")
