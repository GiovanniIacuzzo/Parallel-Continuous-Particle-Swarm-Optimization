"""M2 backend selection and independent graph-seed experiments."""
import argparse
import hashlib
import itertools
import json
import os
import platform
import random
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest

from study import campaign_lock
ROOT = Path(__file__).resolve().parent
ARCH = ROOT / "experiments/04_architectures"
sys.path.insert(0, str(ARCH))
from checkpoint_io import atomic_write_json, read_jsonl_checkpoint

FIELDS = ["objective","dimension","graph","swarm","graph_seed","backend","workers","work","iterations"]
TOPO = ["global","grid","random8"]
COST = [0,128,1024,8192]
METHODS = [("serial",1)] + list(itertools.product(["master","particle"],[1,2,4,8]))
plt.rcParams.update({"pdf.fonttype":42,"font.size":8,"axes.spines.top":False,"axes.spines.right":False})


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def native():
    compiler = shlex.split(os.environ.get("CXX","c++"))
    version = subprocess.check_output(compiler+["--version"],text=True).splitlines()[0]
    src = [ARCH/"decision_native.cpp"] + [ROOT/"cpp/src"/(s+".cpp") for s in
           ("optimizer","objectives","analytic_propagator","topology","random_stream")]
    headers = sorted((ROOT/"cpp/include").rglob("*.hpp")) + [ROOT/"cpp/src/optimizer_internal.hpp"]
    flags = ["-O3","-DNDEBUG","-std=c++20","-pthread"]
    identity = dict(compiler=compiler,version=version,flags=flags,machine=platform.machine(),
                    sources={str(p.relative_to(ROOT)):digest(p) for p in src+headers})
    tag = hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()[:16]
    exe = ROOT/"build/decision"/tag/"decision_native"
    if not exe.exists():
        exe.parent.mkdir(parents=True,exist_ok=True)
        tmp = exe.with_suffix(".tmp")
        subprocess.run(compiler+flags+["-I",str(ROOT/"cpp/include"),*src,"-o",tmp],check=True)
        tmp.replace(exe)
    return exe, dict(identity,binary_sha256=digest(exe))


def plan(stage,session,smoke):
    dims=[30] if smoke else [30,100]
    if stage=="cost":
        levels = [0,128] if smoke else COST
        seeds = range(132001,132003 if smoke else 132013)
        blocks=list(itertools.product(dims,TOPO,levels,seeds))
        rng=random.Random(8917+session)
        rng.shuffle(blocks)
        jobs=[]
        for d,g,work,s in blocks:
            methods=METHODS.copy()
            rng.shuffle(methods)
            jobs.extend(("rastrigin",d,g,s,s,m,w,work,2 if smoke else 48) for m,w in methods)
    else:
        jobs=[]
        for obj,d,i in itertools.product(["rastrigin","diffusion"],dims,range(2 if smoke else 12)):
            s=142001+i
            w=4 if obj=="rastrigin" else 8
            for g in ["grid","random4","random8"]:
                for j in range(1 if g=="grid" else (2 if smoke else 8)):
                    gs=0 if g=="grid" else 152001+8*i+j
                    jobs.append((obj,d,g,s,gs,"particle",w,0,2 if smoke else 96))
        random.Random(92513).shuffle(jobs)
    return jobs


def invoke(exe,job):
    lines=subprocess.check_output([exe,*map(str,job)],text=True).splitlines()
    wall,ev,fe,degree=lines[0].split()
    hist=list(map(float,lines[1].split()))
    pos=list(map(float,lines[2].split()))
    if (int(fe)!=64*(job[-1]+1) or len(hist)!=job[-1]+1 or len(pos)!=job[1]
        or not np.isfinite([*pos,*hist,float(wall),float(ev),float(degree)]).all()
        or float(wall)<=0 or float(ev)<=0 or np.any(np.diff(hist)>0)):
        raise ValueError(f"Invalid native result: {job}")
    return dict(zip(FIELDS,job),seconds=float(wall),eval_seconds=float(ev),fe=int(fe),
                degree=float(degree),initial=hist[0],best=hist[-1],
                trace=hashlib.sha256(("\n".join(lines[1:])).encode()).hexdigest())


def campaign(stage,session,smoke,out):
    exe,build=native()
    jobs=plan(stage,session,smoke)
    folder=out/(f"cost_session_{session}" if stage=="cost" else "graph")
    folder.mkdir(parents=True,exist_ok=True)
    keys=["|".join(map(str,j)) for j in jobs]
    protocol=dict(build=build,plan_sha256=hashlib.sha256("\n".join(keys).encode()).hexdigest(),
                  stage=stage,session=session,jobs=len(jobs),platform=platform.platform(),
                  driver=digest(Path(__file__)),chunk=1)
    manifest=folder/"manifest.json"
    if manifest.exists():
        if json.loads(manifest.read_text())["protocol"]!=protocol:
            raise SystemExit(f"Campaign fingerprint mismatch: {folder}")
    else:
        atomic_write_json(manifest,dict(protocol=protocol,started=datetime.now(timezone.utc).isoformat()))
    raw=folder/"runs.jsonl"
    rows,_=read_jsonl_checkpoint(raw,repair=True)
    done={r["run_key"] for r in rows}
    if done-set(keys):
        raise ValueError("Unexpected cached runs")
    pending=[(j,k) for j,k in zip(jobs,keys) if k not in done]
    if pending:
        for m,w in METHODS:
            invoke(exe,("rastrigin",30,"grid",999,999,m,w,128,4))
    with raw.open("a") as stream:
        for i,(job,key) in enumerate(pending,1):
            row=invoke(exe,job)
            row.update(run_key=key,session=session,utc=datetime.now(timezone.utc).isoformat())
            stream.write(json.dumps(row,allow_nan=False)+"\n")
            stream.flush()
            if i%100==0:
                print(f"{stage} {len(done)+i}/{len(keys)}",flush=True)
            rows.append(row)
    frame=pd.DataFrame(rows)
    if frame.groupby(["objective","dimension","swarm"]).initial.nunique().max()!=1:
        raise ValueError("Initial objective differs across paired runs")
    if stage=="cost" and frame.groupby(["dimension","graph","swarm"]).trace.nunique().max()!=1:
        raise ValueError("Different best history or final position across cost/backend/worker settings")
    return frame


def write(frame,path):
    tmp=path.with_suffix(".tmp")
    frame.to_csv(tmp,index=False)
    tmp.replace(path)


def cost_analysis(frames,out):
    a=pd.concat(frames)
    if a.groupby(["dimension","graph","swarm"]).trace.nunique().max()!=1:
        raise ValueError("Cross-session trajectory mismatch")
    a["method"]=a.backend+"_"+a.workers.astype(str)
    idx=["dimension","graph","work","swarm","session"]
    wide=a.pivot(index=idx,columns="method",values="seconds")
    effects=[]
    for (d,g,lvl),v in wide.groupby(level=idx[:3]):
        # Every session contributes equally for each paired seed.
        per=v.groupby(level="swarm").median()
        for backend,p in METHODS:
            m=f"{backend}_{p}"
            speed=(v[f"{backend}_1"]/v[m]).groupby(level="swarm").median()
            accel=(v["serial_1"]/v[m]).groupby(level="swarm").median()
            effects.append(dict(dimension=d,graph=g,work=lvl,method=m,
                                seconds=per[m].median(),speedup=speed.median(),
                                efficiency=speed.median()/p,serial_speedup=accel.median()))
    write(pd.DataFrame(effects),out/"backend_summary.csv")
    rows=[]
    for (d,g,lvl),v in wide.groupby(level=idx[:3]):
        serial=a[(a.dimension==d)&(a.graph==g)&(a.work==lvl)&(a.backend=="serial")]
        cost=float((1e6*serial.eval_seconds/serial.fe).median())
        for backend in ["particle","master"]:
            ratio=v[backend+"_4"]/v[backend+"_8"]
            paired=ratio.groupby(level="swarm").median().to_numpy()
            rng=np.random.default_rng(41235)
            boot=np.median(rng.choice(paired,(4000,len(paired))),axis=1)
            lo,hi=np.quantile(boot,[.025,.975])
            rows.append(dict(dimension=d,graph=g,work=lvl,backend=backend,
                             serial_eval_us=cost,t4_over_t8=np.median(paired),low=lo,high=hi,
                             sessions_favoring_8=int((ratio.groupby(level="session").median()>1).sum()),
                             sessions=ratio.index.get_level_values("session").nunique()))
    r=pd.DataFrame(rows)
    write(r,out/"crossover.csv")
    brackets=[]
    for (d,g,backend),v in r.groupby(["dimension","graph","backend"]):
        v=v.sort_values("work").reset_index(drop=True)
        supported=(v.low>1)&(v.sessions_favoring_8==v.sessions)
        stable=[i for i in v.index if supported.iloc[i:].all()]
        i=stable[0] if stable else None
        brackets.append(dict(dimension=d,graph=g,backend=backend,
            first_supported_work=v.work[i] if i is not None else None,
            first_supported_us=v.serial_eval_us[i] if i is not None else None,
            preceding_work=v.work[i-1] if i and i>0 else None,
            preceding_us=v.serial_eval_us[i-1] if i and i>0 else None))
    write(pd.DataFrame(brackets),out/"supported_levels.csv")
    dims=sorted(r.dimension.unique())
    fig,axes=plt.subplots(len(dims),3,figsize=(8,2.55*len(dims)),squeeze=False,layout="constrained",sharey=True)
    for (d,g,backend),v in r.groupby(["dimension","graph","backend"]):
        ax=axes[dims.index(d),TOPO.index(g)]
        v=v.sort_values("work")
        ax.plot(v.serial_eval_us,v.t4_over_t8,"o-",label=backend)
        ax.fill_between(v.serial_eval_us,v.low,v.high,alpha=.13)
        ax.axhline(1,color="grey",lw=.5)
        ax.set(xscale="log",title=f"{g}, D={d}",xlabel="Serial evaluation (µs)")
        if g==TOPO[0]: ax.set_ylabel("Time ratio, 4 / 8 workers")
    axes[0,0].legend()
    fig.savefig(out/"fig_crossover.pdf"); plt.close(fig)
    if set(a.session)=={1,2,3}:
        train=wide[wide.index.get_level_values("session")<3]
        test=wide[wide.index.get_level_values("session")==3]
        choices=[]
        for cell,v in train.groupby(level=idx[:3]):
            # Equal weight per session; ties resolve by method name.
            chosen=v.groupby(level="session").median().mean().idxmin()
            held=test.xs(cell,level=idx[:3])
            reference=held.median().idxmin()
            choices.append(dict(zip(idx,cell),chosen=chosen,holdout_best=reference,
                                heldout_speedup_vs_serial=(held.serial_1/held[chosen]).median(),
                                heldout_time_ratio_to_best=(held[chosen]/held[reference]).median()))
        write(pd.DataFrame(choices),out/"heldout_decisions.csv")


def graph_analysis(a,out):
    rows=[]
    for (obj,d),v in a.groupby(["objective","dimension"]):
        pivot=v.groupby(["swarm","graph"]).best.median().unstack()
        for left,right in [("random8","grid"),("random8","random4")]:
            delta=pivot[left]-pivot[right]
            wins=int((delta<0).sum()); losses=int((delta>0).sum())
            ratio=pivot[left]/pivot[right]
            rng=np.random.default_rng(976)
            # Estimand: performance of the eight-graph median for a fresh swarm.
            ci=np.quantile(np.median(rng.choice(ratio,(4000,len(ratio))),axis=1),[.025,.975])
            rows.append(dict(objective=obj,dimension=d,left=left,right=right,swarms=len(delta),
                             wins=wins,losses=losses,ties=len(delta)-wins-losses,
                             median_ratio=ratio.median(),ci_low=ci[0],ci_high=ci[1],
                             p=binomtest(wins,wins+losses).pvalue if wins+losses else 1.))
    r=pd.DataFrame(rows).sort_values("p")
    r["p_holm"]=np.minimum(1,np.maximum.accumulate(r.p.to_numpy()*np.arange(len(r),0,-1)))
    write(r,out/"graph_contrasts.csv")
    detail=a.groupby(["objective","dimension","swarm","graph"]).agg(
        quality=("best","median"),q25=("best",lambda x:x.quantile(.25)),
        q75=("best",lambda x:x.quantile(.75)),mean_degree=("degree","mean")).reset_index()
    write(detail,out/"graph_by_swarm.csv")
    fig,ax=plt.subplots(figsize=(7,3.7),layout="constrained")
    r=r.sort_values(["objective","dimension","right"]).reset_index(drop=True)
    for i,row in r.iterrows():
        ax.plot([row.ci_low,row.ci_high],[i,i],color="#286789");ax.plot(row.median_ratio,i,"o",color="#286789")
    ax.axvline(1,ls=":",color=".4")
    ax.set_yticks(r.index,[f"{v.objective}, D={v.dimension} · Random 8 / {v.right}" for v in r.itertuples()])
    ax.set_xlabel("Paired final objective ratio (lower favors Random 8)")
    fig.savefig(out/"fig_graph_replication.pdf");plt.close(fig)


def phases(source,out):
    rows=[]
    for obj in ["rastrigin","inverse_diffusion"]:
        with (source/obj/"publication_scaling_runs.jsonl").open() as stream:
            for line in stream:
                v=json.loads(line)
                if v["architecture"]!="particle_parallel": continue
                names=["topology_setup","initialization","social_selection","propagation","evaluation_wall"]
                parts={n:v[n+"_seconds"] for n in names}
                parts["other"]=v["elapsed_seconds"]-sum(parts.values())
                if parts["other"] < -1e-8: raise ValueError("Phase timers overlap")
                for name,value in parts.items():
                    rows.append(dict(objective=obj,dimension=v["dimension"],graph=v["topology"],
                                     workers=v["workers"],seed=v["seed"],phase=name,
                                     ms=1000*value, percent=100*value/v["elapsed_seconds"]))
    keys=["objective","dimension","graph","workers","phase"]
    a=pd.DataFrame(rows)
    if len(a)!=55296 or a.groupby(keys+["seed"]).size().ne(3).any():
        raise ValueError("Incomplete topology_m2_v1 timing design")
    summary=a.groupby(keys+["seed"])[["ms","percent"]].median().groupby(level=keys).median().reset_index()
    write(summary,out/"phase_decomposition.csv")
    fig,axes=plt.subplots(2,2,figsize=(7.4,4.8),layout="constrained",sharex=True)
    for ax,((obj,d),v) in zip(axes.flat,summary.groupby(["objective","dimension"])):
        for g in TOPO:
            actual={"grid":"grid_von_neumann","random8":"random_k8"}.get(g,g)
            s=v[(v.phase=="social_selection")&(v.graph==actual)].sort_values("workers")
            ax.plot(s.workers,s.percent,"o-",label=g)
        ax.set(title=f"{obj.replace('_',' ')}, D={d}",xticks=[1,2,4,8],xlabel="Workers",ylabel="Social selection (%)",ylim=(0,None))
    axes[0,0].legend()
    fig.savefig(out/"fig_social_cost.pdf");plt.close(fig)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stage",choices=["phases","cost","graph"])
    ap.add_argument("--session",type=int,choices=[1,2,3],default=1)
    ap.add_argument("--smoke",action="store_true")
    args=ap.parse_args()
    base=ROOT/"results"/("decision_smoke" if args.smoke else "decision_m2_v1")
    with campaign_lock(ROOT/"results"):
        base.mkdir(parents=True,exist_ok=True)
        if args.stage=="phases":
            phases(ROOT/"results/topology_m2_v1",base);return
        os.environ["CPSO_THREADPOOL_CHUNK"]="1"
        data=campaign(args.stage,args.session if args.stage=="cost" else 1,args.smoke,base)
        if args.stage=="graph":
            graph_analysis(data,base);return
        frames=[]
        for s in [1,2,3]:
            folder=base/f"cost_session_{s}"
            if not (folder/"manifest.json").exists(): continue
            meta=json.loads((folder/"manifest.json").read_text())["protocol"]
            current=json.loads((base/f"cost_session_{args.session}"/"manifest.json").read_text())["protocol"]
            for key in ["build","driver","platform","chunk"]:
                if current[key]!=meta[key]: raise ValueError("Incompatible cost sessions")
            rows,_=read_jsonl_checkpoint(folder/"runs.jsonl")
            expected={"|".join(map(str,j)) for j in plan("cost",s,args.smoke)}
            if {r["run_key"] for r in rows}==expected: frames.append(pd.DataFrame(rows))
        cost_analysis(frames,base)


if __name__=="__main__":
    main()
