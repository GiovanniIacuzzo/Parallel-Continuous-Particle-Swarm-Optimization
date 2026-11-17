"""Independent BBOB quality validation. Custom native callbacks remain serial."""
from __future__ import annotations
from dataclasses import asdict,replace
import itertools
import time
import numpy as np
import pandas as pd
from scipy.stats import binomtest
from .common import read,prepare,freeze,cached_run,fingerprint,shuffled,native_config,holm,ROOT,csv
from .comparison import BASELINES
from .coco_logging import short_coco_observer,observer_optimum,archive_interrupted_observer,target_hits
from cpso_serial import run_serial_objective
from cpso_serial.pso_baselines import PSOConfig,minimize_pso

PRIMARY='CPSO serial reference'
TARGETS=np.logspace(2,-8,21).tolist()


def jobs(design):
    cfg=design['external']
    for f,d,i,m in itertools.product(cfg['functions'],cfg['dimensions'],cfg['instances'],[PRIMARY,*BASELINES]):
        yield dict(function=f,dimension=d,instance=i,method=m)


def run_one(base,folder,design,key):
    import cocoex
    f,d,i,m=key['function'],key['dimension'],key['instance'],key['method'];cfg=design['external']
    suite=cocoex.Suite('bbob',f'instances: {i}',f'dimensions: {d} function_indices: {f}')
    problem=suite[0];lo=tuple(problem.lower_bounds);hi=tuple(problem.upper_bounds)
    seed=int(fingerprint([cfg['seed'],f,d,i])[:15],16)
    population=design['population'];budget=cfg['checkpoints'][-1]*d
    if budget%population: raise ValueError('BBOB budget not divisible by population')
    frozen=read(base/'02_parallel/selection/frozen.json');topology=frozen['global_method']['topology'];dt=frozen['dt']
    trace=[];timestamps=[];best=float('inf');started=time.perf_counter()
    def measured(x):
        nonlocal best
        value=float(problem(x));best=min(best,value);trace.append(best);timestamps.append(time.perf_counter()-started)
        return value
    log=folder/'observer'/fingerprint(key)
    archive_interrupted_observer(log)
    with short_coco_observer(problem,log,m):
        started=time.perf_counter()
        if m==PRIMARY:
            c=native_config('sphere',d,seed,budget//population-1,dt,topology,population)
            result=run_serial_objective(replace(c,lower_bounds=lo,upper_bounds=hi),measured)
        else:
            result=minimize_pso(measured,PSOConfig(lo,hi,population_size=population,max_evaluations=budget,seed=seed),method=m)
        elapsed=time.perf_counter()-started
        if len(trace)!=budget or result.function_evaluations!=budget: raise ValueError('COCO FE mismatch')
    optimum=observer_optimum(log)
    fe,seconds=target_hits(trace,timestamps,optimum,TARGETS)
    if trace[-1]<optimum-1e-7: raise ValueError('COCO optimum inconsistent with trace')
    return dict(seed=seed,fe=budget,optimum=optimum,elapsed_seconds=elapsed,
        checkpoints={str(c*d):trace[c*d-1] for c in cfg['checkpoints']},
        targets=[dict(target=t,fe=h,seconds=s) for t,h,s in zip(TARGETS,fe,seconds)],
        topology=topology,dt=dt,backend='serial_native_callback' if m==PRIMARY else 'numpy_serial',
        initial_best=trace[population-1])


def run(base,design,function=None):
    if not (base/'02_parallel/holdout/complete.json').exists(): raise ValueError('First complete framework holdout')
    folder=base/'03_comparison/bbob';prepare(folder,design)
    selected=read(base/'02_parallel/selection/frozen.json')
    freeze(folder/'frozen_recipe.json',dict(topology=selected['global_method']['topology'],dt=selected['dt'],
        native_callback_limitation='serial quality control; this is not a parallel BBOB run'))
    planned=[k for k in jobs(design) if function is None or k['function']==function]
    if not planned: raise ValueError('No BBOB jobs matched')
    for index,key in enumerate(shuffled(planned,871),1):
        cached_run(folder,key,lambda:run_one(base,folder,design,key))
        if index%10==0: print(f'BBOB: {index}/{len(planned)} checked/completed',flush=True)
    try: analyze(base,design)
    except FileNotFoundError: print('BBOB shard complete; other functions remain.',flush=True)


def analyze(base,design):
    folder=base/'03_comparison/bbob';out=folder/'analysis';out.mkdir(parents=True,exist_ok=True)
    rows=[];target_rows=[]
    for key in jobs(design):
        record=read(folder/'runs'/(fingerprint(key)+'.json'))
        if record['key']!=key: raise ValueError('BBOB key mismatch')
        r=record['result']
        for c in design['external']['checkpoints']:
            value=r['checkpoints'][str(c*key['dimension'])]
            rows.append(key|dict(budget_per_dimension=c,fe=c*key['dimension'],value=value,
                                gap=max(0.,value-r['optimum'])))
        for t in r['targets']:
            target_rows.append(key|dict(target=t['target'],hit_fe=t['fe'],hit_seconds=t['seconds'],
                                budget=r['fe'],elapsed=r['elapsed_seconds']))
    a=pd.DataFrame(rows);targets=pd.DataFrame(target_rows)
    a['rank']=a.groupby(['function','dimension','instance','budget_per_dimension']).value.rank()
    a.to_csv(out/'ranked_results.csv',index=False);targets.to_csv(out/'target_hits.csv',index=False)
    summaries=[];contrasts=[]
    for (d,c),g in a.groupby(['dimension','budget_per_dimension']):
        byfun=g.groupby(['function','method'])['rank'].mean().unstack()
        rng=np.random.default_rng(173)
        boot=byfun.to_numpy()[rng.integers(len(byfun),size=(design['inference']['bootstrap'],len(byfun)))].mean(axis=1)
        for j,m in enumerate(byfun.columns):
            summaries.append(dict(dimension=d,budget_per_dimension=c,method=m,mean_rank=byfun[m].mean(),
                lo=np.quantile(boot[:,j],.025),hi=np.quantile(boot[:,j],.975)))
        pivot=g.pivot(index=['function','instance'],columns='method',values='value')
        for m in BASELINES:
            diff=(pivot[PRIMARY]-pivot[m]).groupby(level='function').median()
            w=int((diff<-1e-10).sum());l=int((diff>1e-10).sum())
            contrasts.append(dict(dimension=d,budget_per_dimension=c,opponent=m,wins=w,losses=l,ties=len(diff)-w-l,
                p=binomtest(w,w+l).pvalue if w+l else 1.,mean_rank_difference=float(byfun[PRIMARY].mean()-byfun[m].mean())))
    pd.DataFrame(summaries).to_csv(out/'rank_summary.csv',index=False)
    tests=pd.DataFrame(contrasts);tests['p_holm']=holm(tests.p);tests.to_csv(out/'function_contrasts.csv',index=False)
    ert=[]
    for (f,d,m,t),g in targets.groupby(['function','dimension','method','target']):
        success=g.hit_fe.notna();n=int(success.sum())
        ert.append(dict(function=f,dimension=d,method=m,target=t,successes=n,runs=len(g),
            ert_fe=float(g.hit_fe.fillna(g.budget).sum()/n) if n else None,
            ert_seconds=float(g.hit_seconds.fillna(g.elapsed).sum()/n) if n else None,
            scope='one function and dimension; instances aggregated'))
    csv(out/'ert_by_function.csv',ert)
    freeze(folder/'complete.json',dict(runs=len(targets)//len(TARGETS),scope='serial external quality control'))
    from .figures import external_figures
    external_figures(base,design)
