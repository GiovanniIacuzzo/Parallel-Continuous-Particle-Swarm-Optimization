from __future__ import annotations
from dataclasses import asdict
import itertools
import math
import time
import numpy as np
import pandas as pd
from scipy.stats import rankdata, binomtest, friedmanchisquare
from .common import (read,prepare,freeze,csv,native_run,cached_run,shuffled,objective,BOUNDS,
                     paired_effect,holm,name,method)
from cpso_serial.pso_baselines import PSOConfig,minimize_pso,available_pso_methods

BASELINES=list(available_pso_methods())
PRIMARY='CPSO selected framework'
CONTROL='CPSO serial control'


def cases(design,suite):
    cfg=design['comparison']
    if suite=='native':
        return list(itertools.product([b for b in cfg['functions'] if b!='inverse_diffusion'],cfg['dimensions']))
    return [('inverse_diffusion',d) for d in cfg['inverse_dimensions']]


def recipe(base,b,d):
    selected=read(base/'02_parallel/selection/frozen.json')
    return selected['by_case'].get(f'{b}|{d}',selected['global_method'])


def iterations_for(design,suite):
    return design['comparison']['iterations'] if suite=='native' or design['profile']=='smoke' else (
        [3,7,15] if design['profile']=='pilot' else [7,15,31])


def run_one(base,design,b,d,seed,it,m):
    cfg=design['comparison'];population=design['population'];work=cfg['work'] if b=='inverse_diffusion' else 0
    chosen=recipe(base,b,d);dt=read(base/'01_dt/holdout/complete.json')['dt'];budget=population*(it+1)
    if m in {PRIMARY,CONTROL}:
        actual=chosen if m==PRIMARY else method(topology=chosen['topology'])
        result=native_run(b,d,seed,it,dt,actual,work,population)
        result['implementation']='native_cpp';result['execution']=actual
    else:
        lo,hi=(-2.,2.) if b=='inverse_diffusion' else BOUNDS[b]
        counter=0;hits={};best=float('inf');t0=time.perf_counter()
        def measured(x):
            nonlocal counter,best
            value=objective(b,x,work);counter+=1;best=min(best,value)
            for target in cfg['targets']:
                if target not in hits and best<=target: hits[target]=(counter,time.perf_counter()-t0)
            return value
        c=PSOConfig((lo,)*d,(hi,)*d,population_size=population,max_evaluations=budget,seed=seed)
        started=time.perf_counter();r=minimize_pso(measured,c,method=m);wall=time.perf_counter()-started
        if counter!=budget or r.function_evaluations!=budget: raise ValueError('Baseline FE mismatch')
        result=asdict(r);result.update(wall_seconds=wall,initial_best=r.best_history[0],
            implementation='numpy_python',execution=dict(architecture='serial',workers=1),
            measured_target_hits={str(t):dict(fe=f,seconds=s) for t,(f,s) in hits.items()})
    if not math.isfinite(result['best_value']) or result['wall_seconds']<=0: raise ValueError('Invalid result')
    return result


def run(base,design,suite,benchmark=None):
    if not (base/'02_parallel/holdout/complete.json').exists():
        raise ValueError('Complete held-out framework validation before the final comparison')
    folder=base/'03_comparison'/suite;prepare(folder,design);cfg=design['comparison']
    recipes={f'{b}|{d}':recipe(base,b,d) for b,d in cases(design,suite)}
    freeze(folder/'frozen_recipes.json',recipes)
    jobs=[]
    for (b,d),seed,it,m in itertools.product(cases(design,suite),cfg['seeds'],iterations_for(design,suite),[PRIMARY,CONTROL,*BASELINES]):
        if benchmark is not None and b!=benchmark:continue
        jobs.append(dict(benchmark=b,dimension=d,seed=seed,iterations=it,method=m))
    if not jobs: raise ValueError('No matching comparison cases')
    for i,key in enumerate(shuffled(jobs,771),1):
        cached_run(folder,key,lambda:run_one(base,design,key['benchmark'],key['dimension'],key['seed'],key['iterations'],key['method']))
        if i%20==0:print(f'Comparison {suite}: {i}/{len(jobs)} checked/completed',flush=True)
    try: analyze(base,design,suite)
    except FileNotFoundError:
        print('Requested shard complete; finish remaining functions to build pooled statistics.',flush=True)


def dataset(base,design,suite):
    from .common import fingerprint
    folder=base/'03_comparison'/suite;rows=[]
    cfg=design['comparison']
    for (b,d),seed,it,m in itertools.product(cases(design,suite),cfg['seeds'],iterations_for(design,suite),[PRIMARY,CONTROL,*BASELINES]):
        key=dict(benchmark=b,dimension=d,seed=seed,iterations=it,method=m)
        record=read(folder/'runs'/(fingerprint(key)+'.json'))
        if record['key']!=key: raise ValueError('Comparison key mismatch')
        r=record['result'];fe=design['population']*(it+1)
        if r['function_evaluations']!=fe: raise ValueError('Comparison budget mismatch')
        rows.append(key|dict(fe=fe,value=r['best_value'],seconds=r['wall_seconds'],
            native_or_baseline_seconds=r['elapsed_seconds'],initial=r['initial_best'],
            workers=r['execution']['workers'],architecture=r['execution']['architecture'],
            implementation=r['implementation']))
    return pd.DataFrame(rows)


def analyze(base,design,suite):
    frame=dataset(base,design,suite);folder=base/'03_comparison'/suite;out=folder/'analysis'
    out.mkdir(parents=True,exist_ok=True)
    # Controls do not duplicate the CPSO's credit in the eleven-method ranking.
    ranked=frame[~frame.method.eq(CONTROL)].copy()
    ranked['rank']=ranked.groupby(['benchmark','dimension','seed','iterations']).value.rank(method='average')
    frame.to_csv(out/'measurements.csv',index=False);ranked.to_csv(out/'ranked_results.csv',index=False)
    ranks=[];contrasts=[];function_tests=[];omnibus=[]
    for (d,it),group in ranked.groupby(['dimension','iterations']):
        per=group.groupby(['benchmark','method'])['rank'].mean().unstack()
        rng=np.random.default_rng(19273)
        boot=per.to_numpy()[rng.integers(len(per),size=(design['inference']['bootstrap'],len(per)))].mean(axis=1)
        for j,m in enumerate(per.columns):
            ranks.append(dict(dimension=d,iterations=it,fe=design['population']*(it+1),method=m,
                mean_rank=float(per[m].mean()),lo=float(np.quantile(boot[:,j],.025)),hi=float(np.quantile(boot[:,j],.975)),
                functions=len(per)))
        if len(per)>=2 and np.any(np.ptp(per.to_numpy(),axis=1)>0):
            chi,p=friedmanchisquare(*per.to_numpy().T)
            omnibus.append(dict(dimension=d,iterations=it,statistic=float(chi),p=float(p),blocks=len(per),
                caution='few fixed function blocks; asymptotic omnibus is secondary'))
        pivot=group.pivot(index=['benchmark','seed'],columns='method',values='value')
        for m in BASELINES:
            # Bounded relative gap difference is invariant to common positive scale.
            left=pivot[PRIMARY];right=pivot[m]
            effect=(left-right)/np.maximum(left+right,1e-16)
            by_seed=effect.groupby(level='seed').mean()
            contrasts.append(dict(dimension=d,iterations=it,opponent=m,
                **paired_effect(by_seed.to_numpy(),design['inference']['bootstrap']),
                inference='seed aggregates conditional on this fixed native suite'))
            per_function=effect.groupby(level='benchmark').median()
            wins=int((per_function<-1e-12).sum());losses=int((per_function>1e-12).sum())
            function_tests.append(dict(dimension=d,iterations=it,opponent=m,wins=wins,losses=losses,
                ties=len(per_function)-wins-losses,p=binomtest(wins,wins+losses).pvalue if wins+losses else 1.,
                inference='function-level sign test; small suite'))
    csv(out/'rank_summary.csv',ranks);csv(out/'omnibus.csv',omnibus)
    for filename,rows in [('seed_contrasts.csv',contrasts),('function_contrasts.csv',function_tests)]:
        a=pd.DataFrame(rows);a['p_holm']=holm(a.p);a.to_csv(out/filename,index=False)
    targets=[]
    for (d,it,m),g in ranked.groupby(['dimension','iterations','method']):
        for t in design['comparison']['targets']:
            success=g.value<=t
            targets.append(dict(dimension=d,iterations=it,method=m,target=t,successes=int(success.sum()),
                runs=len(g),fraction=float(success.mean()),
                # Native API exposes full-run time, not per-generation timestamps.
                mean_completed_run_seconds=float(g.seconds.mean()),
                interpretation='final-target success; elapsed cost of full run, not exact time to first target'))
    csv(out/'targets.csv',targets)
    freeze(folder/'complete.json',dict(runs=len(frame),ranked_methods=len(BASELINES)+1,
        timing_scope='implementation performance: native C++ CPSO versus NumPy baselines',
        native_target_timestamps=False))
    from .figures import comparison_figures
    comparison_figures(base,design,suite)
    print(f'Comparison {suite}: {len(frame)} runs, {out}',flush=True)
