"""Controlled experiments responding to the methodological review.

No paper files are modified. Run --help and read REVIEW_EXPERIMENTS.md first.
"""
from __future__ import annotations
import argparse
import hashlib
import itertools
import importlib.metadata
import os
import json
import math
import platform
import random
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'python'))
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution
from scipy.sparse.csgraph import shortest_path
from scipy.stats import friedmanchisquare, wilcoxon
from cpso_serial.config import CpsoConfig
from cpso_serial.native import controlled_graph, run_controlled, _load_library
from study import pin, campaign_lock

GRAPHS = {'global': ('global',4), 'ring': ('ring',4), 'grid': ('grid_von_neumann',4),
          'star': ('wheel',4), 'random4': ('random',4), 'random8': ('random',8),
          'small_world': ('small_world',4), 'scale_free': ('scale_free',4)}
TARGETS = np.logspace(2, -8, 21).tolist()
GROUPS = {**dict.fromkeys(range(1,6),'separable'), **dict.fromkeys(range(6,10),'moderate'),
          **dict.fromkeys(range(10,15),'ill_conditioned'),
          **dict.fromkeys(range(15,20),'multimodal_global'), **dict.fromkeys(range(20,25),'multimodal_weak')}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def provenance():
    files = [Path(__file__), *sorted((ROOT/'python/cpso_serial').glob('*.py')),
             *sorted((ROOT/'cpp').rglob('*.cpp')), *sorted((ROOT/'cpp/include').rglob('*.*')),
             ROOT/'cpp/src/optimizer_internal.hpp']
    return dict(commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                sources={str(p.relative_to(ROOT)):digest(p) for p in files},
                native=digest(_load_library()._name), build=_load_library().cpso_build_info().decode(),
                platform=platform.platform(), python=sys.version,
                packages={name:importlib.metadata.version(name) for name in ['numpy','scipy','pandas','coco-experiment']},
                threads={name:os.environ.get(name) for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','CPSO_THREADPOOL_CHUNK']},
                numpy=np.__version__, scipy=__import__('scipy').__version__)


def seed_for(*parts):
    return int.from_bytes(hashlib.sha256(':'.join(map(str,parts)).encode()).digest()[:8], 'little')


def config(d, seed, graph, iterations, dt):
    topology, degree = GRAPHS[graph]
    return CpsoConfig((-5.,)*d, (5.,)*d, population_size=64, iterations=iterations,
                      seed=seed, dt=dt, topology=topology, topology_degree=degree,
                      grid_rows=8 if graph=='grid' else None, grid_columns=8 if graph=='grid' else None)


def graph_statistics(adjacency):
    a = adjacency.copy()
    np.fill_diagonal(a,False)
    degree = a.sum(axis=1)
    clustering = []
    for neighbors in a:
        ids=np.flatnonzero(neighbors); k=len(ids)
        clustering.append(float(a[np.ix_(ids,ids)].sum()/(k*(k-1))) if k>1 else 0.)
    distances=shortest_path(a.astype(float),directed=False,unweighted=True)
    off=distances[~np.eye(len(a),dtype=bool)]
    return dict(mean_degree=float(degree.mean()), min_degree=int(degree.min()),
                max_degree=int(degree.max()), clustering=float(np.mean(clustering)),
                connected=bool(np.isfinite(off).all()),
                mean_path=float(off.mean()) if np.isfinite(off).all() else None,
                adjacency_sha256=hashlib.sha256(a.tobytes()).hexdigest())


def variants(smoke):
    # Full factorial; discrete dynamics are not duplicated across irrelevant dt.
    for graph, scope in itertools.product(GRAPHS, ['scalar','coordinate']):
        for dyn, dt in [('discrete',1.), *[('analytic',t) for t in ([1.] if smoke else [.38125,1.,3.6])]]:
            yield f'{dyn}_{scope}_{graph}_dt{dt:g}', dyn, scope, graph, dt


def ablation(args):
    smoke=args.profile=='smoke'; pilot=args.profile=='pilot'
    dims=[5] if smoke else [5,20]
    functions=[1,8] if smoke else ([1,8,15,21] if pilot else list(range(1,25)))
    instances=[1] if smoke else ([1,2,3] if pilot else list(range(1,16)))
    multiplier=128 if smoke else (512 if pilot else 2048)
    methods=list(variants(smoke))
    design=dict(suite='bbob',dimensions=dims, functions=functions, instances=instances,
                budget_per_d=multiplier, population=64, variants=methods, targets=TARGETS,
                reference='scipy_DE_best1bin', seed=20261116,
                inference_unit='function, instances aggregated within function',
                interpretation='fixed factorial, not a new function or instance holdout')
    root=ROOT/'results/review_v1/ablation'/args.profile
    pin(root/'design.json',json.loads(json.dumps(design)))
    pin(root/'provenance.json',provenance())
    import cocoex
    expected=len(dims)*len(functions)*len(instances)*(len(methods)+1)
    print(f'{expected} runs; {len(methods)+1} methods; {multiplier}D FE',flush=True)
    completed=0
    for d,f,i in itertools.product(dims,functions,instances):
        seed=seed_for('swarm',d,f,i); gs=seed_for('graph',d,f,i)
        order=[*methods, ('DE','DE','coordinate','global',1.)]
        random.Random(seed).shuffle(order)
        for name,dyn,scope,graph,dt in order:
            path=root/'runs'/f'd{d}_f{f}_i{i}_{name}.json'
            if path.exists():
                old=json.loads(path.read_text())
                if old['method']!=name or old['fe']!=multiplier*d or old['seed']!=seed:
                    raise ValueError(f'Invalid resume record: {path}')
                completed+=1
                continue
            suite=cocoex.Suite('bbob',f'instances: {i}',f'dimensions: {d} function_indices: {f}')
            problem=suite[0]
            # COCO exposes the optimum as a diagnostic-only private property.
            # Also retain official Observer logs, with Fopt in the .dat header.
            fopt=float(problem._best_value)
            log=root/'observer'/path.stem
            if log.exists():
                raise RuntimeError(f'Interrupted Observer directory: archive {log} before retrying')
            log.parent.mkdir(parents=True,exist_ok=True)
            observer=cocoex.Observer('bbob',f'outer_folder: {log.parent} result_folder: {log.name} algorithm_name: {name}')
            problem.observe_with(observer)
            count=0; best=math.inf; hits=[None]*len(TARGETS); hit_seconds=[None]*len(TARGETS)
            initial=[]; trace=[]; started=time.perf_counter()
            def measured(x):
                nonlocal count,best
                value=float(problem(x)); count+=1; best=min(best,value)
                if count<=64: initial.append(list(x))
                trace.append(best)
                for j,target in enumerate(TARGETS):
                    if hits[j] is None and best-fopt<=target:
                        hits[j]=count; hit_seconds[j]=time.perf_counter()-started
                return value
            cfg=config(d,seed,graph,multiplier*d//64-1,dt)
            cfg=__import__('dataclasses').replace(cfg,lower_bounds=tuple(problem.lower_bounds),
                                                upper_bounds=tuple(problem.upper_bounds))
            if dyn=='DE':
                rng=np.random.default_rng(seed)
                init=rng.uniform(problem.lower_bounds,problem.upper_bounds,(64,d))
                de=differential_evolution(measured,list(zip(problem.lower_bounds,problem.upper_bounds)),
                    strategy='best1bin', maxiter=cfg.iterations,init=init,seed=seed % 2**32,
                    tol=0.,atol=-1.,polish=False,updating='deferred',workers=1)
                final=float(de.fun); timing={}
            else:
                result=run_controlled(cfg,measured,dynamics=dyn,randomness=scope,graph_seed=gs)
                final=result.best_value
                timing={k:v for k,v in asdict(result).items() if k.endswith('seconds')}
            elapsed=time.perf_counter()-started
            problem.free()
            dat=list(log.rglob('*.dat'))
            import re
            if len(dat)!=1:
                raise ValueError(f'Missing Observer data in {log}')
            with dat[0].open() as stream:
                header=stream.readline()
            match=re.search(r'Fopt\s*\(([^)]+)\)',header)
            if match is None or not math.isclose(float(match[1]),fopt,rel_tol=1e-12,abs_tol=1e-10):
                raise ValueError('COCO optimum/Observer mismatch')
            if count!=multiplier*d or not math.isfinite(final) or final!=best:
                raise ValueError(f'Budget/value mismatch: {name}: {count}')
            adjacency=controlled_graph(cfg,gs)
            graph_info=graph_statistics(adjacency)
            # For DE graph metrics have no algorithmic meaning.
            record=dict(method=name,dynamics=dyn,scope=scope,graph=graph if dyn!='DE' else 'NA',dt=dt,
                        dimension=d,function=f,instance=i,seed=seed,graph_seed=gs,
                        initial_sha256=hashlib.sha256(np.asarray(initial).tobytes()).hexdigest(),
                        initial_best=trace[63],fe=count,best=final,fopt=fopt,gap=max(0.,final-fopt),
                        hits=hits,hit_seconds=hit_seconds,elapsed_seconds=elapsed,
                        native_phases=timing,graph_metrics=graph_info if dyn!='DE' else {},
                        # Store every generation checkpoint, not every evaluation.
                        history=trace[63::64])
            pin(path,record); completed+=1
        print(f'Completed f{f} D{d} instance{i}: {completed}/{expected}',flush=True)
    analyze_ablation(root)


def holm(pvalues):
    p=np.asarray(pvalues,float); order=np.argsort(p); adjusted=np.empty(len(p))
    adjusted[order]=np.minimum(1.,np.maximum.accumulate(p[order]*np.arange(len(p),0,-1)))
    return adjusted


def bootstrap_median(values):
    values=np.asarray(values,float)
    rng=np.random.default_rng(271828)
    boot=np.median(rng.choice(values,(4000,len(values))),axis=1)
    lo,hi=np.quantile(boot,[.025,.975])
    return dict(relative_effect_low=float(lo),relative_effect_high=float(hi))


def signed_test(delta):
    delta=np.asarray(delta,float); nz=delta[delta!=0]
    if not len(nz): return 1.,0.
    from scipy.stats import rankdata
    ranks=rankdata(abs(nz))
    effect=float((ranks[nz>0].sum()-ranks[nz<0].sum())/ranks.sum())
    return float(wilcoxon(delta,zero_method='wilcox',method='auto').pvalue),effect


def analyze_ablation(root):
    design=json.loads((root/'design.json').read_text())
    records=[json.loads(p.read_text()) for p in sorted((root/'runs').glob('*.json'))]
    expected=len(design['dimensions'])*len(design['functions'])*len(design['instances'])*(len(design['variants'])+1)
    if len(records)!=expected: raise ValueError(f'Incomplete campaign: {len(records)}/{expected}')
    a=pd.DataFrame(records)
    keys=['dimension','function','instance','method']
    if a.duplicated(keys).any(): raise ValueError('Duplicate run')
    native=a[a.dynamics!='DE']
    if not native.groupby(keys[:-1]).initial_sha256.nunique().eq(1).all():
        raise ValueError('Native initial states are not paired')
    out=root/'analysis';out.mkdir(exist_ok=True)
    scalar_columns=[k for k in a.columns if k not in ['hits','hit_seconds','history','native_phases','graph_metrics']]
    a[scalar_columns].to_csv(out/'runs.csv',index=False)
    # Each function is a block; instances are repeated observations, not new functions.
    med=a.groupby(['dimension','function','method']).gap.median().unstack('method')
    ranks=med.rank(axis=1,method='average')
    ranks.groupby(level='dimension').mean().to_csv(out/'mean_ranks.csv')
    groups=ranks.reset_index();groups['group']=groups.function.map(GROUPS)
    groups.drop(columns='function').groupby(['dimension','group']).mean().to_csv(out/'ranks_by_group.csv')
    omnibus=[];contrasts=[];factorial=[]
    for d,table in med.groupby(level='dimension'):
        vals=table.to_numpy()
        if np.all(np.ptp(vals,axis=1)==0): stat,p=0.,1.
        else: stat,p=friedmanchisquare(*vals.T)
        omnibus.append(dict(dimension=d,functions=len(table),statistic=float(stat),p=float(p)))
        primary='analytic_scalar_random8_dt1'
        for other in table.columns:
            if other==primary: continue
            delta=table[primary]-table[other]
            relative=delta/(table[primary]+table[other]).clip(lower=1e-16)
            p,effect=signed_test(relative)
            contrasts.append(dict(dimension=d,primary=primary,opponent=other,p=p,
                                  rank_biserial_positive_favors_opponent=effect,
                                  median_gap_difference=float(delta.median()),
                                  median_relative_effect=float(relative.median()),
                                  **bootstrap_median(relative),
                                  wins=int((delta<0).sum()),ties=int((delta==0).sum()),losses=int((delta>0).sum())))
        # Planned mechanistic comparisons use bounded relative paired effects,
        # aggregated over instances within each function before inference.
        t=a[a.dimension==d].pivot(index=['function','instance'],columns='method',values='gap')
        for scope,graph in itertools.product(['scalar','coordinate'],GRAPHS):
            left=f'analytic_{scope}_{graph}_dt1';right=f'discrete_{scope}_{graph}_dt1'
            denom=(t[left]+t[right]).clip(lower=1e-16)
            effects=((t[left]-t[right])/denom).groupby('function').median()
            p,effect=signed_test(effects)
            factorial.append(dict(dimension=d,contrast='analytic_minus_discrete',scope=scope,graph=graph,
                                  median_relative_effect=float(effects.median()),**bootstrap_median(effects),p=p,rank_biserial=effect))
        for dyn,scope in itertools.product(['analytic','discrete'],['scalar','coordinate']):
            left=f'{dyn}_{scope}_random8_dt1';right=f'{dyn}_{scope}_global_dt1'
            effects=((t[left]-t[right])/(t[left]+t[right]).clip(lower=1e-16)).groupby('function').median()
            p,effect=signed_test(effects)
            factorial.append(dict(dimension=d,contrast='random8_minus_global',scope=scope,graph=dyn,
                                  median_relative_effect=float(effects.median()),**bootstrap_median(effects),p=p,rank_biserial=effect))
        for dyn,graph in itertools.product(['analytic','discrete'],GRAPHS):
            left=f'{dyn}_coordinate_{graph}_dt1';right=f'{dyn}_scalar_{graph}_dt1'
            effects=((t[left]-t[right])/(t[left]+t[right]).clip(lower=1e-16)).groupby('function').median()
            p,effect=signed_test(effects)
            factorial.append(dict(dimension=d,contrast='coordinate_minus_scalar',scope=dyn,graph=graph,
                                  median_relative_effect=float(effects.median()),**bootstrap_median(effects),p=p,rank_biserial=effect))
        # Difference-in-differences: topology advantage changes with dt and dynamics.
        def relative(left,right):
            return (t[left]-t[right])/(t[left]+t[right]).clip(lower=1e-16)
        for scope in ['scalar','coordinate']:
            base=relative(f'analytic_{scope}_random8_dt1',f'analytic_{scope}_grid_dt1')
            for dt in sorted({v[4] for v in design['variants'] if v[1]=='analytic'}):
                if dt==1.: continue
                interaction=relative(f'analytic_{scope}_random8_dt{dt:g}',f'analytic_{scope}_grid_dt{dt:g}')-base
                effects=interaction.groupby('function').median();p,effect=signed_test(effects)
                factorial.append(dict(dimension=d,contrast=f'topology_dt_interaction_{dt:g}_minus_1',
                    scope=scope,graph='random8_vs_grid',median_relative_effect=float(effects.median()),**bootstrap_median(effects),p=p,rank_biserial=effect))
            effects=(relative(f'analytic_{scope}_random8_dt1',f'analytic_{scope}_global_dt1')
                     -relative(f'discrete_{scope}_random8_dt1',f'discrete_{scope}_global_dt1')).groupby('function').median()
            p,effect=signed_test(effects)
            factorial.append(dict(dimension=d,contrast='dynamics_topology_interaction',scope=scope,
                graph='random8_vs_global',median_relative_effect=float(effects.median()),**bootstrap_median(effects),p=p,rank_biserial=effect))
    pd.DataFrame(omnibus).to_csv(out/'friedman.csv',index=False)
    for rows,name in [(contrasts,'all_primary_contrasts'),(factorial,'planned_contrasts')]:
        frame=pd.DataFrame(rows);frame['p_holm']=holm(frame.p)
        frame.to_csv(out/(name+'.csv'),index=False)
    # ECDF includes failures in the denominator. ERT sums full budgets for failures.
    targets=[]
    for (d,method),g in a.groupby(['dimension','method']):
        for j,target in enumerate(TARGETS):
            hits=[r[j] for r in g.hits];times=[r[j] for r in g.hit_seconds]
            success=sum(v is not None for v in hits)
            targets.append(dict(dimension=d,method=method,target=target,successes=success,runs=len(g),
                ert_fe=sum(v if v is not None else fe for v,fe in zip(hits,g.fe))/success if success else None,
                ert_seconds=sum(v if v is not None else elapsed for v,elapsed in zip(times,g.elapsed_seconds))/success if success else None))
    pd.DataFrame(targets).to_csv(out/'target_ert.csv',index=False)
    ecdf=[]
    for (d,method),g in a.groupby(['dimension','method']):
        for budget in np.geomspace(64,design['budget_per_d']*d,60):
            count=sum(v is not None and v<=budget for hits in g.hits for v in hits)
            ecdf.append(dict(dimension=d,method=method,fe_per_d=budget/d,fraction=count/(len(g)*len(TARGETS))))
    ecdf=pd.DataFrame(ecdf);ecdf.to_csv(out/'ecdf.csv',index=False)
    pd.DataFrame([dict(dimension=r['dimension'],function=r['function'],instance=r['instance'],
                       graph=r['graph'],graph_seed=r['graph_seed'],**r['graph_metrics'])
                  for r in records if r['dynamics']=='analytic' and r['scope']=='scalar' and r['dt']==1.]).to_csv(out/'graph_metrics.csv',index=False)
    plot_ecdf(ecdf,out)
    print(f'Analysis complete: {out}',flush=True)


def plot_ecdf(a,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'pdf.fonttype':42,'font.size':9})
    for d,g in a.groupby('dimension'):
        fig,ax=plt.subplots(figsize=(6.4,4.))
        show=['analytic_scalar_global_dt1','analytic_scalar_random8_dt1',
              'discrete_scalar_global_dt1','discrete_scalar_random8_dt1',
              'analytic_coordinate_random8_dt1','discrete_coordinate_random8_dt1','DE']
        for method in show:
            t=g[g.method==method];ax.plot(t.fe_per_d,t.fraction,label=method.replace('_',' '))
        ax.set(xscale='log',ylim=(0,1),xlabel='Function evaluations / dimension',
               ylabel='Fraction of problem–target pairs solved',title=f'BBOB · D={d}')
        ax.legend(fontsize=6);fig.tight_layout();fig.savefig(out/f'fig_ecdf_d{d}.pdf');plt.close(fig)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('stage',choices=['ablation','analyze'])
    ap.add_argument('--profile',choices=['smoke','pilot','paper'],default='smoke')
    args=ap.parse_args()
    with campaign_lock(ROOT/'results'):
        if args.stage=='ablation': ablation(args)
        else: analyze_ablation(ROOT/'results/review_v1/ablation'/args.profile)

if __name__=='__main__': main()
