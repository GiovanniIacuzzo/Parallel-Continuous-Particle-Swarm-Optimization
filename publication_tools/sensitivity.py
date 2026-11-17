from __future__ import annotations
import importlib.util
import itertools
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare
from .common import ROOT, read, freeze, prepare, csv, normalized_score, paired_effect, holm


def driver():
    path=ROOT/'experiments/06_dt_sensitivity/run_dt_sensitivity.py'
    spec=importlib.util.spec_from_file_location('dt_campaign_driver',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def run(base, design, phase):
    folder=base/'01_dt'; cfg=design['dt']; runner=driver()
    prepare(folder,design)
    if phase=='holdout':
        frozen=read(folder/'selection/frozen.json')
        grid=frozen['validation_grid']
    else: grid=cfg['grid']
    for iterations in cfg['iterations']:
        protocol=dict(campaign_id=f"publication_v2_{design['profile']}_{phase}_k{iterations}",
            schedule_seed=901+iterations,benchmarks=cfg['functions'],dimensions=cfg['dimensions'],
            population_size=design['population'],iterations=iterations,seeds=cfg[phase+'_seeds'],
            dt_values=grid,reference_dt=cfg['reference'],topology='global',evidence_class='exploratory')
        path=freeze(folder/phase/f'k{iterations}'/'config.json',protocol)
        raw=path.parent/'raw'
        runner.execute(path,raw,resume=(raw/'study_config.json').exists())
    analyze(base,design,phase)


def data(base,design,phase):
    folder=base/'01_dt'/phase; runner=driver(); arrays=[]; rows=[]
    cfg=design['dt']; seeds=cfg[phase+'_seeds']
    grid=(
        read(base/'01_dt/selection/frozen.json')['validation_grid'] if phase=='holdout' else cfg['grid'])
    cells=list(itertools.product(cfg['functions'],cfg['dimensions']))
    for iterations in cfg['iterations']:
        config=read(folder/f'k{iterations}/config.json');raw=folder/f'k{iterations}/raw'
        keys={runner.run_key(*job) for job in runner.planned_runs(config)}
        records=runner.load_completed_records(raw/'runs.jsonl',expected_keys=keys,
            expected_fe=design['population']*(iterations+1),population=design['population'],
            iterations=iterations,repair_truncated_tail=False)
        if set(records)!=keys: raise ValueError(f'Incomplete dt campaign: {raw}: {len(records)}/{len(keys)}')
        arr=np.empty((len(cells),len(seeds),len(grid)))
        for c,(b,d) in enumerate(cells):
            for s,seed in enumerate(seeds):
                initials=[records[b,d,seed,float(t)]['best_history'][0] for t in grid]
                if len(set(initials))!=1: raise ValueError('Initial states differ across dt')
                for j,t in enumerate(grid):
                    r=records[b,d,seed,float(t)];z=float(normalized_score(r['best_value'],initials[0]))
                    arr[c,s,j]=z
                    history=normalized_score(r['best_history'],initials[0])
                    axis=np.asarray(r['history_evaluations'],float)
                    auc=float(np.sum((history[:-1]+history[1:])*.5*np.diff(axis))/(axis[-1]-axis[0]))
                    rows.append(dict(benchmark=b,dimension=d,iterations=iterations,seed=seed,dt=t,
                        score=z,anytime_log_residual=auc,final_value=r['best_value'],initial=initials[0],
                        fe=r['function_evaluations'],native_seconds=r['elapsed_seconds']))
        arrays.append(arr)
    return np.asarray(arrays),pd.DataFrame(rows),grid,cells


def analyze(base,design,phase):
    arrays,raw,grid,cells=data(base,design,phase)
    out=base/'01_dt'/phase/'analysis';out.mkdir(parents=True,exist_ok=True)
    raw.to_csv(out/'paired_results.csv',index=False)
    # Independent runs at every budget: the inertia schedule reaches its endpoint
    # at every budget. These are not checkpoints of a longer run.
    by_seed=arrays.mean(axis=(0,1));scores=by_seed.mean(axis=0)
    ranked=sorted(range(len(grid)),key=lambda i:(scores[i],grid[i]))
    folder=base/'01_dt'/phase
    if phase=='selection':
        selected=grid[ranked[0]]
        freeze(folder/'frozen.json',dict(dt=selected,
            validation_grid=sorted(set([grid[i] for i in ranked[:3]]+[design['dt']['reference']])),
            rule='minimum equal-function/dimension/budget mean log10 residual on selection seeds',
            selected_on='selection only',iterations=design['dt']['iterations']))
    else: selected=read(base/'01_dt/selection/frozen.json')['dt']
    rows=[]
    for i,t in enumerate(grid):
        stat=paired_effect(by_seed[:,i],design['inference']['bootstrap'])
        rows.append(dict(dt=t,score=float(scores[i]),lo=stat['lo'],hi=stat['hi'],frozen=t==selected))
    csv(out/'ranking.csv',rows)
    contrasts=[]
    si=grid.index(selected)
    # Global contrast primary, plus budget and cell contrasts in ONE Holm family.
    for j,t in enumerate(grid):
        if j==si: continue
        contrasts.append(dict(scope='all_budgets_cells',iterations=0,benchmark='all',dimension=0,
            candidate=selected,opponent=t,**paired_effect(by_seed[:,si]-by_seed[:,j],design['inference']['bootstrap'])))
        for k,it in enumerate(design['dt']['iterations']):
            diff=arrays[k,:,:,si]-arrays[k,:,:,j]
            contrasts.append(dict(scope='budget',iterations=it,benchmark='all',dimension=0,
                candidate=selected,opponent=t,**paired_effect(diff.mean(axis=0),design['inference']['bootstrap'])))
            for c,(b,d) in enumerate(cells):
                contrasts.append(dict(scope='cell',iterations=it,benchmark=b,dimension=d,
                    candidate=selected,opponent=t,**paired_effect(diff[c],design['inference']['bootstrap'])))
    frame=pd.DataFrame(contrasts);frame['p_holm']=holm(frame.p)
    frame['interpretation']='held-out' if phase=='holdout' else 'exploratory post-selection'
    frame.to_csv(out/'contrasts.csv',index=False)
    omnibus=[]
    for k,it in enumerate(design['dt']['iterations']):
        samples=arrays[k].mean(axis=0)
        if len(grid)>=3 and np.any(np.ptp(samples,axis=1)>0):
            chi,p=friedmanchisquare(*samples.T)
        else: chi,p=0.,1.
        omnibus.append(dict(iterations=it,statistic=float(chi),p=float(p),
                             block='paired seed, fixed function/dimension mean'))
    csv(out/'omnibus.csv',omnibus)
    freeze(folder/'complete.json',dict(dt=selected,runs=len(raw),phase=phase,
            seeds=len(design['dt'][phase+'_seeds']),scope='fixed native benchmark suite'))
    from .figures import dt_figures
    dt_figures(base,design,phase)
    print(f'DT {phase}: frozen dt={selected:g}; {len(raw)} runs; {out}',flush=True)
