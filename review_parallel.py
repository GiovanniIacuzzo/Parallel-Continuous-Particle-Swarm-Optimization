"""Native scaling, equal-work replica throughput, held-out overhead prediction."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from dataclasses import asdict, replace
import itertools
import json
import multiprocessing
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
import time
import numpy as np
import pandas as pd
from scipy.optimize import nnls
from review_study import ROOT, config, digest, provenance
from cpso_serial.native import run_controlled
from study import campaign_lock, pin


def task(job):
    d,dyn,seed,work,iterations,backend,workers=job
    cfg=config(d,seed,'global',iterations,1.)
    r=run_controlled(cfg,dynamics=dyn,benchmark='sphere_work',architecture=backend,
                     workers=workers,work_iterations=work)
    return dict(seed=seed,fe=r.function_evaluations,
        trace=__import__('hashlib').sha256(repr((r.best_history,r.best_position)).encode()).hexdigest(),
        phases={k:v for k,v in asdict(r).items() if k.endswith('seconds')})


def calibrate(d,target):
    if target==0: return 0,0.
    work=1024
    for _ in range(5):
        row=task((d,'analytic',999,work,0,'serial',1))
        observed=row['phases']['evaluation_wall_seconds']/row['fe']
        if .8*target<=observed<=1.25*target: return work,observed
        work=max(1,min(100_000_000,round(work*target/max(observed,1e-12))))
    return work,observed


def environment(conditions):
    notes=json.loads(conditions.read_text()) if conditions else dict(
        power_source='unknown',low_power_mode='unknown',thermal_state='unknown',
        background_load='unknown',notes='Record observed conditions; unknown fields are retained.')
    data=dict(system=platform.platform(),machine=platform.machine(),logical_cpus=os.cpu_count(),
              utc=datetime.now(timezone.utc).isoformat(),conditions=notes)
    if platform.system()=='Darwin':
        for key in ['hw.model','hw.perflevel0.physicalcpu','hw.perflevel1.physicalcpu']:
            result=subprocess.run(['sysctl','-n',key],capture_output=True,text=True)
            data[key]=result.stdout.strip() if result.returncode==0 else None
        result=subprocess.run(['pmset','-g','batt'],capture_output=True,text=True)
        data['pmset_batt']=result.stdout.strip()
    try:
        import psutil
        data['physical_cpus']=psutil.cpu_count(logical=False)
        data['load_average']=list(os.getloadavg())
        data['temperatures']={k:[v._asdict() for v in vals] for k,vals in psutil.sensors_temperatures().items()}
    except (ImportError,AttributeError,OSError): pass
    return data


def campaign(args):
    smoke=args.profile=='smoke';pilot=args.profile=='pilot'
    dims=[5] if smoke else ([30] if pilot else [30,100])
    costs=[0.,1e-5] if smoke else ([0.,1e-5,1e-4,1e-3] if pilot else [0.,1e-6,1e-5,1e-4,5e-4,1e-3,5e-3,1e-2])
    workers=[1,2] if smoke else [1,2,4,8]
    batch=2 if smoke else 8
    repetitions=1 if smoke else (2 if pilot else 3)
    design=dict(dimensions=dims,target_costs_seconds=costs,workers=workers,batch=batch,
                repetitions=repetitions, dynamics=['analytic','discrete'],randomness='scalar',graph='global',
                iterations_policy='smoke:2; otherwise max(2,min(32,floor(.1/(64*max(cost,1e-6))))-1)',
                training_sessions=[1,2],holdout_session=3,
                timing='batch wall clock includes Python dispatch and pool creation/destruction')
    root=ROOT/'results/review_v1/parallel'/args.profile
    pin(root/'design.json',design)
    source=provenance();source['review_parallel.py']=digest(__file__)
    pin(root/'provenance.json',source)
    folder=root/f'session_{args.session}'
    if args.plan:
        cells=len(dims)*len(costs)*2*repetitions
        print(json.dumps(dict(design=design,batches_per_session=cells*(3+3*(len(workers)-1)),
                              notice='Paper high-cost levels can take hours; launch sessions separately.'),indent=2))
        return
    if not (folder/'environment_start.json').exists():
        pin(folder/'environment_start.json',environment(args.conditions))
    calibration={}
    for d,c in itertools.product(dims,costs):
        file=folder/'calibration'/f'd{d}_cost{c:g}.json'
        if file.exists(): entry=json.loads(file.read_text())
        else:
            work,observed=calibrate(d,c)
            entry=dict(work=work,observed_seconds=observed,target_seconds=c)
            pin(file,entry)
        calibration[d,c]=entry['work']
    cells=list(itertools.product(dims,costs,['analytic','discrete'],range(repetitions)))
    random.Random(1200+args.session).shuffle(cells)
    for d,c,dyn,rep in cells:
        iterations=2 if smoke else max(2,min(32,int(.1/(64*max(c,1e-6))))-1)
        work=calibration[d,c]
        methods=[('serial',1)]+list(itertools.product(['master_slave','particle_parallel','replicas'],workers[1:]))
        # Also measure each native backend at one worker to quantify its overhead.
        methods += [('master_slave',1),('particle_parallel',1)]
        random.Random(500+rep+args.session).shuffle(methods)
        for method,p in methods:
            key=f'd{d}_{dyn}_cost{c:g}_r{rep}_{method}_p{p}'
            path=folder/'runs'/(key+'.json')
            if path.exists(): continue
            jobs=[(d,dyn,170001+j,work,iterations,
                   'serial' if method=='replicas' else method,1 if method=='replicas' else p) for j in range(batch)]
            started=time.perf_counter()
            if method=='replicas':
                # Includes spawn + shutdown. Same number of independent runs and FE.
                with ProcessPoolExecutor(max_workers=p,mp_context=multiprocessing.get_context('spawn')) as pool:
                    rows=list(pool.map(task,jobs))
            else: rows=[task(job) for job in jobs]
            seconds=time.perf_counter()-started
            pin(path,dict(dimension=d,dynamics=dyn,target_cost=c,work=work,iteration_count=iterations,
                          repetition=rep,session=args.session,method=method,workers=p,
                          batch_seconds=seconds,rows=rows,utc=datetime.now(timezone.utc).isoformat()))
        print(f'Session {args.session}: D{d} {dyn}, cost={c:g}, repeat={rep}',flush=True)
    pin(folder/'environment_end.json',environment(args.conditions))
    analyze(root)


def analyze(root):
    design=json.loads((root/'design.json').read_text());out=root/'analysis';out.mkdir(exist_ok=True)
    records=[];detail=[]
    methods_expected=3+3*(len(design['workers'])-1)
    expected=len(design['dimensions'])*len(design['target_costs_seconds'])*len(design['dynamics'])*design['repetitions']*methods_expected
    for folder in sorted(root.glob('session_*')):
        files=list((folder/'runs').glob('*.json'))
        if len(files)!=expected: raise ValueError(f'Incomplete session {folder}: {len(files)}/{expected}')
        for path in files:
            r=json.loads(path.read_text()); runs=r.pop('rows')
            if len(runs)!=design['batch']: raise ValueError('Incomplete batch')
            phase={k:sum(t['phases'][k] for t in runs) for k in runs[0]['phases']}
            r.update(native_seconds=phase['elapsed_seconds'],evaluation_seconds=phase['evaluation_wall_seconds'],
                     propagation_seconds=phase['propagation_seconds'],social_seconds=phase['social_selection_seconds'],
                     topology_seconds=phase['topology_setup_seconds'],initialization_seconds=phase['initialization_seconds'],
                     fe=sum(t['fe'] for t in runs))
            for t in runs:
                detail.append({k:r[k] for k in ['dimension','dynamics','target_cost','work','session','repetition','method','workers']}|t)
            records.append(r)
    a=pd.DataFrame(records);raw=pd.DataFrame(detail)
    # Per seed, same work/dynamics/configuration must have bit-identical trajectory.
    if not raw.groupby(['dimension','dynamics','work','seed']).trace.nunique().eq(1).all():
        raise ValueError('Serial/parallel/replica trajectory mismatch')
    a.to_csv(out/'batch_times_and_phases.csv',index=False)
    keys=['dimension','dynamics','target_cost','session','repetition']
    serial=a[a.method=='serial'].set_index(keys)
    rows=[]
    for _,r in a.iterrows():
        ref=serial.loc[tuple(r[k] for k in keys)]
        one=a[(a.method==r.method)&(a.workers==1)]
        if r.method=='replicas': native_one=ref.native_seconds
        else: native_one=one.set_index(keys).loc[tuple(r[k] for k in keys)].native_seconds
        rows.append(dict(r,serial_batch_speedup=ref.batch_seconds/r.batch_seconds,
                         serial_batch_efficiency=ref.batch_seconds/r.batch_seconds/r.workers,
                         serial_native_speedup=ref.native_seconds/r.native_seconds if r.method!='replicas' else None,
                         backend_strong_speedup=native_one/r.native_seconds if r.method!='replicas' else None,
                         backend_strong_efficiency=native_one/r.native_seconds/r.workers if r.method!='replicas' else None,
                         native_one_worker_overhead=(native_one/ref.native_seconds-1) if r.workers==1 else None,
                         serial_eval_us=1e6*ref.evaluation_seconds/ref.fe))
    summary=pd.DataFrame(rows);summary.to_csv(out/'speedups.csv',index=False)
    sessions=set(a.session)
    if not {1,2,3}.issubset(sessions):
        print('Timing summaries complete. Model validation requires sessions 1, 2, 3.');return
    model_rows=[];predictions=[]
    for d,dyn,backend in itertools.product(design['dimensions'],design['dynamics'],['master_slave','particle_parallel']):
        # Distinct per-work iteration counts: normalize phases per native iteration
        # using total FE as exposure. Nonnegative model of time per FE:
        # T/FE = a + b*(serial_eval_time/FE)/p + c*p/FE.
        # Actual total batch model: a*FE + b*T_eval(serial)/p + c*p.
        train=summary[(summary.session<3)&(summary.dimension==d)&(summary.dynamics==dyn)&(summary.method==backend)]
        test=summary[(summary.session==3)&(summary.dimension==d)&(summary.dynamics==dyn)&(summary.method==backend)]
        def features(g):
            # Estimate objective cost using only training-session serial timings.
            costs=serial.reset_index();costs=costs[(costs.session<3)&(costs.dimension==d)&(costs.dynamics==dyn)]
            costs=(costs.evaluation_seconds/costs.fe).groupby(costs.target_cost).median()
            e=g.target_cost.map(costs)*g.fe
            return np.column_stack([g.fe,e/g.workers,g.workers])
        x=features(train);scale=np.maximum(np.linalg.norm(x,axis=0),1e-15)
        coef_scaled,_=nnls(x/scale,train.native_seconds.to_numpy());coef=coef_scaled/scale
        model_rows.append(dict(dimension=d,dynamics=dyn,backend=backend,serial_per_fe=coef[0],eval_factor=coef[1],overhead_per_worker=coef[2]))
        pred=features(test)@coef
        for (_,r),v in zip(test.iterrows(),pred):
            predictions.append(dict(r,predicted_seconds=float(v),relative_error=float(v/r.native_seconds-1)))
    pd.DataFrame(model_rows).to_csv(out/'overhead_model.csv',index=False)
    pred=pd.DataFrame(predictions);pred.to_csv(out/'heldout_predictions.csv',index=False)
    decisions=[]
    for cell,g in pred.groupby(['dimension','dynamics','target_cost']):
        med=g.groupby(['method','workers'])[['predicted_seconds','native_seconds']].median()
        # The model selects within each backend's worker counts. Serial and replica
        # throughput are reported separately and are not silently mixed with latency.
        for backend,t in med.groupby(level='method'):
            chosen=t.predicted_seconds.idxmin(); oracle=t.native_seconds.idxmin()
            decisions.append(dict(dimension=cell[0],dynamics=cell[1],target_cost=cell[2],backend=backend,
                chosen_workers=int(chosen[1]),oracle_workers=int(oracle[1]),
                relative_regret=float(t.loc[chosen,'native_seconds']/t.loc[oracle,'native_seconds']-1)))
    pd.DataFrame(decisions).to_csv(out/'heldout_regret.csv',index=False)
    print(f'Complete timing analysis: {out}',flush=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('stage',choices=['run','analyze'])
    ap.add_argument('--profile',choices=['smoke','pilot','paper'],default='smoke')
    ap.add_argument('--session',type=int,choices=[1,2,3],default=1)
    ap.add_argument('--conditions',type=Path)
    ap.add_argument('--plan',action='store_true')
    args=ap.parse_args()
    with campaign_lock(ROOT/'results'):
        if args.stage=='run':campaign(args)
        else:analyze(ROOT/'results/review_v1/parallel'/args.profile)

if __name__=='__main__':main()
