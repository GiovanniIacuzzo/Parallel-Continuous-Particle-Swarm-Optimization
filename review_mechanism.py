"""Measure affine-subspace confinement and boundary-induced escape."""
import argparse
import itertools
import json
from pathlib import Path
import numpy as np
import pandas as pd
from review_study import ROOT, config, pin, provenance, seed_for, digest
from cpso_serial.native import controlled_initial_state, run_controlled
from study import campaign_lock


def run(args):
    smoke=args.profile=='smoke';pilot=args.profile=='pilot'
    dims=[200] if smoke else [5,20,1000]
    seeds=list(range(1,(2 if smoke else 4 if pilot else 31)))
    iterations=3 if smoke else (63 if pilot else 500)
    root=ROOT/'results/review_v1/mechanism'/args.profile
    pin(root/'design.json',dict(dimensions=dims,seeds=seeds,iterations=iterations,population=64,
        graphs=['global','random8'],dynamics=['analytic','discrete'],scope=['scalar','coordinate'],
        boundaries=['none','clamp'],objective='shifted Householder-rotated ellipsoid, condition 1e6',
        interpretation='none is an unconstrained diagnostic, not a benchmark algorithm'))
    prov=provenance();prov['review_mechanism.py']=digest(__file__)
    pin(root/'provenance.json',prov)
    for d,rep in itertools.product(dims,seeds):
        seed=seed_for('mechanism_swarm',d,rep);gs=seed_for('mechanism_graph',d,rep)
        base=config(d,seed,'global',iterations,1.)
        x,v=controlled_initial_state(base)
        origin=x[0]
        directions=np.concatenate((x[1:]-origin,v),axis=0).T
        u,s,_=np.linalg.svd(directions,full_matrices=False)
        rank=int((s>max(directions.shape)*np.finfo(float).eps*s[0]).sum())
        q=u[:,:rank]
        rng=np.random.default_rng(seed_for('mechanism_objective',d,rep))
        shift=rng.uniform(-2,2,d);h=rng.normal(size=d);h/=np.linalg.norm(h)
        weights=np.geomspace(1.,1e6,d)
        for dyn,scope,graph,boundary in itertools.product(['analytic','discrete'],['scalar','coordinate'],['global','random8'],['none','clamp']):
            file=root/'runs'/f'd{d}_r{rep}_{dyn}_{scope}_{graph}_{boundary}.json'
            if file.exists():continue
            contacts=0
            def objective(point):
                nonlocal contacts
                p=np.asarray(point)
                contacts+=int(np.count_nonzero((p==-5.)|(p==5.)))
                z=p-shift;z=z-2*h*np.dot(h,z)
                return float(np.dot(weights*z,z))
            result=run_controlled(config(d,seed,graph,iterations,1.),objective,dynamics=dyn,
                randomness=scope,graph_seed=gs,boundary_policy=boundary)
            delta=np.asarray(result.best_position)-origin
            residual=delta-q@(q.T@delta)
            relative=float(np.linalg.norm(residual)/max(np.linalg.norm(delta),1e-15))
            target=shift-origin
            record=dict(dimension=d,replicate=rep,seed=seed,graph_seed=gs,dynamics=dyn,scope=scope,
                graph=graph,boundary=boundary,initial_affine_rank=rank,fe=result.function_evaluations,
                best=result.best_value,best_relative_subspace_residual=relative,
                target_relative_subspace_residual=float(np.linalg.norm(target-q@(q.T@target))/np.linalg.norm(target)),
                boundary_coordinate_contacts=contacts,elapsed_seconds=result.elapsed_seconds)
            if boundary=='none' and scope=='scalar' and relative>1e-10:
                raise ValueError(f'Unexpected affine-subspace escape: {record}')
            pin(file,record)
        print(f'Mechanism D={d}, replicate={rep}, affine rank={rank}',flush=True)
    files=list((root/'runs').glob('*.json'))
    if len(files)!=len(dims)*len(seeds)*16:raise ValueError('Incomplete mechanism experiment')
    a=pd.DataFrame([json.loads(p.read_text()) for p in files]);out=root/'analysis';out.mkdir(exist_ok=True)
    a.to_csv(out/'subspace_diagnostics.csv',index=False)
    a.groupby(['dimension','dynamics','scope','graph','boundary']).median(numeric_only=True).to_csv(out/'median_diagnostics.csv')


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--profile',choices=['smoke','pilot','paper'],default='smoke')
    args=ap.parse_args()
    with campaign_lock(ROOT/'results'):run(args)

if __name__=='__main__':main()
