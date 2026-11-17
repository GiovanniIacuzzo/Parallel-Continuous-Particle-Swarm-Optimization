from __future__ import annotations
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import subprocess
import json
import math
import os
from pathlib import Path
import platform
import random
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import rankdata, wilcoxon

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'python'), str(ROOT / 'experiments/04_architectures')]
from checkpoint_io import atomic_write_json
from environment_metadata import collect_environment_metadata
from cpso_serial import CpsoConfig, native_build_info, native_version
from cpso_serial.native import _load_library
from run_architecture_study import invoke_method
from run_publication_scaling import configuration
from analysis.analyze_dt_statistical import objective as scalar_objective, BOUNDS

FUNCTIONS = list(BOUNDS)
TOPOS = ['global','ring','grid_von_neumann','wheel','random_k4','random_k8','small_world_k4','scale_free_k4']
LABELS = {'global':'Global','ring':'Ring','grid_von_neumann':'Von Neumann','wheel':'Star',
          'random_k4':'Random 4','random_k8':'Random 8','small_world_k4':'Small-world','scale_free_k4':'Scale-free'}
FLOOR = 1e-16


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,allow_nan=False).encode()).hexdigest()


def freeze(path, value):
    value = json.loads(json.dumps(value,allow_nan=False))
    path = Path(path)
    if path.exists():
        if read(path) != value:
            raise ValueError(f'Frozen protocol differs: {path}. Use a new --output directory.')
    else:
        atomic_write_json(path,value)
    return path


def provenance():
    lib = Path(_load_library()._name)
    paths = sorted((ROOT/'publication_tools').glob('*.py')) + [ROOT/'publication.py',ROOT/'study.py']
    paths += sorted((ROOT/'python/cpso_serial').rglob('*.py'))
    paths += [ROOT/'experiments/06_dt_sensitivity/run_dt_sensitivity.py',
              ROOT/'experiments/04_architectures/run_publication_scaling.py',
              ROOT/'experiments/04_architectures/run_architecture_study.py',
              ROOT/'analysis/analyze_dt_statistical.py']
    return {'native_binary_sha256':sha(lib),'native_version':native_version(),
            'native_build':native_build_info(),
            'runtime':dict(python=sys.version,machine=platform.machine(),
                packages={k:importlib.metadata.version(k) for k in ['numpy','scipy','pandas','matplotlib']},
                thread_environment={k:os.environ.get(k) for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','CPSO_THREADPOOL_CHUNK']}),
            'sources':{str(p.relative_to(ROOT)):sha(p) for p in paths}}


def prepare(folder, design):
    freeze(folder/'protocol.json',design)
    freeze(folder/'provenance.json',provenance())
    if not (folder/'environment.json').exists():
        atomic_write_json(folder/'environment.json',collect_environment_metadata(
            native_version=native_version(),native_build_info=native_build_info()))


def csv(path, rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_csv(path,index=False)


def holm(p):
    p=np.asarray(p,float)
    if not np.isfinite(p).all() or np.any((p<0)|(p>1)):
        raise ValueError('Invalid p-values')
    order=np.argsort(p); result=np.empty_like(p)
    result[order]=np.minimum(1,np.maximum.accumulate(p[order]*np.arange(len(p),0,-1)))
    return result


def paired_effect(values, reps=4000):
    x=np.asarray(values,float)
    if x.ndim!=1 or not len(x) or not np.isfinite(x).all():
        raise ValueError('Invalid paired effects')
    rng=np.random.default_rng(17319)
    boot=x[rng.integers(len(x),size=(reps,len(x)))].mean(axis=1)
    nonzero=x[np.abs(x)>1e-12]
    p=1. if not len(nonzero) else float(wilcoxon(nonzero,method='auto').pvalue)
    ranks=rankdata(np.abs(nonzero))
    biserial=0. if not len(nonzero) else float(np.sum(np.sign(nonzero)*ranks)/ranks.sum())
    return dict(effect=float(x.mean()),lo=float(np.quantile(boot,.025)),hi=float(np.quantile(boot,.975)),
                p=p,rank_biserial=biserial,n_pairs=len(x),wins=int((x<-1e-12).sum()),
                ties=int((np.abs(x)<=1e-12).sum()),losses=int((x>1e-12).sum()))


def native_config(benchmark,d,seed,iterations,dt,topology='global',population=64):
    cfg=configuration(d,seed,topology,population,iterations,dt,benchmark)
    lo,hi=(-2.,2.) if benchmark=='inverse_diffusion' else BOUNDS[benchmark]
    return replace(cfg,lower_bounds=(lo,)*d,upper_bounds=(hi,)*d)


def method(architecture='serial', workers=1, topology='global', **kwargs):
    return dict(architecture=architecture,workers=workers,topology=topology,**kwargs)


def name(m):
    s=f"{m['architecture']}__{m['topology']}__w{m['workers']}"
    if m['architecture']=='islands':
        s+=f"__i{m['islands']}__m{m['migration_interval']}__{m['migration_topology']}"
    return s


def trace_hash(result):
    return fingerprint([result.best_value,result.best_position,result.best_history,result.history_evaluations,
                        result.function_evaluations])


def native_run(benchmark,d,seed,iterations,dt,m,work,population=64):
    cfg=native_config(benchmark,d,seed,iterations,dt,m['topology'],population)
    started=time.perf_counter()
    r=invoke_method(cfg,benchmark,m,work)
    wall=time.perf_counter()-started
    if r.function_evaluations != population*(iterations+1):
        raise ValueError('Native FE budget mismatch')
    result=asdict(r)
    history=np.asarray(r.best_history,float)
    if not np.isfinite(history).all() or np.any(np.diff(history)>1e-10):
        raise ValueError('Invalid native best-so-far history')
    # Independently audit all final native outputs, including inverse diffusion.
    audited=objective(benchmark,r.best_position,work)
    if not math.isclose(audited,r.best_value,rel_tol=2e-8,abs_tol=1e-8):
        raise ValueError(f'Independent objective audit failed: {benchmark}: {audited} != {r.best_value}')
    result.update(wall_seconds=wall,trace_hash=trace_hash(r),initial_best=float(history[0]))
    return result


def inverse_diffusion(x,steps):
    """Independent NumPy implementation of the unchanged native RK4 benchmark.

    A full RK4 solve is performed at each FE; no precomputed/surrogate map.
    Wall times remain implementation-dependent (NumPy vs native C++).
    """
    if steps<16 or steps%4: raise ValueError('RK4 steps must be >=16 and divisible by four')
    x=np.asarray(x,float); d=len(x); j=np.arange(d)
    truth=.4*np.sin(2*np.pi*(j+1)/(d+1)); source=np.tanh(x)-np.tanh(truth)
    decay=1.1+.02*(j%11); h=2./steps; u=np.zeros(d); obs=0.
    def rhs(state,t):
        left=np.concatenate((state[:1],state[:-1]));right=np.concatenate((state[1:],state[-1:]))
        return -decay*state+.2*(left-2*state+right)+(1+.2*np.sin(2*np.pi*t))*source
    for step in range(steps):
        t=h*step; k1=rhs(u,t);k2=rhs(u+.5*h*k1,t+.5*h)
        k3=rhs(u+.5*h*k2,t+.5*h);k4=rhs(u+h*k3,t+h)
        u+=(h/6)*(k1+2*k2+2*k3+k4)
        if (step+1)%(steps//4)==0: obs+=float(u@u)
    return obs/(4*d)+.001*float(np.sum((x-truth)**2))/d


def objective(benchmark,x,work=0):
    return inverse_diffusion(x,work) if benchmark=='inverse_diffusion' else scalar_objective(benchmark,np.asarray(x))


def cached_run(folder,key,execute):
    path=folder/'runs'/(fingerprint(key)+'.json')
    if path.exists():
        r=read(path)
        if r['key']!=key or r.get('result_sha256')!=fingerprint(r['result']):
            raise ValueError('Incompatible or modified cached run')
        return r['result']
    result=execute()
    atomic_write_json(path,dict(key=key,result=result,result_sha256=fingerprint(result)))
    return result


def shuffled(jobs,seed):
    jobs=list(jobs);random.Random(seed).shuffle(jobs);return jobs


def normalized_score(best,initial):
    if initial<=0 or not math.isfinite(initial): raise ValueError('Invalid normalization reference')
    return np.log10(np.clip(np.asarray(best,float)/initial,FLOOR,1.))


def graph_statistics(adjacency):
    """Retained graph diagnostic used by the native regression checks."""
    from scipy.sparse.csgraph import shortest_path
    a=np.asarray(adjacency,bool).copy();np.fill_diagonal(a,False);degree=a.sum(axis=1)
    clustering=[]
    for neighbors in a:
        ids=np.flatnonzero(neighbors);k=len(ids)
        clustering.append(float(a[np.ix_(ids,ids)].sum()/(k*(k-1))) if k>1 else 0.)
    distances=shortest_path(a.astype(float),directed=False,unweighted=True)
    off=distances[~np.eye(len(a),dtype=bool)]
    return dict(mean_degree=float(degree.mean()),min_degree=int(degree.min()),max_degree=int(degree.max()),
        clustering=float(np.mean(clustering)),connected=bool(np.isfinite(off).all()),
        mean_path=float(off.mean()) if np.isfinite(off).all() else None,
        adjacency_sha256=hashlib.sha256(a.tobytes()).hexdigest())


def session_environment():
    data=collect_environment_metadata(native_version=native_version(),native_build_info=native_build_info())
    data['utc']=datetime.now(timezone.utc).isoformat()
    data['observations']={'thermal_state':'not measured','background_load':'not classified'}
    if platform.system()=='Darwin':
        for key in ['hw.model','hw.perflevel0.physicalcpu','hw.perflevel1.physicalcpu']:
            r=subprocess.run(['sysctl','-n',key],capture_output=True,text=True)
            data[key]=r.stdout.strip() if r.returncode==0 else None
        for label,command in [('power',['pmset','-g','batt']),('power_settings',['pmset','-g','custom'])]:
            r=subprocess.run(command,capture_output=True,text=True)
            data[label]=r.stdout.strip() if r.returncode==0 else None
    try:data['load_average']=list(os.getloadavg())
    except OSError:pass
    return data
