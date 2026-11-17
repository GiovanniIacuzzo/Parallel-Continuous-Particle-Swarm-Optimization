"""Checks for protocol integrity, inferential units and unchanged native execution."""
from dataclasses import replace
import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'python')]
from publication_tools.common import (holm,paired_effect,freeze,cached_run,native_config,native_run,
                                      method,objective,BOUNDS)
from publication_tools.design import protocol
from publication_tools.parallel import methods,choose
from publication_tools.sensitivity import driver
from publication_tools.external import TARGETS
from publication import verify_protected


def test_holm_and_zero_effects():
    np.testing.assert_allclose(holm([.03,.01,.04]),[.06,.03,.06])
    stat=paired_effect(np.zeros(8),1000)
    assert stat['p']==1 and stat['lo']==stat['hi']==0 and stat['rank_biserial']==0
    with pytest.raises(ValueError):holm([float('nan')])


def test_frozen_protocol_and_cached_run_are_immutable(tmp_path):
    freeze(tmp_path/'protocol.json',{'grid':[1,2]})
    with pytest.raises(ValueError):freeze(tmp_path/'protocol.json',{'grid':[1,3]})
    counter=[]
    first=cached_run(tmp_path,{'seed':1},lambda:counter.append(1) or {'best':2})
    again=cached_run(tmp_path,{'seed':1},lambda:pytest.fail('must reuse completed run'))
    assert first==again and counter==[1]


@pytest.mark.parametrize('profile',['smoke','pilot','paper'])
def test_design_disjoint_and_architectures_supported(profile):
    d=protocol(profile)
    for stage in ['dt','parallel']:
        assert not set(d[stage]['selection_seeds'])&set(d[stage]['holdout_seeds'])
    allmethods=methods(d)
    assert {m['architecture'] for m in allmethods}=={'serial','master_slave','particle_parallel','asynchronous','islands'}
    assert len({json.dumps(m,sort_keys=True) for m in allmethods})==len(allmethods)
    for m in allmethods:
        n=d['population']//m['islands'] if m['architecture']=='islands' else d['population']
        native_config('sphere',2,1,1,1.,m['topology'],n).validate()
        if m['architecture']=='islands':assert m['workers']<=m['islands']


@pytest.mark.parametrize('b',list(BOUNDS)+['inverse_diffusion'])
def test_native_bounds_and_independent_objective_agree(b):
    lo,hi=(-2.,2.) if b=='inverse_diffusion' else BOUNDS[b]
    c=native_config(b,4,123,3,1.,population=16)
    assert c.lower_bounds==(lo,)*4 and c.upper_bounds==(hi,)*4
    serial=native_run(b,4,123,3,1.,method(),16 if b=='inverse_diffusion' else 0,16)
    pp=native_run(b,4,123,3,1.,method('particle_parallel',2),16 if b=='inverse_diffusion' else 0,16)
    assert serial['trace_hash']==pp['trace_hash']
    assert serial['function_evaluations']==64


def test_incomplete_dt_checkpoint_cannot_be_analyzed(tmp_path):
    from publication_tools.sensitivity import data
    d=protocol('smoke');folder=tmp_path/'01_dt/selection/k2';folder.mkdir(parents=True)
    (folder/'config.json').write_text(json.dumps(dict(campaign_id='missing',schedule_seed=1,
        benchmarks=d['dt']['functions'],dimensions=d['dt']['dimensions'],seeds=d['dt']['selection_seeds'],
        dt_values=d['dt']['grid'],population_size=16,iterations=2,reference_dt=30.,topology='global')))
    with pytest.raises(ValueError,match='Incomplete dt campaign'):data(tmp_path,d,'selection')


def test_selection_keeps_quality_before_time():
    import pandas as pd
    d=protocol('smoke');rows=[]
    for seed in [1,2,3]:
        for m,q,t in [('fast_bad',-1.,.001),('slower_good',-3.,.1)]:
            rows.append(dict(benchmark='sphere',dimension=4,seed=seed,method=m,wall_seconds=t,quality=q))
    chosen,_=choose(pd.DataFrame(rows),d)
    assert chosen=='slower_good'


def test_native_and_paper_files_unchanged():
    assert verify_protected()>20
    assert len(TARGETS)==21 and TARGETS[0]==100. and TARGETS[-1]==1e-8
