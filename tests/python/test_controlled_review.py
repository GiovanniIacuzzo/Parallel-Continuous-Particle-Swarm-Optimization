"""Mechanistic checks for controlled native ablations."""
from dataclasses import replace
import numpy as np
import pytest
from cpso_serial.config import CpsoConfig
from cpso_serial.native import run_controlled, run_serial_objective, controlled_graph


def sphere(x): return sum(v*v for v in x)


def cfg(d=5):
    return CpsoConfig((-5.,)*d,(5.,)*d,population_size=64,iterations=8,seed=91,
                      dt=1.,topology='random',topology_degree=8)


def test_legacy_trajectory_unchanged():
    c=cfg(); old=run_serial_objective(c,sphere)
    new=run_controlled(c,sphere)
    assert new.best_history==old.best_history
    assert new.best_position==old.best_position
    assert new.history_evaluations==old.history_evaluations


@pytest.mark.parametrize('scope',['scalar','coordinate'])
@pytest.mark.parametrize('dynamics',['analytic','discrete'])
def test_synchronous_equivalence_and_budget(scope,dynamics):
    c=cfg()
    serial=run_controlled(c,dynamics=dynamics,randomness=scope,graph_seed=71)
    for backend in ['master_slave','particle_parallel']:
        parallel=run_controlled(c,dynamics=dynamics,randomness=scope,graph_seed=71,
                                architecture=backend,workers=4)
        assert parallel.best_history==serial.best_history
        assert parallel.best_position==serial.best_position
        assert parallel.function_evaluations==64*9


def test_all_variants_share_initial_swarm_and_fe_count():
    initials=[]
    for dynamics in ['analytic','discrete']:
        for scope in ['scalar','coordinate']:
            seen=[]
            run_controlled(replace(cfg(),iterations=1),lambda x: seen.append(x) or sphere(x),
                           dynamics=dynamics,randomness=scope,graph_seed=77)
            assert len(seen)==128
            initials.append(seen[:64])
    assert all(v==initials[0] for v in initials)


def test_one_dimension_random_scopes_identical():
    for dynamics in ['analytic','discrete']:
        a=run_controlled(cfg(1),sphere,dynamics=dynamics,randomness='scalar')
        b=run_controlled(cfg(1),sphere,dynamics=dynamics,randomness='coordinate')
        assert a.best_history==b.best_history


def test_graph_seed_changes_graph_not_initial_swarm():
    c=cfg(); a=controlled_graph(c,11);b=controlled_graph(c,12)
    assert np.array_equal(a,a.T) and np.diag(a).all()
    assert not np.array_equal(a,b)
    starts=[]
    for gs in [11,12]:
        points=[]
        run_controlled(replace(c,iterations=0),lambda x: points.append(x) or sphere(x),graph_seed=gs)
        starts.append(points)
    assert starts[0]==starts[1]


def test_graph_metric_definitions():
    from review_study import graph_statistics
    c=replace(cfg(),topology='wheel')
    m=graph_statistics(controlled_graph(c,42))
    assert m['clustering']==0.
    assert m['mean_degree']==pytest.approx(2*63/64)
    assert m['mean_path']==pytest.approx(2-2/64)


def test_invalid_controls_rejected():
    with pytest.raises(ValueError): run_controlled(cfg(),sphere,graph_seed=-1)
    with pytest.raises(ValueError): run_controlled(cfg(),sphere,dynamics='rk4')


def test_discrete_first_step_matches_inertial_drift():
    from cpso_serial.native import controlled_initial_state
    c=replace(cfg(),iterations=1,cognitive=2.,social=0.)
    x,v=controlled_initial_state(c)
    points=[]
    run_controlled(c,lambda p: points.append(p) or sphere(p),dynamics='discrete')
    assert np.array_equal(x,np.array(points[:64]))
    expected=np.clip(x+c.mu_max*v,-5.,5.)
    np.testing.assert_array_equal(np.array(points[64:]),expected)


@pytest.mark.parametrize('dynamics',['analytic','discrete'])
def test_scalar_unbounded_subspace_and_coordinate_escape(dynamics):
    from cpso_serial.native import controlled_initial_state
    c=replace(cfg(180),iterations=3)
    x,v=controlled_initial_state(c);origin=x[0]
    q=np.linalg.qr(np.concatenate((x[1:]-origin,v)).T,mode='reduced')[0]
    shift=np.linspace(-2,2,180)
    residuals=[]
    for scope in ['scalar','coordinate']:
        seen=[]
        def objective(p):
            seen.append(p)
            return np.sum((np.array(p)-shift)**2)
        run_controlled(c,objective,dynamics=dynamics,randomness=scope,boundary_policy='none')
        delta=np.array(seen[64:])-origin
        residuals.append(np.linalg.norm(delta-(delta@q)@q.T)/np.linalg.norm(delta))
    assert residuals[0]<1e-12
    assert residuals[1]>1e-3


def test_controlled_external_dispatch_and_record_metadata():
    import sys
    from pathlib import Path
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'experiments/05_external_benchmarks'))
    from common import run_method, build_run_record
    for scope in ['scalar','coordinate']:
        method=f'matched_discrete_{scope}_random8'
        r=run_method(method,sphere,[-5.]*5,[5.]*5,population_size=64,
                     max_evaluations=128,seed=91,cpso_dt=1.)
        assert r.function_evaluations==128
        record=build_run_record(suite='test',problem_id='f1',problem_name='sphere',function_id=1,
            dimension=5,method=method,seed=91,max_evaluations=128,result=r,milestones=[64,128])
        assert record['algorithm_parameters']['velocity_limit_fraction'] is None
        assert record['algorithm_parameters']['randomness']==scope
        assert record['algorithm_parameters']['dynamics']=='discrete'
