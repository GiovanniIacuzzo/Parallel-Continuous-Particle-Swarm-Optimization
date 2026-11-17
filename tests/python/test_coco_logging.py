"""COCO optimum extraction must not rely on version-specific private attributes."""
from pathlib import Path
import numpy as np
import pytest
from publication_tools.coco_logging import observer_optimum, target_hits, archive_interrupted_observer


def test_official_header_optimum(tmp_path):
    d=tmp_path/'data_f1';d.mkdir()
    (d/'run.dat').write_text('% evaluations | f-value | Fopt (-1.2345e+02)\n1 3 4\n')
    assert observer_optimum(tmp_path)==-123.45


@pytest.mark.parametrize('header',['% no optimum\n','% Fopt (nan)\n','% Fopt (infinity)\n','% Fopt (abc)\n'])
def test_invalid_optimum_rejected(tmp_path,header):
    (tmp_path/'run.dat').write_text(header)
    with pytest.raises(ValueError):observer_optimum(tmp_path)


def test_first_target_hits_include_initialization_and_failures():
    hits,times=target_hits([110.,105.,105.,100.5],[.1,.2,.3,.4],100.,[10.,5.,1.,.01])
    assert hits==[1,2,4,None]
    assert times==[.1,.2,.4,None]


def test_missing_or_multiple_log_files_rejected(tmp_path):
    with pytest.raises(ValueError):observer_optimum(tmp_path)
    for name in ['one.dat','two.dat']:(tmp_path/name).write_text('% Fopt (0)\n')
    with pytest.raises(ValueError):observer_optimum(tmp_path)


def test_interrupted_observer_is_preserved(tmp_path):
    log=tmp_path/'observer'/'run';log.mkdir(parents=True)
    (log/'partial.dat').write_text('keep me')
    archive_interrupted_observer(log)
    assert not log.exists()
    archived=list((tmp_path/'interrupted_observer').glob('run_*/partial.dat'))
    assert len(archived)==1 and archived[0].read_text()=='keep me'


def test_real_coco_logger_without_private_problem_access(tmp_path):
    cocoex=pytest.importorskip('cocoex')
    suite=cocoex.Suite('bbob','instances: 1','dimensions: 5 function_indices: 1')
    problem=suite[0]
    observer=cocoex.Observer('bbob',f'outer_folder: {tmp_path} result_folder: official')
    problem.observe_with(observer)
    count=0
    class PublicProblem:
        def __getattr__(self,name):
            raise AssertionError(f'Unexpected attribute access: {name}')
        def __call__(self,x):
            nonlocal count
            count+=1
            return float(problem(x))
    callback=PublicProblem()
    trace=[callback(np.zeros(5))]
    problem.free()
    optimum=observer_optimum(tmp_path/'official')
    assert np.isfinite(optimum) and trace[0]>=optimum-1e-10
    assert count==1  # Reading Fopt never consumes an extra FE.


def test_logger_with_destination_longer_than_coco_limit(tmp_path):
    import os
    from publication_tools.coco_logging import short_coco_observer
    cocoex=pytest.importorskip('cocoex')
    destination=tmp_path/('a'*80)/('b'*80)/('c'*50)/'observer'
    previous=Path.cwd()
    suite=cocoex.Suite('bbob','instances: 1','dimensions: 5 function_indices: 1')
    p=suite[0]
    with short_coco_observer(p,destination,'analytic_coordinate_scale_free_dt0.38125') as label:
        assert len(label)==13
        value=p([0.]*5)
    assert Path.cwd()==previous
    assert value>=observer_optimum(destination)-1e-10
    assert list(destination.rglob('*.info'))


def test_logger_preserves_partial_logs_and_restores_cwd(tmp_path):
    from publication_tools.coco_logging import short_coco_observer
    cocoex=pytest.importorskip('cocoex')
    previous=Path.cwd()
    suite=cocoex.Suite('bbob','instances: 1','dimensions: 5 function_indices: 1')
    p=suite[0];destination=tmp_path/'failed'
    with pytest.raises(RuntimeError,match='interrupted'):
        with short_coco_observer(p,destination,'test'):
            p([0.]*5)
            raise RuntimeError('interrupted')
    assert Path.cwd()==previous
    assert observer_optimum(destination)==pytest.approx(79.48)
