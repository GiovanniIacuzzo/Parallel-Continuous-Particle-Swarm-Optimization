"""Preserve prior results and stop batch execution when a subprocess fails."""
import json
from pathlib import Path
import sys
from zipfile import ZipFile
import pytest
import review_support as support


def test_archive_preserves_raw_results(tmp_path,monkeypatch):
    monkeypatch.setattr(support,'ROOT',tmp_path)
    raw=tmp_path/'results/review_v1/ablation/smoke/runs/one.json'
    raw.parent.mkdir(parents=True);raw.write_text('{"fe": 640}')
    support.archive()
    assert not (tmp_path/'results/review_v1').exists()
    saved=list((tmp_path/'results/review_archived').glob('*/review_v1/ablation/smoke/runs/one.json'))
    assert len(saved)==1 and saved[0].read_text()=='{"fe": 640}'
    support.archive()  # No source directory: no destructive second action.
    assert saved[0].exists()


def test_logged_failure_stops_with_exit_code(tmp_path,monkeypatch):
    monkeypatch.setattr(support,'ROOT',tmp_path)
    with pytest.raises(SystemExit) as exc:
        support.logged([sys.executable,'-c','print("diagnostic"); raise SystemExit(7)'],'failure')
    assert exc.value.code==7
    logs=list((tmp_path/'results/review_support').glob('*failure.log'))
    assert len(logs)==1
    assert 'diagnostic' in logs[0].read_text() and 'EXIT_CODE: 7' in logs[0].read_text()


def test_collect_contains_partial_results_and_logs(tmp_path,monkeypatch):
    monkeypatch.setattr(support,'ROOT',tmp_path)
    monkeypatch.setattr(support,'information',lambda: (support.support()/'environment.json').write_text('{"test": true}'))
    raw=tmp_path/'results/review_v1/ablation/smoke/runs/partial.json'
    raw.parent.mkdir(parents=True);raw.write_text('{}')
    (support.support()/'run.log').write_text('failed')
    support.collect()
    with ZipFile(tmp_path/'risultati_review_M2.zip') as archive:
        assert archive.testzip() is None
        assert archive.read('results/review_support/run.log')==b'failed'
        assert 'results/review_v1/ablation/smoke/runs/partial.json' in archive.namelist()


def test_collector_retains_old_timings_and_new_ablation(tmp_path,monkeypatch):
    monkeypatch.setattr(support,'ROOT',tmp_path)
    monkeypatch.setattr(support,'information',lambda: None)
    for path in ['results/review_v1/parallel/pilot/data.json','results/review_v2/ablation/pilot/data.json']:
        p=tmp_path/path;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('{}')
    support.collect()
    with ZipFile(tmp_path/'risultati_review_M2.zip') as archive:
        assert 'results/review_v1/parallel/pilot/data.json' in archive.namelist()
        assert 'results/review_v2/ablation/pilot/data.json' in archive.namelist()
