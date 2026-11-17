"""Official COCO logger handling, extracted from the previous validated audit."""
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import math
import os
from pathlib import Path
import re
import shutil
import tempfile
import numpy as np

TARGETS=np.logspace(2,-8,21).tolist()

def observer_optimum(folder):
    """Read Fopt after COCO has closed/flushed the official run logger."""
    files=list(Path(folder).rglob('*.dat'))
    if len(files)!=1:
        raise ValueError(f'Expected one COCO .dat file in {folder}, found {len(files)}')
    with files[0].open(encoding='utf-8') as stream:
        header=stream.readline()
    match=re.search(r'Fopt\s*\(\s*([^()]+?)\s*\)',header)
    if match is None:
        raise ValueError(f'Missing Fopt in COCO header: {files[0]}')
    try:
        optimum=float(match[1])
    except ValueError as exc:
        raise ValueError(f'Invalid Fopt in COCO header: {files[0]}') from exc
    if not math.isfinite(optimum):
        raise ValueError(f'Nonfinite Fopt in COCO header: {files[0]}')
    return optimum


def target_hits(trace, evaluation_seconds, optimum, targets=TARGETS):
    """Compute first hits offline; add no objective calls and retain censoring."""
    values=np.asarray(trace,float); times=np.asarray(evaluation_seconds,float)
    if (values.ndim!=1 or not len(values) or values.shape!=times.shape
        or not np.isfinite(values).all() or not np.isfinite(times).all()
        or np.any(np.diff(values)>0) or np.any(np.diff(times)<0) or np.any(times<0)
        or not math.isfinite(optimum)):
        raise ValueError('Invalid objective trace/evaluation timestamps')
    hits=[];seconds=[]
    for target in targets:
        if not math.isfinite(target) or target<0:
            raise ValueError('Targets must be finite and nonnegative')
        indices=np.flatnonzero(values-optimum<=target)
        if len(indices):
            j=int(indices[0]);hits.append(j+1);seconds.append(float(times[j]))
        else:
            hits.append(None);seconds.append(None)
    return hits,seconds


def archive_interrupted_observer(log):
    """Keep incomplete logger output instead of appending to it on resume."""
    log=Path(log)
    if not log.exists():return
    archive=log.parent.parent/'interrupted_observer'
    archive.mkdir(parents=True,exist_ok=True)
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    destination=archive/(log.name+'_'+stamp)
    log.rename(destination)
    print(f'Archived interrupted Observer: {destination}',flush=True)


@contextmanager
def short_coco_observer(problem, destination, method):
    """Keep every path passed to COCO short; retain logs at the requested path.

    COCO 2.8.3 can terminate the process on long logger paths. A private
    temporary working directory lets the C logger use './r/...' regardless of
    the repository's path. This serial callback runner never changes cwd while
    other optimization jobs are active. Problem.free closes/flushed the logs
    before copying them back. Failed Python runs retain partial logger files.
    """
    import cocoex
    destination=Path(destination).resolve()
    if destination.exists():raise ValueError(f'Observer destination exists: {destination}')
    previous=Path.cwd()
    label='m'+hashlib.sha256(method.encode()).hexdigest()[:12]
    with tempfile.TemporaryDirectory(prefix='cpso_') as temporary:
        source=Path(temporary)/'r'
        try:
            os.chdir(temporary)
            observer=cocoex.Observer('bbob',
                f'outer_folder: . result_folder: r algorithm_name: {label}')
            problem.observe_with(observer)
            yield label
        finally:
            try:
                problem.free()
            finally:
                os.chdir(previous)
                if source.exists():
                    destination.parent.mkdir(parents=True,exist_ok=True)
                    shutil.copytree(source,destination)

