"""Local recovery, logged execution and result packaging for the review studies."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from zipfile import ZipFile, ZIP_DEFLATED
from study import campaign_lock

ROOT=Path(__file__).resolve().parent


def stamp():return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')


def support():
    folder=ROOT/'results/review_support';folder.mkdir(parents=True,exist_ok=True)
    return folder


def information():
    versions={}
    for name in ['coco-experiment','numpy','scipy','pandas','pytest']:
        try:versions[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:versions[name]=None
    data=dict(utc=datetime.now(timezone.utc).isoformat(),python=sys.version,executable=sys.executable,
              machine=platform.machine(),platform=platform.platform(),packages=versions,
              threads={k:os.environ.get(k) for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS','CPSO_THREADPOOL_CHUNK']})
    try:
        sys.path.insert(0,str(ROOT/'python'))
        from cpso_serial.native import _load_library
        library=_load_library()
        data['native_build']=library.cpso_build_info().decode()
    except Exception as exc:data['native_error']=str(exc)
    path=support()/'environment.json';path.write_text(json.dumps(data,indent=2)+'\n')
    print(json.dumps(data,indent=2),flush=True)
    return data


def archive():
    with campaign_lock(ROOT/'results'):
        current=ROOT/'results/review_v1'
        if not current.exists():
            print('No previous review_v1 results to archive.');return
        destination=ROOT/'results/review_archived'/stamp()/'review_v1'
        destination.parent.mkdir(parents=True,exist_ok=False)
        current.rename(destination)
        record=dict(utc=datetime.now(timezone.utc).isoformat(),archived_path=str(destination.relative_to(ROOT)),
                    reason='Preserve earlier results before applying corrected COCO logging protocol')
        (support()/('archive_'+stamp()+'.json')).write_text(json.dumps(record,indent=2)+'\n')
        print(f'Previous results preserved in: {destination}',flush=True)


def logged(command,label):
    file=support()/(stamp()+'_'+label+'.log')
    print('RUN: '+' '.join(map(str,command)),flush=True)
    print(f'LOG: {file}',flush=True)
    with file.open('w',encoding='utf-8') as output:
        output.write('COMMAND: '+' '.join(map(str,command))+'\n');output.flush()
        process=subprocess.Popen(command,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                                 text=True,encoding='utf-8',errors='replace',bufsize=1)
        try:
            for line in process.stdout:
                print(line,end='',flush=True);output.write(line);output.flush()
            code=process.wait()
        except BaseException:
            process.terminate();process.wait();raise
        output.write(f'\nEXIT_CODE: {code}\n')
    if code:
        print(f'STOP: {label} failed (exit {code}). Send the collected diagnostics.',flush=True)
        raise SystemExit(code)


def execute(action,session):
    information();py=sys.executable
    if action=='smoke':
        jobs=[([py,'-m','pytest','-q'],'tests'),
              ([py,'review_study.py','ablation','--profile','smoke'],'bbob_smoke'),
              ([py,'review_mechanism.py','--profile','smoke'],'mechanism_smoke')]
        jobs += [([py,'review_parallel.py','run','--profile','smoke','--session',str(i)],f'parallel_smoke_s{i}') for i in [1,2,3]]
        jobs += [([py,'review_stability.py'],'stability')]
    elif action=='quality':
        jobs=[([py,'review_study.py','ablation','--profile','pilot'],'bbob_pilot'),
              ([py,'review_mechanism.py','--profile','pilot'],'mechanism_pilot')]
    else:
        jobs=[([py,'review_parallel.py','run','--profile','pilot','--session',str(session)],f'parallel_pilot_s{session}')]
    for command,label in jobs:logged(command,label)
    print(f'COMPLETE: {action}',flush=True)


def collect():
    with campaign_lock(ROOT/'results'):
        information()
        target=ROOT/'risultati_review_M2.zip'
        with ZipFile(target,'w',ZIP_DEFLATED) as out:
            for folder in [ROOT/'results/review_v1',support()]:
                if folder.exists():
                    for path in sorted(folder.rglob('*')):
                        if path.is_file() and not path.is_symlink():
                            out.write(path,path.relative_to(ROOT))
        with ZipFile(target) as archive_file:
            if archive_file.testzip() is not None:raise RuntimeError('Archive integrity failed')
        print(f'SEND: {target}',flush=True)
        print('Includes current raw data, Observer logs, analyses, execution logs and environment. Older archived raw runs stay on your computer.',flush=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('action',choices=['archive','info','smoke','quality','timing','collect'])
    ap.add_argument('--session',type=int,choices=[1,2,3],default=1)
    args=ap.parse_args()
    if args.action=='archive':archive()
    elif args.action=='info':information()
    elif args.action=='collect':collect()
    else:execute(args.action,args.session)

if __name__=='__main__':main()
