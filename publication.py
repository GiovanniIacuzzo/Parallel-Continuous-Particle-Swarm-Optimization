#!/usr/bin/env python3
"""Three core studies: dt sensitivity, native parallel frameworks, PSO comparison."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys
from zipfile import ZipFile,ZIP_DEFLATED

# Limit nested BLAS pools before importing NumPy/SciPy. Native CPSO workers are
# controlled separately by the immutable study design.
for variable in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:
    os.environ.setdefault(variable,'1')
ROOT=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT),str(ROOT/'python')]
from publication_tools.design import protocol
from publication_tools.common import read,freeze,ROOT,sha,provenance
from study import campaign_lock


def plan(design):
    from publication_tools.parallel import methods
    from publication_tools.comparison import BASELINES,cases,iterations_for
    d=design['dt'];p=design['parallel'];c=design['comparison'];e=design['external']
    ns=len(d['functions'])*len(d['dimensions'])*len(d['iterations'])
    return dict(profile=design['profile'],
        dt=dict(selection_runs=ns*len(d['grid'])*len(d['selection_seeds']),
                holdout_runs_maximum=ns*4*len(d['holdout_seeds']),
                steps=d['grid'],independent_iteration_budgets=d['iterations']),
        parallel=dict(selection_methods=len(methods(design)),
            selection_runs=len(p['functions'])*len(p['dimensions'])*len(p['selection_seeds'])*p['repetitions']*len(p['selection_sessions'])*len(methods(design)),
            holdout='frozen global and case selections plus matched serial references; exact count after selection',
            selection_sessions=p['selection_sessions'],holdout_sessions=p['holdout_sessions'],
            iterations=p['iterations'],rk4_steps=p['work']),
        comparison={suite:len(cases(design,suite))*len(c['seeds'])*len(iterations_for(design,suite))*(len(BASELINES)+2) for suite in ['native','inverse']},
        bbob_serial_quality_runs=len(e['functions'])*len(e['dimensions'])*len(e['instances'])*(len(BASELINES)+1),
        note='No runtime estimate is claimed before measuring the target Mac. Pilot precedes paper. Do not run timing campaigns concurrently.')


def pack(base):
    target=ROOT/f'risultati_CPSO_{base.name}.zip'
    if not base.exists(): raise ValueError(f'No results: {base}')
    with ZipFile(target,'w',ZIP_DEFLATED) as z:
        for p in sorted(base.rglob('*')):
            if p.is_file() and p.suffix!='.lock':z.write(p,p.relative_to(ROOT))
    print(target)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['plan','check','dt','parallel','compare','bbob','figures','pack'])
    parser.add_argument('--profile',choices=['smoke','pilot','paper'],default='pilot')
    parser.add_argument('--phase',choices=['selection','holdout','all'],default='all')
    parser.add_argument('--session',type=int,help='One parallel timing session; cannot combine with --phase all')
    parser.add_argument('--suite',choices=['native','inverse'],default='native')
    parser.add_argument('--benchmark',help='Optional native comparison shard, e.g. rastrigin')
    parser.add_argument('--function',type=int,help='Optional BBOB function shard, e.g. 1')
    parser.add_argument('--output',type=Path,help='New results directory, default results/publication_v2/<profile>')
    args=parser.parse_args();design=protocol(args.profile)
    base=(args.output or ROOT/'results/publication_v2'/args.profile).resolve()
    if not base.is_relative_to(ROOT/'results'):
        parser.error('--output must be inside this checkout/results/')
    if args.session is not None and args.phase=='all':parser.error('--session requires an explicit --phase')
    if args.stage=='plan': print(json.dumps(plan(design),indent=2));return
    if args.stage=='check':
        from cpso_serial import native_version,native_build_info,run_serial
        from publication_tools.common import native_config
        r=run_serial(native_config('sphere',2,1,2,1.,population=16),'sphere')
        print(json.dumps(dict(native_version=native_version(),build=native_build_info(),
            fe=r.function_evaluations,protected_paths_verified=verify_protected()),indent=2));return
    with campaign_lock(ROOT/'results'):
        freeze(base/'protocol.json',design)
        if args.stage not in {'figures','pack'}:
            freeze(base/'execution_provenance.json',provenance())
        if args.stage=='pack':pack(base);return
        if args.stage=='dt':
            from publication_tools.sensitivity import run
            for phase in ['selection','holdout'] if args.phase=='all' else [args.phase]:run(base,design,phase)
        elif args.stage=='parallel':
            from publication_tools.parallel import run
            for phase in ['selection','holdout'] if args.phase=='all' else [args.phase]:run(base,design,phase,args.session)
        elif args.stage=='compare':
            from publication_tools.comparison import run
            run(base,design,args.suite,args.benchmark)
        elif args.stage=='bbob':
            from publication_tools.external import run
            run(base,design,args.function)
        elif args.stage=='figures':
            from publication_tools.figures import dt_figures,parallel_figures,comparison_figures,external_figures
            count=0
            for phase in ['selection','holdout']:
                if (base/f'01_dt/{phase}/complete.json').exists():dt_figures(base,design,phase);count+=1
                if (base/f'02_parallel/{phase}/complete.json').exists():parallel_figures(base,design,phase);count+=1
            for suite in ['native','inverse']:
                if (base/f'03_comparison/{suite}/complete.json').exists():comparison_figures(base,design,suite);count+=1
            if (base/'03_comparison/bbob/complete.json').exists():external_figures(base,design);count+=1
            print(f'Regenerated {count} completed figure groups in {base/"figures"}')


def verify_protected():
    expected=read(ROOT/'docs/protected_files_sha256.json')
    missing=[p for p,h in expected.items() if not (ROOT/p).is_file() or sha(ROOT/p)!=h]
    if missing:raise ValueError(f'Protected files changed: {missing}')
    return len(expected)


if __name__=='__main__':main()
