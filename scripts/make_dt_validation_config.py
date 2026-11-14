#!/usr/bin/env python3
"""Freeze top dt candidates selected on fresh training seeds for independent holdout."""
import argparse
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--selection-summary',type=Path,required=True)
    p.add_argument('--selection-config',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--candidate-count',type=int,default=3)
    args=p.parse_args()
    from cpso_serial.output_policy import results_output
    args.output = results_output(args.output, name="frozen holdout config output")
    if not 2<=args.candidate_count<=8:
        p.error('--candidate-count must be between 2 and 8')
    sel=json.loads(args.selection_summary.read_text())
    cfg=json.loads(args.selection_config.read_text())
    if sel['campaign_id']!=cfg['campaign_id']:
        raise ValueError('The selection summary does not match the supplied config')
    chosen=sel['ranked_dt_candidates'][:args.candidate_count]
    frozen=float(sel['candidate_for_validation_dt'])
    if frozen not in chosen:
        raise ValueError('Frozen candidate absent from holdout list')
    reference=float(cfg['reference_dt'])
    selected=sorted(set(chosen+[reference]))
    old=set(cfg['seeds'])
    seeds=list(range(920001,920031))
    if old.intersection(seeds):
        raise ValueError('Holdout seed collision')
    cfg.update({
        'campaign_id':'cpso_dt_independent_holdout_m2',
        'title':'CPSO dt final precommitted held-out validation',
        'description':'Held-out comparison of frozen dt and nearest selection alternatives, independent seed identifiers; frozen_dt must not be changed after observing results.',
        'schedule_seed':20261124,
        'seeds':seeds,
        'dt_values':selected,
        'reference_dt':reference,
        'evidence_class':'confirmatory',
        'frozen_dt':frozen,
        'selection_campaign_id':sel['campaign_id'],
        'selected_candidates_predeclared':chosen,
    })
    args.output.parent.mkdir(parents=True,exist_ok=True)
    if args.output.exists():
        raise FileExistsError(f'Refusing to overwrite frozen holdout: {args.output}')
    args.output.write_text(json.dumps(cfg,indent=2)+'\n')
    print(f'Frozen candidate: {frozen:g}')
    print('Holdout dt grid:',selected)
    print('Fresh seeds:',len(seeds))
    print('Runs:',len(selected)*len(seeds)*len(cfg['benchmarks'])*len(cfg['dimensions']))
    print('Output:',args.output)

if __name__=='__main__':
    main()
