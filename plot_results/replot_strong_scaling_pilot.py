#!/usr/bin/env python3
"""Recompute genuine one-worker-reference strong scaling from CPSO pilot runs.

This deliberately does not overwrite or redefine the serial-reference acceleration.
Three seeds are treated as a descriptive pilot: seed ranges, not inferential CIs.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ARCH = ('master_slave', 'particle_parallel')
LABELS = {'master_slave': 'Master–Slave', 'particle_parallel': 'Particle-Parallel'}
COLORS = {'master_slave': '#1B648A', 'particle_parallel': '#C77028'}


def load_runs(folder: Path):
    manifest_file = folder / 'scaling_pilot_manifest.json'
    runs_file = folder / 'scaling_pilot_runs.jsonl'
    manifest = json.loads(manifest_file.read_text(encoding='utf-8'))
    records = []
    for index, line in enumerate(runs_file.read_text(encoding='utf-8').splitlines(), 1):
        if line.strip():
            record = json.loads(line)
            if record.get('chunk') != manifest['chunk_size']:
                raise ValueError(f'{folder}: chunk mismatch in record {index}')
            records.append(record)
    return manifest, records


def calculate(manifest, records):
    experiment_cols = ('benchmark','dimension','topology','work_iterations','seed')
    all_records = {}
    for item in records:
        k = tuple(item[c] for c in experiment_cols) + (item['architecture'],item['workers'])
        if k in all_records:
            raise ValueError(f'Duplicate run key {k}')
        if not math.isfinite(float(item['elapsed_seconds'])) or float(item['elapsed_seconds']) <= 0:
            raise ValueError(f'Nonpositive/nonfinite runtime {k}')
        all_records[k] = item

    paired = []
    fields = ('benchmark','dimension','topology','work_iterations')
    cases = sorted({tuple(item[c] for c in experiment_cols) for item in records})
    expected_cases = len(manifest['benchmarks'])*len(manifest['dimensions'])*len(manifest['topologies'])*len(manifest['work'])*len(manifest['seeds'])
    if len(cases)!=expected_cases:
        raise ValueError(f'Missing cases: {len(cases)} found, {expected_cases} expected')
    worker_counts = sorted(manifest['workers'])
    if worker_counts[0] != 1:
        raise ValueError('One-worker parallel reference is required')
    for case in cases:
        serial_key = case + ('serial',1)
        serial = all_records.get(serial_key)
        if serial is None:
            raise ValueError(f'Missing serial case {case}')
        for arch in ARCH:
            ref = all_records.get(case + (arch,1))
            if ref is None:
                raise ValueError(f'Missing parallel one-worker reference {case} {arch}')
            for workers in worker_counts:
                item = all_records.get(case + (arch,workers))
                if item is None:
                    raise ValueError(f'Missing worker result {case} {arch} p={workers}')
                for key in ('best_value','best_position','best_history','history_evaluations','function_evaluations'):
                    if item[key] != serial[key]:
                        raise ValueError(f'Exact trajectory mismatch at {case}/{arch}/p={workers}/{key}')
                strong_speedup = float(ref['elapsed_seconds'])/float(item['elapsed_seconds'])
                serial_speedup = float(serial['elapsed_seconds'])/float(item['elapsed_seconds'])
                paired.append({
                    'chunk':manifest['chunk_size'],
                    **dict(zip(experiment_cols,case)),
                    'architecture':arch,'workers':workers,
                    'serial_elapsed_seconds':serial['elapsed_seconds'],
                    'parallel_p1_elapsed_seconds':ref['elapsed_seconds'],
                    'parallel_elapsed_seconds':item['elapsed_seconds'],
                    'strong_speedup':strong_speedup,
                    'strong_efficiency':strong_speedup/workers,
                    'serial_relative_speedup':serial_speedup,
                    'serial_relative_efficiency':serial_speedup/workers,
                })
    summaries=[]
    grouped=defaultdict(list)
    for x in paired:
        group=tuple(x[k] for k in fields) + (x['architecture'],x['workers'])
        grouped[group].append(x)
    for group, values in sorted(grouped.items()):
        ss=np.array([v['strong_speedup'] for v in values],dtype=float)
        ee=np.array([v['strong_efficiency'] for v in values],dtype=float)
        aa=np.array([v['serial_relative_speedup'] for v in values],dtype=float)
        chunk=manifest['chunk_size']
        summaries.append({
            'chunk':chunk, **dict(zip(fields,group[:4])),
            'architecture':group[-2], 'workers':group[-1],
            'n_seeds':len(values),
            'strong_speedup_median':float(np.median(ss)),
            'strong_speedup_min':float(np.min(ss)),'strong_speedup_max':float(np.max(ss)),
            'strong_efficiency_median':float(np.median(ee)),
            'strong_efficiency_min':float(np.min(ee)),
            'strong_efficiency_max':float(np.max(ee)),
            'serial_relative_speedup_median':float(np.median(aa)),
        })
    return paired,summaries


def write_csv(file: Path, records: list[dict]):
    if not records: raise ValueError('No records to write')
    with file.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def create_figures(manifest, summary, out):
    plt.rcParams.update({
        'font.family':'DejaVu Sans','font.size':9,'axes.titlesize':10,
        'axes.labelsize':9,'legend.fontsize':8,
        'pdf.fonttype':42,'ps.fonttype':42,
        'axes.spines.top':False,'axes.spines.right':False,
        'savefig.facecolor':'white',
    })
    for benchmark in manifest['benchmarks']:
        for dimension in manifest['dimensions']:
            for topology in manifest['topologies']:
                costs=sorted(manifest['work'])
                if not costs: continue
                fig,axes=plt.subplots(2,len(costs),figsize=(5.0*len(costs),6.0),squeeze=False,sharex='col')
                for j,cost in enumerate(costs):
                    ax_s=axes[0,j];ax_e=axes[1,j]
                    for arch in ARCH:
                        rows=sorted((r for r in summary if r['benchmark']==benchmark and r['dimension']==dimension and r['topology']==topology and r['work_iterations']==cost and r['architecture']==arch),key=lambda r:r['workers'])
                        x=np.array([r['workers'] for r in rows],dtype=int)
                        if x.size==0: raise ValueError('No data for '+benchmark+'/'+topology+'/'+str(cost))
                        for ax,col in ((ax_s,'strong_speedup'),(ax_e,'strong_efficiency')):
                            med=np.array([r[col+'_median'] for r in rows])
                            low=np.array([r[col+'_min'] for r in rows])
                            high=np.array([r[col+'_max'] for r in rows])
                            ax.plot(x,med,marker='o',linewidth=1.8,markersize=5,label=LABELS[arch],color=COLORS[arch])
                            ax.fill_between(x,low,high,color=COLORS[arch],alpha=.13,linewidth=0)
                    ax_s.plot([1,max(manifest['workers'])],[1,max(manifest['workers'])],linestyle='--',color='#6B7280',linewidth=1,label='Ideal')
                    ax_e.axhline(1,linestyle='--',color='#6B7280',linewidth=1)
                    ax_s.set_title(f'Work = {cost} integration steps' if benchmark=='inverse_diffusion' else f'Work = {cost} synthetic iterations',pad=10)
                    for ax in (ax_s,ax_e):
                        ax.set_xticks(sorted(manifest['workers']))
                        ax.grid(alpha=.18)
                        ax.set_xlim(.85,max(manifest['workers'])+.15)
                    ax_s.set_ylim(bottom=0)
                    ax_e.set_ylim(bottom=0)
                    ax_e.set_xlabel('Workers p')
                axes[0,0].set_ylabel('Strong speedup  T(1) / T(p)')
                axes[1,0].set_ylabel('Strong efficiency  T(1) / [p T(p)]')
                handles,labels=axes[0,0].get_legend_handles_labels()
                fig.legend(handles,labels,loc='upper center',ncol=3,frameon=False,bbox_to_anchor=(.5,.943))
                fig.suptitle(f'{benchmark.replace("_"," ").title()}  ·  D={dimension}  ·  {topology}  ·  chunk={manifest["chunk_size"]}',y=.99,size=12)
                fig.text(.5,.015,f'One-worker PARALLEL reference; medians and seed ranges (n={len(manifest["seeds"])}); descriptive pilot, not final publication evidence.',ha='center',size=7,color='#667085')
                fig.tight_layout(rect=(0,.035,1,.885),h_pad=2)
                out_file=out/f'fig_true_strong_scaling_{benchmark}_{topology}_d{dimension}_chunk{manifest["chunk_size"]}.pdf'
                fig.savefig(out_file,bbox_inches='tight')
                plt.close(fig)
                print(out_file)


def main():
    pa=argparse.ArgumentParser(description=__doc__)
    pa.add_argument('--root',type=Path,default=Path('results/raw/parallel'))
    pa.add_argument('--output',type=Path,default=Path('results/figures/pilot_strong_scaling_corrected'))
    args=pa.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    per_seed=[];summaries=[]
    for chunk in (1,8):
        folder=args.root/f'strong_scaling_pilot_chunk{chunk}'
        manifest,records=load_runs(folder)
        a,b=calculate(manifest,records)
        per_seed.extend(a);summaries.extend(b)
        create_figures(manifest,b,args.output)
    write_csv(args.output/'strong_scaling_by_seed.csv',per_seed)
    write_csv(args.output/'strong_scaling_summary.csv',summaries)
    print('Validated records:',len(per_seed),'| summary rows:',len(summaries))

if __name__=='__main__':
    main()
