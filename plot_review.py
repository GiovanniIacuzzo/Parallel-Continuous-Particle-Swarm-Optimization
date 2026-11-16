"""Regenerate review figures as vector PDF from completed campaign CSVs."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from review_study import ROOT, plot_ecdf

plt.rcParams.update({'pdf.fonttype':42,'font.size':9,'axes.spines.top':False,'axes.spines.right':False})


def parallel(root):
    a=pd.read_csv(root/'analysis/speedups.csv');out=root/'analysis'
    p=int(a.workers.max())
    for (d,dyn),g in a.groupby(['dimension','dynamics']):
        fig,axes=plt.subplots(1,2,figsize=(9,3.6),layout='constrained')
        for method in ['master_slave','particle_parallel','replicas']:
            t=g[(g.method==method)&(g.workers==p)].groupby('target_cost').median(numeric_only=True)
            axes[0].plot(t.serial_eval_us,t.serial_batch_speedup,'o-',label=method.replace('_',' '))
        axes[0].set(xscale='log',xlabel='Measured serial evaluation (µs)',ylabel='Batch speedup versus serial',
                    title=f'{p} workers · equal total FE')
        axes[0].axhline(1,color='grey',lw=.7);axes[0].legend(fontsize=7)
        # Phase medians at the least expensive workload, with native wall time
        # in the denominator. Residual includes synchronization and state commits.
        v=g[(g.target_cost==g.target_cost.min())&(g.method!='replicas')]
        med=v.groupby(['method','workers']).median(numeric_only=True)
        chosen=[idx for idx in med.index if idx[1] in {1,p}]
        med=med.loc[chosen];total=med.native_seconds.to_numpy();bottom=np.zeros(len(med))
        names=['evaluation_seconds','propagation_seconds','social_seconds','topology_seconds','initialization_seconds']
        for name in names:
            fraction=med[name].to_numpy()/total
            axes[1].bar(np.arange(len(med)),fraction,bottom=bottom,label=name.replace('_seconds',''))
            bottom+=fraction
        if np.any(bottom>1.01): raise ValueError('Phase times exceed native elapsed time')
        axes[1].bar(np.arange(len(med)),np.maximum(0,1-bottom),bottom=bottom,label='other')
        axes[1].set(xticks=np.arange(len(med)),xticklabels=[f'{m.replace("_slave","").replace("_parallel","")} {w}' for m,w in med.index],
                    ylabel='Fraction of native wall time',title='Phase decomposition · cheapest workload')
        axes[1].tick_params(axis='x',labelrotation=35);axes[1].legend(fontsize=6)
        fig.suptitle(f'D={d} · {dyn} dynamics');fig.savefig(out/f'fig_parallel_d{d}_{dyn}.pdf');plt.close(fig)


def mechanism(root):
    a=pd.read_csv(root/'analysis/subspace_diagnostics.csv');out=root/'analysis'
    d=int(a.dimension.max());a=a[a.dimension==d]
    fig,axes=plt.subplots(1,2,figsize=(8,3.5),layout='constrained')
    for ax,(dyn,g) in zip(axes,a.groupby('dynamics')):
        for j,((scope,boundary),t) in enumerate(g.groupby(['scope','boundary'])):
            vals=t.best_relative_subspace_residual.clip(lower=1e-16)
            ax.scatter(np.full(len(t),j),vals,s=15,alpha=.6)
        groups=list(g.groupby(['scope','boundary']).groups)
        ax.set(yscale='log',xticks=range(len(groups)),xticklabels=[f'{s}\n{b}' for s,b in groups],
               ylabel='Relative residual outside initial affine space',title=f'{dyn} · D={d}')
    fig.savefig(out/'fig_subspace_residual.pdf');plt.close(fig)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('stage',choices=['ablation','parallel','mechanism'])
    ap.add_argument('--profile',choices=['smoke','pilot','paper'],default='pilot');args=ap.parse_args()
    root=ROOT/'results'/('review_v2' if args.stage=='ablation' else 'review_v1')/args.stage/args.profile
    if args.stage=='ablation':plot_ecdf(pd.read_csv(root/'analysis/ecdf.csv'),root/'analysis')
    elif args.stage=='parallel':parallel(root)
    else:mechanism(root)

if __name__=='__main__':main()
