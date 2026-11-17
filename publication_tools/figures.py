"""Vector PDF figures, English labels, fixed palette and print-sized fonts."""
from __future__ import annotations
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from .common import read,LABELS,name,method

COLORS=['#1f5b85','#c4452d','#26847d','#8c64a4','#c68c24','#677885','#8a5538','#658b3c','#b45b91','#3d6e70','#777777']
ARCH_COLORS=dict(serial='#252525',master_slave='#1f5b85',particle_parallel='#c4452d',asynchronous='#26847d',islands='#8c64a4')
BASE_COLORS=[c for i,c in enumerate(COLORS) if i!=1]
NAMES={'CPSO selected framework':'CPSO selected','classic_gbest':'Classic gbest','inertia_gbest':'Inertia gbest','inertia_lbest_ring':'Inertia ring',
       'constriction_gbest':'Constriction','clpso':'CLPSO','slpso':'SL-PSO','spso2011':'SPSO-2011',
       'fips_ring':'FIPS','cso':'CSO','dms_pso':'DMS-PSO','CPSO serial reference':'CPSO reference (serial)'}


def style():
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.labelsize':9,'axes.titlesize':10,
        'xtick.labelsize':8,'ytick.labelsize':8,'legend.fontsize':8,'axes.spines.top':False,
        'axes.spines.right':False,'lines.linewidth':1.6,'lines.markersize':4,
        'pdf.fonttype':42,'ps.fonttype':42,'axes.grid':True,'grid.alpha':.18,'grid.linewidth':.5})


def save(fig,path):
    path.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(path,bbox_inches='tight',pad_inches=.08)
    plt.close(fig)


def legend(fig,ax,ncol=3):
    handles,labels=ax.get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,-.08),borderaxespad=0,ncol=ncol,frameon=False)


def dt_figures(base,design,phase):
    style();out=base/'01_dt'/phase/'analysis';a=pd.read_csv(out/'paired_results.csv')
    dest=base/'figures'/'01_dt'/phase
    fig,axes=plt.subplots(1,2,figsize=(7.4,3.2),layout='constrained')
    for j,(it,g) in enumerate(a.groupby('iterations')):
        means=g.groupby(['seed','dt']).score.mean().unstack();rng=np.random.default_rng(912)
        boot=means.to_numpy()[rng.integers(len(means),size=(design['inference']['bootstrap'],len(means)))].mean(axis=1)
        x=means.columns.to_numpy(float);y=means.mean().to_numpy()
        axes[0].plot(x,y,'o-',color=COLORS[j],label=f'{design["population"]*(it+1):,} FE')
        axes[0].fill_between(x,np.quantile(boot,.025,axis=0),np.quantile(boot,.975,axis=0),color=COLORS[j],alpha=.12)
        au=g.groupby('dt').anytime_log_residual.mean()
        axes[1].plot(au.index,au.values,'o-',color=COLORS[j])
    frozen=read(base/'01_dt/selection/frozen.json')['dt']
    for ax in axes:
        ax.set_xscale('log');ax.set_xlabel(r'Time step $dt$');ax.axvline(frozen,color='#555',ls=':',lw=1.)
    axes[0].set(title='Final residual',ylabel=r'Mean $\log_{10}(f_B/f_0)$')
    axes[1].set(title='Anytime improvement',ylabel='Mean integrated log residual')
    legend(fig,axes[0]);fig.subplots_adjust(bottom=.22) if not fig.get_constrained_layout() else None
    save(fig,dest/'fig_dt_budget_sensitivity.pdf')
    for d,group in a.groupby('dimension'):
        funcs=list(group.benchmark.unique());cols=min(3,len(funcs));nrows=math.ceil(len(funcs)/cols)
        fig,axes=plt.subplots(nrows,cols,figsize=(7.4,2.5*nrows),squeeze=False,layout='constrained')
        for ax,(b,g) in zip(axes.flat,group.groupby('benchmark')):
            for j,(it,gg) in enumerate(g.groupby('iterations')):
                stats=gg.groupby('dt').score.agg(['mean','std','count'])
                ax.plot(stats.index,stats['mean'],'o-',color=COLORS[j],label=f'{design["population"]*(it+1):,} FE')
            ax.set_xscale('log');ax.set(title=b.capitalize(),xlabel=r'$dt$',ylabel='Mean log residual')
            ax.axvline(frozen,color='#555',ls=':',lw=1.)
        for ax in list(axes.flat)[len(funcs):]:ax.set_visible(False)
        legend(fig,axes.flat[0]);save(fig,dest/f'fig_dt_profiles_d{d}.pdf')
    if phase=='holdout':
        a=pd.read_csv(out/'contrasts.csv');a=a[a.scope.isin(['all_budgets_cells','budget'])]
        fig,ax=plt.subplots(figsize=(7.4,max(3.,.32*len(a)+.8)),layout='constrained')
        y=np.arange(len(a));labels=[f"{r.candidate:g} vs {r.opponent:g} | "+('all budgets' if r.iterations==0 else f'K={r.iterations}') for r in a.itertuples()]
        for i,r in enumerate(a.itertuples()):
            ax.plot([r.lo,r.hi],[i,i],color=COLORS[0]);ax.plot(r.effect,i,'o',color=COLORS[0])
        ax.set_yticks(y,labels);ax.axvline(0,color='#555',ls=':');ax.invert_yaxis()
        ax.set_xlabel('Paired mean log-residual difference (negative favors frozen dt)')
        save(fig,dest/'fig_dt_holdout_effects.pdf')


def parallel_figures(base,design,phase):
    style();plt.rcParams.update({'font.size':11,'axes.labelsize':11,'axes.titlesize':12,'xtick.labelsize':11,'ytick.labelsize':11,'legend.fontsize':11})
    folder=base/'02_parallel'/phase/'analysis';a=pd.read_csv(folder/'scaling_summary.csv')
    dest=base/'figures'/'02_parallel'/phase
    if phase=='selection':
        for (b,d),g in a.groupby(['benchmark','dimension']):
            for metric,label in [('serial_speedup','Speedup vs serial'),('strong_speedup','Backend strong scaling')]:
                nc=min(4,len(design['parallel']['topologies']));nr=math.ceil(len(design['parallel']['topologies'])/nc)
                fig,axes=plt.subplots(nr,nc,figsize=(10. if nc==4 else 7.4,2.7*nr),squeeze=False,layout='constrained')
                for ax,topology in zip(axes.flat,design['parallel']['topologies']):
                    for arch in ['master_slave','particle_parallel']:
                        rows=g[(g.metric==metric)&(g.topology==topology)&(g.architecture==arch)].sort_values('workers')
                        if rows.empty:continue
                        ax.plot(rows.workers,rows.estimate,'o-',color=ARCH_COLORS[arch],label=arch.replace('_',' ').title())
                        ax.fill_between(rows.workers,rows.lo,rows.hi,color=ARCH_COLORS[arch],alpha=.15)
                    p=design['parallel']['workers'];ax.plot(p,p,':',color='#999',lw=1,label='Ideal' if ax==axes.flat[0] else None)
                    ax.set(title=LABELS[topology],xlabel='Workers',ylabel=label);ax.set_xticks(p)
                for ax in list(axes.flat)[len(design['parallel']['topologies']):]:ax.set_visible(False)
                legend(fig,axes.flat[0]);save(fig,dest/f'fig_{metric}_{b}_d{d}.pdf')
            nc=min(4,len(design['parallel']['topologies']));nr=math.ceil(len(design['parallel']['topologies'])/nc)
            fig,axes=plt.subplots(nr,nc,figsize=(10. if nc==4 else 7.4,2.7*nr),squeeze=False,layout='constrained')
            for ax,topology in zip(axes.flat,design['parallel']['topologies']):
                for arch in ['master_slave','particle_parallel']:
                    rows=g[(g.metric=='serial_speedup')&(g.topology==topology)&(g.architecture==arch)].sort_values('workers')
                    if rows.empty:continue
                    ax.plot(rows.workers,100*rows.estimate/rows.workers,'o-',color=ARCH_COLORS[arch],label=arch.replace('_',' ').title())
                    ax.fill_between(rows.workers,100*rows.lo/rows.workers,100*rows.hi/rows.workers,color=ARCH_COLORS[arch],alpha=.15)
                ax.set(title=LABELS[topology],xlabel='Workers',ylabel='Serial-reference efficiency (%)',ylim=(0,120));ax.set_xticks(design['parallel']['workers'])
            for ax in list(axes.flat)[len(design['parallel']['topologies']):]:ax.set_visible(False)
            legend(fig,axes.flat[0]);save(fig,dest/f'fig_efficiency_{b}_d{d}.pdf')
        ranking=pd.read_csv(folder/'selection_ranking.csv');raw=pd.read_csv(folder/'paired_runs.csv')
        fig,ax=plt.subplots(figsize=(7.4,4.),layout='constrained')
        lookup=raw.drop_duplicates('method').set_index('method')
        for arch in ARCH_COLORS:
            points=ranking[ranking.method.map(lookup.architecture)==arch]
            ax.scatter(points.mean_log_time,points.mean_quality,s=22,color=ARCH_COLORS[arch],alpha=.7,label=arch.replace('_',' ').title())
        frozen=read(base/'02_parallel/selection/frozen.json');winner=ranking.set_index('method').loc[name(frozen['global_method'])]
        ax.plot(winner.mean_log_time,winner.mean_quality,'*',ms=15,color='#000',label='Frozen global selection')
        ax.set(xlabel='Mean log10 completed-run wall time (s)',ylabel='Mean log10 residual (lower is better)')
        legend(fig,ax,3);save(fig,dest/'fig_framework_quality_cost.pdf')
    else:
        frozen=read(base/'02_parallel/selection/frozen.json');m=name(frozen['global_method'])
        g=a[(a.method==m)&(a.metric=='serial_speedup')]
        fig,ax=plt.subplots(figsize=(7.4,3.4),layout='constrained')
        for i,r in enumerate(g.itertuples()):
            ax.plot([r.lo,r.hi],[i,i],color=COLORS[0]);ax.plot(r.estimate,i,'o',color=COLORS[0])
        ax.set_yticks(range(len(g)),[f'{r.benchmark}, D={r.dimension}' for r in g.itertuples()]);ax.axvline(1,color='#555',ls=':')
        ax.set_xlabel('Held-out serial runtime ratio (native timer)');ax.invert_yaxis()
        save(fig,dest/'fig_framework_holdout.pdf')


def comparison_figures(base,design,suite):
    style();out=base/'03_comparison'/suite/'analysis';a=pd.read_csv(out/'rank_summary.csv')
    dest=base/'figures'/'03_comparison'/suite
    for it,group in a.groupby('iterations'):
        dims=list(group.dimension.unique());fig,axes=plt.subplots(math.ceil(len(dims)/2),2,
            figsize=(7.4,4.2*math.ceil(len(dims)/2)),squeeze=False,layout='constrained')
        order=group[group.dimension==dims[0]].sort_values('mean_rank').method.tolist()
        for ax,d in zip(axes.flat,dims):
            g=group[group.dimension==d].set_index('method').loc[order]
            for j,(m,r) in enumerate(g.iterrows()):
                color=COLORS[1] if m=='CPSO selected framework' else COLORS[0]
                ax.plot([r.lo,r.hi],[j,j],color=color);ax.plot(r.mean_rank,j,'o',color=color)
            ax.set_yticks(range(len(order)),[NAMES.get(m,m) for m in order]);ax.invert_yaxis()
            ax.set(title=f'D = {d}',xlabel='Mean rank (lower is better)',xlim=(.6,11.4))
        for ax in list(axes.flat)[len(dims):]:ax.set_visible(False)
        save(fig,dest/f'fig_rank_k{it}.pdf')
    raw=pd.read_csv(out/'ranked_results.csv')
    methods=['CPSO selected framework']+[m for m in sorted(raw.method.unique()) if m!='CPSO selected framework']
    # Each point is a separate run budget. Do not invent per-generation times.
    for d,group in raw.groupby('dimension'):
        fig,axes=plt.subplots(1,2,figsize=(7.4,3.5),layout='constrained')
        for j,m in enumerate(methods):
            g=group[group.method==m].groupby('iterations').agg(rank=('rank','mean'),seconds=('seconds','median'),fe=('fe','first'))
            color=COLORS[1] if m=='CPSO selected framework' else BASE_COLORS[j-1]
            axes[0].plot(g.fe,g['rank'],'o-',color=color,label=NAMES.get(m,m))
            axes[1].plot(g.seconds,g['rank'],'o-',color=color)
        for ax in axes: ax.set_ylabel('Mean final rank');ax.set_xscale('log')
        axes[0].set_xlabel('Function evaluations');axes[1].set_xlabel('Completed-run wall time (s)')
        legend(fig,axes[0],3);save(fig,dest/f'fig_completed_run_tradeoff_d{d}.pdf')


def external_figures(base,design):
    style();out=base/'03_comparison/bbob/analysis';a=pd.read_csv(out/'target_hits.csv')
    dest=base/'figures'/'03_comparison/bbob';dims=design['external']['dimensions']
    fig,axes=plt.subplots(1,len(dims),figsize=(7.4,3.5),squeeze=False,layout='constrained')
    for ax,d in zip(axes.flat,dims):
        g=a[a.dimension==d]
        x=np.geomspace(1,design['external']['checkpoints'][-1],150)
        methods=['CPSO serial reference']+[m for m in sorted(g.method.unique()) if m!='CPSO serial reference']
        for j,m in enumerate(methods):
            z=g[g.method==m];hits=z.hit_fe.to_numpy()/d
            y=np.array([(hits<=t).sum()/len(hits) for t in x])
            ax.step(x,y,where='post',label=NAMES.get(m,m),color=COLORS[1] if j==0 else BASE_COLORS[j-1],lw=2.1 if j==0 else 1.1)
        ax.set(xscale='log',xlabel='Evaluations / dimension',ylabel='Solved problem-target fraction',title=f'BBOB, D = {d}',ylim=(0,1))
    legend(fig,axes.flat[0],3);save(fig,dest/'fig_bbob_ecdf_serial_control.pdf')
