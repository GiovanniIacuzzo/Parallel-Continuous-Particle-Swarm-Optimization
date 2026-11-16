#!/usr/bin/env python3
"""Publication plots from completed review studies; never runs an optimizer.

Default: four complementary figures. Optional timing diagnostics are requested
with --supplementary. Output is vector PDF plus captions and provenance JSON.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, ScalarFormatter
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
COLORS=dict(navy='#1E344B',blue='#265D78',rust='#A94124',teal='#3A8D8C',
            gold='#C28A2C',purple='#76558F',grey='#6B7280',light='#BCC4CC')
PRIMARY='analytic_coordinate_random8_dt1'
METHODS=[PRIMARY,'analytic_scalar_random8_dt1','analytic_coordinate_grid_dt1',
         'discrete_scalar_random8_dt1','discrete_coordinate_random8_dt1','DE']
LABELS={PRIMARY:'CPSO R8, coordinate','analytic_scalar_random8_dt1':'CPSO R8, scalar',
 'analytic_coordinate_grid_dt1':'CPSO grid, coordinate',
 'discrete_scalar_random8_dt1':'Discrete R8, scalar',
 'discrete_coordinate_random8_dt1':'Discrete R8, coordinate','DE':'DE'}
STYLES={PRIMARY:(COLORS['rust'],'-', 'o'),'analytic_scalar_random8_dt1':(COLORS['blue'],'--','s'),
 'analytic_coordinate_grid_dt1':(COLORS['teal'],'-','^'),
 'discrete_scalar_random8_dt1':(COLORS['navy'],':','D'),
 'discrete_coordinate_random8_dt1':(COLORS['grey'],'--','v'),'DE':('#222222','-.','x')}
CORE=['ecdf','ablations','mechanism','scaling']


def configure_style(style='paper'):
    # Matches the DejaVu Sans figures already stored in paper/images.
    plt.rcParams.update({'font.family':'DejaVu Sans','mathtext.fontset':'dejavusans',
        'font.size':8,'axes.titlesize':9,'axes.labelsize':8,'xtick.labelsize':7,
        'ytick.labelsize':7,'legend.fontsize':7,'axes.spines.top':False,
        'axes.spines.right':False,'axes.linewidth':.7,'lines.linewidth':1.45,
        'lines.markersize':3.8,'grid.alpha':.2,'grid.linewidth':.5,
        'pdf.fonttype':42,'ps.fonttype':42,'savefig.bbox':'tight','savefig.pad_inches':.04})
    if style=='serif':
        plt.rcParams.update({'font.family':'serif','font.serif':['STIX Two Text','STIXGeneral','DejaVu Serif'],
                             'mathtext.fontset':'stix'})


class Plotter:
    def __init__(self,results,profile,output,gap_floor=1e-12,style='paper'):
        self.results=Path(results).resolve();self.profile=profile
        self.output=Path(output).resolve();self.output.mkdir(parents=True,exist_ok=True)
        self.gap_floor=gap_floor;self.sources={};self.figures=[];self.skipped=[]
        self.quality=None;self.quality_path=None;self.design=None
        configure_style(style)
        self.style=style

    def read(self,path):
        path=Path(path)
        if not path.is_file():raise FileNotFoundError(path)
        self.sources[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
        return pd.read_csv(path)

    def load_quality(self):
        if self.quality is not None:return self.quality
        candidates=[self.results/v/'ablation'/self.profile for v in ['review_v2','review_v1']]
        path=next((p for p in candidates if (p/'analysis/runs.csv').exists()),None)
        if path is None:raise FileNotFoundError('No completed BBOB analysis/runs.csv in review_v2 or review_v1')
        design_path=path/'design.json'
        design=json.loads(design_path.read_text())
        self.sources[str(design_path)]=hashlib.sha256(design_path.read_bytes()).hexdigest()
        a=self.read(path/'analysis/runs.csv')
        keys=['dimension','function','instance','method']
        required=keys+['gap','initial_sha256','fe']
        if set(required)-set(a):raise ValueError('Missing BBOB columns')
        if a.duplicated(keys).any():raise ValueError('Duplicate BBOB run')
        methods=[v[0] for v in design['variants']]+['DE']
        expected=pd.MultiIndex.from_product([design['dimensions'],design['functions'],design['instances'],methods],names=keys)
        observed=pd.MultiIndex.from_frame(a[keys])
        if len(a)!=len(expected) or len(expected.difference(observed)) or len(observed.difference(expected)):
            raise ValueError('Incomplete BBOB campaign; publication plots require all design cells')
        if not np.isfinite(a.gap).all() or (a.gap<0).any():raise ValueError('Invalid BBOB gaps')
        if not (a.fe==a.dimension*design['budget_per_d']).all():raise ValueError('Inconsistent BBOB FE budgets')
        native=a[a.method!='DE']
        if not native.groupby(keys[:3]).initial_sha256.nunique().eq(1).all():
            raise ValueError('Native initial states are not matched')
        if set(METHODS)-set(a.method):raise ValueError('Required matched controls absent from campaign')
        self.quality=a;self.quality_path=path;self.design=design
        return a

    def save(self,fig,name,caption,role='main'):
        path=self.output/(name+'.pdf')
        fig.savefig(path,format='pdf',metadata={'Creator':'CPSO review publication plotting',
                                              'Title':name,'Subject':self.profile+' descriptive study'})
        plt.close(fig)
        self.figures.append(dict(file=path.name,role=role,caption=caption))
        print('WROTE',path,flush=True)

    def ecdf(self):
        self.load_quality();a=self.read(self.quality_path/'analysis/ecdf.csv')
        keys=['dimension','method','fe_per_d']
        if a.duplicated(keys).any() or not np.isfinite(a[['fe_per_d','fraction']]).all().all():
            raise ValueError('Invalid ECDF data')
        if (a.fe_per_d<=0).any() or not a.fraction.between(0,1).all():raise ValueError('Invalid ECDF ranges')
        dims=sorted(self.quality.dimension.unique());fig,axes=plt.subplots(1,len(dims),figsize=(8.1,3.65),squeeze=False,sharey=True)
        for j,(ax,d) in enumerate(zip(axes[0],dims)):
            for method in METHODS:
                g=a[(a.dimension==d)&(a.method==method)].sort_values('fe_per_d')
                if g.empty or np.any(np.diff(g.fraction)<-1e-12):raise ValueError('Missing/nonmonotone ECDF')
                c,ls,_=STYLES[method]
                ax.step(g.fe_per_d,g.fraction,where='post',color=c,ls=ls,label=LABELS[method],lw=1.8 if method==PRIMARY else 1.3)
            ax.set(xscale='log',ylim=(0,1),xlabel='Function evaluations / dimension',title=f'({chr(97+j)}) BBOB, D = {d}')
            ax.grid(True,which='major');ax.set_axisbelow(True)
        axes[0,0].set_ylabel('Fraction of problem-target pairs solved')
        handles,labels=axes[0,0].get_legend_handles_labels()
        fig.legend(handles,labels,loc='lower center',ncol=3,frameon=False,bbox_to_anchor=(.5,.005))
        fig.subplots_adjust(left=.075,right=.99,top=.9,bottom=.29,wspace=.14)
        nfunc=len(self.design['functions']);ninst=len(self.design['instances'])
        self.save(fig,'fig_review_bbob_ecdf',
            f'BBOB target ECDF for {nfunc} functions and {ninst} instances per dimension, '
            f'with a fixed budget of {self.design["budget_per_d"]}D FE. The {len(self.design["targets"])} '
            'logarithmic gap targets range from 1e2 to 1e-8. Failed problem-target pairs remain in the denominator. '
            'Curves use the saved FE grid; no uncertainty or superiority claim is implied. '
            'All illustrated PSO controls use the matched native implementation without a velocity limit. '
            'DE is the separate SciPy quality reference. '+self.pilot_note())

    def pilot_note(self):
        return ('Pilot results are exploratory and require independent confirmation.' if self.profile!='paper' else
                'Report only the completed design and accompany comparisons with the prespecified statistical analysis.')

    def ablations(self):
        a=self.load_quality();dims=sorted(a.dimension.unique())
        fig,axes=plt.subplots(2,len(dims),figsize=(8.1,5.8),squeeze=False)
        specs=[('analytic_scalar_random8_dt1','discrete_scalar_random8_dt1','Scalar coefficients',COLORS['blue'],'s'),
               (PRIMARY,'discrete_coordinate_random8_dt1','Coordinate coefficients',COLORS['rust'],'o')]
        scopes=[('analytic_coordinate_global_dt1','analytic_scalar_global_dt1','Global',COLORS['navy'],'s'),
                (PRIMARY,'analytic_scalar_random8_dt1','R8',COLORS['rust'],'o')]
        for col,d in enumerate(dims):
            funcs=sorted(a[a.dimension==d].function.unique());positions=np.arange(len(funcs))
            for row,comparisons in enumerate([specs,scopes]):
                ax=axes[row,col];extents=[]
                for k,(left,right,label,color,marker) in enumerate(comparisons):
                    effects=paired_log_effects(a[a.dimension==d],left,right,self.gap_floor)
                    summary=effects.groupby('function').effect.agg(['median','min','max']).reindex(funcs)
                    y=positions+(k-.5)*.18
                    ax.hlines(y,summary['min'],summary['max'],color=color,lw=1,alpha=.6)
                    ax.scatter(summary['median'],y,color=color,marker=marker,s=22,label=label,zorder=3)
                    extents.extend(summary[['min','max']].to_numpy().ravel())
                ax.axvline(0,color=COLORS['grey'],ls=':',lw=1)
                lo=min(0.,float(np.nanmin(extents)));hi=max(0.,float(np.nanmax(extents)))
                padding=max(.15,(hi-lo)*.1)
                ax.set(xlim=(lo-padding,hi+padding),yticks=positions,yticklabels=[f'F{f:02d}' for f in funcs],
                       ylim=(len(funcs)-.5,-.5),title=f'({chr(97+row*len(dims)+col)}) D = {d}')
                ax.set_xlabel(r'$\log_{10}(g_{\rm continuous}/g_{\rm discrete})$' if row==0 else
                              r'$\log_{10}(g_{\rm coordinate}/g_{\rm scalar})$')
                ax.grid(axis='x');ax.set_axisbelow(True);ax.xaxis.set_major_locator(MaxNLocator(5))
                if col==0:ax.set_ylabel('BBOB function')
        handles,labels=axes[0,0].get_legend_handles_labels()
        fig.legend(handles,labels,frameon=False,loc='upper center',bbox_to_anchor=(.5,1.),ncol=2)
        handles,labels=axes[1,0].get_legend_handles_labels()
        fig.legend(handles,labels,frameon=False,loc='center',bbox_to_anchor=(.5,.49),ncol=2)
        fig.subplots_adjust(left=.075,right=.99,top=.91,bottom=.1,wspace=.24,hspace=.7)
        self.save(fig,'fig_review_matched_ablations',
            'Matched final-gap ablations at dt=1. Top row compares continuous and discrete propagation on the same R8 graphs '
            'for each coefficient scope. Bottom row compares coordinate and scalar coefficients within analytic CPSO '
            'for Global and R8. Ratios are formed within each function/instance before taking the median across instances. '
            'Segments show the full observed instance range, not confidence intervals; the vertical zero line indicates equality. '
            f'Only for logarithmic display, gaps below {self.gap_floor:g} are clipped to that value. '
            'Negative values favor continuous propagation (top) or coordinate sampling (bottom). '+self.pilot_note())

    def mechanism(self):
        path=self.results/'review_v1/mechanism'/self.profile/'analysis/subspace_diagnostics.csv'
        a=self.read(path);required=['dimension','replicate','dynamics','scope','graph','boundary','best',
              'best_relative_subspace_residual','initial_affine_rank']
        if set(required)-set(a):raise ValueError('Missing mechanism columns')
        if a.duplicated(['dimension','replicate','dynamics','scope','graph','boundary']).any():raise ValueError('Duplicate geometric run')
        d=int(a.dimension.max());g=a[(a.dimension==d)&(a.dynamics=='analytic')]
        reps=sorted(g.replicate.unique())
        expected=pd.MultiIndex.from_product([reps,['scalar','coordinate'],['global','random8'],['none','clamp']],
                                            names=['replicate','scope','graph','boundary'])
        if len(g)!=len(expected) or len(expected.difference(pd.MultiIndex.from_frame(g[list(expected.names)]))):
            raise ValueError('Incomplete geometric diagnostic')
        fig,axes=plt.subplots(1,2,figsize=(8.1,3.7));ax=axes[0]
        categories=[('scalar','none'),('coordinate','none'),('scalar','clamp'),('coordinate','clamp')]
        for x,(scope,boundary) in enumerate(categories):
            color=COLORS['blue'] if scope=='scalar' else COLORS['rust']
            for offset,graph,marker in [(-.12,'global','s'),(.12,'random8','o')]:
                t=g[(g.scope==scope)&(g.boundary==boundary)&(g.graph==graph)].sort_values('replicate')
                values=t.best_relative_subspace_residual.to_numpy().clip(min=1e-16)
                jitter=np.linspace(-.035,.035,len(t))
                ax.scatter(x+offset+jitter,values,marker=marker,s=23,facecolors='none' if graph=='global' else color,
                           edgecolors=color,linewidths=.9,zorder=3)
                ax.plot([x+offset-.055,x+offset+.055],[np.median(values)]*2,color=color,lw=1.4)
        ax.set(yscale='log',ylim=(1e-16,1.2),xticks=range(4),
            xticklabels=['Scalar\nunbounded','Coordinate\nunbounded','Scalar\nclamped','Coordinate\nclamped'],
            ylabel='Relative residual of final best',title=f'(a) Affine-space escape, D = {d}')
        ax.grid(axis='y',which='major');ax.set_axisbelow(True)
        ax.legend(handles=[Line2D([],[],marker='s',color=COLORS['grey'],mfc='none',ls='none',label='Global'),
                           Line2D([],[],marker='o',color=COLORS['grey'],ls='none',label='R8')],frameon=False,loc='center left',fontsize=6.5)
        ax=axes[1]
        for j,graph in enumerate(['global','random8']):
            t=g[(g.graph==graph)&(g.boundary=='clamp')].pivot(index='replicate',columns='scope',values='best')
            if t.isna().any().any():raise ValueError('Unpaired geometric objective values')
            ratios=t.coordinate.clip(lower=self.gap_floor)/t.scalar.clip(lower=self.gap_floor)
            color=COLORS['navy'] if graph=='global' else COLORS['rust']
            ax.plot([j,j],[ratios.min(),ratios.max()],color=color,lw=1,alpha=.6)
            ax.scatter(j+np.linspace(-.045,.045,len(ratios)),ratios,color=color,s=23,zorder=3)
            ax.plot([j-.11,j+.11],[ratios.median()]*2,color=color,lw=1.5)
        ax.axhline(1,color=COLORS['grey'],ls=':',lw=1)
        ax.set(yscale='log',xticks=[0,1],xticklabels=['Global','R8'],ylabel='Final objective ratio (coordinate / scalar)',
               title='(b) Clamped-run quality, paired replicates')
        lo,hi=ax.get_ylim()
        ticks=sorted({v*10.**e for e in range(-12,13) for v in [1.,1.5,2.,3.,5.,7.5] if lo<=v*10.**e<=hi})
        if len(ticks)>8:ticks=ticks[::math.ceil(len(ticks)/8)]
        ax.set_yticks(ticks)
        ax.grid(axis='y',which='major');ax.yaxis.set_major_formatter(ScalarFormatter())
        ax.yaxis.set_minor_formatter(plt.NullFormatter())
        fig.subplots_adjust(left=.09,right=.985,top=.9,bottom=.22,wspace=.37)
        rank=sorted(g.initial_affine_rank.unique());rank_text=', '.join(map(lambda v:str(int(v)),rank))
        self.save(fig,'fig_review_subspace_mechanism',
            f'Shifted, Householder-rotated ellipsoid diagnostic at D={d}; initial affine rank {rank_text}. '
            f'Each graph/scope/boundary cell contains {len(reps)} replicates. Left: relative projection residual of the final best '
            'outside the initial affine space; unbounded runs are mechanistic diagnostics, not benchmark competitors. '
            'Values below 1e-16 are displayed at that floor. Right: paired coordinate/scalar final-objective ratios with the '
            'standard clamp, for Global and R8; ratios below one favor coordinate sampling. Points are replicates, short '
            'horizontal ticks are medians and vertical segments show ranges. Escaping the initial space does not guarantee '
            'better optimization. This plot concerns one diagnostic function, not CEC. '+self.pilot_note())

    def load_speedups(self):
        path=self.results/'review_v1/parallel'/self.profile/'analysis/speedups.csv'
        a=self.read(path)
        if not np.isfinite(a[['native_seconds','serial_eval_us']]).all().all() or (a.native_seconds<=0).any():
            raise ValueError('Invalid native times')
        if a.duplicated(['dimension','dynamics','target_cost','session','repetition','method','workers']).any():
            raise ValueError('Duplicate timing batch')
        return a

    def scaling(self):
        a=self.load_speedups()
        for d,dimension in a.groupby('dimension'):
            dyns=[v for v in ['analytic','discrete'] if v in set(dimension.dynamics)]
            fig,axes=plt.subplots(len(dyns),2,figsize=(8.1,2.55*len(dyns)+.6),squeeze=False)
            handles=[];labels=[]
            for row,dyn in enumerate(dyns):
                g=dimension[(dimension.dynamics==dyn)&(dimension.method=='particle_parallel')]
                workers=sorted(g.workers.unique());colors=[COLORS[k] for k in ['blue','rust','teal','gold','purple','grey']]
                for k,(cost,t) in enumerate(g.groupby('target_cost')):
                    s=session_ratios(t);measured=float(t.serial_eval_us.median())
                    label=format_cost(measured);color=colors[k%len(colors)];line=None
                    for col,metric in enumerate(['serial_native_speedup','normalized_efficiency']):
                        by=s.groupby('workers')[metric].agg(['median','min','max']).reindex(workers)
                        ax=axes[row,col]
                        line,=ax.plot(workers,by['median'],marker=['o','s','^','D','v','P'][k%6],color=color,label=label)
                        ax.fill_between(workers,by['min'],by['max'],color=color,alpha=.1,lw=0)
                    if row==0:handles.append(line);labels.append(label)
                for col,ax in enumerate(axes[row]):
                    ax.set(xticks=workers,xlabel='Workers',title=f'({chr(97+row*2+col)}) '+('CPSO' if dyn=='analytic' else 'Discrete PSO'))
                    ax.grid(True);ax.set_axisbelow(True)
                    if col==0:
                        ax.plot(workers,workers,ls=':',color=COLORS['grey'],lw=1,zorder=0)
                        ax.set_ylabel('Speedup relative to native serial');ax.set_ylim(bottom=0)
                    else:
                        ax.axhline(1,color=COLORS['grey'],ls=':',lw=1)
                        ax.set_ylabel('Normalized efficiency, S(p) / p');ax.set_ylim(bottom=0)
            fig.legend(handles,labels,loc='lower center',ncol=min(4,len(handles)),frameon=False,bbox_to_anchor=(.5,.005),
                       title='Measured serial evaluation cost',title_fontsize=7)
            fig.subplots_adjust(left=.08,right=.985,top=.93,bottom=.2,hspace=.44,wspace=.26)
            self.save(fig,f'fig_review_cost_regimes_d{int(d)}',
                f'Synchronous particle-parallel scaling for the native synthetic Sphere-work objective at D={int(d)} '
                'and Global topology. S(p) uses the separately measured native serial time for the same batch, FE and seeds; '
                'the one-worker parallel point is not forced to one. Repetitions are first reduced to one median per session '
                'and worker count; lines show medians across sessions and shading the full session-median range, not a confidence '
                'interval. Legend values are measured serial costs, rather than requested target costs. Dotted references are '
                'ideal speedup and unit efficiency. S(p)/p is a conventional normalization on heterogeneous M2 cores. '
                'Different cost levels can use different generation counts; compare worker counts within one cost level. '+self.pilot_note())

    def phases(self):
        a=self.load_speedups();a=a[(a.dynamics=='analytic')&(a.method!='replicas')]
        for d,g in a.groupby('dimension'):
            levels=[float(g.target_cost.min()),float(g.target_cost.max())]
            fig,axes=plt.subplots(1,2,figsize=(8.1,3.9),squeeze=False,sharey=True)
            phase_cols=['evaluation_seconds','propagation_seconds','initialization_seconds','social_seconds','topology_seconds','other_seconds']
            colors=[COLORS[k] for k in ['rust','blue','teal','gold','purple','light']]
            labels=['Evaluation','Propagation','Initialization','Social selection','Graph setup','Other']
            for j,(ax,cost) in enumerate(zip(axes[0],levels)):
                t=g[g.target_cost==cost].copy();f=phase_fractions(t)
                f=f.join(t[['session','method','workers']])
                # Equal session weighting after averaging repetitions of fractions.
                means=f.groupby(['session','method','workers'])[phase_cols].mean().groupby(['method','workers']).mean()
                allowed={1,4,int(t.workers.max())}
                chosen=sorted([idx for idx in means.index if idx[0]=='serial' or idx[1] in allowed],
                              key=lambda idx:({'serial':0,'master_slave':1,'particle_parallel':2}[idx[0]],idx[1]))
                means=means.loc[chosen];bottom=np.zeros(len(means));x=np.arange(len(means))
                for name,color,label in zip(phase_cols,colors,labels):
                    values=means[name].to_numpy();ax.bar(x,values,bottom=bottom,color=color,label=label,width=.72);bottom+=values
                names=[('Serial' if m=='serial' else ('MS' if m=='master_slave' else 'PP')+f' {p}') for m,p in means.index]
                ax.set(xticks=x,xticklabels=names,ylim=(0,1.01),title=f'({chr(97+j)}) '+format_cost(t.serial_eval_us.median())+' per evaluation')
                ax.tick_params(axis='x',rotation=35);ax.grid(axis='y');ax.set_axisbelow(True)
            axes[0,0].set_ylabel('Fraction of native wall time')
            handles,names=axes[0,0].get_legend_handles_labels();fig.legend(handles,names,ncol=3,loc='lower center',frameon=False,bbox_to_anchor=(.5,.0))
            fig.subplots_adjust(left=.075,right=.99,top=.9,bottom=.29,wspace=.13)
            self.save(fig,f'fig_review_phase_decomposition_d{int(d)}',
                'Native CPSO phase decomposition at the smallest and largest recorded evaluation costs. MS denotes master-slave; '
                'PP particle-parallel. Each batch is normalized by its native time before averaging repetitions and then sessions '
                'with equal weight. Other is the nonoverlapping residual, including unlabelled synchronization/state work. '
                'Process startup and Python dispatch are outside these native phase times. '+self.pilot_note(),role='supplementary')

    def selector(self):
        base=self.results/'review_v1/parallel'/self.profile/'analysis'
        a=self.read(base/'heldout_regret.csv');pred=self.read(base/'heldout_predictions.csv')
        if (a.relative_regret<0).any() or not np.isfinite(a.relative_regret).all():raise ValueError('Invalid regret')
        for d,g in a.groupby('dimension'):
            fig,axes=plt.subplots(1,2,figsize=(8.1,3.25),sharey=True)
            for j,(ax,dyn) in enumerate(zip(axes,['analytic','discrete'])):
                for method,color,marker in [('master_slave',COLORS['blue'],'s'),('particle_parallel',COLORS['rust'],'o')]:
                    t=g[(g.dynamics==dyn)&(g.backend==method)].sort_values('target_cost')
                    costmap=pred[(pred.dimension==d)&(pred.dynamics==dyn)].groupby('target_cost').serial_eval_us.median()
                    x=t.target_cost.map(costmap)
                    ax.plot(x,100*t.relative_regret,marker=marker,color=color,label=method.replace('_','-'))
                ax.set(xscale='log',xlabel=r'Measured serial evaluation cost ($\mu$s)',
                       title=f'({chr(97+j)}) '+('CPSO' if dyn=='analytic' else 'Discrete PSO'))
                ax.axhline(0,color=COLORS['grey'],ls=':',lw=1);ax.grid(True);ax.set_axisbelow(True)
            axes[0].set_ylabel('Conditional held-out regret (%)');axes[0].legend(frameon=False,fontsize=7)
            fig.subplots_adjust(left=.085,right=.99,top=.88,bottom=.2,wspace=.17)
            self.save(fig,f'fig_review_selector_regret_d{int(d)}',
                'Session-held-out worker-selection diagnostic. Regret equals chosen time / minimum observed time - 1, '
                'conditional on the candidate workers of each fixed backend. It is not a global oracle across serial, '
                'parallel backends and replica scheduling. One reserved session provides these decisions; no confidence interval '
                'is implied. The cheap-workload failures should be reported, not hidden. If the selector is revised using these '
                'results, a new held-out session is required.',role='supplementary')

    def finalize(self):
        self.sources[str(Path(__file__).resolve())]=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        captions=['# Suggested figure captions','',f'Profile: {self.profile}. Style: {self.style}.',
                  '','Use the main figures selectively. Pilot figures are exploratory; no significance stars or CD diagrams are generated.','']
        for item in self.figures:
            captions += [f'## {item["file"]}',f'Role: {item["role"]}.','',item['caption'],'']
        if self.skipped:captions+=['## Missing inputs','',*['- '+s for s in self.skipped],'']
        (self.output/'FIGURE_CAPTIONS.md').write_text('\n'.join(captions),encoding='utf-8')
        floor_count=0
        if self.quality is not None:
            floor_count=int(((self.quality.method.isin(METHODS+['analytic_scalar_global_dt1','analytic_coordinate_global_dt1']))&
                            (self.quality.gap<self.gap_floor)).sum())
        (self.output/'plot_manifest.json').write_text(json.dumps(dict(profile=self.profile,style=self.style,
            gap_floor=self.gap_floor,selected_run_gaps_below_display_floor=floor_count,
            figures=self.figures,skipped=self.skipped,sources=self.sources),indent=2)+'\n')


def format_cost(microseconds):
    return (f'{microseconds/1000:.2g} ms' if microseconds>=500 else f'{microseconds:.2g} '+r'$\mu$s')


def paired_log_effects(a,left,right,floor):
    keys=['dimension','function','instance']
    t=a[a.method.isin([left,right])]
    if t.duplicated(keys+['method']).any():raise ValueError('Duplicate paired run')
    wide=t.pivot(index=keys,columns='method',values='gap')
    if left not in wide or right not in wide or wide[[left,right]].isna().any().any():
        raise ValueError(f'Unpaired ablation: {left} / {right}')
    if (wide[[left,right]]<0).any().any():raise ValueError('Negative objective gap')
    effects=np.log10(wide[left].clip(lower=floor))-np.log10(wide[right].clip(lower=floor))
    return effects.rename('effect').reset_index()


def session_ratios(t):
    if not np.isfinite(t.serial_native_speedup).all() or (t.serial_native_speedup<=0).any():
        raise ValueError('Invalid serial-relative speedup')
    # The input contains paired ratios, not unpaired median times.
    s=t.groupby(['session','workers']).serial_native_speedup.median().reset_index()
    expected=pd.MultiIndex.from_product([sorted(t.session.unique()),sorted(t.workers.unique())],names=['session','workers'])
    if len(expected.difference(pd.MultiIndex.from_frame(s[['session','workers']]))):
        raise ValueError('Incomplete session/worker curve')
    s['normalized_efficiency']=s.serial_native_speedup/s.workers
    return s


def phase_fractions(a):
    names=['evaluation_seconds','propagation_seconds','initialization_seconds','social_seconds','topology_seconds']
    values=a[names].astype(float).copy();total=a.native_seconds.astype(float)
    if (total<=0).any() or not np.isfinite(values).all().all() or (values<0).any().any():
        raise ValueError('Invalid phase times')
    residual=total-values.sum(axis=1)
    if (residual < -1e-8*total).any():raise ValueError('Phase timers overlap or exceed native wall time')
    values['other_seconds']=residual.clip(lower=0)
    return values.div(total,axis=0)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--results-root',type=Path,default=ROOT/'results')
    ap.add_argument('--profile',choices=['smoke','pilot','paper'],default='pilot')
    ap.add_argument('--output',type=Path)
    ap.add_argument('--only',nargs='+',choices=CORE+['phases','selector'])
    ap.add_argument('--supplementary',action='store_true')
    ap.add_argument('--style',choices=['paper','serif'],default='paper')
    ap.add_argument('--gap-floor',type=float,default=1e-12)
    ap.add_argument('--allow-missing',action='store_true',help='Skip only absent files; malformed or incomplete data still fail')
    args=ap.parse_args()
    if not math.isfinite(args.gap_floor) or args.gap_floor<=0:ap.error('--gap-floor must be finite and positive')
    output=args.output or args.results_root/'review_publication_figures'/args.profile
    p=Plotter(args.results_root,args.profile,output,args.gap_floor,args.style)
    selected=list(dict.fromkeys((args.only or CORE)+(['phases','selector'] if args.supplementary else [])))
    for name in selected:
        try:getattr(p,name)()
        except FileNotFoundError as exc:
            if not args.allow_missing:raise
            message=f'{name}: {exc}';p.skipped.append(message);print('SKIP',message,flush=True)
    p.finalize()
    if not p.figures:raise SystemExit('No figures generated: check the input paths')
    print(f'COMPLETE: {len(p.figures)} vector PDFs; captions in {p.output/"FIGURE_CAPTIONS.md"}',flush=True)

if __name__=='__main__':main()
