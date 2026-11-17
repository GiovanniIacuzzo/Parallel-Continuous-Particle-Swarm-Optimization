#!/usr/bin/env python3
"""Read-only replot of six publication figures from existing CSV/JSON outputs.

No optimizer, native library, campaign protocol or statistical result is changed.
This standalone root script is outside the original execution fingerprint.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
import textwrap

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, MaxNLocator, MultipleLocator, PercentFormatter
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
PRIMARY = 'CPSO selected framework'
REFERENCE = 'CPSO serial reference'
RED = '#bf432e'
BLUE = '#24678d'
INK = '#273746'
ARCH_ORDER = ['serial', 'master_slave', 'particle_parallel', 'asynchronous', 'islands']
ARCH = {'serial': 'Serial', 'master_slave': 'Master-slave', 'particle_parallel': 'Particle-parallel',
        'asynchronous': 'Asynchronous', 'islands': 'Islands'}
TOPO = {'global': 'Global', 'ring': 'Ring', 'grid_von_neumann': 'Von Neumann', 'wheel': 'Star',
        'random_k4': 'Random 4', 'random_k8': 'Random 8', 'small_world_k4': 'Small-world',
        'scale_free_k4': 'Scale-free'}
LABEL = {PRIMARY: 'CPSO selected', REFERENCE: 'CPSO reference (serial)',
         'classic_gbest': 'Classic gbest', 'inertia_gbest': 'Inertia gbest',
         'inertia_lbest_ring': 'Inertia ring', 'constriction_gbest': 'Constriction',
         'clpso': 'CLPSO', 'fips_ring': 'FIPS', 'spso2011': 'SPSO-2011',
         'slpso': 'SL-PSO', 'cso': 'CSO', 'dms_pso': 'DMS-PSO'}
REFERENCE = 'CPSO serial reference'
FAMILIES = [
    ('Velocity updates', ['classic_gbest', 'inertia_gbest', 'inertia_lbest_ring', 'constriction_gbest']),
    ('Learning and informants', ['clpso', 'fips_ring', 'spso2011']),
    ('Social and competitive updates', ['slpso', 'cso', 'dms_pso']),
]
WORKER_COLORS = ['#31688e', '#d19b25', '#35a079', '#8c6bb1', '#66717d']
PEER_COLORS = ['#24678d', '#26847d', '#9966a3', '#b38120']
PEER_STYLES = ['--', '-.', ':', (0, (5, 1.5, 1, 1.5))]
BUDGET_MARKERS = ['o', 's', '^', 'D', 'v', 'P']


@contextmanager
def plotting_style():
    with plt.rc_context({'font.family': 'DejaVu Sans', 'font.size': 10,
        'axes.titlesize': 11, 'axes.labelsize': 10, 'xtick.labelsize': 9,
        'ytick.labelsize': 9, 'legend.fontsize': 9, 'axes.spines.top': False,
        'axes.spines.right': False, 'axes.grid': False, 'pdf.fonttype': 42,
        'ps.fonttype': 42, 'lines.linewidth': 1.6, 'savefig.pad_inches': .08}):
        yield


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def read_csv(path, required):
    if not path.is_file():
        raise FileNotFoundError(f'Missing analysis input: {path}. Use the full results folder, not only figures/.')
    a = pd.read_csv(path)
    missing = sorted(set(required) - set(a.columns))
    if missing or a.empty:
        raise ValueError(f'{path}: empty data or missing columns {missing}')
    return a


def finite(a, columns, label):
    values = a[columns].apply(pd.to_numeric, errors='raise').to_numpy(float)
    if not np.isfinite(values).all():
        raise ValueError(f'{label}: non-finite values in {columns}')


def canonical_method(m):
    result = f"{m['architecture']}__{m['topology']}__w{m['workers']}"
    if m['architecture'] == 'islands':
        result += f"__i{m['islands']}__m{m['migration_interval']}__{m['migration_topology']}"
    return result


def recipe_label(m):
    text = f"{ARCH.get(m['architecture'], m['architecture'])} | {TOPO.get(m['topology'], m['topology'])} | {m['workers']} workers"
    if m['architecture'] == 'islands':
        text += f" | {m['islands']} islands | migration {m['migration_interval']}"
    return text


def decimal_tick(x, _pos=None):
    return f'{x:,.0f}' if x >= 100 else f'{x:g}'


def log_ticks(ax, low, high, axis='x'):
    setter = ax.set_xticks if axis == 'x' else ax.set_yticks
    formatter = ax.xaxis if axis == 'x' else ax.yaxis
    choices = [k * 10.**p for p in range(math.floor(math.log10(low)), math.ceil(math.log10(high))+1)
               for k in [1, 2, 5] if low <= k * 10.**p <= high]
    if len(choices) > 8:
        choices = [10.**p for p in range(math.floor(math.log10(low)), math.ceil(math.log10(high))+1)
                   if low <= 10.**p <= high]
    setter(choices)
    formatter.set_major_formatter(FuncFormatter(decimal_tick))
    formatter.set_minor_formatter(plt.NullFormatter())


def holdout_inputs(base):
    folder = base/'02_parallel/holdout/analysis'
    a = read_csv(folder/'scaling_summary.csv', ['benchmark','dimension','method','metric','estimate','lo','hi'])
    frozen = read_json(base/'02_parallel/selection/frozen.json')['global_method']
    g = a[(a.method == canonical_method(frozen)) & (a.metric == 'serial_speedup')].copy()
    if g.empty or g.duplicated(['benchmark','dimension']).any():
        raise ValueError('Frozen holdout method is missing or duplicated in scaling_summary.csv')
    finite(g, ['estimate','lo','hi'], 'Holdout')
    if (g.lo <= 0).any() or (g.lo > g.estimate).any() or (g.estimate > g.hi).any():
        raise ValueError('Invalid holdout confidence intervals')
    return g, frozen


def plot_holdout(g, frozen):
    benchmarks = sorted(g.benchmark.unique(), key=lambda x: (x != 'inverse_diffusion', x))
    fig, axes = plt.subplots(1, len(benchmarks), figsize=(8.0, 4.2), squeeze=False)
    fig.subplots_adjust(left=.105, right=.96, bottom=.33, top=.76, wspace=.33)
    low = min(.7, float(g.lo.min())*.86)
    high = max(1.5, float(g.hi.max())*1.24)
    for ax, b in zip(axes.flat, benchmarks):
        z = g[g.benchmark == b].sort_values('dimension')
        ax.set_xscale('log', base=2)
        ax.set_xlim(low, high)
        ax.axvspan(low, 1, color='#f5ece9', zorder=0)
        ax.axvline(1, ls='--', lw=1, color='#66717d')
        for y, row in enumerate(z.itertuples()):
            color = RED if row.estimate < 1 else BLUE
            ax.errorbar(row.estimate, y, xerr=[[row.estimate-row.lo], [row.hi-row.estimate]],
                        fmt='o', color=color, capsize=4, lw=2, ms=6, zorder=3)
            ax.annotate(f'{row.estimate:.2f}x', (row.estimate, y), xytext=(0, 12),
                        textcoords='offset points', ha='center', fontsize=10, color=color, weight='bold')
        ax.set_yticks(range(len(z)), [f'D = {int(v)}' for v in z.dimension])
        ax.set_ylim(len(z)-.5, -.6)
        ax.set_title('Inverse diffusion' if b == 'inverse_diffusion' else b.replace('_',' ').title(), pad=12)
        ticks = [x for x in [.25, .5, .75, 1, 1.5, 2, 3, 4, 6, 8, 12, 16] if low <= x <= high]
        ax.set_xticks(ticks)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:g}x'))
        ax.xaxis.set_minor_formatter(plt.NullFormatter())
        ax.grid(axis='x', color='#dde3e7', alpha=.7)
        ax.set_xlabel('Serial time / selected-framework time', labelpad=9)
    kind = 'Synchronous speedup' if frozen['architecture'] in ['serial','master_slave','particle_parallel'] else 'Runtime ratio - changed search dynamics'
    fig.suptitle('Held-out framework performance\n'+textwrap.fill(recipe_label(frozen),72), fontsize=11, y=.98)
    fig.text(.5, .14, kind + ' | Native timer | 95% paired-seed bootstrap intervals', ha='center', fontsize=9)
    fig.text(.5, .075, 'Below 1x: slower than serial. Above 1x: faster. Serial reference uses the same social topology.',
             ha='center', fontsize=8.5)
    return fig


def pareto_mask(time, quality):
    t, q = np.asarray(time, float), np.asarray(quality, float)
    return np.array([not np.any((t <= t[i]) & (q <= q[i]) & ((t < t[i]) | (q < q[i]))) for i in range(len(t))])


def selection_inputs(base):
    folder = base/'02_parallel/selection/analysis'
    rank = read_csv(folder/'selection_ranking.csv', ['method','mean_log_time','mean_quality','eligible'])
    raw = read_csv(folder/'paired_runs.csv', ['method','architecture','workers','topology'])
    meta = raw[['method','architecture','workers','topology']].drop_duplicates()
    if meta.method.duplicated().any() or rank.method.duplicated().any():
        raise ValueError('Inconsistent framework identity in selection data')
    a = rank.merge(meta, on='method', how='left', validate='one_to_one')
    if a[['architecture','workers','topology']].isna().any().any():
        raise ValueError('Selection configuration metadata are incomplete')
    if not set(a.architecture) <= set(ARCH_ORDER):
        raise ValueError('Unknown framework architecture')
    finite(a, ['mean_log_time','mean_quality','workers'], 'Selection')
    mapping = {'true': True, 'false': False, '1': True, '0': False}
    eligible = a.eligible.astype(str).str.lower().map(mapping)
    if eligible.isna().any():
        raise ValueError('Invalid eligibility flags')
    a['eligible'] = eligible
    a['milliseconds'] = 1000*np.power(10., a.mean_log_time)
    a['residual_percent'] = 100*np.power(10., a.mean_quality)
    if not np.isfinite(a[['milliseconds','residual_percent']]).all().all() or (a.milliseconds <= 0).any():
        raise ValueError('Invalid transformed framework metrics')
    frozen = read_json(base/'02_parallel/selection/frozen.json')['global_method']
    if canonical_method(frozen) not in set(a.method):
        raise ValueError('Frozen framework is not in the selection data')
    return a, frozen


def plot_selection(a, frozen):
    fig, axes = plt.subplots(3, 2, figsize=(8.1, 8.5))
    fig.subplots_adjust(left=.09, right=.975, top=.83, bottom=.20, hspace=.49, wspace=.27)
    workers = sorted(a.workers.astype(int).unique())
    if len(workers) > len(WORKER_COLORS):
        raise ValueError('Too many distinct worker levels for the prescribed palette')
    colors = dict(zip(workers, WORKER_COLORS))
    chosen = a[a.method == canonical_method(frozen)].iloc[0]
    low, high = float(a.milliseconds.min())*.86, float(a.milliseconds.max())*1.17
    qmin, qmax = float(a.residual_percent.min()), float(a.residual_percent.max())
    pad = max(1., .15*(qmax-qmin))
    qlim = (max(0.,qmin-pad), qmax+pad)
    for ax, arch in zip(axes.flat, ARCH_ORDER):
        g = a[a.architecture == arch]
        for p in workers:
            z = g[g.workers == p]
            for eligible in [False, True]:
                zz = z[z.eligible == eligible]
                ax.scatter(zz.milliseconds, zz.residual_percent, s=32,
                    edgecolors=colors[p], facecolors=colors[p] if eligible else 'none',
                    alpha=.75 if eligible else .48, linewidths=.8, zorder=3)
        if chosen.architecture == arch:
            ax.scatter([chosen.milliseconds], [chosen.residual_percent], marker='*', s=150,
                       color='black', linewidths=.6, zorder=5)
        ax.set_title(ARCH[arch])
    ax = axes.flat[-1]
    front = a[pareto_mask(a.milliseconds, a.residual_percent)].sort_values('milliseconds')
    ax.plot(front.milliseconds, front.residual_percent, color='#8b969e', lw=1.1, zorder=1)
    for p in workers:
        z = front[front.workers == p]
        ax.scatter(z.milliseconds, z.residual_percent, s=42, color=colors[p], zorder=3)
    ax.scatter([chosen.milliseconds], [chosen.residual_percent], marker='*', s=150, color='black', zorder=5)
    ax.set_title('Global frontier + frozen choice',fontsize=10)
    for i, ax in enumerate(axes.flat):
        ax.set_xscale('log'); ax.set_xlim(low, high); ax.set_ylim(*qlim)
        log_ticks(ax, low, high)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:g}%'))
        ax.yaxis.set_major_locator(MaxNLocator(4))
        ax.grid(color='#e2e7ea', alpha=.7, zorder=0)
        if i % 2 == 0: ax.set_ylabel('Geometric residual $f_B/f_0$')
        if i >= 4: ax.set_xlabel('Geometric wall time (ms)')
    handles = [Line2D([], [], marker='o', color='none', markerfacecolor=colors[p], markeredgecolor=colors[p],
                      label=f'{p} worker' if p == 1 else f'{p} workers') for p in workers]
    handles.append(Line2D([], [], marker='*', color='black', ls='none', ms=10, label='Frozen choice'))
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.5,.095), ncol=len(handles), frameon=False)
    fig.suptitle('Framework selection: completed-run cost and quality', y=.98, fontsize=12)
    fig.text(.5, .925, textwrap.fill(recipe_label(frozen),72), ha='center', fontsize=10)
    fig.text(.5, .067, 'Each point is a configuration. Solid: eligible under the existing rule. Open: not eligible.',
             ha='center', fontsize=8.3)
    fig.text(.5, .032, 'Lower and left are better; time axes are logarithmic. All configurations remain visible.\nThe last panel shows the nondominated frontier, with the frozen choice overlaid.',
             ha='center', fontsize=8.3)
    return fig


def validate_balanced(a, keys, label):
    if a.duplicated(keys+['method']).any():
        raise ValueError(f'{label}: duplicate observations')
    groups = a.groupby('method', sort=False)
    expected = None
    for name, g in groups:
        observed = set(map(tuple, g[keys].to_numpy()))
        if expected is None: expected = observed
        elif observed != expected:
            raise ValueError(f'{label}: incomplete paired design for {name}')


def ecdf_values(hits, dimension, budget):
    """Exact ECDF; failed targets remain in the denominator."""
    values = pd.to_numeric(pd.Series(hits), errors='raise').to_numpy(float)
    observed = values[~np.isnan(values)]
    if not len(values) or not np.isfinite(observed).all() or (observed <= 0).any() or (observed > budget*dimension).any():
        raise ValueError('Invalid target first-hit evaluations')
    h = np.sort(observed/dimension)
    x = np.unique(np.concatenate(([1.], h[(h >= 1.) & (h <= budget)], [float(budget)])))
    y = np.searchsorted(h, x, side='right')/len(values)
    return x, y


def bbob_inputs(base):
    a = read_csv(base/'03_comparison/bbob/analysis/target_hits.csv',
                 ['function','dimension','instance','method','target','hit_fe','budget'])
    expected = {REFERENCE, *[m for _, methods in FAMILIES for m in methods]}
    if set(a.method) != expected:
        raise ValueError(f'BBOB expects all eleven methods, found {sorted(a.method.unique())}')
    finite(a, ['dimension','target','budget'], 'BBOB')
    if (a.dimension <= 0).any() or (a.budget <= 0).any(): raise ValueError('Invalid BBOB dimensions/budgets')
    validate_balanced(a, ['function','dimension','instance','target'], 'BBOB')
    for d, g in a.groupby('dimension'):
        if len(set(g.budget)) != 1: raise ValueError('BBOB final budget varies within a dimension')
        for _, z in g.groupby('method'):
            ecdf_values(z.hit_fe, d, float(z.budget.iloc[0])/d)
    return a


def plot_bbob(a):
    dims = sorted(a.dimension.unique())
    if len(dims) > 3: raise ValueError('Use at most three dimensions per BBOB figure')
    fig = plt.figure(figsize=(8.1 if len(dims) <= 2 else 10.5, 10.0))
    grid = fig.add_gridspec(6, len(dims), height_ratios=[1,.42,1,.42,1,.42],
                          left=.095, right=.975, bottom=.075, top=.91, hspace=.40, wspace=.22)
    max_fraction = 0.
    for (d,m),g in a.groupby(['dimension','method']):
        max_fraction = max(max_fraction, float(g.hit_fe.notna().mean()))
    # Start at zero; use the same data-adaptive scale for every panel.
    ymax = min(1., max(.1, math.ceil((max_fraction+.025)/.05)*.05))
    for row,(family, methods) in enumerate(FAMILIES):
        handles = None
        for col,d in enumerate(dims):
            ax = fig.add_subplot(grid[2*row,col]);g = a[a.dimension == d]
            budget = float(g.budget.iloc[0])/d
            for j,m in enumerate([REFERENCE,*methods]):
                z = g[g.method == m];x,y = ecdf_values(z.hit_fe,d,budget)
                ax.step(x,y,where='post',color=RED if j == 0 else PEER_COLORS[j-1],
                        ls='-' if j == 0 else PEER_STYLES[j-1],lw=2.2 if j == 0 else 1.5,
                        label=LABEL[m],zorder=5 if j == 0 else 3)
            ax.set_xscale('log');ax.set_xlim(1,budget);ax.set_ylim(0,ymax)
            step = .05 if ymax <= .15 else .1 if ymax <= .5 else .2
            ax.set_yticks(np.arange(0,ymax+1e-12,step));ax.yaxis.set_major_formatter(PercentFormatter(1,decimals=0))
            ax.set_xticks([x for x in [1,10,100,budget] if x <= budget]);ax.xaxis.set_major_formatter(FuncFormatter(decimal_tick))
            ax.xaxis.set_minor_formatter(plt.NullFormatter());ax.grid(color='#e2e7ea',alpha=.8)
            ax.set_title(f'{family}\nD = {int(d)}',fontsize=10.5)
            if col == 0: ax.set_ylabel('Solved problem-target pairs')
            ax.set_xlabel('Evaluations / dimension',labelpad=3)
            handles = ax.get_legend_handles_labels()
        leg = fig.add_subplot(grid[2*row+1,:]);leg.axis('off')
        leg.legend(*handles,loc='center',ncol=3,frameon=False,fontsize=8.5,
                   handlelength=2.5,columnspacing=1.4)
    fig.suptitle('BBOB target attainment - serial quality control',fontsize=12,y=.98)
    fig.text(.5,.028,'All problem-target pairs for each dimension are retained, including failures. Vertical scale is shared.',
             ha='center',fontsize=8.5)
    return fig


def tradeoff_inputs(base):
    a = read_csv(base/'03_comparison/native/analysis/ranked_results.csv',
                 ['benchmark','dimension','seed','iterations','method','rank','seconds','fe'])
    if PRIMARY not in set(a.method): raise ValueError('Native primary CPSO method is missing')
    if len(set(a.method)) != 11: raise ValueError('Native rank plot requires the original eleven-method comparison')
    finite(a, ['dimension','seed','iterations','rank','seconds','fe'], 'Native comparison')
    if (a.seconds <= 0).any() or (a.fe <= 0).any() or (a['rank'] < 1).any() or (a['rank'] > 11).any():
        raise ValueError('Invalid native runtimes, FE or ranks')
    validate_balanced(a, ['benchmark','dimension','seed','iterations'], 'Native comparison')
    if a.groupby(['dimension','iterations']).fe.nunique().max() != 1:
        raise ValueError('Unmatched native FE budgets')
    return a


def plot_tradeoff(a, dimension):
    a = a[a.dimension == dimension]
    its = sorted(a.iterations.unique())
    if len(its) > len(BUDGET_MARKERS): raise ValueError('Too many independent budgets for this layout')
    summary = a.groupby(['iterations','method']).agg(mean_rank=('rank','mean'),
                milliseconds=('seconds',lambda x:1000*float(x.median())),fe=('fe','first')).reset_index()
    order = summary[summary.iterations == its[-1]].sort_values(['mean_rank','method']).method.tolist()
    fig,(ax_rank,ax_time) = plt.subplots(1,2,figsize=(8.1,6.5),sharey=True,
                                       gridspec_kw={'width_ratios':[1,1.05]})
    fig.subplots_adjust(left=.18,right=.975,top=.84,bottom=.27,wspace=.19)
    tones = ['#9baebf','#547f9e','#173f5f','#788493','#326e69','#43546a']
    offsets = np.linspace(-.18,.18,len(its))
    for y,m in enumerate(order):
        rows = summary[summary.method == m].set_index('iterations').loc[its]
        if m == PRIMARY:
            for ax in [ax_rank,ax_time]:ax.axhspan(y-.41,y+.41,color='#fae9e4',zorder=0)
        for j,it in enumerate(its):
            row = rows.loc[it]
            for ax,x in [(ax_rank,row.mean_rank),(ax_time,row.milliseconds)]:
                ax.scatter(x,y+offsets[j],marker=BUDGET_MARKERS[j],s=35,color=RED if m == PRIMARY else tones[j],
                           edgecolor='white',linewidth=.45,zorder=3)
    ax_rank.set_yticks(range(len(order)),[LABEL.get(m,m) for m in order])
    for tick,m in zip(ax_rank.get_yticklabels(),order):
        if m == PRIMARY:tick.set_color(RED);tick.set_fontweight('bold')
    ax_rank.set_ylim(len(order)-.55,-.55);ax_rank.set_xlim(.6,11.4)
    ax_rank.set_xticks([1,3,5,7,9,11]);ax_rank.set_title('Solution quality',pad=12)
    ax_rank.set_xlabel('Mean final rank (lower is better)',labelpad=8)
    ax_time.set_xscale('log');low=summary.milliseconds.min()*.75;high=summary.milliseconds.max()*1.35
    ax_time.set_xlim(low,high);log_ticks(ax_time,low,high)
    ax_time.set_title('Completed-run cost',pad=12);ax_time.set_xlabel('Median wall time (ms, log scale)',labelpad=8)
    for ax in [ax_rank,ax_time]:ax.grid(axis='x',color='#e0e6ea',alpha=.8);ax.tick_params(axis='y',length=0)
    handles = [Line2D([],[],marker=BUDGET_MARKERS[j],color=tones[j],ls='none',ms=6,
                      label=f"{int(summary[summary.iterations == it].fe.iloc[0]):,} FE") for j,it in enumerate(its)]
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.57,.13),ncol=min(3,len(handles)),frameon=False)
    fig.suptitle(f'CPSO and PSO variants | D = {int(dimension)}',fontsize=12,y=.97)
    fig.text(.5,.905,'Rows use the same algorithm order in both panels; CPSO is highlighted.',ha='center',fontsize=9)
    fig.text(.5,.065,'Each marker is a separate run budget. Rank is a mean; time is a median across the same problem/seed sample.',
             ha='center',fontsize=8)
    fig.text(.5,.031,'Timing compares the delivered C++ CPSO and NumPy baseline implementations; these are not exact time-to-target curves.',
             ha='center',fontsize=8)
    return fig


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def regenerate(base, only='all', backup=True):
    """Validate all requested inputs before replacing any PDF. CSVs stay read-only."""
    base = Path(base).resolve()
    tasks = []; inputs = []
    if only in ['all','holdout']:
        g,frozen = holdout_inputs(base)
        tasks.append(('02_parallel/holdout/fig_framework_holdout.pdf',lambda g=g,f=frozen:plot_holdout(g,f)))
        inputs += [base/'02_parallel/holdout/analysis/scaling_summary.csv',base/'02_parallel/selection/frozen.json']
    if only in ['all','selection']:
        a,frozen = selection_inputs(base)
        tasks.append(('02_parallel/selection/fig_framework_quality_cost.pdf',lambda a=a,f=frozen:plot_selection(a,f)))
        inputs += [base/'02_parallel/selection/analysis/selection_ranking.csv',base/'02_parallel/selection/analysis/paired_runs.csv',base/'02_parallel/selection/frozen.json']
    if only in ['all','bbob']:
        a = bbob_inputs(base)
        tasks.append(('03_comparison/bbob/fig_bbob_ecdf_serial_control.pdf',lambda a=a:plot_bbob(a)))
        inputs += [base/'03_comparison/bbob/analysis/target_hits.csv']
    if only in ['all','tradeoff']:
        a = tradeoff_inputs(base)
        for d in sorted(a.dimension.unique()):
            tasks.append((f'03_comparison/native/fig_completed_run_tradeoff_d{int(d)}.pdf',lambda a=a,d=d:plot_tradeoff(a,d)))
        inputs += [base/'03_comparison/native/analysis/ranked_results.csv']
    if not tasks: raise ValueError(f'Unknown figure group: {only}')
    input_hashes = {str(p.relative_to(base)):sha(p) for p in set(inputs)}
    figures = base/'figures';figures.mkdir(parents=True,exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    outputs = []
    with tempfile.TemporaryDirectory(prefix='cpso_plot_',dir=figures) as tmp, plotting_style():
        staged = []
        try:
            for i,(relative,create) in enumerate(tasks):
                fig = create();temp = Path(tmp)/f'{i}.pdf'
                fig.savefig(temp,format='pdf',bbox_inches='tight',pad_inches=.10,
                            metadata={'Title':Path(relative).stem,'Subject':'Read-only visualization of existing CPSO analysis results'})
                plt.close(fig);staged.append((relative,temp))
        finally:plt.close('all')
        if {str(p.relative_to(base)):sha(p) for p in set(inputs)} != input_hashes:
            raise RuntimeError('Analysis inputs changed during plotting; PDFs were not replaced')
        for relative,temp in staged:
            target = figures/relative;target.parent.mkdir(parents=True,exist_ok=True)
            if backup and target.exists():
                old = figures/'_before_readability'/stamp/relative;old.parent.mkdir(parents=True,exist_ok=True)
                shutil.copy2(target,old)
            os.replace(temp,target);outputs.append(relative);print(f'Updated: {target}',flush=True)
    manifest = dict(created_utc=stamp,script_sha256=sha(Path(__file__)),inputs_sha256=input_hashes,
                    replaced_pdfs=outputs,experiment_data_modified=False,
                    ecdf='exact first-hit support; failures retained; shared zero-based percentage scale')
    record = figures/'_readability_logs'/f'{stamp}.json';record.parent.mkdir(parents=True,exist_ok=True)
    record.write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    return manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--profile',choices=['smoke','pilot','paper'],default='pilot')
    p.add_argument('--results',type=Path,help='Full existing results directory (contains 01_dt, 02_parallel, 03_comparison)')
    p.add_argument('--only',choices=['all','holdout','selection','bbob','tradeoff'],default='all')
    p.add_argument('--no-backup',action='store_true',help='Do not preserve replaced PDFs; CSVs are always read-only')
    args = p.parse_args()
    base = args.results or ROOT/'results/publication_v2'/args.profile
    try:regenerate(base,args.only,backup=not args.no_backup)
    except (ValueError,FileNotFoundError) as exc:p.exit(2,f'Plotting stopped: {exc}\n')


if __name__ == '__main__':main()
