from __future__ import annotations
from dataclasses import replace
import itertools
import numpy as np
import pandas as pd
from .common import (ROOT,read,freeze,prepare,csv,native_config,native_run,method,name,cached_run,
                     shuffled,paired_effect,normalized_score,holm,session_environment)


def methods(design):
    cfg=design['parallel']; output=[]
    for topology in cfg['topologies']:
        output.append(method(topology=topology))
        for architecture,p in itertools.product(['master_slave','particle_parallel','asynchronous'],cfg['workers']):
            output.append(method(architecture,p,topology))
        for p,interval,mt in itertools.product(cfg['workers'],cfg['migration_intervals'],['ring','fully_connected']):
            if p>cfg['islands'] or (interval==0 and mt=='fully_connected'): continue
            m=method('islands',p,topology,islands=cfg['islands'],migration_interval=interval,migration_topology=mt)
            # Every sub-swarm must independently satisfy the native topology rules.
            try:
                c=native_config('sphere',2,1,1,1.,topology,design['population']//cfg['islands'])
                if topology=='grid_von_neumann': continue
                c.validate()
            except ValueError: continue
            output.append(m)
    return output


def design_for_phase(base,design,phase):
    all_methods=methods(design)
    if phase=='holdout':
        frozen=read(base/'02_parallel/selection/frozen.json')
        # Validate only frozen selections and their serial references. No retuning.
        wanted=[frozen['global_method'],*frozen['by_case'].values()]
        wanted += [method(topology=m['topology']) for m in wanted]
        # Global topology reference supplies a common quality normalization.
        wanted.append(method())
        unique={name(m):m for m in wanted}
        all_methods=list(unique.values())
    return all_methods


def jobs(design,phase,session,selected):
    cfg=design['parallel']
    for b,d,seed,rep in itertools.product(cfg['functions'],cfg['dimensions'],cfg[phase+'_seeds'],range(cfg['repetitions'])):
        for m in selected:
            yield dict(phase=phase,session=session,benchmark=b,dimension=d,seed=seed,repetition=rep,method=m)


def run(base,design,phase,session=None):
    folder=base/'02_parallel'/phase; prepare(folder,design)
    dt=read(base/'01_dt/holdout/complete.json')['dt']
    selected=design_for_phase(base,design,phase);cfg=design['parallel']
    freeze(folder/'methods.json',selected)
    sessions=cfg[phase+'_sessions']
    if session is not None and session not in sessions: raise ValueError(f'Allowed sessions: {sessions}')
    active=sessions if session is None else [session]
    for s in active:
        from checkpoint_io import atomic_write_json
        from datetime import datetime,timezone
        stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')
        atomic_write_json(folder/'session_environment'/f's{s}_{stamp}_start.json',session_environment())
        # Full size, one untimed update per candidate and cell before collection.
        for b,d in itertools.product(cfg['functions'],cfg['dimensions']):
            for m in selected:
                native_run(b,d,8675309,1,dt,m,cfg['work'] if b=='inverse_diffusion' else 0,design['population'])
        for index,key in enumerate(shuffled(jobs(design,phase,s,selected),1110+s),1):
            b,d,seed,m=key['benchmark'],key['dimension'],key['seed'],key['method']
            cached_run(folder,key,lambda: native_run(b,d,seed,cfg['iterations'],dt,m,
                cfg['work'] if b=='inverse_diffusion' else 0,design['population']))
            if index%25==0: print(f'Parallel {phase}, session {s}: {index} runs checked/completed',flush=True)
        freeze(folder/f'session_{s}_complete.json',dict(session=s,completed=True))
        atomic_write_json(folder/'session_environment'/f's{s}_{stamp}_end.json',session_environment())
    if all((folder/f'session_{s}_complete.json').exists() for s in sessions):
        analyze(base,design,phase)
    else: print('Session complete. Remaining sessions must finish before selection.',flush=True)


def dataset(base,design,phase):
    folder=base/'02_parallel'/phase;selected=read(folder/'methods.json');rows=[]
    for s in design['parallel'][phase+'_sessions']:
        for key in jobs(design,phase,s,selected):
            from .common import fingerprint
            path=folder/'runs'/(fingerprint(key)+'.json')
            if not path.exists(): raise ValueError(f'Missing planned parallel run: {key}')
            record=read(path)
            if record['key']!=key: raise ValueError('Parallel checkpoint mismatch')
            r=record['result'];m=key['method']
            rows.append({k:v for k,v in key.items() if k!='method'}|dict(
                method=name(m),architecture=m['architecture'],workers=m['workers'],topology=m['topology'],
                native_seconds=r['elapsed_seconds'],wall_seconds=r['wall_seconds'],value=r['best_value'],
                initial=r['initial_best'],fe=r['function_evaluations'],trace=r['trace_hash'],
                evaluation_seconds=r['evaluation_wall_seconds'],propagation_seconds=r['propagation_seconds'],
                selection_seconds=r['social_selection_seconds'],initialization_seconds=r['initialization_seconds'],
                migration_seconds=r['migration_seconds']))
    frame=pd.DataFrame(rows)
    block=['session','benchmark','dimension','seed','repetition','topology']
    ref=frame[frame.architecture.eq('serial')][block+['native_seconds','wall_seconds','trace','value','initial','fe']]
    frame=frame.merge(ref,on=block,validate='many_to_one',suffixes=('','_serial'))
    if not frame.fe.eq(frame.fe_serial).all(): raise ValueError('Unmatched total FE')
    sync=frame.architecture.isin(['master_slave','particle_parallel'])
    if not frame.loc[sync,'trace'].eq(frame.loc[sync,'trace_serial']).all():
        raise ValueError('Synchronous trajectory differs from matched serial; do not report scaling')
    frame['serial_speedup']=frame.native_seconds_serial/frame.native_seconds
    frame['serial_efficiency']=frame.serial_speedup/frame.workers
    frame['serial_wall_speedup']=frame.wall_seconds_serial/frame.wall_seconds
    frame['ratio_kind']=np.where(frame.architecture.isin(['serial','master_slave','particle_parallel']),
                                  'same-work synchronous speedup','runtime ratio; changed search dynamics')
    # Strong scaling only for equivalent synchronous backends.
    strong=frame[frame.workers.eq(1)&frame.architecture.isin(['master_slave','particle_parallel'])]
    strong=strong[block+['architecture','native_seconds']].rename(columns={'native_seconds':'backend_one_seconds'})
    frame=frame.merge(strong,on=block+['architecture'],how='left',validate='many_to_one')
    frame['strong_speedup']=frame.backend_one_seconds/frame.native_seconds
    frame['strong_efficiency']=frame.strong_speedup/frame.workers
    # Same global-serial initial best normalizes every topology/architecture.
    global_ref=ref[ref.topology.eq('global')].drop(columns='topology')
    global_ref=global_ref[['session','benchmark','dimension','seed','repetition','initial']].rename(columns={'initial':'global_initial'})
    frame=frame.merge(global_ref,on=['session','benchmark','dimension','seed','repetition'],validate='many_to_one')
    frame['quality']=np.log10(np.clip(frame.value/frame.global_initial,1e-16,1.))
    if not np.isfinite(frame[['native_seconds','wall_seconds','quality','serial_speedup']]).all().all():
        raise ValueError('Nonfinite parallel measurements')
    return frame


def choose(frame,design):
    # Repetitions and sessions are first collapsed within optimizer seed.
    # Methods are then compared over equally weighted benchmark/dimension cells.
    keys=['benchmark','dimension','seed','method']
    seed=frame.groupby(keys).agg(log_time=('wall_seconds',lambda x:np.log10(x).mean()),
                                quality=('quality','mean')).reset_index()
    aggregate=seed.groupby(['seed','method'])[['log_time','quality']].mean().reset_index()
    q=aggregate.pivot(index='seed',columns='method',values='quality')
    t=aggregate.pivot(index='seed',columns='method',values='log_time')
    quality_best=q.mean().idxmin();rows=[]
    for m in q.columns:
        delta=paired_effect((q[m]-q[quality_best]).to_numpy(),design['inference']['bootstrap'])
        rows.append(dict(method=m,mean_quality=q[m].mean(),mean_log_time=t[m].mean(),
                         quality_loss=delta['effect'],quality_loss_hi=delta['hi'],
                         eligible=delta['hi']<=design['parallel']['quality_margin_log10']))
    ranking=pd.DataFrame(rows).sort_values(['eligible','mean_log_time'],ascending=[False,True])
    return str(ranking[ranking.eligible].iloc[0]['method']),ranking


def analyze(base,design,phase):
    frame=dataset(base,design,phase);folder=base/'02_parallel'/phase;out=folder/'analysis'
    out.mkdir(parents=True,exist_ok=True);frame.to_csv(out/'paired_runs.csv',index=False)
    summaries=[]
    for key,g in frame.groupby(['benchmark','dimension','method']):
        b,d,m=key
        for metric in ['serial_speedup','serial_wall_speedup','strong_speedup']:
            values=g.groupby('seed')[metric].agg(lambda x:np.log10(x.dropna()).mean())
            values=values.dropna()
            if values.empty: continue
            stat=paired_effect(values.to_numpy(),design['inference']['bootstrap'])
            summaries.append(dict(benchmark=b,dimension=d,method=m,metric=metric,
                estimate=10**stat['effect'],lo=10**stat['lo'],hi=10**stat['hi'],
                seed_clusters=len(values),workers=int(g.workers.iloc[0]),architecture=g.architecture.iloc[0],
                topology=g.topology.iloc[0],ratio_kind=g.ratio_kind.iloc[0]))
    csv(out/'scaling_summary.csv',summaries)
    all_methods={name(m):m for m in read(folder/'methods.json')}
    if phase=='selection':
        winner,ranking=choose(frame,design);ranking.to_csv(out/'selection_ranking.csv',index=False)
        by_case={}
        for (b,d),g in frame.groupby(['benchmark','dimension']):
            local,_=choose(g,design);by_case[f'{b}|{d}']=all_methods[local]
        freeze(folder/'frozen.json',dict(global_method=all_methods[winner],by_case=by_case,
            dt=read(base/'01_dt/holdout/complete.json')['dt'],
            rule='minimum geometric-mean completed-run wall time among candidates within the prespecified quality margin',
            margin=design['parallel']['quality_margin_log10'],
            caveat='selection is exploratory; quality eligibility is not an equivalence claim'))
    else:
        frozen=read(base/'02_parallel/selection/frozen.json');contrasts=[]
        targets={name(frozen['global_method']),*[name(m) for m in frozen['by_case'].values()]}
        for (b,d),g in frame.groupby(['benchmark','dimension']):
            for m in targets:
                x=g[g.method.eq(m)].groupby('seed').quality.mean()
                topo=all_methods[m]['topology'];reference=name(method(topology=topo))
                y=g[g.method.eq(reference)].groupby('seed').quality.mean()
                stat=paired_effect((x-y).to_numpy(),design['inference']['bootstrap'])
                contrasts.append(dict(benchmark=b,dimension=d,method=m,reference=reference,**stat))
        report=pd.DataFrame(contrasts);report['p_holm']=holm(report.p)
        report['quality_margin_ci_criterion_met']=report.hi<=design['parallel']['quality_margin_log10']
        report.to_csv(out/'holdout_quality_contrasts.csv',index=False)
    freeze(folder/'complete.json',dict(phase=phase,runs=len(frame),equivalent_sync_rows=int(frame.architecture.isin(['master_slave','particle_parallel']).sum())))
    from .figures import parallel_figures
    parallel_figures(base,design,phase)
    print(f'Parallel {phase} complete: {len(frame)} matched measurements. {out}',flush=True)
