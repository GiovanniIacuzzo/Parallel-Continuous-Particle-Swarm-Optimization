"""Visualization checks on synthetic fixtures, never optimization evidence.

Run from the checkout root: python -m unittest discover -s tests/plotting -v
"""
from pathlib import Path
import hashlib
import importlib.util
import itertools
import json
import tempfile
import unittest
import sys

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import replot_readable as plotting

METHODS=[m for _,group in plotting.FAMILIES for m in group]
PRIMARY='CPSO selected framework'
FROZEN=dict(architecture='particle_parallel',topology='random_k8',workers=8)


def fixture(base):
    rng=np.random.default_rng(1942)
    def write(relative,a):
        p=base/relative;p.parent.mkdir(parents=True,exist_ok=True);a.to_csv(p,index=False)
    p=base/'02_parallel/selection/frozen.json';p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(dict(global_method=FROZEN)))
    topologies=['global','ring','grid_von_neumann','wheel','random_k4','random_k8','small_world_k4','scale_free_k4']
    meta=[];ranking=[]
    def add(arch,topo,w,**extra):
        m=dict(architecture=arch,topology=topo,workers=w,**extra);name=plotting.canonical_method(m)
        duration={'serial':12,'master_slave':7,'particle_parallel':5,'asynchronous':5.5,'islands':8}[arch]*(1+.1*rng.normal())/max(1,w)**.18
        quality=float(-.32-.28*rng.random())
        ranking.append(dict(method=name,mean_log_time=np.log10(duration/1000),mean_quality=quality,eligible=quality<-.52))
        meta.append(dict(method=name,architecture=arch,workers=w,topology=topo))
    for topo in topologies:
        add('serial',topo,1)
        for arch,w in itertools.product(['master_slave','particle_parallel','asynchronous'],[1,2,4,8]):add(arch,topo,w)
        if topo!='grid_von_neumann':
            for w,interval,mt in itertools.product([1,2,4],[0,2,5],['ring','fully_connected']):
                if interval==0 and mt=='fully_connected':continue
                add('islands',topo,w,islands=4,migration_interval=interval,migration_topology=mt)
    write('02_parallel/selection/analysis/selection_ranking.csv',pd.DataFrame(ranking))
    write('02_parallel/selection/analysis/paired_runs.csv',pd.DataFrame(meta))
    hold=[]
    for b,d,value in [('inverse_diffusion',30,4.02),('inverse_diffusion',100,4.16),('rastrigin',30,.87),('rastrigin',100,1.72)]:
        hold.append(dict(benchmark=b,dimension=d,method=plotting.canonical_method(FROZEN),metric='serial_speedup',estimate=value,lo=value*.94,hi=value*1.06))
    write('02_parallel/holdout/analysis/scaling_summary.csv',pd.DataFrame(hold))
    rows=[];method_names=[PRIMARY,*METHODS]
    for d,b,s,it in itertools.product([5,20,100],['sphere','rastrigin','ackley','rosenbrock','griewank','salomon'],range(4),[15,63,127]):
        score=rng.normal(5,2,len(method_names));score[0]=2 if it==15 else 4 if it==63 else 5
        order=pd.Series(score).rank().to_numpy()
        for j,m in enumerate(method_names):
            rows.append(dict(benchmark=b,dimension=d,seed=s,iterations=it,method=m,rank=order[j],
                seconds=(.0005 if j==0 else .002*(1+j/10))*(it+1)*(.9+.2*rng.random()),fe=64*(it+1)))
    write('03_comparison/native/analysis/ranked_results.csv',pd.DataFrame(rows))
    rows=[]
    for d,f,i,m in itertools.product([5,20],range(1,25),range(1,4),[plotting.REFERENCE,*METHODS]):
        base_hit=1+rng.random()*35
        for j,target in enumerate(np.logspace(2,-8,21)):
            hit=base_hit*d*1.9**j
            rows.append(dict(function=f,dimension=d,instance=i,method=m,target=target,
                             hit_fe=int(np.ceil(hit)) if hit<=512*d else np.nan,budget=512*d))
    write('03_comparison/bbob/analysis/target_hits.csv',pd.DataFrame(rows))


class ReadabilityTests(unittest.TestCase):
    def test_pareto_retains_tradeoffs_and_ties(self):
        np.testing.assert_array_equal(plotting.pareto_mask([1,2,3,1],[3,2,4,3]),[True,True,False,True])

    def test_ecdf_keeps_failed_targets_and_exact_jump(self):
        x,y=plotting.ecdf_values([5,10,np.nan,np.nan],5,100)
        self.assertEqual(y[-1],.5)
        self.assertEqual(y[np.flatnonzero(x==1)[0]],.25)
        self.assertEqual(y[np.flatnonzero(x==2)[0]],.5)
        self.assertTrue(np.all(np.diff(y)>=0))
        self.assertEqual(plotting.ecdf_values([np.nan,np.nan],5,100)[1][-1],0)
        with self.assertRaises(ValueError):plotting.ecdf_values([0],5,100)

    def test_incomplete_method_is_rejected(self):
        a=pd.DataFrame(dict(method=['A','A','B'],case=[1,2,1]))
        with self.assertRaisesRegex(ValueError,'incomplete'):
            plotting.validate_balanced(a,['case'],'example')

    def test_six_outputs_preserve_inputs_and_back_up_old_pdfs(self):
        with tempfile.TemporaryDirectory() as td:
            base=Path(td);fixture(base)
            expected=['02_parallel/holdout/fig_framework_holdout.pdf',
                '02_parallel/selection/fig_framework_quality_cost.pdf',
                '03_comparison/bbob/fig_bbob_ecdf_serial_control.pdf',
                *[f'03_comparison/native/fig_completed_run_tradeoff_d{d}.pdf' for d in [5,20,100]]]
            for n in expected:
                p=base/'figures'/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b'original output')
            source={p:hashlib.sha256(p.read_bytes()).hexdigest() for p in base.rglob('*') if p.suffix in ['.csv','.json']}
            result=plotting.regenerate(base)
            self.assertEqual(set(result['replaced_pdfs']),set(expected))
            self.assertEqual(source,{p:hashlib.sha256(p.read_bytes()).hexdigest() for p in source})
            self.assertEqual(len(list((base/'figures/_before_readability').rglob('*.pdf'))),6)
            for n in expected:self.assertTrue((base/'figures'/n).read_bytes().startswith(b'%PDF'))
            self.assertNotIn('cpso_serial',sys.modules)

    def test_missing_inputs_fail_before_replacing_existing_pdf(self):
        with tempfile.TemporaryDirectory() as td:
            base=Path(td);fixture(base)
            p=base/'figures/02_parallel/holdout/fig_framework_holdout.pdf';p.parent.mkdir(parents=True);p.write_bytes(b'keep')
            (base/'03_comparison/bbob/analysis/target_hits.csv').unlink()
            with self.assertRaises(FileNotFoundError):plotting.regenerate(base)
            self.assertEqual(p.read_bytes(),b'keep')


if __name__=='__main__':unittest.main()
