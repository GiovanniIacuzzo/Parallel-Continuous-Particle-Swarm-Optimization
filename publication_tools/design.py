"""Prespecified plans. Pilot outputs never feed the independent paper protocol."""
from .common import FUNCTIONS, TOPOS


def protocol(profile):
    if profile not in {'smoke','pilot','paper'}: raise ValueError(profile)
    smoke=profile=='smoke'; pilot=profile=='pilot'
    start=100000 if smoke else 200000 if pilot else 300000
    return dict(version=2,profile=profile,population=16 if smoke else 64,
        dt=dict(functions=FUNCTIONS[:2] if smoke else FUNCTIONS,
            dimensions=[2,4] if smoke else [30,100],
            iterations=[2,4] if smoke else [15,63,127] if pilot else [31,127,511],
            grid=[.175,.38125,1.,3.6,30.] if smoke else
                [.04,.12,.175,.25,.35,.38125,.5,.7,1.,2.,3.6,30.] if pilot else
                [.01,.02,.04,.08,.12,.175,.2,.25,.3,.35,.38125,.5,.7,1.,1.5,2.,3.,3.6,5.,7.,10.,15.,30.,50.],
            selection_seeds=list(range(start+100,start+100+(2 if smoke else 8 if pilot else 30))),
            holdout_seeds=list(range(start+200,start+200+(2 if smoke else 12 if pilot else 30))),reference=30.),
        parallel=dict(functions=['rastrigin','inverse_diffusion'],dimensions=[2,4] if smoke else [30,100],
            topologies=['global','random_k8'] if smoke else TOPOS,
            workers=[1,2] if smoke else [1,2,4,8], iterations=2 if smoke else 15 if pilot else 63,
            work=16 if smoke else 128 if pilot else 512,
            selection_seeds=list(range(start+300,start+300+(2 if smoke else 4 if pilot else 8))),
            holdout_seeds=list(range(start+400,start+400+(2 if smoke else 8 if pilot else 12))),
            selection_sessions=[1] if smoke or pilot else [1,2],holdout_sessions=[2] if smoke or pilot else [3],
            repetitions=1 if smoke else 2 if pilot else 3,
            quality_margin_log10=.10,islands=2 if smoke else 4,
            migration_intervals=[0,2] if smoke else [0,2,5] if pilot else [0,10,50]),
        comparison=dict(functions=FUNCTIONS[:2] if smoke else FUNCTIONS+['inverse_diffusion'],
            dimensions=[2,4] if smoke else [5,20,100],
            iterations=[2,4] if smoke else [15,63,127] if pilot else [31,127,511],
            seeds=list(range(start+500,start+500+(2 if smoke else 8 if pilot else 30))),
            inverse_dimensions=[2,4] if smoke else [20],
            work=16 if smoke else 128 if pilot else 512,
            targets=[100.,10.,1.,.1,.01,.001,1e-5,1e-8]),
        external=dict(functions=[1,8] if smoke else list(range(1,25)),
            dimensions=[2] if smoke else [5,20],instances=[1] if smoke else list(range(11,14)) if pilot else list(range(21,36)),
            checkpoints=[32,64] if smoke else [64,192,512], seed=start+600),
        inference=dict(bootstrap=1000 if smoke else 5000 if pilot else 10000,
                       seed=829173,alpha=.05,
                       primary='paired seed aggregates conditional on fixed functions and dimensions',
                       secondary='function-level contrasts report cross-function heterogeneity separately'))
