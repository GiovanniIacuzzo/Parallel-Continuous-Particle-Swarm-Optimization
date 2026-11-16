"""Check pairing and normalization in the publication plotting transformations."""
import numpy as np
import pandas as pd
import pytest
from plot_review_publication import paired_log_effects,phase_fractions,session_ratios


def test_gap_ratios_are_paired_before_aggregation():
    a=pd.DataFrame([dict(dimension=20,function=1,instance=i,method=m,gap=g)
        for i,l,r in [(1,1.,100.),(2,100.,1.),(3,2.,100.)]
        for m,g in [('left',l),('right',r)]])
    effects=paired_log_effects(a,'left','right',1e-12)
    assert effects.effect.median()==pytest.approx(np.log10(.02))
    # A missing partner must not silently change the matched estimand.
    with pytest.raises(ValueError,match='Unpaired'):
        paired_log_effects(a.iloc[:-1],'left','right',1e-12)


def test_measured_one_worker_overhead_is_not_forced_away():
    t=pd.DataFrame(dict(session=[1,1,1,1],workers=[1,1,4,4],serial_native_speedup=[.7,.9,2.8,3.2]))
    s=session_ratios(t).set_index('workers')
    assert s.loc[1,'serial_native_speedup']==pytest.approx(.8)
    assert s.loc[4,'normalized_efficiency']==pytest.approx(.75)


def test_phase_fractions_retain_residual_and_detect_overlap():
    a=pd.DataFrame(dict(native_seconds=[10.],evaluation_seconds=[4.],propagation_seconds=[2.],
        initialization_seconds=[1.],social_seconds=[.5],topology_seconds=[.5]))
    f=phase_fractions(a)
    assert f.other_seconds.iloc[0]==pytest.approx(.2)
    assert f.sum(axis=1).iloc[0]==pytest.approx(1.)
    a.loc[0,'evaluation_seconds']=14.
    with pytest.raises(ValueError,match='overlap'):phase_fractions(a)
