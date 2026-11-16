import numpy as np
import pytest
from review_study import holm, signed_test, graph_statistics, bootstrap_median


def test_holm_preserves_original_order_and_monotonicity():
    np.testing.assert_allclose(holm([.04,.001,.03]),[.06,.003,.06])
    np.testing.assert_allclose(holm([1.,1.,1.]),[1.,1.,1.])


def test_paired_inference_handles_complete_ties_and_direction():
    assert signed_test(np.zeros(24))==(1.,0.)
    p,effect=signed_test(-np.arange(1.,25.))
    assert p<.001 and effect==-1.
    assert bootstrap_median(np.ones(24))==dict(relative_effect_low=1.,relative_effect_high=1.)


def test_disconnected_graph_does_not_report_finite_mean_path():
    m=graph_statistics(np.eye(4,dtype=bool))
    assert not m['connected']
    assert m['mean_path'] is None
    assert m['clustering']==0.
