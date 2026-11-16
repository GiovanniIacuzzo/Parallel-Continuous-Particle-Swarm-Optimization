"""Guard the topology actually passed to the external objective callback."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "python"), str(ROOT / "experiments/05_external_benchmarks")]
import common


def test_cpso_external_graph_dispatch(monkeypatch):
    monkeypatch.setattr(common, "run_serial_objective", lambda config, objective: config)
    def configuration(method):
        return common.run_method(method, sum, (-5.,)*5, (5.,)*5,
                                 population_size=64, max_evaluations=1920, seed=57, cpso_dt=1.)
    primary = configuration("cpso_random_k8")
    control = configuration("cpso_continuous")
    assert (primary.topology, primary.topology_degree) == ("random", 8)
    assert control.topology == "global"
    assert (primary.seed, primary.dt, primary.iterations) == (control.seed, control.dt, control.iterations) == (57, 1., 29)
