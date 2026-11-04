from pathlib import Path

import numpy as np
import pytest

from cpso_hpc import CpsoEngine, load_config

ROOT = Path(__file__).resolve().parents[1]


def test_config_validation():
    cfg = load_config(ROOT / "configs" / "smoke.toml")
    assert cfg["problem"]["dimension"] == 10
    assert cfg["algorithm"]["population_size"] >= cfg["algorithm"]["workers"]


def test_engine_smoke_and_accounting():
    cfg = load_config(ROOT / "configs" / "smoke.toml")
    result = CpsoEngine().run(cfg)
    assert np.isfinite(result.best_value)
    assert result.function_evaluations == 40 * 101
    assert len(result.history) == 101
    assert len(result.history_fes) == 101
    assert all(b <= a + 1e-12 for a, b in zip(result.history, result.history[1:]))


def test_star_is_worker_partition_invariant():
    cfg = load_config(ROOT / "configs" / "smoke.toml")
    cfg["algorithm"]["workers"] = 1
    r1 = CpsoEngine().run(cfg)
    cfg["algorithm"]["workers"] = 4
    r4 = CpsoEngine().run(cfg)
    assert r1.history == r4.history
    assert r1.best_position == r4.best_position


def test_bad_population_rejected():
    cfg = load_config(ROOT / "configs" / "smoke.toml")
    cfg["algorithm"]["population_size"] = 2
    cfg["algorithm"]["workers"] = 4
    with pytest.raises(ValueError):
        from cpso_hpc.config import validate_config
        validate_config(cfg)
