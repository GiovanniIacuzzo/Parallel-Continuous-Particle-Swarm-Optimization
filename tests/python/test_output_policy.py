from pathlib import Path
import pytest
from cpso_serial.output_policy import REPOSITORY_ROOT, RESULTS_ROOT, results_output


def test_allowed_output_is_under_results():
    path = results_output(RESULTS_ROOT / "figures/dt/my_study")
    assert path == RESULTS_ROOT / "figures/dt/my_study"


def test_reject_paper_even_if_explicit():
    with pytest.raises(ValueError, match="must be inside"):
        results_output(REPOSITORY_ROOT / "paper/figures/should_not_exist")


def test_reject_external_directory(tmp_path):
    with pytest.raises(ValueError, match="must be inside"):
        results_output(tmp_path / "my-output")
