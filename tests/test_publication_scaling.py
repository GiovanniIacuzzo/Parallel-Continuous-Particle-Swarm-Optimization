"""End-to-end CLI smoke test with a deterministic fake native CPSO backend.

This checks orchestration, restart, pairing, CSVs, PDF generation, and exact
identity checks. It does NOT replace the actual C++ tests on the user's Mac.
"""
from __future__ import annotations
import csv
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

RUNNER = Path(__file__).resolve().parents[1] / "experiments/04_architectures/run_publication_scaling.py"

FAKE_INIT = '''
from dataclasses import dataclass
@dataclass
class CpsoConfig:
    lower_bounds: tuple
    upper_bounds: tuple
    population_size: int
    iterations: int
    seed: int
    dt: float
    mu_max: float
    mu_min: float
    cognitive: float
    social: float
    initial_velocity_fraction: float
    boundary_restitution: float
    topology: str
    topology_degree: int
    topology_rewiring_probability: float
    grid_rows: int = 0
    grid_columns: int = 0
    def validate(self):
        assert len(self.lower_bounds) == len(self.upper_bounds)

class R:
    def __init__(self, cfg, kind, workers):
        self.best_value = cfg.seed / 1000000.0
        self.best_position = [0.1] * len(cfg.lower_bounds)
        self.best_history = [2.0, self.best_value]
        self.history_evaluations = [cfg.population_size, cfg.population_size*(cfg.iterations+1)]
        self.function_evaluations = cfg.population_size*(cfg.iterations+1)
        size = 0.1 + len(cfg.lower_bounds) / 100.0
        self.elapsed_seconds = size * (1 if kind == 'serial' else 1.02 / workers)
        self.evaluation_wall_seconds = self.elapsed_seconds * 0.9
        self.propagation_seconds = self.elapsed_seconds * 0.05
        self.social_selection_seconds = self.elapsed_seconds * 0.01
        self.topology_setup_seconds = self.elapsed_seconds * 0.005
        self.initialization_seconds = self.elapsed_seconds * 0.01
        self.evaluation_worker_seconds = self.elapsed_seconds * workers
        self.propagation_worker_seconds = self.elapsed_seconds * 0.05

def run_serial(c, benchmark, work_iterations): return R(c, 'serial', 1)
def run_master_slave(c, benchmark, workers, work_iterations): return R(c, 'master_slave', workers)
def run_particle_parallel(c, benchmark, workers, work_iterations): return R(c, 'particle_parallel', workers)
def native_build_info(): return 'build_type=Release;compiler=FAKE'
'''
FAKE_NATIVE = '''
from pathlib import Path
class Fake:
    _name = str(Path(__file__).parent / 'fake_binary.so')
def _load_library(): return Fake()
'''


def run():
    with tempfile.TemporaryDirectory(prefix="cpso_pub_runner_test_") as base:
        root = Path(base)
        script = root / "experiments/04_architectures/run_publication_scaling.py"
        script.parent.mkdir(parents=True)
        shutil.copy2(RUNNER, script)
        pkg = root / "python/cpso_serial"
        pkg.mkdir(parents=True)
        (pkg / "__init__.py").write_text(FAKE_INIT)
        (pkg / "native.py").write_text(FAKE_NATIVE)
        (pkg / "fake_binary.so").write_bytes(b"FAKE-SHARED-LIBRARY-BYTES")
        base_args = [sys.executable, str(script), "--dimensions", "30", "--topologies", "global,ring",
                     "--workers", "1,2,4", "--architectures", "master_slave,particle_parallel",
                     "--seeds", "1,2", "--repetitions", "2", "--iterations", "1",
                     "--population", "8", "--work", "16", "--bootstrap", "100",
                     "--output", "results/raw/parallel/publication_mock"]
        def call(*extra):
            return subprocess.run(base_args + list(extra), cwd=root, text=True,
                                  capture_output=True, check=True)
        dry = call("--dry-run")
        assert "8 repeated paired blocks" in dry.stdout, dry.stdout
        call("--max-blocks", "1")
        out = root / "results/raw/parallel/publication_mock"
        assert len((out / "publication_scaling_runs.jsonl").read_text().splitlines()) == 7
        call()
        raw = [json.loads(s) for s in (out / "publication_scaling_runs.jsonl").read_text().splitlines()]
        assert len(raw) == 56, len(raw)
        call("--no-warmup")  # rerun is fully resumable: no new records
        assert len((out / "publication_scaling_runs.jsonl").read_text().splitlines()) == 56
        paired = list(csv.DictReader((out / "publication_scaling_paired.csv").open()))
        summary = list(csv.DictReader((out / "publication_scaling_summary.csv").open()))
        assert len(paired) == 48
        assert len(summary) == 2 * 2 * 3 * 5  # topologies=2; architectures=2; workers=3; metrics=5
        for record in paired:
            if record["workers"] == "1":
                assert float(record["strong_speedup"]) == 1
                assert float(record["strong_efficiency"]) == 1
        assert len(list(out.glob("*.pdf"))) == 2
        print("PASS: dry-run, partial, exact-pairing, resume, CSV stats, p=1, PDF output")


if __name__ == "__main__":
    run()
