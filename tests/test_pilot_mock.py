"""Offline logic test only: a fake native solver validates pairing/plotting logic.

This is NOT a timing validation or replacement for the repository's C++ tests.
"""
from __future__ import annotations
import contextlib
import csv
import io
import os
import shutil
import runpy
import sys
import tempfile
import types
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
NATIVE=types.ModuleType("cpso_serial")
POLICY=types.ModuleType("cpso_serial.output_policy")

class CpsoConfig:
    def __init__(self,**kwargs): self.__dict__.update(kwargs)

class Result:
    def __init__(self,config,benchmark,steps,arch,workers):
        self.best_value=42.0
        self.best_position=tuple(0.1 for _ in range(len(config.lower_bounds)))
        self.best_history=(75.0, 42.0)
        self.history_evaluations=(config.population_size,config.population_size*(config.iterations+1))
        self.function_evaluations=config.population_size*(config.iterations+1)
        self.evaluation_wall_seconds=.01
        self.propagation_seconds=.005
        self.social_selection_seconds=.001
        factor={"serial":1.,"master_slave":max(.2,1.2/workers),
                "particle_parallel":max(.15,1./workers)}[arch]
        self.elapsed_seconds=factor*(.03+.0001*steps)

def native_info():return 'OFFLINE-MOCK-NOT-REAL'
def run_serial(c,b,work_iterations=0):return Result(c,b,work_iterations,'serial',1)
def run_master_slave(c,b,workers=2,work_iterations=0):return Result(c,b,work_iterations,'master_slave',workers)
def run_particle_parallel(c,b,workers=2,work_iterations=0):return Result(c,b,work_iterations,'particle_parallel',workers)
def output(p): return Path(p).resolve()

for name,val in list(locals().items()):
    if name in ("CpsoConfig","run_serial","run_master_slave","run_particle_parallel"):
        setattr(NATIVE,name,val)
NATIVE.native_build_info=native_info
POLICY.results_output=output
sys.modules["cpso_serial"]=NATIVE
sys.modules["cpso_serial.output_policy"]=POLICY

with tempfile.TemporaryDirectory(prefix="cpso_pilot_mock_") as tmp:
    filename=ROOT/"experiments/04_architectures/run_scaling_pilot.py"
    sys.argv=[str(filename),"--output",tmp,"--dimensions","4", "--benchmarks","sphere_work,inverse_diffusion",
              "--topologies","global,ring","--work","16,32", "--workers","1,2",
              "--seeds","94001,94002","--iterations","2","--no-warmup"]
    try:
        with contextlib.redirect_stdout(io.StringIO()) as result:
            runpy.run_path(str(filename),run_name="__main__")
    except SystemExit as exc:
        assert exc.code in (0,None),exc
    with (Path(tmp)/"scaling_pilot_summary.csv").open() as handle:
        rows=list(csv.DictReader(handle))
    # 2 objectives x 1 dimension x 2 topologies x 2 costs x 2 arch x 2 workers
    assert len(rows)==32, len(rows)
    assert len(list(Path(tmp).glob("fig_scaling_*.pdf")))==4
    assert all(float(r["speedup_median"]) > 0 for r in rows)
    if os.environ.get("KEEP_MOCK"):
        shutil.copyfile(sorted(Path(tmp).glob("fig_scaling_*.pdf"))[0],
                        Path(os.environ["KEEP_MOCK"]))
print("Pilot runner: pairing, CSV output, 4 vector PDFs PASS (mocked solver)")
