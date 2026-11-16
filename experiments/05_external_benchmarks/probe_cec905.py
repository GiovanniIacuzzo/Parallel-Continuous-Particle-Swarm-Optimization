#!/usr/bin/env python3
"""Independent CEC F13/F14 905-dimensional objective smoke.

Run BEFORE the 22-run optimizer smoke; if the native extension segfaults,
the shell sees the nonzero exit status and no campaign data are touched.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'python'))
from cpso_serial.benchmarks.cec2013_lsgo import CEC2013LSGOAdapter

if __name__=='__main__':
    fun=int(sys.argv[1]) if len(sys.argv)>1 else 13
    if fun not in (13,14):
        raise SystemExit('Only F13/F14 supported by this probe')
    p=CEC2013LSGOAdapter().get_problem(fun)
    assert p.dimension == 905, (fun,p.dimension)
    base=np.linspace(-0.3,0.3,905,dtype=np.float64)
    v=p.objective(tuple(base))
    assert np.isfinite(v), (fun,v)
    # Test that the native objective ignores the extra 95 elements: this is
    # essential for treating the benchmark as truly 905-dimensional.
    from cec2013lsgo.cec2013 import Benchmark
    native_benchmark=Benchmark()
    native=native_benchmark.get_function(fun)
    padded=np.zeros(1000,dtype=np.float64)
    padded[:905]=base
    normal=float(native(padded))
    padded[905:]=np.linspace(-0.6,0.6,95)
    changed=float(native(padded))
    assert np.isfinite(normal) and np.isfinite(changed), (normal,changed)
    if not np.isclose(normal,changed,rtol=1e-11,atol=1e-8):
        raise RuntimeError(f'CEC F{fun}: trailing 95 inputs change fitness: {normal} -> {changed}')
    if not np.isclose(v,normal,rtol=1e-11,atol=1e-8):
        raise RuntimeError(f'CEC F{fun}: adapter/native fitness mismatch: {v} != {normal}')
    print(f'PASS CEC F{fun}: effective D={p.dimension}, f(x)={v:.10g}, ignored native tail verified')
