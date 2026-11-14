#!/usr/bin/env python3
"""A light-weight end-to-end consistency audit of native CPSO paths.

No algorithmic modification, no statistical publication claims. Run on Mac M2
with CPSO_SERIAL_LIBRARY pointing to the built native .dylib.
"""
from __future__ import annotations
import math
from cpso_serial import CpsoConfig, run_serial, run_master_slave, run_particle_parallel


def same(a,b,rel=1e-11,abs_=1e-10):
    if len(a)!=len(b):
        raise AssertionError('Mismatched lengths')
    for x,y in zip(a,b):
        if not math.isclose(x,y,rel_tol=rel,abs_tol=abs_):
            raise AssertionError(f'Mismatched trajectories: {x} vs {y}')


def run():
    count=0
    for benchmark,bounds in [('sphere',(-100,100)),('rastrigin',(-5.12,5.12))]:
        for dt in (0.01,0.2,0.3,3.6,30.0):
            for dim in (5,30):
                c=CpsoConfig(lower_bounds=(bounds[0],)*dim,upper_bounds=(bounds[1],)*dim,
                    population_size=24,iterations=50,seed=970001,dt=dt,topology='global',
                    mu_max=.9,mu_min=.4,cognitive=2.05,social=2.05,
                    initial_velocity_fraction=.1,boundary_restitution=.5)
                serial=run_serial(c,benchmark)
                master=run_master_slave(c,benchmark,workers=2)
                particle=run_particle_parallel(c,benchmark,workers=2)
                for other in (master,particle):
                    assert serial.function_evaluations==other.function_evaluations==24*51
                    same(serial.best_history,other.best_history)
                    same(serial.best_position,other.best_position)
                repeated=run_serial(c,benchmark)
                same(serial.best_history,repeated.best_history,rel=0,abs_=0)
                count+=1
    print(f'PASS: {count} test cells, serial/master-slave/particle-parallel and serial reproducibility')
    print('NOTE: This is an engineering contract check, not independent proof of the original CPSO formulation.')

if __name__=='__main__':
    run()
