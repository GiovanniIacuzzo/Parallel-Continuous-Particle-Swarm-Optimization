# Test protocol

The test suite is split between deterministic C++ unit/integration tests and
Python binding tests.

## C++ tests

1. **Analytic propagator versus an independent reference:** fourth-order
   Runge--Kutta integration with a fine fixed substep is used only as a test
   oracle. The analytic flow is checked in underdamped, critically damped, and
   overdamped regimes.
2. **Semigroup property:** for frozen coefficients, propagating once over
   \(h_1+h_2\) is compared with propagation over \(h_1\) followed by \(h_2\).
3. **Degenerate and equilibrium cases:** zero interval, stationary
   equilibrium, zero frequency with damping, and zero frequency without
   damping.
4. **Boundary map:** lower and upper violations, restitution, and inward
   velocity handling.
5. **Objective implementations:** known values and minimizers for the
   implemented benchmark functions; Rosenbrock rejects dimension one.
6. **Optimizer contract:** fixed-seed repeatability, history monotonicity,
   bounds on the reported incumbent, history length, and exact FE accounting.
7. **Input/error handling:** invalid bound dimensions and non-finite objective
   output are rejected.
8. **Master--slave equivalence:** for 1, 2, 4, and 8 worker threads, final best
   value, position, full history, and FE count must equal the serial reference
   exactly for deterministic objectives.
9. **Concurrent dispatch:** a controlled blocking objective verifies that
   initial-population calls overlap in time.
10. **Parallel failure handling:** objective exceptions and non-finite return
    values must propagate without deadlock; invalid worker counts are rejected.
11. **Timing/result contract:** timing fields are non-negative and worker count
    is reported consistently.

## Python binding tests

1. Configuration validation.
2. Native ABI/version check.
3. End-to-end smoke run through `ctypes`.
4. Fixed-seed repeatability.
5. Zero-transition, one-dimensional run.
6. Exact serial/master--slave result equivalence across worker counts.
7. Master--slave result timing fields and worker-count validation.

## Commands

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCPSO_BUILD_TESTS=ON
cmake --build build --parallel
ctest --test-dir build --output-on-failure

python -m pip install -e . --no-build-isolation
python -m pip install pytest
PYTHONPATH=python pytest -q
python scripts/run_smoke.py
PYTHONPATH=python python experiments/02_master_slave/benchmark_scaling.py --workers 2,4
```

The smoke experiment checks execution and output contracts; its optimization
score is not a scientific performance claim. Benchmark-quality conclusions
require a separate preregistered seed/instance protocol and statistical
analysis.
