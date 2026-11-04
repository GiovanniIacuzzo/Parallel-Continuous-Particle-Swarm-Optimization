# CPSO Serial Reference and Synchronous Master--Slave Engine

Version **0.3.0** extends the clean C++20 serial reference and synchronous
master--slave executor with a controlled workload and paired scaling-sweep
framework. The algorithmic reference remains the serial,
global-best, synchronous continuous particle swarm optimizer based on the
frozen damped-oscillator formulation associated with Orlando and Ricciardello.

## What is implemented

- Closed-form propagation of the frozen damped oscillator in underdamped,
  critically damped, and overdamped regimes.
- Serial synchronous global-best CPSO.
- Synchronous master--slave CPSO that parallelizes objective evaluations only.
- A persistent `std::thread` pool with indexed result collection and stable
  particle-order commits.
- Deterministic per-particle pseudo-random streams for fixed-seed repeatability.
- Explicit linear inertia schedule and bounded-domain clamp/restitution map.
- Objective-evaluation accounting, best-so-far history, and separated timing
  fields.
- Six built-in sanity benchmarks: Sphere, Rastrigin, Rosenbrock, Ackley,
  Griewank, and Salomon.
- C++ API, C ABI 3, Python `ctypes` wrapper, and C++/Python regression tests.
- A paired serial/master--slave local scaling experiment driver.
- A `sphere_work` synthetic arithmetic objective with a configurable number of loop iterations per evaluation (performance microbenchmark only).
- A multi-factor sweep over dimension, population, objective work, worker count and paired seeds.
- Median paired speedup, MAD, bootstrap intervals, efficiency, and PDF/SVG/PNG analysis plots.

## Build and test

Requirements: CMake 3.20+, a C++20 compiler, Python 3.10+, and (for Python
tests) `pytest`.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCPSO_BUILD_TESTS=ON
cmake --build build --parallel
ctest --test-dir build --output-on-failure

python -m pip install -e . --no-build-isolation
python -m pip install pytest
PYTHONPATH=python pytest -q
```

The shared library is written to `build/lib`. The Python wrapper discovers it
there automatically. For a custom location, set `CPSO_SERIAL_LIBRARY` to the
full library path. For Apple Silicon setup, see [`docs/build_macos.md`](docs/build_macos.md).

## Run a smoke experiment

```bash
python scripts/run_smoke.py
```

## Run a paired serial/master--slave scaling sweep

The 0.3.0 sweep varies problem dimension, population, controlled objective cost,
and worker count. Example (a local exploratory run):

```bash
PYTHONPATH=python python experiments/02_master_slave/run_scaling_sweep.py \
  --objective sphere_work --dimensions 32 --populations 32 \
  --iterations 30 --seeds 5 --workers 1,2,4,8 \
  --work-iterations 0,1000,10000,50000
```

The driver writes `scaling_runs.csv`, `scaling_summary.csv`, and
`scaling_metadata.json` to a new run directory. It verifies exact serial /
master--slave output equivalence for every seed before accepting timings.
Generate vector figures and a report with:

```bash
python experiments/02_master_slave/analyze_scaling_sweep.py \
  results/raw/master_slave_sweep/<run_directory>
```

The controlled `sphere_work` objective is not a recognized optimization
benchmark. It estimates the workload break-even region of the executor; it
does not establish performance on arbitrary real objectives. Read
[`docs/master_slave_scaling_protocol.md`](docs/master_slave_scaling_protocol.md)
before conducting or interpreting a sweep.

## Run the original single-configuration diagnostic

```bash
PYTHONPATH=python python experiments/02_master_slave/benchmark_scaling.py \
  --benchmark rastrigin --dimension 64 --population 96 --iterations 100 \
  --seeds 10 --workers 2,4,8
```

The script warms each code path, executes paired seed/configuration runs in a
deterministically shuffled order, and refuses to accept a timing pair unless
the final value, best position, complete best-so-far history, and function-
evaluation count match exactly. It writes a CSV of raw run records and a JSON
metadata/summary file under `results/raw/master_slave/` by default.

These are local exploratory measurements. Built-in analytic objectives are
cheap and may be dominated by thread scheduling and synchronization overhead.
Do not infer general scalability or make publication claims from one machine or
one objective family.

## Model and implementation conventions

Read before using this implementation in research:

- [`docs/mathematical_model.md`](docs/mathematical_model.md)
- [`docs/implementation_decisions.md`](docs/implementation_decisions.md)
- [`docs/master_slave.md`](docs/master_slave.md)
- [`docs/master_slave_scaling_protocol.md`](docs/master_slave_scaling_protocol.md)
- [`docs/test_protocol.md`](docs/test_protocol.md)

The closed-form flow is exact for each frozen constant-coefficient oscillator
segment, not for the complete switched stochastic optimization process. The
interval length `dt` changes the algorithmic observation/update schedule; it is
not a numerical ODE accuracy parameter. The source-fidelity register still
lists equation-level checks against the complete Orlando--Ricciardello paper.

## C++ API

- `analytic_propagator.hpp`: local oscillator flow and boundary map;
- `objectives.hpp`: built-in objective functions;
- `optimizer.hpp`: `Config`, `Result`, `minimize_serial`, and
  `minimize_master_slave`;
- `thread_pool.hpp`: persistent worker pool and indexed `parallel_for`;
- `c_api.h`: stable C ABI used by Python, including native build metadata
  (`native_build_info()`) for experiment manifests.

A custom objective can be passed through the C++ API. In master--slave mode the
objective must be deterministic and reentrant/thread-safe. The Python
interface exposes the built-in pure objectives and the explicitly labelled
`sphere_work` performance microbenchmark. Custom C++ objectives are
still available through the C++ API.

## Scope boundary

This release does not implement particle-neighborhood topologies, island
migration, asynchronous scheduling, Cooperative Co-evolution, COCO/CEC
adapters, standard PSO comparison algorithms, or publication-grade benchmark
orchestration. These remain later milestones.
