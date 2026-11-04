# CPSO-HPC — C++ engine with Python orchestration

Version 0.1.0 is the first controlled migration of the CPSO research code. The numerical hot path is C++17; Python is used for TOML configuration, experiment launch, results serialization and analysis. The C++ library is exposed through a small stable C ABI and loaded with Python `ctypes`, so pybind11 is not required.

## Architecture

```text
configs/*.toml
      │
      ▼
Python wrapper (config validation, seeds, output)
      │ ctypes / C ABI
      ▼
C++17 core ── exact frozen-ODE propagator
      │       ├─ CPSO population and personal bests
      │       ├─ independent / Star / synchronous Ring
      │       ├─ native benchmark functions
      │       └─ OpenMP shared-memory parallelism
      ▼
JSON + CSV results
```

**Important scope note:** this first migration provides shared-memory parallelism on one host through OpenMP. It is not yet an MPI multi-node implementation. The existing source used Python multiprocessing and MATLAB process pools; the paper should not claim multi-node distributed-memory execution until an MPI backend is implemented and validated on an actual cluster.

## Build (Linux)

Requirements: CMake 3.20+, a C++17 compiler, OpenMP (recommended), Python 3.11+ and NumPy.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCPSO_ENABLE_OPENMP=ON
cmake --build build --parallel
ctest --test-dir build --output-on-failure
python -m pip install -e .
```

On macOS, install a compiler toolchain and CMake (for example with Homebrew). Apple Clang does not provide OpenMP by default; install `libomp` and configure the OpenMP flags, or build with `-DCPSO_ENABLE_OPENMP=OFF` for the serial fallback. The code remains CPU-only and does not require PyTorch.

## Run

```bash
cpso-hpc --config configs/smoke.toml --output results/smoke
```

Or:

```bash
PYTHONPATH=python python scripts/run_experiment.py \
  --config configs/lsgo_rastrigin_500d.toml \
  --output results/rastrigin_500d_star
```

For a repeated strong-scaling sweep (the profile contains a fixed nonzero 500-D shift,
500,000 FEs/run and 30 paired seeds):

```bash
PYTHONPATH=python python scripts/run_scaling.py \
  --config configs/hpc_strong_scaling_500d.toml \
  --output-dir results/strong_scaling_500d
```

Edit the worker list to match the physical cores actually available on the target host.
The supplied 96-worker profile is a template, not a claim that the current development
environment can run 96 workers.

The configuration has three sections: `[problem]`, `[algorithm]`, `[cpso]`. You can change dimension, bounds, optimum position, global population, workers, topology, communication interval, seed, and CPSO coefficients without changing C++ code.

`population_size` is the **total population across all workers**. It is not the population per worker. For example, 100 particles and 4 workers means 25 particles per island (subject to deterministic remainder distribution).

## Topologies

- `independent`: each island uses only its own incumbent.
- `star`: all islands receive the best incumbent at a communication event.
- `ring`: synchronous one-hop diffusion around a directed ring.

Use `sync_interval = 1` for a clean Star strong-scaling comparison: with fixed seed and fixed total population, the Star trajectory is invariant to thread count and the worker count changes execution partitioning rather than the optimization trajectory. Ring is an algorithmic variation, not just a parallel implementation of the same trajectory.

## Tests

```bash
ctest --test-dir build --output-on-failure
pytest -q
```

The tests cover benchmark minima, deterministic seeds, FE accounting, monotone best-so-far logging, and Star invariance to the number of OpenMP workers.

## Research cautions

- The exact propagator solves the local ODE with frozen random coefficients and frozen personal/social attractors. It does not solve the entire switched stochastic CPSO process as one global closed-form ODE.
- `dt` controls the interval between refreshes and is an algorithmic parameter, not a local-truncation-error control parameter.
- The random stream construction is deterministic and scheduling invariant; it does not prove pairwise statistical independence.
- The native benchmark functions and their coordinate translations are documented in `docs/mathematical_model.md`.
- The legacy files are retained for provenance but are not used by the new engine.

## Directory layout

```text
cpp/                    C++17 engine and C ABI
python/cpso_hpc/         Python configuration and wrapper
configs/                 Reproducible TOML experiment files
scripts/                 Command-line entry points
tests/, cpp/tests/       Python and C++ tests
docs/                    Mathematical specification and migration audit
legacy/original_project/ Original uploaded scripts, kept inactive
data/gnbg_iii/           GNBG-III provenance and usage notes
```
