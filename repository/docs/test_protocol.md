# Test protocol

The repository has two automated layers: native C++ tests for numerical and
concurrency invariants, and Python tests for the `ctypes` ABI, public wrapper,
experiment validators and campaign analysis. Tests are correctness checks; the
smoke campaigns are not evidence of optimization superiority or speedup.

## C++ tests

The CMake test executable covers:

1. The closed-form damped-oscillator propagator against a fine-step RK4 oracle
   in underdamped, critically damped and overdamped regimes.
2. The semigroup property, equilibrium and degenerate-frequency cases.
3. Boundary handling, restitution and inward/outward velocity cases.
4. Built-in objective values, minimizers and invalid-dimension handling.
5. Random-stream determinism, topology construction and social-leader selection.
6. Fixed-seed repeatability, monotone incumbent history, bounds and exact
   function-evaluation accounting.
7. Exact serial equivalence for synchronous master--slave and particle-parallel
   execution, including deterministic commit order.
8. Controlled overlap of concurrent objective calls, worker-count validation,
   and propagation of callback exceptions/non-finite results without deadlock.
9. Asynchronous event/history contracts and deterministic single-worker
   behavior covered by the native regression cases.
10. Island partitioning, migration semantics, migration accounting and result
    metadata.

## Python tests

The Python suite covers configuration validation, ABI/version/build metadata,
end-to-end calls through `ctypes`, fixed-seed repeatability, exact serial versus
synchronous-parallel equivalence, particle topologies, asynchronous and island
contracts, result-history/FE invariants, and validation/statistical utilities
for the research campaign. It also checks that an architecture-study dry-run
is read-only and does not require the compiled native library.

## Reproducible commands

Run from the repository root. The native library is built separately from the
Python package:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCPSO_BUILD_TESTS=ON
cmake --build build --parallel
ctest --test-dir build --output-on-failure

python -m pip install -e ".[analysis,test]" --no-build-isolation
CPSO_SERIAL_LIBRARY="$PWD/build/lib/libcpso_serial.so" PYTHONPATH=python \
  python -m pytest -q
PYTHONPATH=python python scripts/run_smoke.py
```

On macOS, the native library is normally `build/lib/libcpso_serial.dylib`;
Windows uses a `.dll` in the CMake runtime output directory. The wrapper can
also discover the default `build/` location. Set `CPSO_SERIAL_LIBRARY` when the
library is elsewhere.

Run the small integration-only architecture smoke study:

```bash
PYTHONPATH=python python experiments/04_architectures/run_architecture_study.py \
  configs/experiments/architecture_study_smoke.json \
  --output /tmp/cpso-architecture-smoke --no-plots
```

Run the campaign validator without creating output or executing runs:

```bash
PYTHONPATH=python python experiments/04_architectures/run_research_campaign.py \
  configs/campaigns/definitive_architecture_campaign.json --dry-run
```

For memory/undefined-behavior and data-race checks, separate Debug builds with
AddressSanitizer + UndefinedBehaviorSanitizer and ThreadSanitizer were used
on the Linux validation host. These sanitizer runs should be repeated on each
supported compiler/platform before a release.

A smoke run checks the execution, accounting and output contracts. Its
optimization score is not a scientific performance claim. Publication-grade
conclusions require the declared seed/instance protocol, external benchmark
adapters where claimed, suitable baseline methods, and appropriate statistical
inference.
