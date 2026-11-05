# CPSO — continuous particle-swarm optimization and parallel execution

This repository contains a C++20 implementation of a continuous-time PSO
(CPSO) model, a Python `ctypes` interface, and experimental drivers for
comparing execution architectures. The code separates algorithmic dynamics
from execution strategy: some parallel modes preserve the serial synchronous
trajectory exactly, while asynchronous and island modes intentionally change
information flow and therefore define different search dynamics.

## Implemented execution modes

| Mode | Parallel work | Relation to serial reference |
|---|---|---|
| `serial` | None | Reference algorithm |
| `master_slave` | Objective-function evaluations | Exact trajectory equivalence is required and tested |
| `particle_parallel` | Initialization, particle propagation and objective evaluations | Exact trajectory equivalence is required and tested |
| `asynchronous` | Event-driven particle updates | Different, schedule-dependent update dynamics for multiple workers |
| `islands` | Independent sub-swarms and optional migration | Different population information flow |

Social topologies are global-best, periodic ring and periodic toroidal von
Neumann grid. The total objective-evaluation budget used in the architecture
studies is held at `N * (K + 1)` for population `N` and `K` update steps.

## Repository map

- `cpp/`: C++20 numerical core, optimizers, thread-pool execution, social
  topologies and C ABI.
- `python/cpso_serial/`: Python configuration/results and native-library
  loader. The import package retains the historical `cpso_serial` name.
- `experiments/02_master_slave/`: early synchronous scaling studies.
- `experiments/03_topologies/`: social-topology comparison and analysis.
- `experiments/04_architectures/`: unified five-mode study runner, campaign
  orchestrator and paired statistical analysis.
- `configs/`: smoke, architecture-study and predeclared campaign JSON files.
- `analysis/`: reusable analysis and plotting utilities for earlier studies.
- `docs/`: mathematical model, implementation decisions, protocols and audit.
- `paper/`: current manuscript draft and bibliography.
- `tests/`: C++ and Python regression suites.

## Build and test

The native library must be compiled before running Python calls. From the
repository root:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCPSO_BUILD_TESTS=ON
cmake --build build --parallel
ctest --test-dir build --output-on-failure

python -m pip install -e ".[analysis,test]" --no-build-isolation
CPSO_SERIAL_LIBRARY="$PWD/build/lib/libcpso_serial.so" PYTHONPATH=python \
  python -m pytest -q
PYTHONPATH=python python scripts/run_smoke.py
```

The shared-library filename is platform-specific: `.so` on Linux, `.dylib` on
macOS and `.dll` on Windows. Usually the wrapper discovers the CMake `build/`
folder automatically; `CPSO_SERIAL_LIBRARY` overrides discovery when needed.

## Continuous integration

`.github/workflows/ci.yml` defines GitHub Actions checks on Ubuntu and macOS
for Python 3.10 and 3.12. It configures/builds the C++ library, runs CTest and
the Python regression suite, and checks Python syntax. The workflow is
configured in the repository but still needs to be executed by GitHub; local
verification for this development pass was performed on Linux/GCC.

## Experimental entry points

Validate the full predeclared campaign and print its run count, without
creating output files or requiring the native library:

```bash
PYTHONPATH=python python experiments/04_architectures/run_research_campaign.py \
  configs/campaigns/definitive_architecture_campaign.json --dry-run
```

Run the small integration campaign (not for scientific inference):

```bash
PYTHONPATH=python python experiments/04_architectures/run_research_campaign.py \
  configs/campaigns/campaign_smoke.json \
  --output /tmp/cpso-campaign-smoke --no-plots
```

The definitive campaign contains 9,360 optimization runs and 273,888,000
objective evaluations. Use the protocol in
`docs/definitive_experimental_campaign.md`; preserve raw JSONL checkpoints and
manifests. Do not edit generated records by hand. The `results/` directory is
intentionally not ignored by Git because raw experiments are research data.

## Scope and scientific limitations

The native engine currently exposes six analytic optimization objectives and
one separately marked synthetic `sphere_work` objective used only to probe
evaluation overhead. The current research campaign is an internal architecture
study, not an external benchmark-suite submission: COCO/BBOB, BBOB-Large-Scale,
CEC 2013 LSGO adapters, standard-PSO baselines, MPI and GPU backends are not
implemented in this repository state. No superiority or state-of-the-art claim
on those suites or architectures follows from the current code.

The exact mathematical fidelity of the serial implementation to the complete
source article by Orlando and Ricciardello (2021) remains an explicit audit
item in `docs/implementation_decisions.md`. Until that equation-by-equation
comparison is completed, describe the implementation as a reconstruction
with documented operational conventions rather than a verified strict
reproduction.

## Package and ABI naming

The distribution metadata is `cpso-parallel`, while the Python import package,
CMake target and shared library retain the historical names `cpso_serial` and
`libcpso_serial`. This is intentional compatibility debt for now; avoid
renaming these interfaces without a migration plan. The project/native version
is `0.6.0`; the C ABI version is `5` (these are separate version numbers).

The current `paper/paper.pdf` is a working draft containing the introduction,
model/semantics and experimental protocol. It does not yet contain definitive
results or conclusions. The full status and unresolved prerequisites are
recorded in `docs/repository_audit.md`; licensing is intentionally unset until
the rights holder confirms it (`docs/license_status.md`).

See `docs/repository_audit.md` for the current audit scope, verified checks,
known limitations and recommended development priorities.
