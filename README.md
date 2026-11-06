# CPSO — continuous particle-swarm optimization and parallel execution

This repository contains a C++20 implementation of a continuous-time PSO
(CPSO) model, a Python `ctypes` interface, five execution architectures, and a
separate external-benchmark layer with discrete-PSO reference baselines. The
code distinguishes algorithmic dynamics from execution strategy: synchronous
parallel modes are required to preserve the serial CPSO trajectory, whereas
asynchronous and island modes intentionally change information flow.

## Implemented execution modes

| Mode | Parallel work | Relation to serial reference |
|---|---|---|
| `serial` | None | Reference algorithm |
| `master_slave` | Objective-function evaluations | Exact trajectory equivalence is required and tested |
| `particle_parallel` | Initialization, propagation and objective evaluations | Exact trajectory equivalence is required and tested |
| `asynchronous` | Event-driven particle updates | Different, schedule-dependent update dynamics for multiple workers |
| `islands` | Independent sub-swarms and optional migration | Different population information flow |

Social topologies are global-best, periodic ring and periodic toroidal von
Neumann grid. The internal architecture studies keep the total objective
budget at `N * (K + 1)` for population `N` and `K` update steps.

## External optimization benchmarks

The external-objective interface is intentionally serial: Python/Cython
benchmark callbacks are not assumed to be reentrant or thread-safe. The native
CPSO engine exposes a callback-based serial entry point; the external benchmark
campaigns do not measure or claim parallel speedup.

| Suite | Scope | Confirmatory configuration |
|---|---|---|
| COCO `bbob` | 24 noiseless single-objective functions; dimensions 2, 3, 5, 10, 20, 40; instances 1–15 | `configs/benchmarks/bbob_confirmatory.json` |
| COCO `bbob-largescale` | 24 noiseless single-objective functions; dimensions 20, 40, 80, 160, 320, 640; instances 1–15 | `configs/benchmarks/bbob_largescale_confirmatory.json` |
| CEC 2013 LSGO | 15 large-scale functions; D=1000; 25 seeds; 3,000,000 FE per run | `configs/benchmarks/cec2013_confirmatory.json` |

Both COCO suites are driven through the official `coco-experiment` (`cocoex`)
package; the functions are not reimplemented locally. COCO observer data are
retained for standard `cocopp` post-processing. CEC 2013 LSGO is accessed through
an optional third-party Python wrapper, which is not vendored. Review its
license before installing or redistributing an environment containing it.

The benchmark campaign compares eleven algorithms under an exactly matched
function-evaluation budget:

- `cpso_continuous`: the native continuous-time CPSO implementation;
- `pso_classic_gbest`: classic global-best PSO;
- `pso_inertia_gbest`: inertia-weight global-best PSO;
- `pso_constriction_gbest`: constriction-factor global-best PSO;
- `pso_inertia_lbest_ring`: inertia-weight local-best PSO on a periodic ring;
- `pso_clpso`: comprehensive-learning PSO with per-coordinate pbest exemplars;
- `pso_slpso`: social-learning PSO designed for scalable/large-scale optimization;
- `pso_spso2011`: SPSO-2011 informant topology and hypersphere update;
- `pso_fips`: fully informed PSO on a periodic degree-two ring;
- `pso_cso`: competitive swarm optimizer with pairwise winner/loser learning;
- `pso_dms`: dynamic multi-swarm PSO with periodic random regrouping.

The four classical variants provide transparent reference baselines;
CLPSO, SL-PSO, SPSO-2011, FIPS, CSO and DMS-PSO add literature-derived
learning, neighborhood, competition and multi-swarm mechanisms. This is a
representative comparison set, not a universal or exhaustive current SOTA set.
The SL-PSO implementation uses the project-fixed population to preserve equal
FE accounting, rather than the paper's dimension-dependent population default.
Algorithm-specific and project-adapted boundary rules are recorded in
`docs/stage_c_pso_baselines.md` and `docs/external_benchmarks_and_pso_baselines.md`. Methods share population,
search bounds, deterministic problem/seed key and exact FE budget; algorithmic
random streams and initial coordinates are not forced to match.

## Repository map

- `cpp/`: C++20 numerical core, optimizers, thread-pool execution, social
  topologies and C ABI.
- `python/cpso_serial/`: Python configuration/results, native-library loader,
  custom serial objective API, continuous-space PSO reference baselines and external benchmark
  adapters. The import package retains the historical `cpso_serial` name.
- `experiments/02_master_slave/`: synchronous scaling studies.
- `experiments/03_topologies/`: social-topology comparisons.
- `experiments/04_architectures/`: five-mode internal architecture campaign.
- `experiments/05_external_benchmarks/`: COCO/BBOB, BBOB-Large-Scale and CEC
  2013 LSGO campaign drivers.
- `configs/`: smoke and confirmatory configurations.
- `analysis/`: statistical and plotting utilities.
- `docs/`: mathematical model, implementation decisions, protocols and audit.
- `paper/`: current manuscript draft and bibliography.
- `tests/`: C++ and Python regression and integration-contract tests.

## Build and core test

From the repository root:

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
The distribution metadata is `cpso-parallel`, while the Python import package
and CMake shared-library target retain the historical `cpso_serial` name.

## Run external-benchmark smoke tests

Install the optional dependencies after compiling the native library:

```bash
python -m pip install -r requirements-benchmarks-coco.txt
# Optional post-processing is included above as `cocopp`.

python -m pip install -r requirements-benchmarks-cec2013.txt
# Review the wrapper license before using/redistributing it.
```

First validate the plans without requiring external packages or creating
output directories. The preflight prints the expected number of problems,
method-runs and total FEs, and rejects budgets that are not exact population
multiples:

```bash
PYTHONPATH=python python experiments/05_external_benchmarks/run_bbob.py \
  --config configs/benchmarks/bbob_smoke.json --dry-run
PYTHONPATH=python python experiments/05_external_benchmarks/run_bbob.py \
  --config configs/benchmarks/bbob_largescale_smoke.json --dry-run
PYTHONPATH=python python experiments/05_external_benchmarks/run_cec2013_lsgo.py \
  --config configs/benchmarks/cec2013_smoke.json --dry-run
```

For an intermediate experiment, use the `*_preconfirmatory.json` files. They
cover more functions/instances than smoke tests but deliberately reduce
problem subsets or FE budgets relative to full campaigns.

Then execute each smoke campaign:

```bash
PYTHONPATH=python python experiments/05_external_benchmarks/run_bbob.py \
  --config configs/benchmarks/bbob_smoke.json
PYTHONPATH=python python experiments/05_external_benchmarks/run_bbob.py \
  --config configs/benchmarks/bbob_largescale_smoke.json
PYTHONPATH=python python experiments/05_external_benchmarks/run_cec2013_lsgo.py \
  --config configs/benchmarks/cec2013_smoke.json
```

The smoke configurations are integration checks only; they are not suitable
for ranking algorithms or drawing scientific conclusions. Start with a smaller
pre-confirmatory subset and pin the external package versions before launching
any confirmatory campaign. The BBOB and BBOB-Large-Scale configurations each
contain 23,760 method-problem runs at the selected full suite settings. CEC
2013 LSGO contains 4,125 method-function-seed runs and 12.375 billion total FE
across the eleven methods. The corresponding FE totals are 3.168 billion for
BBOB and 49.896 billion for BBOB-LargeScale. These are substantial compute campaigns: do not launch
them until the smoke tests, package environment, storage capacity and run-time
budget have been checked.

All external results are stored as resumable JSONL records. COCO also writes
per-method observer data under `coco_data/` for `cocopp`. Preserve both the
JSONL sidecars and the raw COCO observer output. Do not edit generated records
by hand.

## CPSO serial-versus-parallel comparison

The focused campaign checks exact trajectory equivalence of `serial`,
`master_slave` and `particle_parallel` on the native C++ objectives, then
measures paired wall time on the explicitly synthetic `sphere_work` cost
instrument. It does not mix execution speed with external benchmark quality.

```bash
PYTHONPATH=python python experiments/04_architectures/run_research_campaign.py \
  configs/campaigns/cpso_serial_parallel_smoke.json --dry-run
PYTHONPATH=python python experiments/04_architectures/run_research_campaign.py \
  configs/campaigns/cpso_serial_parallel_smoke.json \
  --output results/raw/cpso_serial_parallel_smoke_mac --no-plots
```

After the smoke passes, use `configs/campaigns/cpso_serial_parallel_pilot.json`
for the larger pilot. Choose worker counts no larger than the Mac's available
logical CPUs, and keep the same power mode and foreground workload during
timing measurements.

## Internal architecture campaign

Validate and count the predeclared internal campaign without creating output
files or requiring the native library:

```bash
PYTHONPATH=python python experiments/04_architectures/run_research_campaign.py \
  configs/campaigns/definitive_architecture_campaign.json --dry-run
```

Run the small internal integration campaign (not for scientific inference):

```bash
PYTHONPATH=python python experiments/04_architectures/run_research_campaign.py \
  configs/campaigns/campaign_smoke.json \
  --output /tmp/cpso-campaign-smoke --no-plots
```

The internal campaign contains 9,360 optimization runs and 273,888,000
objective evaluations. Its scaling stage uses `sphere_work` exclusively as an
execution-cost instrument; those results must not be mixed with external
optimization-quality tables.

## Scope and scientific limitations

The current native engine exposes six analytic objectives and one separately
marked synthetic `sphere_work` objective. External suite adapters and ten
reference PSO variants (including SPSO-2011, FIPS, CSO and DMS-PSO) are
implemented in a separate serial evaluation layer;
this does not make the native parallel execution backends compatible with
arbitrary Python objectives. COCO/CEC runs compare search quality at matched FE
budgets, not parallel wall-time speedup. No external benchmark results are
included in this development snapshot, and no superiority or state-of-the-art
claim follows from the integration tests.

The exact mathematical fidelity of the serial CPSO implementation to the
complete source article by Orlando and Ricciardello (2021) remains an explicit
audit item in `docs/implementation_decisions.md`. Until an equation-by-equation
comparison with the complete article is completed, describe the native
implementation as a documented reconstruction rather than a certified strict
reproduction.

## Package and ABI versions

Project/package version: `0.7.4`. Native C ABI version: `7`. The ABI 7
addition provides a custom serial objective callback; it is deliberately
restricted to the serial execution architecture. The version numbers are
separate and are checked when the Python wrapper loads the native library.

The current `paper/paper.pdf` is a working draft describing the internal
architecture study and the newly specified external benchmark protocol. It
does not contain definitive external benchmark results or conclusions. The
project license is intentionally unset until the rights holder confirms it;
see `docs/license_status.md`.

## Stage C — extended PSO comparison

The Stage C pilots use all eleven methods (CPSO plus ten PSO-derived
comparators). They are separate from the older smoke/pre-confirmatory configs:

```bash
PYTHONPATH=python python experiments/05_external_benchmarks/run_bbob.py \
  --config configs/benchmarks/bbob_stageC_smoke.json --dry-run
PYTHONPATH=python python experiments/05_external_benchmarks/run_bbob.py \
  --config configs/benchmarks/bbob_largescale_stageC_smoke.json --dry-run
PYTHONPATH=python python experiments/05_external_benchmarks/run_cec2013_lsgo.py \
  --config configs/benchmarks/cec2013_stageC_smoke.json --dry-run
```

After preflight, run the same commands without `--dry-run` and provide an
explicit `--output` directory. Only after all three real smoke campaigns pass,
run `bbob_stageC_pilot.json`, `cec2013_stageC_pilot.json`, and the broader
`bbob_largescale_preconfirmatory.json` campaign. The methods, update laws,
constraints and parameter policies are documented in
`docs/stage_c_pso_baselines.md`.
