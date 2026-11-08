# CPSO — continuous particle-swarm optimization and parallel execution

This repository contains a C++20 implementation of a continuous-time PSO
(CPSO) model, a Python `ctypes` interface, five execution architectures, and a
separate external-benchmark layer with discrete-PSO reference baselines. This
root `README.md` is the authoritative repository and operations guide; the
project intentionally has no separate `docs/` directory. The code distinguishes
algorithmic dynamics from execution strategy: synchronous parallel modes are
required to preserve the serial CPSO trajectory, whereas asynchronous and
island modes intentionally change information flow.

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
retained for standard `cocopp` post-processing. The direct COCO dependencies are
pinned in `requirements-benchmarks-coco.txt` (`coco-experiment==2.8.3`,
`cocopp==2.9.1`) and in the matching optional extras. CEC 2013 LSGO is accessed
through the optional third-party wrapper `cec2013lsgo==2.2`, which is not
vendored and is a legacy integration. Review its GPL-3.0 terms and test its
build separately before installing it on Apple Silicon or redistributing an
environment containing it.

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
The implemented update laws, fixed parameters, initialization choices and
boundary-handling policies are defined in `python/cpso_serial/pso_baselines.py`
and the external-suite adapters. Methods share population, search bounds,
problem identity and exact FE budget; algorithmic random streams and initial
coordinates are not forced to match. These are documented reconstructions of
the cited methods with explicit project adaptations, not claims of bitwise
reproduction of every original implementation.

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

## MacBook Air M2 — measured exploratory profiles

The repository includes dedicated Apple-Silicon profiles. They preserve the
full per-run FE budgets of the broader internal pilot studies and reduce the
number of paired seeds, rather than shortening every optimization trajectory.
This makes the measured quality curves useful for checking the analysis and
figure pipeline and for inspecting whether qualitative trends are plausible.
The smaller seed count remains exploratory and is not a substitute for the
confirmatory HPC design.

Build natively on Apple Silicon (use an ARM64 terminal, not Rosetta):

```bash
uname -m  # must print arm64
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e ".[analysis,test]"

cmake -S . -B build-m2 -DCMAKE_BUILD_TYPE=Release -DCPSO_BUILD_TESTS=ON
cmake --build build-m2 --parallel 4
ctest --test-dir build-m2 --output-on-failure

export CPSO_SERIAL_LIBRARY="$PWD/build-m2/lib/libcpso_serial.dylib"
export PYTHONPATH="$PWD/python"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
python -m pytest -q
python scripts/run_smoke.py
```

Execute the three internal profiles sequentially. Start with `--dry-run` for
each; do not launch them concurrently on a fanless Air:

```bash
python experiments/04_architectures/run_research_campaign.py \
  configs/campaigns/cpso_serial_parallel_m2_pilot.json \
  --output results/raw/cpso_serial_parallel_m2_pilot --dry-run
python experiments/04_architectures/run_research_campaign.py \
  configs/campaigns/cpso_serial_parallel_m2_pilot.json \
  --output results/raw/cpso_serial_parallel_m2_pilot --no-plots

python experiments/04_architectures/run_research_campaign.py \
  configs/campaigns/cpso_dynamics_m2_pilot.json \
  --output results/raw/cpso_dynamics_m2_pilot --dry-run
python experiments/04_architectures/run_research_campaign.py \
  configs/campaigns/cpso_dynamics_m2_pilot.json \
  --output results/raw/cpso_dynamics_m2_pilot --no-plots

python experiments/04_architectures/run_research_campaign.py \
  configs/campaigns/cpso_topology_expansion_m2_pilot.json \
  --output results/raw/cpso_topology_expansion_m2_pilot --dry-run
python experiments/04_architectures/run_research_campaign.py \
  configs/campaigns/cpso_topology_expansion_m2_pilot.json \
  --output results/raw/cpso_topology_expansion_m2_pilot --no-plots
```

The quality stages retain six analytic functions, D=30 and D=100, N=64,
200 updates and five paired seeds (12,864 FE per run). The synthetic timing
stages are isolated from optimization quality and use `sphere_work` only as a
controlled execution-cost instrument. The three internal profiles contain
560, 720 and 525 runs, respectively; the separate `dt` screening contains 900
runs at 32,064 FE each. Execute them one at a time. The architecture runner
checkpoints individual runs: after an interruption, rerun the identical
command with the same configuration and output directory. Do not edit a
configuration while resuming it.

The M2 quality pilots preserve the per-run objective budget and the first five
predeclared seed values from the broader pilot designs. Their final objective
values and FE-indexed convergence curves can therefore be compared with HPC
records at matching benchmark, dimension, method, parameters, seed and FE
budget. Small cross-platform floating-point differences may still alter
near-tie decisions, so exact cross-machine trajectory identity must be measured,
not assumed. Five seeds remain exploratory and do not provide the confirmatory
power of the full HPC design. M2 wall times describe the M2 and must not be
compared directly to HPC wall times. Only synchronous runs whose exact
trajectory check passes are eligible for speedup interpretation.

Generate the internal pilot figures from the three explicitly selected
campaign outputs. The script writes a `FIGURE_CATALOGUE.md` with every
figure generated or omitted; it returns a non-zero exit code if any expected
figure is unavailable, even when the other figures were created:

```bash
python analysis/postprocess_publication_figures.py \
  --profile exploratory \
  --synchronous-campaign results/raw/cpso_serial_parallel_m2_pilot \
  --dynamics-campaign results/raw/cpso_dynamics_m2_pilot \
  --topology-campaign results/raw/cpso_topology_expansion_m2_pilot \
  --output paper/figures/internal_pilots
```

The paired `dt` screening profile retains the full 15-point grid and 500-update
run budget, but uses five paired seeds (900 runs, 28,857,600 FE). It is
resumable after interruption:

```bash
python experiments/06_dt_sensitivity/run_dt_sensitivity.py \
  --config configs/campaigns/cpso_dt_sensitivity_m2_screening.json \
  --output results/raw/cpso_dt_sensitivity_m2_screening --dry-run
python experiments/06_dt_sensitivity/run_dt_sensitivity.py \
  --config configs/campaigns/cpso_dt_sensitivity_m2_screening.json \
  --output results/raw/cpso_dt_sensitivity_m2_screening
# If interrupted, repeat the last command with --resume.
python analysis/analyze_dt_sensitivity.py \
  --input results/raw/cpso_dt_sensitivity_m2_screening \
  --output-dir results/analysis/cpso_dt_sensitivity_m2_screening \
  --figures-dir paper/figures/dt_m2_screening
```

A result is comparable to an HPC result in algorithmic quality only when the
benchmark, implementation, parameter values, seed, initialization and FE
budget are matched. Cross-machine timing comparisons require a separate
normalization/benchmarking protocol and are not inferred from these pilots.

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

The native CPSO implementation is a documented numerical reconstruction of
the continuous-time model attributed to Orlando and Ricciardello (2021). The
current repository regression tests establish software properties and
executor contracts; they do not certify mathematical fidelity to the complete
source article. A traceable equation-by-equation comparison with the primary
source, including the stochastic coefficients, frozen-coefficient
assumption, analytic propagator, initialization and boundary conventions,
remains required before describing the code as a strict reproduction.

## Audit status and remaining release decisions

The repository audit performed on 2026-11-07 corrected the manuscript/figure
filename contract, separated confirmatory and exploratory post-processing,
added a generated figure catalogue and non-zero exit on incomplete figure
sets, fixed the campaign equivalence-audit CSV writer, and made the `dt`
screening resumable with configuration-hash and record validation. The M2
profiles are explicitly `exploratory`; they preserve full per-run FE budgets
and reduce paired seeds. No generated pilot output is shipped in this source
archive.

The root README is intentionally the single operational guide; `docs/` is not
required and has not been recreated. Optional COCO/Cocopp/CEC entry-point
versions are pinned, but the project does not yet ship a complete
platform-specific transitive lockfile. The analysis/test dependencies in
`pyproject.toml` and `requirements.txt` retain compatible lower bounds. For a
fully frozen production environment, generate and review a lock on the target
OS/architecture after the native build and external adapters have been tested.

Two decisions cannot safely be guessed by a code change: (i) the rights holder
must choose the project's redistribution license and add a matching `LICENSE`
file; (ii) strict mathematical reproduction of the primary CPSO source still
requires the equation-by-equation audit described above. Until these are
resolved, do not claim an assigned project license or strict source-paper
reproduction.

## Package and ABI versions

Project/package version: `0.7.4`. Native C ABI version: `7`. The ABI 7
addition provides a custom serial objective callback; it is deliberately
restricted to the serial execution architecture. The version numbers are
separate and are checked when the Python wrapper loads the native library.

The current `paper/paper.pdf` is a working draft describing the internal
architecture study and the newly specified external benchmark protocol. It
does not contain definitive external benchmark results or conclusions. The
No `LICENSE` file or package license metadata is supplied yet. Do not assume
that the repository is licensed for redistribution or reuse until the rights
holder selects a license and adds the corresponding file.

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
constraints and parameter policies are implemented and described in
`python/cpso_serial/pso_baselines.py` and `python/cpso_serial/benchmarks/`;
project-specific adaptations are recorded in their module docstrings and
configuration files.
