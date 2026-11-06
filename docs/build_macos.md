# Build on macOS

The project uses standard C++20 and the standard-library `std::thread` runtime
for all five execution modes: serial, master--slave, particle-parallel,
event-driven asynchronous, and islands. The current backend is shared-memory;
there are no OpenMP, MPI, or CUDA execution backends.

On an Apple Silicon Mac, open a terminal in the repository root and run:

```bash
xcode-select --install   # only if command-line developer tools are absent
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCPSO_BUILD_TESTS=ON
cmake --build build --parallel
ctest --test-dir build --output-on-failure

python3 -m pip install -e ".[analysis,test]" --no-build-isolation
PYTHONPATH=python python3 -m pytest -q
PYTHONPATH=python python3 scripts/run_smoke.py
PYTHONPATH=python python3 experiments/02_master_slave/benchmark_scaling.py --workers 2,4
PYTHONPATH=python python3 experiments/03_topologies/run_topology_comparison.py --dimensions 10,30 --population 40 --seeds 20
```

A recent Apple Clang with C++20 support and CMake 3.20 or newer are required.
CMake normally selects the native architecture of the running toolchain; do not
force `arm64` or `x86_64` unless building intentionally for a different target.

The Python wrapper searches `build/lib` for the platform shared library. If the
library is placed elsewhere, set `CPSO_SERIAL_LIBRARY` to its absolute path.

The macOS commands above are documentation only; the repository audit build
was performed on Linux/GCC. Run the native and Python test suites on the target
Mac before using its timing measurements or treating macOS support as verified.

## First CPSO architecture and PSO algorithm comparisons

Run these commands from the repository root after the build and core tests above.
Use a virtual environment and keep raw results outside the source tree if possible.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e ".[analysis,test]" --no-build-isolation
export PYTHONPATH="$PWD/python"

# Verify the exact-trajectory CPSO serial/synchronous-parallel plan.
python experiments/04_architectures/run_research_campaign.py \
  configs/campaigns/cpso_serial_parallel_smoke.json --dry-run
python experiments/04_architectures/run_research_campaign.py \
  configs/campaigns/cpso_serial_parallel_smoke.json \
  --output results/raw/cpso_serial_parallel_smoke_mac --no-plots

# After reviewing the smoke outputs, run the larger pilot (1,400 runs).
python experiments/04_architectures/run_research_campaign.py \
  configs/campaigns/cpso_serial_parallel_pilot.json --dry-run
python experiments/04_architectures/run_research_campaign.py \
  configs/campaigns/cpso_serial_parallel_pilot.json \
  --output results/raw/cpso_serial_parallel_pilot_mac --no-plots

# Install the official COCO Python package and check external algorithm runs.
python -m pip install -r requirements-benchmarks-coco.txt
python experiments/05_external_benchmarks/run_bbob.py \
  --config configs/benchmarks/bbob_smoke.json --dry-run
python experiments/05_external_benchmarks/run_bbob.py \
  --config configs/benchmarks/bbob_smoke.json
python experiments/05_external_benchmarks/run_bbob.py \
  --config configs/benchmarks/bbob_largescale_smoke.json --dry-run
python experiments/05_external_benchmarks/run_bbob.py \
  --config configs/benchmarks/bbob_largescale_smoke.json
```

The BBOB smoke commands compare native serial CPSO with the ten discrete-time, continuous-space PSO
reference methods at matched FE budgets. They do not compare parallel CPSO
architectures; the external Python/Cython objective callback is serial-only.
The internal architecture campaign above is the source for parallel scaling.
For CEC 2013 LSGO, inspect the license of the optional GPL-3.0 wrapper before
installing `requirements-benchmarks-cec2013.txt`; then use
`experiments/05_external_benchmarks/run_cec2013_lsgo.py` with
`configs/benchmarks/cec2013_smoke.json` first.

Do not execute pre-confirmatory or confirmatory campaigns until the real COCO/CEC
smoke runs complete successfully and their output metadata and FE counters have
been inspected. Never use the smoke results for algorithm rankings.

### Asynchronous and island dynamics

The synchronous CPSO smoke above tests exact trajectory equivalence. Asynchronous
updates and island migration intentionally change the algorithmic dynamics, so
they are evaluated separately against serial references at equal FE budgets.
First run the existing 27-run integration smoke, then inspect the 720-run pilot:

```bash
python experiments/04_architectures/run_research_campaign.py \
  configs/campaigns/campaign_smoke.json \
  --output results/raw/cpso_all_architectures_smoke_mac --no-plots
python experiments/04_architectures/run_research_campaign.py \
  configs/campaigns/cpso_dynamics_pilot.json --dry-run
python experiments/04_architectures/run_research_campaign.py \
  configs/campaigns/cpso_dynamics_pilot.json \
  --output results/raw/cpso_dynamics_pilot_mac --no-plots
```

The dynamics pilot has 360 asynchronous and 360 island runs. It uses six
built-in objectives, dimensions 30/100, five seeds, population 64 and 200
iterations. It is a pilot for quality effects, not a substitute for the
30-seed confirmatory architecture campaign.
