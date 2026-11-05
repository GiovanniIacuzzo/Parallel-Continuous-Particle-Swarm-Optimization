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
