# Master--slave scaling protocol

## Purpose

Version 0.3.0 adds a controlled experiment driver for the synchronous master--slave implementation. The purpose is to locate the workload range in which parallel objective evaluation amortizes task dispatch, synchronization, and worker-pool lifecycle costs.

This is an implementation/performance characterization stage. It is not yet a comparison against COCO/CEC benchmark suites and it does not establish general scalability.

## Experimental factors

The sweep driver varies:

- decision-space dimension `D`;
- population size `N`;
- recurrence work per objective evaluation `W`;
- worker count `p`;
- paired random seeds.

The default objective is `sphere_work`. Its value is

\[
f_W(x)=\sum_{j=1}^{D}x_j^2+10^{-30}|a_W(x)|,
\]

where `a_W` is a deterministic floating-point recurrence executed `W` times. The second term is deliberately negligible at ordinary optimization scales; its purpose is to keep the recurrence observable to optimizing compilers. This objective is a controlled microbenchmark, not a recognized optimization test function. At `W=0`, it returns the ordinary Sphere objective exactly.

## Comparability constraints

For every `(D, N, W, seed)` tuple:

1. One serial reference run is executed.
2. Master--slave runs are performed for every requested worker count.
3. Execution order is deterministically shuffled to reduce systematic order effects.
4. Warm-up runs are performed but excluded from the raw measurement table.
5. A parallel run is accepted only if best value, best position, complete best-so-far history, completed transitions, and function-evaluation count match the serial reference exactly.

The comparison therefore preserves the optimizer's algorithmic trajectory for the tested deterministic objective. It measures execution backend overhead and speedup, not a change in optimization quality.

## Timings and definitions

The raw CSV records:

- `elapsed_seconds`: end-to-end native optimizer wall time; in master--slave mode it includes creation and shutdown of the worker pool;
- `evaluation_batch_wall_seconds`: sum of wall-clock durations of complete objective batches, including dispatch and waiting;
- `propagation_seconds`: measured sequential particle propagation time;
- `residual_seconds`: non-negative residual after subtracting evaluation-batch and propagation time from total elapsed time. This residual includes state updates and other orchestration; it is not a direct measurement of scheduler overhead;
- `evaluations_per_second`: completed objective evaluations divided by total elapsed time.

For each seed, total-time speedup is

\[
S_p=\frac{T_{\mathrm{serial}}}{T_{p,\mathrm{master-slave}}},
\]

with the same seed, problem dimension, population, transition budget and objective workload. Parallel efficiency is `S_p/p`.

The reported bootstrap interval is a percentile interval for the median paired speedup. It is exploratory for small seed counts; it must not replace larger repeated experiments or formal statistical analysis.

## Recommended local workflow

Build a clean Release library first:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCPSO_BUILD_TESTS=ON
cmake --build build --parallel
ctest --test-dir build --output-on-failure
PYTHONPATH=python python -m pytest -q
```

Run a short pilot:

```bash
PYTHONPATH=python python experiments/02_master_slave/run_scaling_sweep.py \
  --objective sphere_work \
  --dimensions 32 \
  --populations 32 \
  --iterations 30 \
  --seeds 5 \
  --workers 1,2,4,8 \
  --work-iterations 0,1000,10000,50000
```

Analyze the run (replace the path with the generated run directory):

```bash
python experiments/02_master_slave/analyze_scaling_sweep.py \
  results/raw/master_slave_sweep/<run_directory>
```

For an initial Apple Silicon characterization, use a moderate population and transition budget. Increase `W` only after checking that the wall times are long enough to be measured reliably. The sweep is a full Cartesian product, so the number of measured runs is

\[
N_{\mathrm{runs}}=N_D N_N N_W N_{\mathrm{seeds}}(1+N_p).
\]

Warm-ups are additional and are not included in that count.

## Machine-control requirements

For a meaningful local run:

- close CPU-intensive applications and avoid running multiple sweeps simultaneously;
- record the exact Mac model/chip, macOS version, compiler version, build type and logical CPU count from the metadata, and add the performance/efficiency-core layout if known;
- do not change power mode or thermal conditions between paired configurations;
- perform at least one preliminary run to verify runtime and memory needs;
- repeat the selected sweep on separate occasions if thermal drift or background activity is suspected;
- do not interpret logical CPU count as a count of homogeneous cores.

The local Linux pilot included with the development release is only a test of the experimental pipeline. It is not evidence about Apple Silicon performance.

## From controlled workload to real objectives

After finding the synthetic break-even region, repeat selected worker/population/dimension combinations with the built-in analytic functions and later with COCO/CEC adapters. The synthetic objective answers: "How much objective-side arithmetic is needed for this executor to amortize its overhead?" It does not answer: "Does master--slave improve performance on an arbitrary scientific objective?"


## Interaction with particle-social topology

Version 0.4.0 adds global-best, ring, and grid social-neighborhood algorithms.
Do not mix topology changes into a pure master--slave speedup pair. For a
strong-scaling comparison, keep one topology fixed between serial and
master--slave runs. For a topology-quality comparison, use identical seeds and
function-evaluation budgets and run the variants through the same executor;
the dedicated protocol is documented in `particle_topologies.md`.
