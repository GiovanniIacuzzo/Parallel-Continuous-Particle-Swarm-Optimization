# Controlled experiments for the methodological review

This revision changes code and experimental protocols. `paper/` remains untouched.
The experiments test explanations; they do not select results to make CPSO win.

## What each experiment resolves

| Review issue | New experiment and evidence | Remaining limitation |
|---|---|---|
| Propagator, graph and coefficient scope were confounded | `review_study.py`: analytic/discrete × scalar/coordinate × eight actual native graphs; analytic dt grid | Mechanistic comparison uses common parameters, not separately tuned algorithms |
| R8 may also help discrete PSO | Both native dynamics use identical R8 realizations and initial x/v | Random connected ER graphs have variable degree; R4 is degree-matched **in expectation**, not exactly, to the grid |
| Topology × dt interaction | R8 versus grid relative effects at dt 0.38125, 1 and 3.6, with planned interaction contrasts | This is a fixed grid, not a newly tuned dt selection |
| Missing structural explanation | Actual adjacency hashes, degree distribution endpoints, mean degree, clustering and shortest-path mean | Structural associations do not establish causation |
| Scalar coefficients may confine high-dimensional motion | `review_mechanism.py`: measure initial affine rank and final-best projection residual at D=5,20,1000 | Unconstrained runs are diagnostic; the ordinary clamp can break the invariant subspace |
| Weak BBOB reporting | Full profile: all 24 functions, 15 instances, D=5/20; official Observer files; fixed target ECDF; censored ERT; DE reference | Fixed-budget study, not a claim of completing the entire standard COCO evaluation protocol |
| Missing inferential detail | Function-block Friedman, all primary Wilcoxon–Holm contrasts, planned contrasts, effect sizes and bootstrap intervals; group-wise ranks | Functions are inference units; instances are aggregated within functions. Pilot/smoke inference is exploratory |
| Parallelism is not CPSO-specific | Same synchronous native framework now also runs discrete PSO; equivalence checks cover both dynamics/scopes | Does not make the framework novel by itself |
| Expensive objectives and replica parallelism missing | `review_parallel.py`: measured workloads from microseconds to 10 ms; compare native within-run and across-replica concurrency at equal total FE | Synthetic cost instrument; report its measured cost, not just the requested target |
| Phases, overhead and held-out regret absent | Phase CSVs, speedup against serial, within-backend speedup, both efficiencies, one-worker overhead; nonnegative overhead model trained on sessions 1–2 and tested on session 3 | A temporal session holdout is not an independent machine replication |
| CEC budgets too small | `review_cec_subset.json`: 3,000,000 FE, 5 declared functions, 5 seeds and eight matched variants | Subset, not the complete CEC competition protocol; no competition-winner comparison is implemented |
| dt lacks a dynamical interpretation | `review_stability.py`: frozen spectral radii, envelope contraction, phase and Clerc–Kennedy reference | Frozen-attractor analysis does not prove stochastic/global convergence |

## Install and verify

Use the same native arm64 environment as the existing study. Rebuild after changing the C++ sources.

```bash
python -m pip install -e '.[analysis,test]'
python -m pip install -r requirements-review.txt
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel 4
ctest --test-dir build --output-on-failure
python -m pytest -q
```

Existing C ABI structures and the default CPSO trajectory are retained. The added C entry points expose controlled synchronous experiments and graph/initial-state diagnostics. Existing binaries must be rebuilt to use those symbols.

The matched discrete control is the unit PSO update `v_next=mu*v+a*(pbest-x)+b*(lbest-x)`, `x_next=x+v_next`. Both controls use the existing native clamp with restitution, the same inertia schedule, coefficients 2.05, population 64 and initial velocity fraction 0.1. **Neither uses a velocity limit.** This isolates the propagator from the baseline implementation's velocity clamp. Keep this matched control distinct from the existing literature baselines.

For a given coefficient scope, both dynamics consume identical native random streams. Scalar mode draws one pair per particle/generation; coordinate mode draws one pair per coordinate. Streams after the first generation differ between scopes by design. Graph seeds are independent of swarm seeds in the BBOB and geometric experiments. Initial x/v are identical across all native variants of one problem.

## Launch in order

First verify the complete pipeline:

```bash
python review_study.py ablation --profile smoke
OPENBLAS_NUM_THREADS=1 python review_mechanism.py --profile smoke
python review_parallel.py run --profile smoke --session 1
python review_parallel.py run --profile smoke --session 2
python review_parallel.py run --profile smoke --session 3
python review_stability.py
```

Then obtain the first useful quality results:

```bash
python review_study.py ablation --profile pilot
OPENBLAS_NUM_THREADS=1 python review_mechanism.py --profile pilot
```

BBOB pilot: **1,560 runs**, 4 functions (1,8,15,21), 3 instances, D=5/20, 65 methods, 512D FE. BBOB full: **46,800 runs**, 24 functions, 15 instances, D=5/20, 65 methods, 2048D FE. The full experiment consumes **1,198,080,000 FE**, so measure pilot runtime before launching it on the M2. No full-run checkpoint is presented as an independently scheduled shorter-budget run.

```bash
python review_study.py ablation --profile paper
OPENBLAS_NUM_THREADS=1 python review_mechanism.py --profile paper
```

The instances used here are **not described as a new holdout**: some appeared in previous experiments. dt values and contrasts are fixed before this campaign; this does not replace the existing `study.py dt --profile paper` screening if a new tuned dt is to be selected. That screening and its independent validation must precede any newly tuned comparison.

### Parallel timing on the M2

Run while connected to the intended power source, with a documented Low Power Mode setting and controlled background activity. A MacBook Air's heterogeneous cores and thermal behavior must be discussed when interpreting p=4/8. Native worker threads are not pinned to P/E cores; scheduling is left to the operating system.

Optionally provide a JSON file via `--conditions path.json` containing `power_source`, `low_power_mode`, `thermal_state`, `background_load`, `notes`. Automatic snapshots retain available CPU model, per-level physical-core counts, power-source output, load and sensor readings. Unavailable readings stay unknown; no temperatures are inferred.

```bash
python review_parallel.py run --profile paper --plan
python review_parallel.py run --profile pilot --session 1
python review_parallel.py run --profile pilot --session 2
python review_parallel.py run --profile pilot --session 3
```

Repeat with `--profile paper` after reviewing pilot measurements. Use genuinely separated sessions and document their timing/conditions. Sessions 1–2 fit the model; session 3 is reserved for evaluation. Do not tune the model after inspecting session 3 and still call it held out.

The paper cost targets are 0, 1, 10, 100, 500, 1000, 5000, 10000 µs; calibrated work counts and actual serial evaluation costs are saved. At each cost, the same eight seed-defined runs and FE are completed under every scheduling strategy. Across-replica measurements include Python process spawn/shutdown; within-run batch measurements include native pool construction for each run. Report this startup-inclusive throughput comparison as such. Replica throughput and the latency of one optimization run answer different questions.

Generation count decreases at high evaluation cost to bound campaign expense, identically for all scheduling strategies in that cell. The fitted exposure-adjusted model is `T_batch_native = a*FE_batch + b*T_eval_serial/p + c*p` per dynamics, dimension and backend. Coefficients are nonnegative. Prediction errors and **relative regret** are saved for session 3. The model estimates native within-run batch time; it does not predict across-replica process startup. Regret is conditional on each backend's candidate worker counts, not a claimed global executor oracle.

Synthetic Sphere work retains the existing tiny deterministic correction to keep the arithmetic loop observable. It is a cost instrument, not a standard quality benchmark. Exact trajectory checks compare backends at the **same** work count. Run the same campaign on a separately identified x86 machine to assess portability; the current profiles use up to 8 workers and do not implement 32/64-core weak scaling or an OpenMP reference.

### Optional full-budget CEC subset

This is deliberately not part of the quick sequence. Inspect the plan before committing the substantial runtime. Install the existing optional CEC requirements first.

```bash
python experiments/05_external_benchmarks/run_cec2013_lsgo.py --config configs/benchmarks/review_cec_subset.json --dry-run
python experiments/05_external_benchmarks/run_cec2013_lsgo.py --config configs/benchmarks/review_cec_subset.json --output results/review_v1/cec_subset/raw
python analysis/postprocess_external_benchmarks.py --results-root results/review_v1/cec_subset/raw --output results/review_v1/cec_subset/analysis
```

The declared subset is F1/F4/F8/F12/F15; 200 runs consume **600,000,000 FE**. It is a partial full-budget study, not the official 15-function/25-run competition. Recommended-population and equal-tuning comparisons of the entire literature baseline panel remain a separate task. Existing F13/F14 wrapper checks remain applicable if those functions are added later.

## Read results and regenerate figures

Outputs are under `results/review_v1/{ablation,mechanism,parallel}/{smoke,pilot,paper}`. Each design and source/build fingerprint is frozen. A code, dependency or binary change requires a new output directory; do not merge runs from different fingerprints.

BBOB keeps one JSON per complete run and official Observer logs. Re-running skips complete records. If a run is interrupted after its Observer directory is created but before its JSON is committed, archive the named Observer directory before retrying; partial traces must not be appended as though complete. Analysis rejects incomplete campaigns.

Key analysis outputs:

- `mean_ranks.csv`, `ranks_by_group.csv`: all variants, all eight graphs.
- `all_primary_contrasts.csv`, `planned_contrasts.csv`, `friedman.csv`: complete contrasts, corrections and effect sizes. Holm families span dimensions within each file; bootstrap intervals are descriptive, unadjusted function-resampling intervals.
- `graph_metrics.csv`: native graph realization diagnostics, excluding self from graph degree/clustering/path statistics.
- `ecdf.csv`, `target_ert.csv`: 21 fixed logarithmic targets 10² to 10⁻⁸. Failures stay in the ECDF denominator and contribute full budgets to ERT; no successes produces an undefined ERT.
- `subspace_diagnostics.csv`: initial affine rank, target and final-best residuals, boundary contacts. A final-best residual does not describe every visited candidate.
- `speedups.csv`, `batch_times_and_phases.csv`, `overhead_model.csv`, `heldout_predictions.csv`, `heldout_regret.csv`: timing, phases and held-out model evidence.

BBOB obtains the optimum from COCO's diagnostic `_best_value` property and cross-checks it against the official Observer `Fopt` header. The optimizer never receives that optimum. DE is a quality calibration reference, not a native timing competitor. Recorded external time-to-target includes callback and target-monitor instrumentation; do not equate it with uninstrumented native scaling timings.

```bash
python review_study.py analyze --profile pilot
python review_parallel.py analyze --profile pilot
python plot_review.py ablation --profile pilot
python plot_review.py mechanism --profile pilot
python plot_review.py parallel --profile pilot
```

Figures are English vector PDFs, with no tables rendered as images. `star` labels the existing hub-and-spoke implementation; `wheel` remains its backward-compatible API name. Nothing in `paper/` is rewritten until the completed experiments justify the claims.
