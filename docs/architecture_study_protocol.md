# Unified CPSO architecture study protocol

## Purpose

`experiments/04_architectures/run_architecture_study.py` is the single entry
point that executes all configured CPSO architectures, checkpoints each
completed run, verifies invariants, materializes tables, computes descriptive
paired uncertainty intervals, and generates figures.

## Run matrix

The JSON configuration defines:

- objective functions and dimensions;
- fixed total population size and number of local updates/generations;
- seed list;
- CPSO continuous-model parameters and per-objective bounds;
- a named list of architecture configurations (executor workers, island
  count, migration period/topology, and particle-social topology).

All methods in a case receive the same objective, bounds, total population,
seed, CPSO coefficients, `dt`, and nominal number of updates. The total FE
budget is exactly `N(K+1)` for every method. Island populations must divide
evenly. The driver requires exactly one serial reference for each social
topology represented by a non-serial method when paired analysis is enabled.

## Architectures and what is compared

1. **Serial synchronous**: reference behavior.
2. **Master--slave synchronous**: serial propagation/commits, parallel objective
   batches.
3. **Particle-parallel synchronous**: parallel initialization and propagation
   as well as parallel objective batches; generation social snapshots and
   deterministic commits preserve serial semantics.
4. **Event-driven asynchronous**: one event advances one particle by one
   continuous segment; the updated personal best is published immediately.
   The event queue re-enqueues that particle only after its current event has
   completed.
5. **Parallel islands**: a fixed total population is divided across islands;
   each island advances synchronously and islands execute concurrently.
   Optional migration is synchronized at generation barriers.

## Required equivalence checks

For every benchmark, dimension, seed and particle-social topology, the study
must include exactly one serial reference if master--slave or particle-parallel
is included. The driver verifies exact equality of best value, best position,
complete history, FE axis and FE count for those two synchronous parallel
variants. If this check fails, the study aborts rather than retaining an
unverified performance comparison.

No exact-equivalence requirement applies to asynchronous or island algorithms:
they intentionally alter the update schedule or population information flow.
Their final quality is compared in paired-by-seed summaries against a serial
reference with the same named particle-social topology, but their time ratios
are not called parallel speedup. For island methods, the interpretation must
also acknowledge that a local island has fewer particles than the serial
reference even though the total FE budget is held fixed.

## Checkpointing and provenance

Each completed run is appended to `runs.jsonl` and flushed immediately. This
supports resume after interruption. `run_manifest.json` stores the exact
configuration hash, a SHA-256 fingerprint of the C++/Python source and driver,
native library version/build information, operating system, Python runtime and
logical CPU count. Resuming is refused if either the configuration or source
fingerprint differs from the original run.

## Outputs

At completion the script writes:

- `runs.jsonl`: one complete JSON record per run, used for safe resume;
- `runs.csv`: one row per optimization run;
- `convergence.csv`: best-so-far observations with their actual FE coordinate;
- `summary.csv`: median, quartiles, MAD, success rate, runtime and phase times;
- `paired_comparisons.csv`: run-level comparison with the serial baseline;
- `paired_summary.csv`: paired median log-objective differences and descriptive
  bootstrap intervals, plus speedup only for exact-equivalent synchronous modes;
- `ARCHITECTURE_STUDY.md`: interpretation limits;
- `figures/`: FE-axis convergence, final quality and runtime distributions, and
  paired speedup for synchronous equivalents, in PNG/PDF/SVG.

## Statistical interpretation

The bootstrap intervals use resampling of paired seed-level differences and
are descriptive uncertainty intervals. They are not p-values and do not
replace a pre-specified multiple-comparison procedure for a publication.
For publication-grade reporting, use enough independent seeds (normally at
least 30 as a starting point), prespecify target values, and add an appropriate
non-parametric omnibus/post-hoc analysis with multiplicity correction.

The built-in functions are development benchmarks only; `sphere_work` is a
synthetic execution-cost instrument. This driver does not yet call COCO/BBOB,
BBOB-Large-Scale or CEC 2013 LSGO. Adapters for those suites must be introduced
with their own version/provenance metadata and FE-accounting tests before
large-scale scientific claims are made.

## Timing rules

Total elapsed wall time includes thread-pool or worker-thread lifecycle. Do not
subtract thread creation for one architecture and include it for another.
For asynchronous CPSO, the event-loop wall timer includes propagation,
evaluation, queueing and waiting; objective-call work is also reported as an
aggregate worker-seconds metric. Do not compare `evaluation_wall_seconds`
across sync/async as though it represented the same interval.
