# Parallel CPSO architectures (0.5.0)

This document distinguishes execution architectures from changes to the
optimization dynamics. The five modes exposed by the C++ and Python APIs are
not interchangeable; they answer different experimental questions.

## 1. Serial synchronous reference

`minimize_serial` is the baseline. All particles use one frozen social snapshot
per generation; propagation, objective evaluations and state commits are
performed serially. One history value is recorded after initialization and
one after each complete generation. The evaluation count at each history point
is `N, 2N, ..., (K+1)N`, where `N` is population size and `K` is the number of
generations.

## 2. Synchronous master--slave

`minimize_master_slave` retains serial population initialization and serial
particle propagation. Objective evaluations for one generation are dispatched
to a persistent worker pool. The master collects indexed results and commits
them in particle-index order after the batch barrier. For deterministic,
thread-safe objectives, fixed seed, and fixed configuration, it must match the
serial reference exactly.

This is parallel objective evaluation, not an asynchronous optimizer.

## 3. Synchronous particle-parallel

`minimize_particle_parallel` parallelizes particle initialization and the
independent propagation of particles after the generation's social attractors
have been frozen. It then executes a parallel objective batch and commits all
results at the generation barrier. Each particle owns its PRNG state, so task
scheduling does not alter random-number consumption.

For the same objective, topology, seed and parameters, this mode must match the
serial reference exactly. Its performance difference from master--slave
quantifies the additional value or overhead of parallelizing initialization
and particle propagation.

## 4. Event-driven asynchronous CPSO

`minimize_asynchronous` uses a ready-event queue. Initially, each particle has
one event. A worker pops one particle event, snapshots the latest social
information under a state mutex, propagates one continuous segment, evaluates
the candidate and publishes the updated personal best immediately. The same
particle is requeued for its next local update only after its current event has
completed; therefore two workers never update the same particle concurrently.

There is no generation-wide barrier. For global-best, the latest incumbent is
read at event start. For ring/grid, the current personal-best values in the
particle's neighborhood are inspected at event start. Each particle performs
exactly `K` updates, so the total FE budget remains `N(K+1)`.

The history contains the initial incumbent and then one value after every
committed event. Its FE axis starts at `N` and increases by one per event. With
multiple workers, event completion order depends on scheduling and objective
latencies; identical seeds therefore do not imply bitwise-identical histories
across separate multi-worker asynchronous runs. A one-worker run is
reproducible but remains asynchronous in update semantics (particle events are
requeued round-robin rather than synchronized by generations).

## 5. Synchronous island model

`minimize_islands` partitions the fixed total population `N` into `M` equal
subpopulations. Each island performs its own synchronous CPSO generation; the
island tasks run concurrently in a shared worker pool. No worker pool is
created inside an island, avoiding nested parallelism and oversubscription.

After each complete generation, an optional migration barrier is executed:

- `migration_interval = 0`: no migration; islands remain independent.
- `ring`: every destination receives the best personal-best individual from
  its predecessor island in a one-directional ring.
- `fully_connected`: every destination receives the best external island
  incumbent among all other islands.

Migration is best-to-worst and elitist: a destination's worst personal-best
particle is replaced only if the incoming individual is strictly better. The
recipient's velocity is reset to zero; the incoming candidate is already
scored, so migration consumes no additional function evaluation. The reported
`migrations_performed` counts accepted replacements, not attempted transfers.

The island backend currently supports global-best or ring topology inside each
island. Grid topology within islands is deliberately rejected until a precise
mapping from the total population grid to local island geometry is specified.

## 6. Metrics and interpretation

All modes preserve the exact FE budget `N(K+1)`. The `history_evaluations`
array associates every best-so-far history value with its cumulative FE count,
so synchronous generation histories and asynchronous event histories can be
plotted on a common FE axis.

- `elapsed_seconds`: total run wall time, including pool/thread creation and
  shutdown for parallel modes.
- `evaluation_wall_seconds`: wall time across synchronous evaluation batches.
  In asynchronous mode this covers the initial evaluation batch only; update
  evaluations overlap with propagation and are summarized by worker time.
- `propagation_seconds`: wall time across synchronous propagation phases.
  In asynchronous mode per-task propagation time is reported instead.
- `async_update_wall_seconds`: wall time for the full asynchronous event
  scheduler, including propagation, objective calls, queueing and waiting.
- `propagation_worker_seconds` and `evaluation_worker_seconds`: sums of task
  durations. These are aggregate worker-seconds and can exceed wall time.
- `migration_seconds` and `migrations_performed`: island migration overhead
  and the number of accepted replacements.

Only master--slave and particle-parallel are required to preserve the serial
algorithm exactly; their paired wall-time ratios may be called speedup. The
asynchronous and island models alter information availability or population
structure. Their runtimes must not be described as speedup over serial without
an explicitly defined algorithmic work-equivalence argument.

## 7. Deliberately excluded at this stage

Cooperative Co-evolution is not included in these five execution modes. It
changes the decision-space decomposition and the objective context vector; it
is a separate algorithmic family, not simply another thread scheduling policy.
It should be added only as a distinct experimental branch with a specified
variable-grouping method (for example, a Differential Grouping variant) and
separate controls for decomposition cost and subproblem evaluation budget.
