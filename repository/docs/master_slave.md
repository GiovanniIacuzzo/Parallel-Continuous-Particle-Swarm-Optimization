# Synchronous CPSO execution (master--slave introduced in 0.2.0; topologies extended in 0.4.0)

## Scope

The master--slave implementation is a synchronous execution variant of the
serial global-best CPSO reference. It parallelizes **objective evaluations
only**. It does not parallelize particle propagation, random-number generation,
personal-best updates, global-best reduction, or social communication.

The intended use is a deterministic strong-semantics comparison against the
serial reference. The same configuration and seed must produce exactly the
same best-so-far history, final best position, final objective value, and
function-evaluation count for deterministic, reentrant objective functions.

## Generation protocol

For each generation `k`:

1. Freeze the social information used by that generation. Global-best copies
   the incumbent global-best vector once; ring and grid select one local
   personal-best leader per particle from the same pre-generation snapshot.
2. Visit particles in index order and propagate each coordinate using the
   unchanged frozen-oscillator propagator and that particle's RNG stream.
3. Store all candidate positions in a generation buffer.
4. Dispatch one indexed objective task per candidate to a persistent C++
   thread pool. Workers claim candidate indices from an atomic counter.
5. Wait for the complete batch. Results are stored by particle index, not by
   task-completion order.
6. Commit current states and personal bests in particle-index order; update the
   global best using the same strict-improvement rule as the serial version.

Initialization follows the same batch-evaluation structure. Every particle's
RNG is seeded from the run seed and its one-based ID through `std::seed_seq`,
so evaluating the initial population in a different order cannot change its
stream. This derivation was corrected in 0.4.0 because the former
`SplitMix64(seed XOR particle_id)` scheme could reuse streams across different
seed/particle pairs. See `implementation_decisions.md`; 0.4.0 trajectories are
not bitwise comparable with 0.3.0, although the global-best update semantics
are preserved.

## Determinism and objective contract

The algorithmic result is independent of worker scheduling provided that the
objective is deterministic and safe to invoke concurrently. In the generic C++
API, the caller must ensure the `Objective` target is reentrant/thread-safe and
does not mutate shared state without synchronization. Stateful objectives,
random objectives, file writes, shared counters, and external services require a
separate thread-safe adapter or must run serially.

Built-in C API benchmark functions are deterministic and safe for concurrent
evaluation. ABI 2 (release 0.2.0) introduced timing and worker-count fields;
ABI 3 (release 0.3.0) appended the `objective_work_iterations` parameter for
the synthetic `sphere_work` microbenchmark. ABI 4 (release 0.4.0) added the
particle topology identifier and grid shape. ABI 5 (release 0.5.0) adds the
architecture selector and execution configuration for particle-parallel,
event-driven asynchronous, and island modes. The Python wrapper checks the ABI
version before use.

The `worker_count` result field records the requested pool size. A batch with
fewer candidates than workers uses only as many tasks as candidates.

## Timing fields

- `elapsed_seconds`: end-to-end optimizer wall time, including thread-pool
  construction and destruction in master--slave mode.
- `evaluation_wall_seconds`: sum of the wall-clock durations of the evaluation
  batches. It includes dispatch and wait overhead but excludes pool startup and
  teardown. The clock is read once per batch rather than once per function call
  to avoid contaminating measurements of very cheap objective functions.
- `propagation_seconds`: time spent propagating particles and applying the
  boundary map. Propagation remains sequential in this milestone.
- `topology_setup_seconds`: one-time construction time for the immutable
  particle-neighborhood representation. Global-best stores one shared O(N)
  index list; ring/grid store bounded-size neighborhoods per particle.
- `social_selection_seconds`: accumulated time per generation to freeze the
  social attractors. It includes selecting local leaders and copying the
  corresponding attractor vectors for ring/grid, or copying the incumbent
  vector for global-best.

The topology and social-selection timers are diagnostic subcomponents, not
independent wall-clock measurements to be summed blindly with the total time.
They are useful for explaining overhead across population sizes and topology
families; `elapsed_seconds` remains the primary end-to-end runtime.

For primary wall-time speedup use paired runs:

\[
S(p)=\frac{T_{\mathrm{serial}}}{T_{\mathrm{master-slave}}(p)}.
\]

Use identical objective, dimension, population, transitions, bounds, CPSO
parameters, and seed. Verify the final best position/history and function-
evaluation count match before accepting a timing pair. Repeat across seeds and
report dispersion. Include worker-pool startup/shutdown in total time, as the
implementation does. Do not compare master--slave to a variant with a
changed topology or update policy and call the ratio speedup.

## Limitations

- Only built-in objectives are exposed through Python in this release. C++
  callers can provide their own objective, subject to the thread-safety contract.
- Cheap analytic objectives can be dominated by dispatch and synchronization
  overhead; lack of speedup on them is not a thread-pool correctness failure.
- This is shared-memory CPU threading, not MPI, GPU, island parallelism, or
  asynchronous CPSO.
- No performance advantage is claimed until measured on the target hardware
  under a documented protocol.


The executor is topology-agnostic: global-best, ring, and toroidal von-Neumann
grid variants all use the same synchronous batch dispatch. See
[`particle_topologies.md`](particle_topologies.md) for exact neighborhood
definitions and the paired algorithmic comparison protocol.
