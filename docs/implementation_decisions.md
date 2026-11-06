# Implementation decisions and source-fidelity register

This note separates the mathematical model from choices required to turn it
into an executable bounded optimizer.

| Item | Choice in 0.2.0 / 0.1.0 baseline | Status |
|---|---|---|
| Local motion | Second-order damped oscillator with frozen coefficients and closed-form flow | Core model represented in the cited article; equation-by-equation audit against full text remains open |
| Random coefficients | One scalar `r1,r2` pair per particle and interval, shared across coordinates | Explicit implementation convention; verify against complete paper |
| Inertia schedule | Linear from `mu_max` on the first transition to `mu_min` on the last | Explicit endpoint-preserving convention |
| Observation time | Objective evaluated once at each interval endpoint | Explicit algorithmic observation policy |
| Social information | Global-best snapshot frozen for the entire synchronous batch | Serial reference semantics |
| Initial positions | Independent uniform sample in the box | Implementation convention |
| Initial velocities | Coordinate-wise uniform sample scaled by each box width | Implementation convention |
| Bounds | Endpoint clamp with outward-velocity restitution | Added bounded-domain policy; not part of the unconstrained ODE |
| Personal best | Strict objective improvement | Standard elitist memory update |
| Early stopping | Disabled | Keeps requested transition and FE budgets exact |
| Random generator | Per-particle `std::mt19937_64` streams seeded through `std::seed_seq` from separate run-seed and particle-ID words | Deterministic stream ownership; tested for duplicate fingerprints on the release seed/ID regression set; not a formal statistical-independence proof |
| Objective values | Six standard analytic functions | Sanity/smoke test objectives, not a replacement for COCO or CEC suites |

## Source-audit item before publication

The accessible publisher landing page for Orlando and Ricciardello (2021) states
that the article proposes a continuous PSO formulation and its analytic
solution, but the public preview does not expose the full mathematical
specification. The previous code archive contained an implementation with
explicit comments referring to equation numbers in that article. The present
clean implementation has been reconstructed from that mathematical mapping
and makes its operational conventions visible. Before describing the code as
a strict reproduction, compare the complete paper equation-by-equation and
resolve at least:

- whether `r1,r2` are scalar per particle or coordinate-wise random vectors;
- the exact inertia schedule and its time indexing;
- the initial position/velocity distributions;
- how the original work treats domain constraints and out-of-domain states;
- whether the paper's algorithm evaluates the objective at every frozen
  interval endpoint exactly as implemented here.

If the complete paper specifies a different convention, the implementation
and its tests must be updated before the serial baseline is considered frozen.


## Master--slave implementation decisions (0.2.0)

| Item | Choice | Status |
|---|---|---|
| Parallelized work | Objective evaluations only | Preserves propagation and update semantics |
| Propagation | Sequential, particle-index order | Same code path as serial reference |
| Batch scheduling | Persistent `std::thread` pool; atomic dynamic index assignment | Implemented and tested |
| Result collection | Store by particle index; commit in particle-index order | Stable tie-breaking and exact deterministic outputs for pure objectives |
| Initial batch | Generate every particle from its own RNG stream, then evaluate the batch | Same initial random states as serial reference |
| Objective contract | Deterministic and reentrant/thread-safe for concurrent use | Required for generic C++ callers |
| Worker count | Explicit user setting, 1 to 4096 | Requested count recorded in results |
| Timing | End-to-end wall, evaluation-batch wall, propagation wall | Timed at batch level to avoid per-evaluation instrumentation overhead; definitions in `docs/master_slave.md` |


## Controlled workload addition (0.3.0)

The C ABI was raised to version 3 to add `objective_work_iterations`. This field is accepted only with the explicitly synthetic `sphere_work` objective and is bounded to 100,000,000 recurrence iterations per evaluation. With zero work iterations, `sphere_work` returns the Sphere function exactly. For a positive workload, it adds a negligible deterministic positive term to keep the arithmetic recurrence observable under compiler optimization. This objective is intended solely to characterize executor overhead and break-even behavior; it is not part of the standard benchmark suite and must not be reported as an ordinary Sphere run.


## Particle-social topologies (0.4.0)

The default global-best behavior is preserved, including incumbent retention
on equal objective values. The ring is periodic with radius one. The grid is a
periodic row-major von-Neumann graph with self plus four orthogonal neighbors;
indices are deduplicated for narrow grids. For ring/grid, equal local
personal-best objective values are resolved by the smallest particle index.
All social leaders are selected from a frozen pre-propagation generation
snapshot. No intra-generation update is visible to another particle.

The topology is independent of the serial/master--slave executor. The new
Python/C ABI configuration fields select topology and grid shape; ABI 4 is
intentionally incompatible with older native libraries, and the wrapper fails
fast on version mismatch.


## Random-stream derivation correction (0.4.0)

The 0.3.0 expression `SplitMix64(run_seed XOR particle_id)` can reuse the same
pre-hash value for different `(run_seed, particle_id)` pairs. This creates
identical pre-hash values for some different run-seed/particle-ID pairs and can reduce the
independence intended by the multi-seed experiment design. In 0.4.0 the two
64-bit identifiers are split into separate 32-bit words and passed, together
with fixed project tags, to `std::seed_seq` to initialize each
`std::mt19937_64` engine. The mapping is deterministic and tested for repeated
stream fingerprints over the regression set. It is not a proof of statistical
independence. Consequently, 0.4.0 preserves the global-best update/social
semantics but does not preserve bitwise trajectories from 0.3.0; all new
multi-seed campaigns must use the 0.4.0 RNG derivation consistently.

## Parallel execution architectures (0.5.0)

| Mode | Operational semantics | Equivalence status |
|---|---|---|
| Serial synchronous | Frozen social snapshot per generation; sequential particle propagation, objective calls and commits | Reference |
| Master--slave synchronous | Serial propagation and commits; objective batches evaluated by a persistent worker pool | Exact serial equivalence required |
| Particle-parallel synchronous | Particle initialization, propagation after the frozen social snapshot, and objective batches are parallel; commits remain deterministic and sequential | Exact serial equivalence required |
| Event-driven asynchronous | One queued event advances one particle by one segment; each completed personal-best update is published immediately and the particle is requeued | Different algorithmic schedule; completion order is schedule-dependent for multiple workers |
| Synchronous islands | Fixed total population split evenly among independent islands; island tasks advance concurrently with generation barriers | Different population information flow; not serial-equivalent |

For the island model, migration sources are copied into a compact immutable
snapshot before any destination is changed. This gives simultaneous migration
semantics and avoids order-dependent cascading transfers. Ring migration sends
each island's incumbent to the next island; fully-connected migration sends
each destination the best incumbent among all other islands. A destination's
worst personal-best particle is replaced only by a strictly better immigrant;
its velocity is reset to zero. The incumbent is already evaluated, so migration
adds no function evaluations. A migration event after the final generation is
skipped because no subsequent update can use the immigrant.

The total evaluation budget is fixed across modes at `N * (K + 1)` for total
population `N` and `K` particle updates. Synchronous histories are recorded at
generation boundaries; asynchronous histories are recorded after each
committed particle event and carry an explicit FE axis. Parallel callbacks must
be reentrant/thread-safe. Worker-time sums can exceed wall time and are kept
separate from elapsed wall-time measurements.

The C ABI version was raised to 5 in release 0.5.0. Project/package version
0.6.0 retains ABI 5 because its version-alignment and tooling changes do not
alter the public C structs or calling convention. The unified architecture-study driver
stores both a study-configuration SHA-256 and a source-code SHA-256 in its run
manifest; it refuses to resume an output directory if either differs. Its
bootstrap intervals are descriptive only. The current development study uses
built-in objectives; COCO/BBOB and CEC adapters remain outside this release.

## Development remediation — project 0.6.0 follow-up

The following changes are part of the current development line and do not
change the C ABI or the mathematical propagator:

- The JSONL checkpoint reader accepts a valid final JSON record without a
  newline and can repair only an incomplete, unterminated final record during
  resume. Any malformed newline-terminated record is treated as interior
  corruption and causes a hard error. Analysis is read-only and refuses an
  incomplete tail until the study is resumed.
- Completed-study validation checks exact coverage of every configured
  `(function, dimension, seed, method)` tuple, rejects duplicate/unexpected or
  missing runs, checks the exact FE budget, verifies aligned finite histories,
  requires a non-increasing best-so-far series, and checks that its final value
  matches the reported incumbent.
- Confirmatory quality inference treats the six objective functions as the
  top-level clusters. Dimension is nested within function and seed within the
  function-dimension cell. The twelve function-dimension blocks remain
  descriptive; paired Wilcoxon and Friedman summaries use six family-level
  units. The nested bootstrap resamples families, dimensions, and paired seeds.
  With only six objective families, statistical power and external
  generalizability remain limited.
- Run manifests now include non-identifying CPU/platform, Python/package,
  thread-environment and native build metadata. CMake cache/build flags and
  physical-core details must still be archived explicitly for each definitive
  timing campaign where the environment helper cannot identify them.
- The manuscript has been narrowed to the implemented five shared-memory
  execution modes, three social topologies, six built-in analytic functions,
  and the synthetic timing objective. External benchmarks and discrete-PSO
  baselines are not represented as implemented results.

The full source-equation audit against Orlando--Ricciardello (2021) remains
blocked on obtaining the complete article text. The local analytic propagator
is tested against an independent RK4 implementation and its semigroup
property, but those tests cannot establish that every modelling convention
matches the published source.


## External benchmark and baseline layer — project 0.7.0 (development)

The project version is advanced to 0.7.0 and the native C ABI to 6. ABI 6
adds a custom objective callback entry point for serial CPSO runs. It rejects
non-serial execution architectures; the callback is synchronous and no
thread-safety or reentrancy guarantee is implied. The Python wrapper preserves
Python exceptions and reports them after control returns from the native call.

The external quality layer now provides adapters and drivers for COCO `bbob`,
COCO `bbob-largescale`, and CEC 2013 LSGO, plus four independent NumPy PSO
reference variants (classic gbest, inertia gbest, constriction gbest, and
inertia lbest ring). All methods receive equal population and exact FE budget.
The C++ CPSO uses its own per-particle random streams; the NumPy baselines use
NumPy's generator. Matching problem/seed labels therefore create paired
replicates but do not guarantee identical initial coordinates or common random
numbers.

The external objective layer is quality-only. It must not be used for
master--slave/particle-parallel scaling claims because the Python callback is
serial-only. COCO writes native observer records for post-processing; CEC writes
FE checkpoints in JSONL. Both experiment drivers are resumable by a stable run
key and require exact FE divisibility by population size.

The COCO confirmatory plans select 24 functions, six dimensions and 15
instances for each of BBOB and BBOB-LargeScale, with project-selected budget
`10,000 * D` per problem/method. The BBOB-LargeScale dimensions are
20, 40, 80, 160, 320 and 640. The CEC 2013 LSGO plan follows 15 functions,
D=1000, 25 seeds, 3,000,000 FE and checkpoints at 120,000, 600,000 and
3,000,000 FE. These configurations are predeclared but are not results.

External package integration was tested locally through controlled fake COCO
and CEC objects, including end-to-end execution, FE parity, JSONL output and
resume behavior. The official `coco-experiment` and CEC wrapper packages were
not installed in the offline environment, so real-suite execution remains a
release gate. The commonly used CEC wrapper is GPL-3.0; its code is not vendored.

The six discrete-time, continuous-space PSO implementations are explicit reference algorithms, not
full standardized SPSO-2007/2011 implementations. Their randomization,
initialization, velocity caps, constriction parameters, local-best neighborhood
and boundary map are recorded in
`docs/external_benchmarks_and_pso_baselines.md`.


## External algorithm-comparison extension — project 0.7.1

Project/package version 0.7.1 adds a literature-derived CLPSO baseline alongside
the four previously implemented classical PSO baselines. The implementation
uses per-coordinate personal-best exemplars, the particle-dependent learning
probability, two-way tournament selection, refresh gap 7, acceleration 1.49445,
and a linearly decreasing inertia weight from 0.9 to 0.4, following Liang et
al. (2006). The project retains its common bounded-domain clamp/restitution
policy and documents this as a protocol-level adaptation. CLPSO runs a
sequential particle sweep; it must not be described as a synchronous
generation-snapshot method. All external algorithms retain exactly matched FE
budgets.

Two campaign configurations were added for internal execution studies:
`cpso_serial_parallel_smoke.json` checks exact CPSO trajectory equivalence and
smoke-level timing collection; `cpso_serial_parallel_pilot.json` expands the
quality-equivalence and synthetic strong-scaling studies. These use the
existing research campaign runner and do not duplicate its implementation.


## PSO baselines and CPSO architecture pilot — project 0.7.1

The external algorithm comparison now contains the four explicit classical
PSO baselines, CLPSO (Liang et al., 2006) and SL-PSO (Cheng and Jin, 2015).
CLPSO selects personal-best exemplars per coordinate using tournaments and
updates from a frozen generation snapshot. SL-PSO orders the current swarm
worst-to-best and learns from better-ranked demonstrators, with the
dimension-dependent probability and social-influence formulas from the cited
paper. For equal FE comparisons, the project fixes the population across
methods instead of using SL-PSO's recommended dimension-dependent population
size; this deviation is stored in run metadata and described in the protocol.
Both learning variants use the common project boundary handling, also
disclosed as a project convention.

The architecture smoke/pilot configurations compare native serial CPSO with
synchronous master-slave and particle-parallel execution. Exact quality
equivalence is checked on built-in analytic objectives. `sphere_work` is used
only as a synthetic execution-cost instrument. External COCO/CEC adapters
remain serial objective-callback comparisons and are not used to infer native
parallel speedup.
