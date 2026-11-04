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
| Random generator | Per-particle `std::mt19937_64` streams seeded from SplitMix64 | Deterministic stream ownership; not a statistical-independence proof |
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
