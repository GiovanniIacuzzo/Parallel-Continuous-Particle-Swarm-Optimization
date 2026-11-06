# CPSO: mathematical specification and implemented execution semantics

**Implementation reviewed:** project 0.7.0, native C ABI 6.  
**Source-fidelity status:** the local ODE and its analytic flow are verified; a strict equation-by-equation reproduction of Orlando--Ricciardello (2021) is not yet certified because the complete source text was not present in the repository or accessible in the public preview. The open conventions are listed in `implementation_decisions.md`.

## 1. Optimization problem and particle state

The implementation minimizes a bounded scalar objective

\[
  \min_{x\in\Omega} J(x), \qquad
  \Omega=\prod_{j=1}^{D}[\ell_j,u_j].
\]

Particle \(i\) stores its position \(x_i\in\Omega\), velocity \(v_i\in\mathbb R^D\),
personal-best position \(p_i\), current objective value, personal-best value,
and a particle-owned pseudorandom-number-generator state. The social attractor
is selected from a frozen snapshot according to the configured topology:
global-best, radius-one periodic ring, or toroidal von-Neumann grid.

## 2. Frozen-coefficient continuous dynamics

For particle \(i\), update event/generation \(k\), and a scalar random pair
\(r_{1,i}^k,r_{2,i}^k\in[0,1)\), the implementation defines

\[
  a_{i,k}=c_1r_{1,i}^k,\qquad b_{i,k}=c_2r_{2,i}^k,
  \qquad \omega_{i,k}^{2}=a_{i,k}+b_{i,k},
\]

and, for a selected social attractor \(s_i^k\),

\[
  F_{i,k}=a_{i,k}p_i^k+b_{i,k}s_i^k.
\]

For each coordinate \(j=1,\ldots,D\), the frozen local equation is

\[
  \ddot{x}_{i,j}+(1-\mu_k)\dot{x}_{i,j}
  +\omega_{i,k}^{2}x_{i,j}=F_{i,k,j}.
\]

When \(\omega_{i,k}^{2}>0\), its equilibrium is

\[
  x_{i,k,j}^{\mathrm{eq}}=\frac{F_{i,k,j}}{\omega_{i,k}^{2}}.
\]

With \(\alpha_k=(1-\mu_k)/2\) and
\(u_{i,j}=x_{i,j}-x_{i,k,j}^{\mathrm{eq}}\), the homogeneous equation becomes

\[
  \ddot u+2\alpha_k\dot u+\omega_{i,k}^{2}u=0.
\]

`propagate_frozen_oscillator` evaluates the exact state transition (up to
floating-point roundoff) in the underdamped, critically damped, and overdamped
regimes. The zero-frequency limit is handled separately by the same limiting
characteristic-root representation. The C++ test suite compares these flows
against an independently implemented RK4 reference, checks the semigroup
property, and covers zero-time, zero-frequency, and equilibrium cases.

The adjective *frozen* is essential: the coefficients, social attractor,
personal best, and \(\mu_k\) remain fixed during one interval of length
\(\Delta t\). They are refreshed at an update boundary. The complete optimizer
is therefore a switched stochastic process; it is not the exact flow of a
single ODE with continuously varying coefficients.

## 3. Random-coefficient convention and reproducibility

In project 0.7.0, one scalar pair \((r_1,r_2)\) is sampled per particle and
per update interval and shared by all coordinates of that particle. This is an
explicit vector-ODE convention, not a statement that the source article
necessarily uses the same convention. It remains subject to the source audit.

Each particle owns a `std::mt19937_64` stream initialized deterministically
from the 64-bit run seed and one-based particle identifier via `std::seed_seq`
and fixed project tags. A particle consumes only its own stream. This
construction provides deterministic, separately derived streams and
schedule-independent random-number consumption for synchronous particle
parallelism. It is not a mathematical proof of statistical independence or
absence of all possible cross-stream correlations.

## 4. Inertia schedule and observation policy

For a run requesting \(K\) transitions, the schedule is

\[
  \mu_k=\mu_{\max}-(\mu_{\max}-\mu_{\min})
  \frac{k}{K-1},\qquad k=0,\ldots,K-1,
\]

when \(K>1\). Thus the first and last transitions use the two endpoints. For
\(K\leq1\), the single (or absent) transition uses \(\mu_{\max}\).

The state is observed at the end of each frozen interval and the objective is
evaluated at that endpoint. Consequently, \(\Delta t\) controls the length of
the frozen-coefficient interval; it is an algorithmic parameter, not a local
error tolerance of a numerical ODE solver.

## 5. Bounds and boundary map

Initial positions are sampled independently and uniformly in each coordinate
interval. Initial velocities are sampled as

\[
  v_{i,j}^{0}\sim U[-\rho(u_j-\ell_j),\n                     \rho(u_j-\ell_j)],
\]

where \(\rho\) is `initial_velocity_fraction`.

After each analytic propagation, an endpoint outside a coordinate interval is
clamped to the violated endpoint. If its velocity points outwards, that
velocity is reversed and multiplied by a restitution factor
\(e\in[0,1]\). This is a discrete boundary map added for bounded benchmark
problems; it is not part of the unconstrained oscillator ODE. The implementation
does not inspect intermediate points inside the analytic segment and does not
resolve multiple crossings/bounces within one interval. Any comparison with
another optimizer must use a stated, compatible bound-handling policy.

## 6. Synchronous update semantics and social topology

At the beginning of a synchronous generation, every particle's social
attractor is selected from the same pre-propagation personal-best snapshot.
All candidate positions are propagated and evaluated before personal-best
states are committed. Commit is deterministic in particle-index order and
uses strict improvement:

\[
 p_i^{k+1}=\begin{cases}
 x_i^{k+1},&J(x_i^{k+1})<J(p_i^k),\\
 p_i^k,&\text{otherwise}.
 \end{cases}
\]

Global-best is recomputed after all commits. The best-so-far objective is
therefore non-increasing. Social topology is defined as follows:

- **Global-best:** every particle uses the best personal-best vector in the
  population snapshot.
- **Ring:** the neighborhood is the particle itself and its periodic left and
  right neighbors; duplicate indices are removed for small populations.
- **Toroidal von-Neumann grid:** the particle and its four orthogonal periodic
  neighbors on a row-major \(R\times C\) grid; diagonal neighbors are excluded
  and duplicate indices are removed. \(R,C\geq2\) and \(RC=N\).

Ties in social leader selection are resolved by the smallest particle index.
The social graph changes the optimization dynamics even when the executor is
serial.

## 7. Implemented execution architectures

The five modes exposed by the C++ and Python APIs have distinct semantics:

| Mode | Parallel work and update policy | Relation to serial reference |
|---|---|---|
| `serial` | Synchronous frozen-snapshot update, sequential propagation/evaluation/commit | Reference |
| `master_slave` | Parallel objective-evaluation batches; propagation and commit remain sequential | Exact equivalence required |
| `particle_parallel` | Parallel initialization and propagation after the frozen snapshot, parallel evaluation, ordered commit | Exact equivalence required |
| `asynchronous` | Event queue; each completed particle event publishes its personal-best update immediately and requeues that particle | Different dynamics |
| `islands` | Fixed total population split into equal sub-swarms with generation barriers and optional migration | Different dynamics |

For `master_slave` and `particle_parallel`, exact equivalence means equality
of the final best position/value, full best-history, FE-history, and total FE
count for a deterministic objective under the same configuration, topology,
and seed. Objective callbacks used concurrently must be reentrant and
thread-safe. They must also be deterministic for exact output equivalence.

In asynchronous mode one event advances one particle by one frozen segment;
its personal-best improvement becomes visible immediately. Each particle
performs exactly \(K\) updates, but there is no generation-wide barrier. With
multiple workers, completion order is scheduling- and latency-dependent. The
single-worker mode is repeatable but still has event-based rather than
generation-synchronous semantics.

In island mode the fixed total population \(N\) is split evenly across \(M\)
islands. Each island advances its own synchronous swarm. If enabled, migration
occurs after a completed generation and before the next generation (never
after the final generation). Emigrants are copied from a frozen island-best
snapshot. Ring migration sends each island incumbent to its successor; fully
connected migration sends each destination the best incumbent among the other
islands. A recipient's worst personal-best particle is replaced only by a
strictly better immigrant and its velocity is reset to zero. The already-known
objective value is transferred, so migration does not incur a new FE. The
island inner topology supports global-best and ring; grid is explicitly
rejected.

All implementations currently use a shared-memory C++ `std::thread` backend.
There is no MPI, distributed-memory, CUDA, or GPU implementation in this
repository.

## 8. Evaluation accounting and histories

For fixed population \(N\), \(K\) requested particle updates, and no early
stopping, every architecture performs

\[
  N_{\mathrm{FE}}=N(K+1)
\]

objective evaluations. Synchronous histories contain the initial population
incumbent and one value per completed generation, with FE axis
\(N,2N,\ldots,(K+1)N\). Asynchronous histories contain the initial incumbent
and one value per committed particle event, with FE axis increasing by one.

`elapsed_seconds` is the end-to-end wall time for a run and includes worker
pool/thread lifecycle. Worker-time fields are sums of task durations and may
exceed wall time. They must not be substituted for elapsed wall time.

## 9. Scope of current validation and claims

- The local frozen oscillator has a closed-form propagator; the optimizer is
  not a generic ODE time-stepping method.
- No convergence theorem for the complete stochastic optimizer is claimed.
- No global-optimality or guaranteed-success claim is made.
- The current built-in optimization objectives are Sphere, Rastrigin,
  Rosenbrock, Ackley, Griewank, and Salomon. `sphere_work` is only a synthetic
  execution-cost instrument and is not an optimization benchmark result.
- External COCO/BBOB, BBOB-Large-Scale and CEC 2013 LSGO benchmark drivers,
  together with six discrete-time, continuous-search-space PSO quality
  baselines, are implemented in a separate serial evaluation layer. They do not alter the native parallel
  execution architectures. Actual runs with the external benchmark packages
  are still required. Cooperative co-evolution, MPI and GPU backends are not
  implemented.
- A strict reproduction claim for Orlando--Ricciardello (2021) remains blocked
  until the complete source is audited for random-variable conventions,
  parameter schedules, initialization, boundary handling, and observation
  times.
