# Serial CPSO: mathematical and numerical specification

## 1. Optimization problem and state

The implementation minimizes a bounded scalar objective

\[
  \min_{x\in\Omega} J(x), \qquad
  \Omega=\prod_{j=1}^{D}[\ell_j,u_j].
\]

Particle \(i\) stores position \(x_i\), velocity \(v_i\), personal-best position
\(p_i\), and the value of its current and personal-best positions. The social
attractor in this first release is the global-best position \(g\) of the swarm.

## 2. Frozen-coefficient CPSO dynamics

For particle \(i\), generation \(k\), and a scalar pair of random coefficients
\(r_{1,i}^k,r_{2,i}^k\in[0,1)\), define

\[
  \omega_{i,k}^{2}=c_1r_{1,i}^k+c_2r_{2,i}^k,
\]

\[
  F_{i,k}=c_1r_{1,i}^kp_i^k+c_2r_{2,i}^kg^k.
\]

The local coordinate equation is

\[
  \ddot{x}_{i,j}+(1-\mu_k)\dot{x}_{i,j}
  +\omega_{i,k}^{2}x_{i,j}=F_{i,k,j}.
\]

When \(\omega_{i,k}^{2}>0\), its equilibrium is

\[
  x_{i,k,j}^{\mathrm{eq}}=F_{i,k,j}/\omega_{i,k}^{2}.
\]

Writing \(\alpha_k=(1-\mu_k)/2\) and
\(u_{i,j}=x_{i,j}-x_{i,k,j}^{\mathrm{eq}}\), the homogeneous equation is

\[
  \ddot u+2\alpha_k\dot u+\omega_{i,k}^{2}u=0.
\]

`propagate_frozen_oscillator` evaluates its closed-form state transition in
three regimes: underdamped, critically damped, and overdamped. The limiting
case \(\omega^2=0\) is handled by the same characteristic-root formulas (with
the critical case covering \(\alpha=0\)).

The coefficients \(r_1,r_2\), personal best, global best, and \(\mu_k\) are
frozen during one interval of length \(\Delta t\). The resulting flow is exact
up to floating-point roundoff for that local constant-coefficient ODE. The
complete optimizer is a switched stochastic process because these quantities
are refreshed at generation boundaries; it is not the exact solution of one
single smooth ODE with continuously changing coefficients.

## 3. Random-coefficient convention

This release samples one pair \((r_{1,i}^k,r_{2,i}^k)\) per particle and per
generation. The same scalar pair is shared by all coordinates of that
particle during the frozen interval. Particle streams are initialized
independently through a deterministic 64-bit seed mixer and advanced only by
the owning particle.

This is an explicit vector-ODE convention. It must be checked against the
coordinate-level random-variable convention in the complete source paper
before the implementation is described as a line-by-line reproduction. The
published publisher landing page verifies the article's existence and its
stated analytic-continuous formulation, but its public preview does not expose
the full derivation.

## 4. Inertia schedule and observation policy

For a requested run with \(T\) transitions, the implementation uses

\[
  \mu_k=\mu_{\max}-(\mu_{\max}-\mu_{\min})
  \frac{k}{T-1},\qquad k=0,\ldots,T-1,
\]

when \(T>1\). Thus the first and last transitions use the two endpoints. For
\(T\leq1\), the single (or absent) transition uses \(\mu_{\max}\).

The optimizer observes the state at the end of each frozen interval and
evaluates the objective there. Consequently, \(\Delta t\) controls the length
of an interval over which the stochastic coefficients and attractors remain
frozen. It is an algorithmic parameter, not the local error tolerance of a
numerical ODE solver.

## 5. Synchronous serial update semantics

At the beginning of generation \(k\), the global-best vector \(g^k\) is
copied. Every particle uses this same snapshot and its own personal-best
snapshot to propagate. All new positions are evaluated before any personal
best is committed. Then personal bests are updated by strict improvement,

\[
 p_i^{k+1}=\begin{cases}
 x_i^{k+1},&J(x_i^{k+1})<J(p_i^k),\\
 p_i^k,&\text{otherwise}.
 \end{cases}
\]

The global best is recomputed from the updated personal bests. The reported
best-so-far value is therefore non-increasing. This two-phase update is the
serial reference semantics intended to be preserved by a future synchronous
master--slave executor.

## 6. Initialization and bounds

Initial positions are sampled independently and uniformly in each box
coordinate. Initial velocities are sampled coordinate-wise as

\[
  v_{i,j}^0\sim U[-\rho(u_j-\ell_j),\rho(u_j-\ell_j)],
\]

where \(\rho\) is `initial_velocity_fraction`.

After each analytic propagation, an endpoint outside a coordinate interval is
clamped to the violated endpoint. If its velocity points outwards, that
velocity is reversed and multiplied by `boundary_restitution` \(e\in[0,1]\).
This is a discrete boundary map chosen for bounded test problems; it is not
part of the unconstrained oscillator ODE. The implementation does not inspect
intermediate points inside the analytic segment and does not model multiple
bounces during one segment. Comparisons with other optimizers must use the
same box and a documented compatible boundary policy.

## 7. Evaluation accounting

There is one objective evaluation per particle at initialization and one after
each completed transition. With fixed population and no early stopping,

\[
  N_{\mathrm{FE}}=N_{\mathrm{pop}}(T+1).
\]

The C++ optimizer checks that every returned objective value is finite. The
history contains the initial best value and one incumbent value after each
completed generation.

## 8. What is and is not claimed

- The local oscillator propagation is closed-form, not an ODE time-stepping
  approximation.
- No convergence theorem for the complete stochastic optimizer is claimed.
- No claim of global optimality or guaranteed optimization success is made.
- The current release is a serial, synchronous, global-best baseline only.
- Topologies other than global-best, island models, asynchronous scheduling,
  parallel workers, COCO/CEC adapters, and PSO comparisons are not included.
- The exact correspondence of all parameter conventions to every equation and
  algorithmic detail in the full Orlando--Ricciardello paper remains a required
  source-audit item.
