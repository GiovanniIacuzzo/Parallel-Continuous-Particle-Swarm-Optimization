# Synchronous particle-social topologies

Version 0.4.0 adds three particle-level social topologies to the existing
synchronous CPSO engine. The population size, initialization, per-particle RNG
streams, frozen-oscillator propagator, inertia schedule, bounds map, observation
interval, objective-evaluation budget, and synchronous commit semantics are
shared by all three variants. Only the social attractor supplied to a particle
changes.

## Definitions

Let particle `i` have personal-best state `(p_i, f_i)`, where `p_i` is its
best-known position and `f_i` its objective value. All neighborhood leader
selections are made from a frozen snapshot at the beginning of a generation;
no particle can observe a personal-best update produced in the same generation.

### Global-best

Every particle uses the incumbent population-wide best personal-best position.
The incumbent is updated only on strict objective improvement. Equal values do
not replace its position. This deliberately preserves the 0.3.0 global-best
reference's tie/update semantics.

### Periodic ring

For a population of size `N`, particles are indexed on a periodic ring. The
neighborhood is

`N_i = {(i-1) mod N, i, (i+1) mod N}`.

It is the radius-one ring, includes the particle itself, and removes duplicate
indices for `N=1` and `N=2`. The social attractor is the personal-best position
with minimum objective value in `N_i`. Equal values are resolved by choosing
the smallest particle index.

### Toroidal von-Neumann grid

The population is placed row-major on an `R x C` periodic grid, with `R*C=N`.
The neighborhood contains the particle itself and its four orthogonal
neighbors (up, down, left, right), with periodic wrap-around in both
coordinates. Both grid dimensions must be at least two. Duplicate indices are
removed, which matters for grids with a row or column of length two. The social attractor is the best personal
best in this set; ties use the smallest particle index.

This is specifically a von-Neumann neighborhood, not a Moore neighborhood:
diagonal cells are not included.

## Configuration

C++:

```cpp
config.topology = cpso::TopologyKind::GlobalBest;
config.topology = cpso::TopologyKind::Ring;
config.topology = cpso::TopologyKind::GridVonNeumann;
config.grid_rows = 5;
config.grid_columns = 8; // must multiply to population_size
```

Python:

```python
CpsoConfig(..., topology="global")
CpsoConfig(..., topology="ring")
CpsoConfig(
    ..., population_size=40, topology="grid_von_neumann",
    grid_rows=5, grid_columns=8,
)
```

Grid dimensions are mandatory for grid topology and forbidden for the other
topologies. Both dimensions must be at least two and their product must equal
the population size. Prime population sizes cannot form a non-degenerate 2D
grid under this definition.

## Deterministic comparison protocol

For algorithmic comparisons, run all topologies with the same:

- objective and search bounds;
- dimension and population size;
- CPSO parameters and observation interval;
- seed set and total function-evaluation budget.

Each seed is run under all three topologies. The social graph is deterministic,
and each particle retains the same seed-derived RNG stream across topologies.
In 0.4.0, streams are initialized from the separate run-seed and particle-ID
words via `std::seed_seq`; this corrects the cross-run stream collisions
possible under the previous XOR-based derivation. Therefore 0.4.0 results are
not bitwise comparable to 0.3.0 even for the global-best topology.
The resulting trajectories are nevertheless expected to differ because the
social attractors differ. This is an algorithmic comparison, not a hardware
speedup experiment.

The serial and master--slave executors must produce exactly equal results for
a fixed topology, seed, configuration, and deterministic thread-safe
objective. Changing the topology is not expected to preserve results.

## Implementation boundaries

- Particle-level topology is independent of the executor (serial or
  master--slave).
- The topology is fixed for the complete run in this release.
- There is no particle migration, island model, asynchronous update, adaptive
  neighborhood, or time-varying communication graph.
- Global-best, ring, and grid are not asserted to have equal convergence
  properties; this must be evaluated empirically.
- Global-best stores one shared list of all particle indices, rather than N
  duplicated neighborhoods. Ring/grid store only their bounded-size local
  neighborhoods. The optimizer records one-time topology construction and
  accumulated per-generation social-selection times separately; total
  end-to-end wall time remains the primary runtime measure.
