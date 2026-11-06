# Particle communication topologies

**Implementation version:** 0.7.3  
**Scope:** static, undirected, particle-level social graphs in the shared-memory CPSO engine.

This document describes the graph used to choose each particle's social attractor. It is distinct from (i) the execution backend (serial, master--slave, particle-parallel, asynchronous, or islands) and (ii) the island-to-island migration graph. Every particle neighborhood includes the particle itself. Graphs are constructed once per run, are immutable during that run, and use deterministic topology seeds independent of the particle-level random streams.

## Implemented topology families

| Identifier | Definition | Main parameterization | Connectivity |
|---|---|---|---|
| `global` | Complete graph / global-best information | None | Connected |
| `ring` | Periodic radius-one ring | None | Connected |
| `grid_von_neumann` | Toroidal 2-D grid, four orthogonal neighbors | `grid_rows * grid_columns = N` | Connected |
| `wheel` | Canonical PSO focal-particle wheel: one focal particle adjacent to all peripheral particles; no peripheral-to-peripheral edges | Randomly relabelled focal particle | Connected |
| `random` | Erdős--Rényi-like `G(N,p)` sample followed by random bridges between connected components | `p = d_target/(N-1)`; default target degree 4 | Connected by construction |
| `small_world` | Watts--Strogatz rewiring of a periodic ring lattice | Even `k=4`, rewiring probability `beta=0.15` by default | Not forcibly repaired after rewiring |
| `scale_free` | Barabási--Albert preferential attachment | `m=2` by default (`topology_degree=2m=4`) | Connected by construction |

The wheel identifier follows the hub-and-spoke PSO neighborhood described in the particle-swarm topology literature; it is **not** the graph-theoretic wheel with a peripheral cycle. The focal role is randomly relabelled on each seeded graph so that particle index zero is not privileged across all seeds.

The connected-random construction is deliberately described as **ER-like with bridge augmentation**. It first samples each undirected edge independently with probability `p=d_target/(N-1)`. If the sampled graph is disconnected, random inter-component edges are added until it is connected. This is not an exact sample from `G(N,p)` conditioned on connectedness, and the final mean degree can be slightly above the target.

The small-world graph follows the Watts--Strogatz ring-lattice rewiring mechanism. Rewiring preserves the number of edges but can produce a disconnected graph for some seeds; connectivity is not repaired, because doing so would alter the specified Watts--Strogatz model. The scale-free graph uses preferential attachment to an initial clique of `m+1` nodes, with each subsequent node attaching to `m` distinct existing nodes. At the finite population sizes used here, “scale-free” identifies the Barabási--Albert construction; it is not a statistical claim that a fitted power law has been demonstrated.

## Configuration

Python examples:

```python
from cpso_serial import CpsoConfig

base = dict(lower_bounds=(-5.0,) * 30, upper_bounds=(5.0,) * 30,
            population_size=64, iterations=200, seed=47001)

cfg_wheel = CpsoConfig(**base, topology="wheel")
cfg_random = CpsoConfig(**base, topology="random", topology_degree=4)
cfg_small_world = CpsoConfig(
    **base, topology="small_world", topology_degree=4,
    topology_rewiring_probability=0.15,
)
cfg_scale_free = CpsoConfig(**base, topology="scale_free", topology_degree=4)
```

`topology_degree` has topology-specific meaning: target expected degree for the connected-random graph; initial ring degree `k` for Watts--Strogatz; twice the attachment count (`2m`) for Barabási--Albert. For the wheel it is unused. Von-Neumann grid dimensions are mandatory for `grid_von_neumann` and forbidden for other topologies. The current island backend does not accept a von-Neumann inner-island topology; it accepts the graph-based families subject to minimum local-population constraints.

## Reproducibility and fair comparison

For each paired topology comparison, hold fixed the benchmark, dimension, population, CPSO coefficients, time step, boundary rule, observation schedule, evaluation budget, and run seed. The graph is generated from a deterministic topology seed derived from the run seed. Particle streams remain assigned by particle index. Thus, topology effects are paired at the run level, while graph-specific randomness is reproducible.

The seven graph families do **not** have identical degree distributions: global is dense; ring and wheel are sparse but structurally different; the toroidal grid has degree four; random, small-world, and scale-free use approximately four-neighbor sparse constructions (with graph-specific degree variation). The experiment therefore compares canonical communication structures as complete graph families; it does not identify a pure “shape-only” effect independent of degree. Reports must include the construction parameters and should avoid attributing every observed difference to a single graph statistic.

For a fixed topology, seed, configuration, and deterministic objective, serial, master--slave, and particle-parallel synchronous execution must retain exactly the same trajectory. Changing the topology is an algorithmic change and is not expected to preserve the trajectory. The asynchronous and island backends also alter information timing or population structure; their results must not be labelled as speedup of the same serial trajectory.

## Sources and scope of novelty

The comparison follows established PSO communication structures and network-generation models: early PSO work on neighborhood topologies, the Watts--Strogatz small-world construction, and Barabási--Albert preferential attachment. Scale-free interactions have also been studied directly in PSO. The present work does not claim the graph families themselves as new; its focus is their controlled integration and evaluation within the continuous-PSO parallel execution framework.
