# Seven-topology expansion: Stage B protocol

**Software revision:** 0.7.3  
**Execution model:** CPU shared memory.  
**Purpose:** evaluate the effects of established particle communication graphs within the CPSO framework; not to re-study the intrinsic oscillator dynamics already addressed by Orlando--Ricciardello.

## Topologies

The comparison contains seven particle-level social graphs:

1. Global-best (complete graph).
2. Periodic radius-one ring.
3. Toroidal von-Neumann grid (8 x 8 for N=64).
4. Canonical PSO wheel (one focal particle connected to all peripherals, no peripheral edges).
5. Connected ER-like random graph: edge probability `p=4/(N-1)`, followed by random bridges if required to connect components.
6. Watts--Strogatz small-world: ring degree `k=4`, rewiring probability `beta=0.15`.
7. Barabasi--Albert scale-free: preferential attachment with `m=2`.

The four added families are wheel, connected-random, small-world, and scale-free. The first three graph types remain anchors for the historical literature. These are fixed graphs during each optimization run. A deterministic topology seed is derived from the run seed and does not consume the particle random streams.

## Pilot design

The pilot configuration is `configs/campaigns/cpso_topology_expansion_pilot.json`.

**Quality stage:** six native deterministic objectives; dimensions D=30 and D=100; N=64; 200 updates; ten paired seeds; seven serial topologies. This yields 840 runs, each with `64*(200+1)=12,864` function evaluations.

**Synchronous execution-cost stage:** `sphere_work`, D=30, N=64, 150 updates, ten paired seeds; for each topology, serial, master--slave with four workers, and particle-parallel with four workers. This yields 210 runs, each with `64*(150+1)=9,664` function evaluations. Exact trajectory equivalence is checked within each topology before reporting paired time ratios.

**Total pilot:** 1,050 runs. The quality stage studies topology under a fixed serial execution backend. The performance stage asks whether graph construction/social-selection overhead changes the cost of the synchronous backends. `sphere_work` is a controlled microbenchmark and is not used as an optimization-quality benchmark.

## Confirmatory design

`configs/campaigns/cpso_topology_expansion_confirmatory.json` repeats the same design with 30 seeds: 2,520 quality runs plus 630 performance runs, for 3,150 total runs. Run this only after reviewing the pilot and freezing the software/configuration revision. Do not change topology parameters after inspecting confirmatory outcomes.

## Statistical unit and reporting

Seeds are paired across topologies within each benchmark-dimension cell. The six objective functions are the top-level families; D=30 and D=100 are repeated measures nested within a family. Confirmatory inference must therefore reduce the paired seed results within each function-dimension cell and then the two dimensions within each function before treating the six functions as independent top-level units. The 12 function-dimension blocks and all run-level observations may be shown descriptively, but not counted as 12 independent benchmark families.

Report median paired log-objective differences, effect sizes, family-wise intervals, exact signed-rank/permutation results with Holm correction, success rates, FE budgets, and all graph construction parameters. The small number of independent function families limits statistical power and should be stated.

The seven topologies have different degree profiles by construction. The experiment is a comparison of canonical graph families, not a degree-matched causal isolation of graph geometry. Do not write that the graph shape alone caused an outcome unless an additional degree-matched sensitivity design supports that claim.

## Commands (macOS)

From the repository root, with `.venv` active, C++ library built, and `PYTHONPATH="$PWD/python"`:

```bash
# Validate the pilot and confirm its run count (1,050)
python experiments/04_architectures/run_research_campaign.py \
  configs/campaigns/cpso_topology_expansion_pilot.json --dry-run

# Execute the pilot; checkpointing makes it resumable
python experiments/04_architectures/run_research_campaign.py \
  configs/campaigns/cpso_topology_expansion_pilot.json \
  --output results/raw/cpso_topology_expansion_pilot_mac --no-plots

# After checking the pilot, validate (3,150) and run the confirmatory campaign
python experiments/04_architectures/run_research_campaign.py \
  configs/campaigns/cpso_topology_expansion_confirmatory.json --dry-run

python experiments/04_architectures/run_research_campaign.py \
  configs/campaigns/cpso_topology_expansion_confirmatory.json \
  --output results/raw/cpso_topology_expansion_confirmatory_mac --no-plots
```

After each campaign, regenerate publication figures without rerunning optimization:

```bash
python analysis/postprocess_publication_figures.py \
  --results-root results \
  --output paper/figures/publication_stage_A
```

The `--no-plots` option suppresses in-campaign diagnostic plots only; the separate post-processing script writes vector PDF figures from the completed CSV/JSONL records.
