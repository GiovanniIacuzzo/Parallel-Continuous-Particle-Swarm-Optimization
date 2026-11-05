# Definitive architecture campaign: design and execution protocol

## Status and scope

This is the pre-registered internal campaign for characterizing CPSO execution
architectures. It answers three questions: (i) how social topology affects the
search dynamics; (ii) how event-driven asynchrony and island decomposition
change search quality; and (iii) when synchronous parallel execution reduces
wall-clock time without changing the serial trajectory.

The campaign is reproducible and checkpointed, but it is **not yet a complete
external benchmark submission**. The current native engine exposes six
built-in analytic functions. COCO/BBOB, BBOB-Large-Scale and CEC 2013 LSGO
adapters are not part of this release. No claim of benchmark-suite superiority
or state-of-the-art performance may be based on this campaign alone. External
suite adapters, version pinning, official instance handling and FE accounting
must be implemented and tested before those claims are made.

`sphere_work` is exclusively an execution-cost instrument. Its results belong
to the systems/scaling analysis and must not be pooled with optimization
benchmark results.

## Entry points

Run the entire campaign:

```bash
PYTHONPATH=python python experiments/04_architectures/run_research_campaign.py \
  configs/campaigns/definitive_architecture_campaign.json \
  --output results/raw/cpso_architecture_campaign_v1
```

Validate and count all runs without execution:

```bash
PYTHONPATH=python python experiments/04_architectures/run_research_campaign.py \
  configs/campaigns/definitive_architecture_campaign.json --dry-run
```

Run selected stages, or reanalyze completed stage outputs:

```bash
PYTHONPATH=python python experiments/04_architectures/run_research_campaign.py \
  configs/campaigns/definitive_architecture_campaign.json \
  --stages 04_strong_scaling --output results/raw/cpso_architecture_campaign_v1

PYTHONPATH=python python experiments/04_architectures/run_research_campaign.py \
  configs/campaigns/definitive_architecture_campaign.json \
  --output results/raw/cpso_architecture_campaign_v1 --analyze-only
```

The output manifest stores the full campaign hash, source hash and deterministic
study execution schedule, plus software, CPU, thread-environment and native
build metadata. `campaign_batches.jsonl` records the stage/study selection used
by each invocation. Each sub-study uses an append-only run-level JSONL
checkpoint; each completed run is flushed and synchronized to the file before
the next task starts. Manifest replacement is atomic. On resume, only an
incomplete unterminated final JSONL record may be discarded; malformed
newline-terminated records and duplicate run keys are hard errors. A changed
configuration or source tree is rejected on resume.
The analyzer writes `campaign_runs.csv`, `campaign_validation.csv`,
pairwise/omnibus statistical tables, scaling summaries, a consolidated report
and publication figures. Raw `runs.jsonl`, per-study manifests and checkpoints
remain the authoritative record; derived CSVs and figures can be regenerated.

## Stage 1 — social topology effect

Question: does the social-information graph change CPSO search behavior when
execution remains serial and synchronous?

- Functions: Sphere, Rastrigin, Rosenbrock, Ackley, Griewank, Salomon.
- Dimensions: 30 and 100.
- Population: 64 particles.
- Generations: 500.
- Seeds: 30 independent seeds (11001–11030).
- Methods: serial global-best, serial ring, serial von-Neumann toroidal grid
  (8 x 8).
- Comparisons: ring vs global-best and grid vs global-best.
- Runs: 1,080; FE per run: 32,064.

This is the clean topology ablation: no thread scheduling or migration changes
are mixed into the comparison.

## Stage 2 — asynchronous dynamics

Question: what is the effect of event-driven publication, and how does
concurrent event completion alter the result?

- Same six functions, dimensions, population, generations and 30 seeds as
  Stage 1.
- References: serial global, ring and grid.
- Asynchronous variants: each of the three social topologies with 1 and 4
  workers.
- Runs: 3,240; FE per run: 32,064.

The one-worker asynchronous case isolates the event-update semantics from
simultaneous multiworker completion. The four-worker case adds concurrent
scheduling. Multiworker asynchronous runs are not expected to be bitwise
reproducible across separate executions because event completion order is
schedule-dependent. They are analyzed statistically over paired seeds.

## Stage 3 — island dynamics

Question: how do island count, local social topology and migration affect
solution quality at fixed total population and FE budget?

- Same six functions, dimensions, total population 64, 500 generations and
  30 seeds.
- Serial references: global-best and ring.
- Island counts: 2, 4 and 8 (local populations 32, 16 and 8).
- Migration ablations at four islands: none, ring migration every 5, 20 and 50
  generations, fully-connected migration every 20 generations.
- Inner topology ablations at four islands: global-best and ring, with and
  without ring migration.
- Runs: 3,960; FE per run: 32,064.

Migration does not consume objective evaluations in the current implementation
because the migrant is transferred with its already-known objective value.
The campaign will record accepted replacements, not attempted transfers.

## Stage 4 — strong scaling of equivalent synchronous backends

Question: under what evaluation costs and problem sizes does the synchronous
parallel backend reduce elapsed wall time while preserving the exact serial
trajectory?

- Objective: `sphere_work` only.
- Dimensions: 30 and 100.
- Total population: 64 and 128, analyzed as separate sub-studies.
- Generations: 80.
- Work levels: 0, 512 and 4,096 recurrence iterations per objective call.
- Timing repetitions: 10 paired seeds (22001–22010).
- Methods: serial, master–slave and particle-parallel with 1, 2, 4 and 8
  workers for each parallel backend.
- Runs: 1,080 across six sub-studies.

For each configuration, strong-scaling speedup is

\[
S(p)=\frac{T_{\mathrm{serial}}}{T_p},\qquad
E(p)=\frac{S(p)}{p}.
\]

This label is used only for master–slave and particle-parallel because their
best position, best value, full history, FE axis and total FE count are checked
for exact equality against the serial reference. The synthetic work levels are
not optimization benchmark results. The run timer includes the lifecycle of
the worker pool; no architecture receives a timing exemption.

## Total campaign size

| Stage | Runs | Total FE |
|---|---:|---:|
| Topology effect | 1,080 | 34,629,120 |
| Asynchronous dynamics | 3,240 | 103,887,360 |
| Island dynamics | 3,960 | 126,973,440 |
| Strong scaling | 1,080 | 8,398,080 |
| **Total** | **9,360** | **273,888,000** |

The full campaign is intentionally checkpointed and can be executed stage by
stage. The campaign runner randomizes sub-study order using the fixed schedule
seed and the underlying study runner interleaves method order within each
benchmark/dimension/seed block.

## Statistical analysis plan

### Experimental units and pairing

Within each benchmark-function x dimension cell, the same seed is used for
every method. Seed-level runs are paired within that cell. The six analytic
benchmark functions are the highest-level independent clusters. The two
dimensions (30 and 100) are repeated measurements nested within a function;
they are not counted as twelve independent function families. Seed replicates
are nested within function x dimension.

For confirmatory inference, first compute paired seed-level log-objective
differences within each function x dimension cell, then take the median across
dimensions to obtain one paired summary per function family (n=6). The
Wilcoxon signed-rank tests and Friedman omnibus analysis operate on these six
family-level summaries. The twelve function x dimension cell summaries are
retained for transparent descriptive reporting and dimension-specific
inspection, but must not be described as twelve independent benchmark families.
The small number of independent function families limits inferential power;
report effect sizes, paired win/loss counts and uncertainty intervals, and do
not make strong generalization claims from p-values alone.

### Quality outcomes

Primary continuous outcome: final objective, analyzed as
\(\log_{10}(\max(f,10^{-16}))\). Primary comparisons are predeclared in the
campaign configuration and expressed as paired differences against their
matched reference method. Report:

- median block-level paired log-objective difference;
- nested bootstrap 95% interval (resample function families, then dimensions
  within selected families, then paired seeds within selected function x
  dimension cells);
- exact two-sided signed-rank sign-flip test on the six function-family
  median differences (all sign configurations are enumerated; zero differences
  are omitted from the rank set);
- Holm-adjusted p-values within each study's predeclared comparison family;
- paired win/loss/tie counts and win rate, retained as descriptive run-level
  summaries rather than independent observations;
- Friedman omnibus statistic on family-level ranks, with a blocked
  Monte-Carlo permutation calibration (20,000 within-family method-label
  permutations). The conventional chi-square and Iman–Davenport p-values are
  retained as asymptotic diagnostics, not as the confirmatory omnibus result.

Also report success rate and expected running time (ERT) for the predeclared
\(10^{-8}\) target. Failed runs are censored at their FE budget; ERT is infinite
when no run reaches target. Target results must not be interpreted without the
success count.

The nested bootstrap interval is descriptive uncertainty; exact family-level
signed-rank sign-flip tests and the blocked permutation Friedman test provide
the small-sample inferential calibration. With only six family clusters, the
smallest attainable two-sided signed-rank p-value is 0.03125 before Holm
correction; after correcting multiple comparisons, statistical significance
may be unattainable even for a consistent effect. The campaign does not search
post-hoc for the most favorable target or comparison.

### Performance outcomes

For each paired seed, calculate elapsed-time ratio against the serial run of
the exact same synchronous algorithm and topology. Report median speedup,
bootstrap 95% interval, parallel efficiency, worker count, population,
dimension and synthetic objective-work level. Do not compute or report a
parallel speedup for asynchronous or island algorithms.

Report both native elapsed time and the phase timing counters. Worker-seconds
are aggregate task durations and may exceed wall-clock time; they are not
substitutes for elapsed time.

## Machine and build protocol

Each study manifest records OS/platform, Python version, native build metadata,
logical CPU count and source fingerprint. For the target Mac run:

1. use a Release build and record the exact compiler/version and CMake cache;
2. close unrelated CPU-intensive tasks and keep the machine on a stable power
   profile;
3. do not run multiple campaigns concurrently;
4. use worker counts no greater than the available logical CPUs; edit the JSON
   worker list on machines with fewer than eight;
5. preserve raw `runs.jsonl`, manifests and CSVs without manual edits;
6. rerun each scaling study at least once in a separate session if thermal or
   background-load stability is in doubt, retaining both sessions rather than
   overwriting data.

The development container results are software-validation data only. The
publication timing results must come from the target hardware and its recorded
build.

## Required publication figures

1. Best-so-far versus FE for global, ring and grid topology (one panel per
   function/dimension or a carefully selected subset).
2. Final objective distributions for topology ablations.
3. Best-so-far versus FE for synchronous and asynchronous event semantics,
   distinguishing one and four workers.
4. Final objective distributions for the asynchronous variants.
5. Island-count ablation (M=2,4,8) at no migration.
6. Migration-policy comparison (none, ring, fully connected) at M=4.
7. Inner island topology comparison (global vs ring).
8. Strong-scaling speedup and efficiency versus workers, separated by N, D and
   synthetic work level.
9. ERT/success-rate table and inferential summary table.

All figure files are generated from saved raw run records by the campaign
analyzer and exported to PNG, PDF and SVG.
