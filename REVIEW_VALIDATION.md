# Validation of the controlled review experiments

Base revision: `bbdedd02346cb62209aa596ffb8a076d01ee2ae3`. Validation date: 2026-11-16. The environment is Linux x86_64, not the author's Mac M2. `paper/` was not edited.

## Executed checks

- 118 Python tests passed; one optional CEC test was skipped because `cec2013lsgo` is not installed.
- All four native CTest cases passed in Release.
- The existing publication-scaling standalone check passed (pairing, resume, CSV statistics, PDF output).
- Bit-for-bit comparison against a separately compiled original revision matched all 63 legacy trajectories: three dimensions × seven topology families × three seeds, N=64, 20 generations, dt=1, Rastrigin. Best positions, histories and FE axes were compared.
- Controlled synchronous equivalence checks passed for both dynamics and coefficient scopes.
- BBOB smoke completed 66 runs; BBOB pilot completed 1,560 runs with official Observer logs and complete FE budgets.
- Geometric smoke completed 16 runs; geometric pilot completed 144 runs (D=5,20,1000; three replicates; 16 variants).
- Parallel smoke completed three sessions, 72 batches total (144 optimizer runs), including model fitting, held-out prediction/regret and plot regeneration. These are pipeline checks, not M2 scaling results.
- CEC subset design passed the runner's dry-run validation (200 runs at 3,000,000 FE); no full-budget CEC run was executed.

## Preliminary quality results

BBOB pilot covers only functions 1/8/15/21 and instances 1/2/3, at 512D FE. Ranks below are among **65** controlled variants/references. Instances are aggregated within each function; all four functions have equal weight. Lower is better.

| Variant at dt=1 | D=5 mean rank | D=20 mean rank |
|---|---:|---:|
| CPSO scalar Global | 28.75 | 39.25 |
| CPSO scalar R8 | 18.50 | 30.50 |
| CPSO coordinate Global | 15.25 | 21.25 |
| CPSO coordinate R8 | 10.00 | 2.25 |
| Discrete scalar Global | 34.50 | 21.25 |
| Discrete scalar R8 | 38.00 | 22.50 |
| Discrete coordinate Global | 48.75 | 29.50 |
| Discrete coordinate R8 | 56.25 | 34.00 |
| DE | 16.50 | 8.50 |

The original scalar CPSO R8 remains behind the matched scalar discrete R8 at D=20. Changing the analytic coefficient scope to per-coordinate sampling gives a large improvement in this pilot. This is a distinct experimental variant, not a retroactive improvement of the published scalar propagator. All planned and primary Wilcoxon contrasts have Holm-adjusted p=1 in this four-function pilot. The full suite is needed before interpreting superiority.

## Geometric finding

At D=1000 the observed initial affine rank is 127 (=2N−1), versus full rank at D=5/20. In the unconstrained diagnostic, the median final-best residual outside that affine space is approximately 1.2e−15 for scalar analytic dynamics and 1.7e−15 for scalar discrete dynamics. Coordinate sampling escapes it. Since the initial space is already full rank at D=20, confinement in that initial affine space cannot explain the D=20 pilot improvement. The normal clamp also allows scalar trajectories to escape; therefore the unconstrained invariant cannot be asserted unchanged for bounded benchmark runs.

The shifted, rotated ill-conditioned ellipsoid does **not** show a universal quality advantage for coordinate sampling: scalar analytic runs have lower median objectives in this small high-dimensional diagnostic. Escaping the initial subspace is a mechanism, not a guarantee of better optimization.

## Claims still awaiting evidence

Full 24-function BBOB confirmation, stronger independent graph replication, separately tuned/recommended literature-baseline populations, full-budget CEC results, M2 multi-session timings, a second hardware platform, 32/64-worker scaling and an OpenMP reference remain open. External instrumented time-to-target is not equated with uninstrumented native scaling.

See [REVIEW_EXPERIMENTS.md](REVIEW_EXPERIMENTS.md) for launch commands, output definitions, correction families, timing scope and recovery behavior. Raw data and source/build fingerprints from this validation are retained in the accompanying results archive.
