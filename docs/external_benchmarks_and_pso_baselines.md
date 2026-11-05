# External benchmark suites and discrete-PSO reference baselines

## Scope and separation of experiments

The external quality layer covers three benchmark families:

1. COCO `bbob` for general-purpose, noise-free continuous black-box optimization.
2. COCO `bbob-largescale` for the same benchmark lineage at dimensions up to 640.
3. CEC 2013 LSGO for high-dimensional large-scale global optimization at D=1000.

These studies answer a different question from the internal CPSO architecture
campaign. They compare search quality under a matched function-evaluation (FE)
budget. They do not measure native parallel speedup. All external-objective
calls currently enter the native CPSO engine through a serial Python callback;
the callback ABI deliberately rejects parallel architectures because a
third-party Python/Cython objective cannot be assumed reentrant or
thread-safe. The Python PSO baselines are also single-process. Their wall times
are recorded for diagnostics only and must not be interpreted as a controlled
comparison of implementation speed between C++ and Python.

The internal analytic/synthetic architecture campaign remains separate.
`sphere_work` is not an external benchmark and must not be mixed into COCO/CEC
quality tables.

## COCO suites: BBOB and BBOB-LargeScale

Both suites use the official `coco-experiment` (`cocoex`) package and COCO
observers. The benchmark functions are not reimplemented locally.

### Standard BBOB

The `bbob` suite contains 24 noiseless single-objective scalable real-parameter
functions. The confirmatory configuration selects dimensions
`2, 3, 5, 10, 20, 40` and instances `1-15`:

- 24 functions;
- 6 dimensions;
- 15 instances per function/dimension;
- population 40;
- project-selected budget `10,000 * D` FE per problem and method;
- five algorithms (CPSO plus four discrete PSO baselines).

Configuration: `configs/benchmarks/bbob_confirmatory.json`. The intermediate
configuration `bbob_preconfirmatory.json` uses 24 functions, dimensions 2, 5
and 10, instances 1-5 and a budget of `2,000 * D` FE. It is intended to expose
integration/performance problems before the full suite, not to serve as a
substitute for the confirmatory protocol.

### BBOB-LargeScale

The `bbob-largescale` suite is the large-dimensional extension of BBOB, based
on permuted block-diagonal transformations. The official suite dimensions are
`20, 40, 80, 160, 320, 640`; it contains the 24 single-objective functions and
15 instances. The confirmatory campaign uses:

- 24 functions;
- 6 dimensions: `20, 40, 80, 160, 320, 640`;
- 15 instances per function/dimension;
- population 40;
- project-selected budget `10,000 * D` FE per problem and method;
- five algorithms.

Configuration: `configs/benchmarks/bbob_largescale_confirmatory.json`.
The intermediate configuration `bbob_largescale_preconfirmatory.json` selects
eight representative function indices, dimensions 20, 80, 320 and 640,
instances 1-3 and a budget of `250 * D` FE. It is an engineering/pilot stage;
its limited problem set and budget do not justify a broad generalization claim.
The `10,000 * D` budget is the project's predeclared comparison budget; it is
not described as a mandatory COCO competition termination rule.

### COCO execution semantics

The driver creates a fresh `Suite` and `Observer` for each algorithm so that
evaluation counters and logging state are not shared between methods. The
observer data are written under `coco_data/<method>/` for standard `cocopp`
post-processing. Each COCO problem is wrapped using its public ID, name,
dimension, bounds and callable objective. The driver checks that COCO's
`problem.evaluations` agrees exactly with the optimizer FE counter.

The optimizer seed is a deterministic 64-bit hash of the COCO problem ID and
campaign seed. Thus each method receives the same problem/seed label, while
each method retains its own algorithm-specific random stream. This is a paired
seed design, not a claim of identical initial particle coordinates or common
random numbers across the native C++ and NumPy random-number generators.

The smoke configuration selects a reduced subset and budget. It is an
integration check only. Full confirmatory configurations each contain
`24 * 6 * 15 * 5 = 10,800` method-problem runs. For BBOB, the sum of selected
dimensions is 80, corresponding to 1.44 billion FE over all methods. For
BBOB-LargeScale, the sum is 1,260, corresponding to 22.68 billion FE. These
costs should be considered before running the full campaigns.

## CEC 2013 Large-Scale Global Optimization

The CEC 2013 LSGO adapter uses an optional external `cec2013lsgo` package. The
adapter reads problem metadata and bounds from the wrapper; the driver rejects
any function whose reported dimension is not 1000.

The confirmatory configuration follows the published benchmark settings:

- 15 minimization functions;
- D=1000;
- 25 independent seeds per function;
- maximum 3,000,000 FE per run;
- checkpoints at 120,000, 600,000 and 3,000,000 FE.

Configuration: `configs/benchmarks/cec2013_confirmatory.json`. The
pre-confirmatory configuration `cec2013_preconfirmatory.json` selects functions
F1-F5, five seeds and the first official checkpoint budget of 120,000 FE.
This is an initial broad-scale pilot, not the complete CEC protocol. The full
configuration is 1,875 method-function-seed runs and 5.625 billion FE across
five methods. The objective functions are expected to return non-finite values when
queried outside their bounds in some wrappers; the present study applies the
same explicit clamp-and-restitution boundary map to all five algorithms. This
is a project-level algorithmic convention, not a property of the benchmark
functions themselves, and must be disclosed when presenting results.

**Licensing note:** commonly used `cec2013lsgo` wrappers are distributed under
GPL-3.0. Their source is not vendored here. Install an external wrapper only
after reviewing its license and implications for the research and
redistribution context. The adapter accepts the legacy
`Benchmark().get_info(fid)/get_function(fid)` interface and compatible
function-bound wrappers such as `Benchmark(fid).get_info()/eval()`.

## Discrete-PSO reference baselines

The reference variants are independent Python/NumPy implementations in
`python/cpso_serial/pso_baselines.py`. They are included as search-quality
comparators; they are not claims to implement every detail of SPSO-2007 or
SPSO-2011. Let `x_i`, `v_i`, `p_i` and `s_i` denote particle position,
velocity, personal best and social attractor. Random values are independent
and coordinate-wise uniform on [0,1].

### 1. Classic global-best PSO

\[
 v_i^{t+1}=v_i^t+c_1 r_{1,i}^t\odot(p_i^t-x_i^t)
                  +c_2 r_{2,i}^t\odot(g^t-x_i^t),\qquad
 x_i^{t+1}=x_i^t+v_i^{t+1},
\]
with `c1 = c2 = 2.0`, global-best information and velocity cap
`0.2 * (upper-lower)`. This is the canonical early gbest-style velocity
update, with the project initialization and boundary conventions specified
below.

### 2. Inertia-weight global-best PSO

The same global-best structure is used with `c1 = c2 = 2.05` and a linear
inertia schedule `w_t` decreasing from 0.9 to 0.4:

\[
 v_i^{t+1}=w_t v_i^t+c_1 r_{1,i}^t\odot(p_i^t-x_i^t)
                    +c_2 r_{2,i}^t\odot(g^t-x_i^t).
\]
The velocity cap is `0.2 * (upper-lower)`.

### 3. Constriction-factor global-best PSO

\[
 v_i^{t+1}=\chi\left[v_i^t+c_1 r_{1,i}^t\odot(p_i^t-x_i^t)
                         +c_2 r_{2,i}^t\odot(g^t-x_i^t)\right],
\]
where `c1 = c2 = 2.05`, `phi = c1+c2 > 4` and

\[
 \chi=\frac{2}{\phi-2+\sqrt{\phi^2-4\phi}}.
\]
For these parameters, `chi` is approximately 0.7298437881. No additional
velocity cap is applied to this variant; the constriction factor and boundary
map remain separate controls.

### 4. Inertia-weight local-best ring PSO

The inertia-weight update is used with `c1 = c2 = 2.05`, `w_t` decreasing
from 0.9 to 0.4, and a periodic ring neighborhood of radius one. Each particle
is attracted to the best personal-best point in its neighborhood. The velocity
cap is `0.2 * (upper-lower)`.

### Common initialization, update and FE conventions

All five methods use the same population size, search bounds, initial-velocity
fraction, problem/seed label and exact FE budget. The seed is passed into each
algorithm's own random-number generator; identical initial coordinates are
not guaranteed across C++ and NumPy implementations. Initialization consumes
exactly `N` objective evaluations. Subsequent updates are synchronous and
use a frozen generation snapshot. Confirmatory budgets are configured as
multiples of `N`, so no partial generation is needed in the external campaigns.

All methods use uniform initialization within the benchmark box, initial
velocity components sampled uniformly from `[-0.1, 0.1]` times the coordinate
width, and the same boundary transformation: clamp the position to the
violated bound and reverse the outward velocity with restitution 0.5. The
native CPSO uses a continuous-time analytic propagation between objective
observations; the four discrete variants use the velocity-position equations
above. Equal FE budget is guaranteed, but the algorithms' internal update
laws are not made mathematically equivalent.

The `classic_gbest`, `inertia_gbest`, `constriction_gbest` and
`inertia_lbest_ring` labels identify the explicit implementations in this
repository. They should not be relabelled as a fully standardized SPSO-2007 or
SPSO-2011 implementation without separately implementing and testing the
respective standard's initialization, neighborhood, random sampling and
boundary details.

## Installation and commands

Build the native library and install the core requirements as described in the
main README. For COCO:

```bash
python -m pip install -r requirements-benchmarks-coco.txt
PYTHONPATH=python python experiments/05_external_benchmarks/run_bbob.py \
  --config configs/benchmarks/bbob_smoke.json --dry-run
PYTHONPATH=python python experiments/05_external_benchmarks/run_bbob.py \
  --config configs/benchmarks/bbob_largescale_smoke.json --dry-run
```

For a real run, remove `--dry-run` after verifying package versions and
storage paths. For CEC 2013 LSGO, first review and install the optional GPL-3.0
wrapper, then use:

```bash
python -m pip install -r requirements-benchmarks-cec2013.txt
PYTHONPATH=python python experiments/05_external_benchmarks/run_cec2013_lsgo.py \
  --config configs/benchmarks/cec2013_smoke.json --dry-run
```

Do not launch confirmatory configurations until every smoke test has been
run with the actual installed benchmark package and the complete environment
has been pinned. Do not edit raw JSONL records manually. The JSONL sidecar is
resumable by run key; COCO observer data must be preserved alongside it.

## Verification status and limitations

- Local integration tests exercise the COCO and CEC drivers against controlled
  fake suite objects, including exact FE accounting and resume behavior.
- The official `coco-experiment`/`cocopp` packages were not installed in the
  offline development environment. Real COCO/BBOB and BBOB-LargeScale
  end-to-end execution is therefore still required.
- The external CEC wrapper was not installed; its adapter and driver were
  tested with controlled stubs, not real CEC objective functions.
- No external benchmark results are included in this repository snapshot.
- The custom-objective C ABI is version 6 and serial-only by design.
- The external quality campaigns do not establish speedup of CPSO's parallel
  execution architectures.
