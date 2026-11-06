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
- eleven algorithms (CPSO plus ten discrete-time, continuous-space PSO baselines).

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
- eleven algorithms (CPSO plus ten discrete-time, continuous-space PSO baselines).

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
`24 * 6 * 15 * 11 = 23,760` method-problem runs. For BBOB, the sum of selected
dimensions is 80, corresponding to 3.168 billion FE over all methods. For
BBOB-LargeScale, the sum is 1,260, corresponding to 49.896 billion FE. These
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
configuration is 4,125 method-function-seed runs and 12.375 billion FE across
eleven methods. The objective functions are expected to return non-finite values when
queried outside their bounds in some wrappers; the present study applies the
same explicit clamp-and-restitution boundary map to all eleven algorithms. This
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
`python/cpso_serial/pso_baselines.py` and `python/cpso_serial/pso_advanced.py`.
They are serial search-quality comparators; their objective callbacks are not
used to claim speedup from the native parallel execution modes. Let `x_i`,
`v_i`, `p_i` and `s_i` denote particle position, velocity, personal best and
social attractor. Algorithm-specific update laws and randomization are
summarized below. The four classical variants are explicit baselines, whereas
SPSO-2011, FIPS, CSO and DMS-PSO have their own documented update mechanisms.

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

### 5. Comprehensive-learning PSO (CLPSO)

CLPSO uses per-coordinate exemplars selected by pairwise tournaments over
personal-best values. The learning probability is

\[
P_{c,i}=0.05+0.45\frac{\exp(10i/(N-1))-1}{\exp(10)-1},
\qquad i=0,\ldots,N-1,
\]

with acceleration 1.49445, inertia decreasing from 0.9 to 0.4, velocity cap
0.2 times the coordinate width, and exemplar refresh after seven consecutive
generations without personal-best improvement. The implementation uses a
frozen generation snapshot; this prevents accidental asynchronous information
flow. The common project boundary map is an explicit adaptation and is not
claimed to reproduce every boundary convention of the original study. See
Liang et al. (2006), DOI `10.1109/TEVC.2005.857610`.

### 6. Social-learning PSO (SL-PSO)

SL-PSO (Cheng and Jin, 2015) sorts the swarm from worst to best. Every particle
except the best may learn from demonstrators with better current fitness. With
rank \(i=0,\ldots,N-2\), dimension \(D\), reference size \(M=100\),
\(\alpha=0.5\), and \(\beta=0.01\), the learning probability and
social-influence factor are

\[
P_i=\left(1-\frac{i}{N}\right)^{\alpha\log(\lceil D/M\rceil)},
\qquad \gamma=\beta\frac{D}{M}.
\]

When learning is activated, each coordinate independently chooses a better
ranked demonstrator and the displacement is updated as

\[
\Delta x_{i,j}^{t+1}=r_1\Delta x_{i,j}^{t}
+r_2(x_{k(i,j),j}^{t}-x_{i,j}^{t})
+r_3\gamma(\bar{x}_{j}^{t}-x_{i,j}^{t}),
\]

where \(r_1,r_2,r_3\sim U(0,1)\), and \(\bar{x}^{t}\) is the current
swarm mean. If learning is not activated, the particle is unchanged. The best
particle is not an imitator. The code uses the same fixed population across
methods to preserve exact FE parity; the paper's recommended population
\(M+\lceil D/10\rceil\) is therefore not used in this matched-population
comparison. The project also applies its common clamp-and-restitution boundary
map. This adaptation is disclosed rather than presented as a byte-for-byte
reproduction. See Cheng and Jin (2015), DOI
`10.1016/j.ins.2014.08.039`.



### 7. SPSO-2011

SPSO-2011 uses an adaptive directed random informant graph and a geometric
hypersphere update. The default uses `K=3` random outgoing informant links per
particle plus self-information, `w=1/(2 ln 2)` and `c=0.5+ln 2`. Its gravity
centre is `g=x+c(p+l-2x)/3`, with the specified two-point case when `l=p`.
The default hypersphere sampler uses a uniform direction and uniform radius.
Initial velocity is uniform between the lower-bound displacement and
upper-bound displacement in a normalized search box; an out-of-box component
is clipped and its velocity component multiplied by `-0.5`. See
`docs/stage_c_pso_baselines.md` and Clerc's SPSO description for the exact
implementation convention.

### 8. Fully Informed PSO (FIPS)

FIPS sums independent random attraction terms from every member of the
neighborhood. The current baseline fixes the neighborhood to the two adjacent
particles in a periodic ring and uses `phi=4.1` with Clerc's constriction
factor. This is a deliberately specified FIPS/ring configuration, not a
claim that FIPS is tied to only one topology.

### 9. Competitive Swarm Optimizer (CSO)

CSO randomly forms pairs; the winner is retained and the loser learns from
the winner, its previous velocity and the previous population mean. The
implementation uses `phi=0.1`, zero initial velocity and an even population.
Only the losers are evaluated after initialization, so each complete
competition generation consumes `N/2` FE. All histories and checkpoints are
indexed by actual FE, not an assumed full-generation counter. The project
boundary projection/restitution is an explicitly disclosed adaptation.

### 10. Dynamic Multi-Swarm PSO (DMS-PSO)

DMS-PSO partitions the population into small groups, uses the current group's
best personal-best as the social attractor, and randomly regroups the full
population every five generations. The implementation uses target group size
three, linearly decreasing inertia `0.9 -> 0.2`, and merges a final singleton
remainder into the preceding group. It is the basic regrouping method without
a local-search or cooperative-learning extension.

### Common initialization, update and FE conventions

All ten discrete-time, continuous-space PSO methods use the same population
size, search bounds, problem/seed label and exact FE budget as the native CPSO.
The seed is passed into each algorithm's own random-number generator; identical
initial coordinates are not guaranteed across C++ and NumPy implementations.
Initialization consumes exactly `N` objective evaluations. Subsequent updates
use a frozen generation snapshot, except that CSO updates/evaluates only the
loser of each pair, and therefore advances by `N/2` FE per complete competition
generation. Histories and checkpoints are indexed by actual FE. Confirmatory
budgets are configured as multiples of `N`, so all methods can consume the exact
same total FE budget.

All methods initialize positions uniformly within the benchmark box. The classical
PSO methods, CLPSO, FIPS and DMS-PSO sample initial velocity components uniformly
from the configured fraction of each coordinate width (0.1 by default); SL-PSO
and CSO start with zero displacement; SPSO-2011 uses its own normalized-box
initialization.
The shared project boundary transformation is used by the methods that do not
require an algorithm-specific boundary rule. SPSO-2011 uses its standard
coordinatewise clip and half-velocity reversal. CSO uses the project
clamp-and-restitution adaptation, which is declared in its metadata. The native CPSO
uses continuous-time analytic propagation between objective observations; the
discrete variants use their own velocity/displacement update equations. Equal
FE budget is guaranteed, but the algorithms' internal update laws are not made
mathematically equivalent.

The `classic_gbest`, `inertia_gbest`, `constriction_gbest` and
`inertia_lbest_ring` labels identify explicit classical implementations, not
formal SPSO standards. SPSO-2011 is implemented as a separate variant with its
standard geometric update and default informant scheme, but no bitwise match
with third-party software is claimed. CLPSO, SL-PSO, FIPS, CSO and DMS-PSO
retain the implementation adaptations documented in `docs/stage_c_pso_baselines.md`.
The ten-method set is representative, not exhaustive SOTA across continuous
optimization.

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
- The custom-objective C ABI is version 7 and serial-only by design.
- The external quality campaigns do not establish speedup of CPSO's parallel
  execution architectures.
