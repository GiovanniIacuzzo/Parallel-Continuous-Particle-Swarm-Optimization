# Parallel CPSO — clean research workspace

C++20 / Python continuous PSO (Orlando–Ricciardello analytic kernel), asynchronous/island variants, ten discrete PSO reference baselines, research campaign runners, and COCO/CEC adapters. This is a **research implementation**, not a bit-for-bit reproduction of every undocumented choice in the source article.

## Start here (Mac M2)

```bash
python3 -m venv .venv && source .venv/bin/activate
python -m pip install -e ".[analysis,test]"
cmake -S . -B build-m2 -DCMAKE_BUILD_TYPE=Release -DCPSO_BUILD_TESTS=ON
cmake --build build-m2 --parallel 4
export CPSO_SERIAL_LIBRARY="$PWD/build-m2/lib/libcpso_serial.dylib"
export PYTHONPATH="$PWD/python"
ctest --test-dir build-m2 --output-on-failure
python -m pytest -q
```

## Current scientific task: dt statistical selection

Two active dt configs are in `configs/campaigns/`: a **historical** 100-point sweep and a **current** independent 30-seed selection. The second is the current statistical study; if it has already run, **do not rerun it**: keep its existing `results/` records and send them for review.

```bash
python experiments/06_dt_sensitivity/run_dt_sensitivity.py \
  --config configs/campaigns/cpso_dt_statistical_selection_m2.json \
  --output results/raw/dt/cpso_dt_statistical_selection_m2 --dry-run

caffeinate -i python experiments/06_dt_sensitivity/run_dt_sensitivity.py \
  --config configs/campaigns/cpso_dt_statistical_selection_m2.json \
  --output results/raw/dt/cpso_dt_statistical_selection_m2

python analysis/analyze_dt_statistical.py \
  --input results/raw/dt/cpso_dt_statistical_selection_m2 \
  --output-dir results/analysis/dt/cpso_dt_statistical_selection_m2
```

**Existing outputs on your Mac may still be in the earlier `results/raw/cpso_...` paths.** Do not move or overwrite them while a campaign is running. All new commands use the grouped layout. The raw manifest records which config generated each run.

## Research flow

1. `experiments/06_dt_sensitivity/run_dt_sensitivity.py`: serial dt calibration, then `analysis/analyze_dt_statistical.py` (selection) or `analysis/analyze_dt_sweep_100.py` (historical exploratory study).
2. `experiments/04_architectures/run_research_campaign.py`: paired parallel architecture/topology campaigns; campaign configs are deliberately archived pending a locked dt.
3. `experiments/05_external_benchmarks/run_bbob.py` and `run_cec2013_lsgo.py`: external method comparisons. Full confirmatory configurations are archived until the protocol and computational budget are finalized; only smoke configs remain live.
4. `plot_results/run_all.py`: one command to build a **PDF-only** figure atlas from recorded outputs. You may also invoke the specialist plotters (`plot_dt_atlas.py`, `plot_internal_atlas.py`, `plot_external_atlas.py`).

```bash
python plot_results/run_all.py --kind dt
python plot_results/run_all.py --kind parallel
python plot_results/run_all.py --kind external
```

Automatic files are always under `results/`. The built-in CLI rejects `--output paper/figures/...`, even if explicitly supplied. **Only the manuscript author copies chosen figures into `paper/`.** This archive contains no manuscript files and the cleanup has not modified any.

The original full README, obsolete plotters and historical configs are preserved in the companion recovery ZIP. See `CLEANUP_REPORT.md` for the exact file-by-file inventory.

## Safe update of an existing local checkout

The supplied ZIP is a full clean source snapshot, **not** a patch that deletes legacy files by extraction. Do not simply unzip it over your live project, or old scripts will remain.

Extract beside your checkout, then preview/apply the migration:

```bash
python /path/to/cpso_cleaned/scripts/apply_cleanup.py \
    --target /path/to/Parallel-Continuous-Particle-Swarm-Optimization
python /path/to/cpso_cleaned/scripts/apply_cleanup.py \
    --target /path/to/Parallel-Continuous-Particle-Swarm-Optimization --apply
```

The migration backs up replaced and removed source files under `results/cleanup_backups/`. It never reads, overwrites or removes `paper/`, `results/raw`, or existing scientific runs; it preserves `.git` and unknown local files. Commit the resulting source changes only after checking `git diff` and rerunning the C++ and Python tests.
