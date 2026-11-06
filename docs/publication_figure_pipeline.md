# Publication figure pipeline

The figure pipeline is a **post-processing step**. It reads existing CSV/JSONL outputs and never launches optimization runs. All figures are written as vector PDF; no PNG, JPEG, or SVG deliverables are created by this script.

## Run

From the repository root, with the project virtual environment active and the `analysis` extra installed:

```bash
python analysis/postprocess_publication_figures.py \
  --results-root results \
  --output paper/figures/publication_stage_A
```

If the results are extracted under another directory, pass the directory containing `raw/` and `external/`, or its parent. The script recognizes nested `results/` and `risultati/` folders.

## Current figure set

1. `fig_01_synchronous_equivalence.pdf` — exact equality of best value, best position, and complete best-so-far history for paired synchronous executors.
2. `fig_02_strong_scaling.pdf` — paired speedup versus worker count, split by backend and synthetic objective cost.
3. `fig_03_parallel_efficiency.pdf` — speedup divided by worker count for the high-cost synthetic workload.
4. `fig_04_runtime_components.pdf` — measured execution-time decomposition; social-selection timing is not double-counted because it is part of propagation.
5. `fig_05_asynchronous_quality.pdf` — exploratory paired quality differences for asynchronous CPSO.
6. `fig_06_island_quality.pdf` — exploratory paired quality differences for the island model.
7. `fig_07_legacy_topology_quality_exploratory.pdf` — historical topology data from software 0.5.0; explicitly exploratory and never pooled with 0.7.x.
8. `fig_08_current_topology_quality.pdf` — generated automatically after the 0.7.3 seven-topology pilot or confirmatory campaign is complete.
9. `fig_09_particle_topology_schematics.pdf` — explanatory diagrams for all seven particle communication topologies.

A `FIGURE_CATALOGUE.md` file accompanies the PDFs and records figures that could not be generated because their source data are not yet present. For the archive supplied on 2026-11-06, figure 08 is expected to be absent until the new topology campaign has run.

## Interpretation controls

- Figures 01--04 use the 0.7.1 synchronous CPSO pilot provided in the result archive. The speedup experiment uses a synthetic workload, not BBOB/CEC objective calls.
- Figures 05--06 are five-seed exploratory pilots and are not confirmatory statistical evidence.
- Figure 07 uses the legacy 0.5.0 topology campaign and must be labelled historical/exploratory.
- Figure 08 is the current 0.7.3 topology expansion result. It prefers confirmatory records when present; otherwise it uses the pilot. For display, seed medians are formed within each function-dimension cell and dimensions are then aggregated within each function, so each plotted point represents one of the six function families rather than a run-level pseudo-replicate. Formal inference must still use the declared function-family hierarchy.
- Figure 09 is a schematic, not an experimental result. The von-Neumann drawing omits periodic wrap-around edges for visual clarity.
- Official COCO ECDFs and performance tables must be generated from COCO Observer output with `cocopp`; the CPSO JSONL/CSV files alone do not contain the complete official COCO post-processing data.

## Vector validation

The output is PDF with Matplotlib's PDF backend and embedded vector text/paths. Before submission, inspect the final PDFs at normal column width and verify fonts and line widths in the journal's typesetting pipeline. Avoid converting the plots to raster images when inserting them into LaTeX.
