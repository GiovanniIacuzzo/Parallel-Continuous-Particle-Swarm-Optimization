#!/usr/bin/env python3
"""Build the exact result figures referenced by the current CPSO manuscript.

Consumes recorded M2 campaign outputs only. No optimizer is executed and no
missing observation is imputed. All figures built from the supplied M2 runs
are explicitly labelled exploratory. Official COCO ECDFs are never replaced
with JSONL final-value ranks.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE.parent) not in sys.path:
    sys.path.insert(0, str(HERE.parent))

from plot_results.common import (  # noqa: E402
    COLORS, PALETTE, as_float, as_int, configure_style, manifest_evidence,
    read_csv, read_json, save_figure,
)
from plot_results import plot_internal as internal_plots  # noqa: E402

LOG_FLOOR = 1.0e-16
EXPECTED_FUNCTIONS = ("sphere", "rastrigin", "rosenbrock", "ackley", "griewank", "salomon")
PAPER_FIGURES = [
    ("fig_sync_equivalence.pdf", "RQ1", "Exact synchronous trajectory-equivalence audit"),
    ("fig_strong_scaling.pdf", "RQ2", "Strong scaling of synchronous backends"),
    ("fig_async_island_anytime.pdf", "RQ3", "Best-so-far curves for asynchronous and island modes"),
    ("fig_topology_quality.pdf", "RQ4", "Paired quality effects for the seven social topologies"),
    ("fig_coco_bbob_ecdf.pdf", "RQ5", "Official COCO BBOB ECDF/runtime-to-target output"),
    ("fig_coco_bbob_ls_ecdf.pdf", "RQ5", "Official COCO BBOB-LargeScale ECDF/runtime-to-target output"),
    ("fig_cec2013_checkpoints.pdf", "RQ5", "CEC 2013 LSGO checkpoint comparison"),
    ("fig_dt_sensitivity_summary.pdf", "Appendix E", "Paired final-quality sensitivity to Delta-t"),
    ("fig_runtime_decomposition.pdf", "Appendix E", "Measured runtime-component decomposition"),
    ("fig_internal_function_forest.pdf", "Appendix E", "Function/dimension paired-effect forest plot"),
    ("fig_migration_diagnostics.pdf", "Appendix E", "Attempted/accepted island migrations"),
    ("fig_external_function_heatmaps.pdf", "Appendix E", "BBOB pilot median within-instance ranks by function and dimension"),
    ("fig_cec_function_breakdown.pdf", "Appendix E", "CEC 2013 LSGO per-function checkpoint breakdown"),
]
EXTERNAL_FIGURES = {
    "fig_coco_bbob_ecdf.pdf", "fig_coco_bbob_ls_ecdf.pdf",
    "fig_cec2013_checkpoints.pdf", "fig_cec_function_breakdown.pdf",
}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"JSONL record must be an object: {path}:{line_number}")
            rows.append(row)
    return rows


def _campaign_records(campaign: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted(campaign.rglob("runs.jsonl")):
        if "figures" in path.parts:
            continue
        rel = path.parent.relative_to(campaign)
        stage = rel.parts[0] if len(rel.parts) >= 2 else "study"
        study = rel.parts[1] if len(rel.parts) >= 2 else path.parent.name
        for raw in _read_jsonl(path):
            row = dict(raw)
            row.setdefault("stage", stage)
            row.setdefault("study", study)
            records.append(row)
    if records:
        return records
    csv_path = campaign / "campaign_runs.csv"
    if csv_path.is_file():
        with csv_path.open("r", newline="", encoding="utf-8-sig") as stream:
            for raw in csv.DictReader(stream):
                row: dict[str, Any] = dict(raw)
                for field in ("best_history", "history_evaluations"):
                    value = row.get(field)
                    if isinstance(value, str) and value.strip():
                        try:
                            row[field] = json.loads(value)
                        except json.JSONDecodeError:
                            row[field] = []
                records.append(row)
    return records


def _history(row: dict[str, Any]) -> tuple[np.ndarray, np.ndarray] | None:
    x, y = row.get("history_evaluations"), row.get("best_history")
    if not isinstance(x, list) or not isinstance(y, list) or not x or len(x) != len(y):
        return None
    xx, yy = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    valid = np.isfinite(xx) & np.isfinite(yy)
    xx, yy = xx[valid], yy[valid]
    if not xx.size:
        return None
    order = np.argsort(xx, kind="stable")
    xx, yy = xx[order], np.minimum.accumulate(yy[order])
    ux: list[float] = []
    uy: list[float] = []
    for fe in np.unique(xx):
        ux.append(float(fe))
        uy.append(float(np.min(yy[xx == fe])))
    return np.asarray(ux), np.asarray(uy)


def _step_interpolate(row: dict[str, Any], grid: np.ndarray) -> np.ndarray | None:
    pair = _history(row)
    if pair is None:
        return None
    x, y = pair
    idx = np.searchsorted(x, grid, side="right") - 1
    idx = np.clip(idx, 0, len(y) - 1)
    return y[idx]


def _label_method(name: str) -> str:
    labels = {
        "serial_global": "Synchronous global reference",
        "serial_ring": "Synchronous ring reference",
        "async_global_w2": "Async global · 2 workers",
        "async_global_w4": "Async global · 4 workers",
        "async_ring_w2": "Async ring · 2 workers",
        "async_ring_w4": "Async ring · 4 workers",
        "island_global_m2_no_migration": "2 islands · no migration",
        "island_global_m4_ring_mig20": "4 islands · ring migration",
        "island_global_m4_full_mig20": "4 islands · fully connected migration",
        "island_ring_m4_ring_mig20": "4 islands · ring intra / ring migration",
    }
    return labels.get(name, name.replace("_", " "))


def _display_bbob_method(name: str) -> str:
    mapping = {
        "cpso_continuous": "CPSO",
        "pso_classic_gbest": "PSO-gbest",
        "pso_clpso": "CLPSO",
        "pso_constriction_gbest": "Constr.-gbest",
        "pso_cso": "CSO",
        "pso_dms": "DMS-PSO",
        "pso_fips": "FIPS",
        "pso_inertia_gbest": "Inertia-gbest",
        "pso_inertia_lbest_ring": "Inertia-ring",
        "pso_slpso": "SLPSO",
        "pso_spso2011": "SPSO2011",
    }
    return mapping.get(name, name.replace("pso_", "").replace("_", "-"))


def _cell_text_color(value: float, vmin: float, vmax: float, cmap_name: str) -> str:
    if not np.isfinite(value):
        return COLORS["ink"]
    normed = 0.5 if vmax <= vmin else (value - vmin) / (vmax - vmin)
    r, g, b, _ = plt.get_cmap(cmap_name)(normed)
    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return COLORS["white"] if luminance < 0.45 else COLORS["ink"]


def _save_named(fig: plt.Figure, output: Path, filename: str) -> list[Path]:
    return list(save_figure(fig, output, Path(filename).stem))


def _copy_named(source_paths: list[Path], output: Path, target_stem: str) -> list[Path]:
    output.mkdir(parents=True, exist_ok=True)
    copied = []
    for source in source_paths:
        if source.suffix.lower() not in {".pdf", ".png", ".svg"}:
            continue
        target = output / f"{target_stem}{source.suffix.lower()}"
        shutil.copy2(source, target)
        copied.append(target)
    return copied


def plot_async_island_anytime(dynamics: Path, output: Path) -> list[Path]:
    records = _campaign_records(dynamics)
    if not records:
        raise FileNotFoundError(f"No dynamics runs.jsonl/campaign_runs.csv found in {dynamics}")

    async_rows = [r for r in records if str(r.get("architecture")) in {"serial", "asynchronous"}
                  and str(r.get("benchmark")) != "sphere_work" and _history(r) is not None
                  and "async" in str(r.get("study", ""))]
    island_rows = [r for r in records if str(r.get("architecture")) in {"serial", "islands"}
                   and str(r.get("benchmark")) != "sphere_work" and _history(r) is not None
                   and "island" in str(r.get("study", ""))]
    if not async_rows or not island_rows:
        raise ValueError("Both asynchronous and island run histories are required")

    configure_style()
    dims = sorted({as_int(r, "dimension", -1) for r in async_rows + island_rows if as_int(r, "dimension", -1) > 0})
    if not dims:
        raise ValueError("No valid dimensions in dynamics records")
    async_methods = ["serial_global", "async_global_w2", "async_global_w4",
                     "serial_ring", "async_ring_w2", "async_ring_w4"]
    island_methods = ["serial_global", "island_global_m2_no_migration", "island_global_m4_ring_mig20",
                      "island_global_m4_full_mig20", "serial_ring", "island_ring_m4_ring_mig20"]
    colors = {
        "serial_global": COLORS["ink"], "serial_ring": COLORS["grey"],
        "async_global_w2": COLORS["blue"], "async_global_w4": COLORS["teal"],
        "async_ring_w2": COLORS["gold"], "async_ring_w4": COLORS["purple"],
        "island_global_m2_no_migration": COLORS["blue"],
        "island_global_m4_ring_mig20": COLORS["teal"],
        "island_global_m4_full_mig20": COLORS["gold"],
        "island_ring_m4_ring_mig20": COLORS["purple"],
    }
    fig, axes = plt.subplots(max(1, len(dims)), 2, figsize=(12.0, 3.15 * max(1, len(dims))), squeeze=False)
    for row_idx, dimension in enumerate(dims):
        for col_idx, (panel_rows, methods, title) in enumerate([
            (async_rows, async_methods, "Event-driven asynchronous mode"),
            (island_rows, island_methods, "Island model"),
        ]):
            ax = axes[row_idx, col_idx]
            subset = [r for r in panel_rows if as_int(r, "dimension", -1) == dimension]
            valid_rows = [r for r in subset if str(r.get("method", "")) in methods]
            available_methods = [m for m in methods if any(str(r.get("method")) == m for r in valid_rows)]
            missing_methods = sorted(set(methods) - set(available_methods))
            if missing_methods:
                raise ValueError(f"Incomplete method coverage for {title}, D={dimension}; missing={missing_methods}")
            if not valid_rows:
                raise ValueError(f"No anytime records for {title}, D={dimension}")
            # This pilot is designed with six functions and five paired seeds.
            present_functions = {str(r.get("benchmark")) for r in valid_rows}
            missing_functions = sorted(set(EXPECTED_FUNCTIONS) - present_functions)
            unexpected_functions = sorted(present_functions - set(EXPECTED_FUNCTIONS))
            if missing_functions or unexpected_functions:
                raise ValueError(
                    f"Unexpected function coverage for {title}, D={dimension}; "
                    f"missing={missing_functions}, unexpected={unexpected_functions}"
                )
            min_fe = max(float(_history(r)[0][0]) for r in valid_rows)
            max_fe = min(float(_history(r)[0][-1]) for r in valid_rows)
            if max_fe <= min_fe:
                raise ValueError(f"No common FE interval for {title}, D={dimension}")
            grid = np.linspace(min_fe, max_fe, 180)
            benchmark_names = sorted(present_functions)
            for method in available_methods:
                family_curves = []
                for benchmark in benchmark_names:
                    cell_rows = [r for r in valid_rows if str(r.get("benchmark")) == benchmark
                                 and str(r.get("method")) == method]
                    seeds = [as_int(r, "seed", -1) for r in cell_rows]
                    if len(cell_rows) != 5 or len(set(seeds)) != 5:
                        raise ValueError(
                            f"Expected exactly five distinct paired seeds for "
                            f"{method}/{benchmark}/D={dimension}; got {len(cell_rows)} rows, "
                            f"{len(set(seeds))} distinct seeds"
                        )
                    run_curves = []
                    for record in cell_rows:
                        curve = _step_interpolate(record, grid)
                        if curve is not None:
                            run_curves.append(np.log10(np.maximum(curve, LOG_FLOOR)))
                    if run_curves:
                        family_curves.append(np.nanmedian(np.vstack(run_curves), axis=0))
                if len(family_curves) != len(EXPECTED_FUNCTIONS):
                    raise ValueError(f"Method {method} lacks complete six-function coverage at D={dimension}")
                matrix = np.vstack(family_curves)
                median = np.nanmedian(matrix, axis=0)
                q1, q3 = np.nanquantile(matrix, [0.25, 0.75], axis=0)
                color = colors.get(method, PALETTE[available_methods.index(method) % len(PALETTE)])
                ax.plot(grid, median, color=color, lw=1.45, label=_label_method(method))
                ax.fill_between(grid, q1, q3, color=color, alpha=0.10, linewidth=0)
            ax.set_title(f"{title} · D={dimension}")
            ax.set_xlabel("Function evaluations (FE)")
            ax.set_ylabel(r"$\log_{10}(f_{best})$ (native functions have $f^*=0$)")
            ax.grid(axis="y")
            ax.legend(frameon=False, fontsize=6.0, ncol=2)
    fig.suptitle("Best-so-far anytime profiles at matched FE budgets", y=1.005, weight="bold")
    fig.text(0.5, 0.003,
             "Within each function: median over five paired seeds; line: median across six fixed functions; band: IQR across function medians. Values are floored at 1e-16 before log10. Exploratory M2 pilot.",
             ha="center", fontsize=6.4, color=COLORS["grey"])
    fig.tight_layout(rect=(0, 0.025, 1, 0.985))
    return _save_named(fig, output, "fig_async_island_anytime.pdf")


def plot_dt_summary(dt_campaign: Path, output: Path) -> list[Path]:
    data = _read_jsonl(dt_campaign / "runs.jsonl")
    if not data:
        raise FileNotFoundError(f"No dt runs.jsonl found in {dt_campaign}")
    config_obj = read_json(dt_campaign / "study_config.json")
    config = config_obj.get("study_config", config_obj.get("plan", config_obj))
    reference_dt = as_float(data[0], "reference_dt", as_float(config, "reference_dt", math.nan))
    observed_benchmarks = {str(r.get("benchmark")) for r in data}
    benchmark_order = [b for b in EXPECTED_FUNCTIONS if b in observed_benchmarks]
    dimensions = sorted({as_int(r, "dimension", -1) for r in data if as_int(r, "dimension", -1) > 0})
    dt_values = sorted({as_float(r, "dt") for r in data if math.isfinite(as_float(r, "dt"))})
    if not math.isfinite(reference_dt) or reference_dt not in dt_values:
        raise ValueError(f"Reference dt={reference_dt} is absent from recorded data")
    if not benchmark_order or not dimensions or not dt_values:
        raise ValueError("Incomplete dt dimensions/functions/value set")
    lookup: dict[tuple[str, int, int, float], float] = {}
    for r in data:
        key = (str(r.get("benchmark")), as_int(r, "dimension", -1), as_int(r, "seed", -1), as_float(r, "dt"))
        value = as_float(r, "best_value")
        if key in lookup:
            raise ValueError(f"Duplicate dt observation for {key}")
        if math.isfinite(value):
            lookup[key] = value
    expected_seeds = sorted({as_int(r, "seed", -1) for r in data})
    if set(benchmark_order) != set(EXPECTED_FUNCTIONS):
        raise ValueError(f"Unexpected dt function set: {sorted(observed_benchmarks)}")
    if len(expected_seeds) != 5:
        raise ValueError(f"Expected five paired dt seeds, found {expected_seeds}")
    expected_cells = len(benchmark_order) * len(dimensions) * len(expected_seeds) * len(dt_values)
    if len(lookup) != expected_cells:
        raise ValueError(f"Incomplete dt grid: {len(lookup)} finite cells, expected {expected_cells}")

    configure_style()
    matrices: list[np.ndarray] = []
    spread_q1: list[np.ndarray] = []
    spread_q3: list[np.ndarray] = []
    for dim in dimensions:
        mat = np.full((len(benchmark_order), len(dt_values)), np.nan)
        q1_mat = np.full_like(mat, np.nan)
        q3_mat = np.full_like(mat, np.nan)
        for i, benchmark in enumerate(benchmark_order):
            for j, dt in enumerate(dt_values):
                deltas = []
                for seed in expected_seeds:
                    val = lookup[(benchmark, dim, seed, dt)]
                    ref = lookup[(benchmark, dim, seed, reference_dt)]
                    deltas.append(math.log10(max(val, LOG_FLOOR)) - math.log10(max(ref, LOG_FLOOR)))
                mat[i, j] = float(np.median(deltas))
                q1_mat[i, j], q3_mat[i, j] = np.quantile(deltas, [0.25, 0.75])
        matrices.append(mat)
        spread_q1.append(q1_mat)
        spread_q3.append(q3_mat)

    finite_parts = [m[np.isfinite(m)] for m in matrices if np.isfinite(m).any()]
    finite = np.concatenate(finite_parts) if finite_parts else np.asarray([], dtype=float)
    vmax = max(0.05, float(np.quantile(np.abs(finite), 0.95))) if finite.size else 1.0
    cmap_name = "RdBu_r"
    fig, axes = plt.subplots(len(dimensions), 1, figsize=(10.4, 2.15 * len(benchmark_order) + 0.45 * len(dimensions)), squeeze=False, constrained_layout=True)
    image = None
    ref_idx = dt_values.index(reference_dt)
    for ax, dim, mat in zip(axes[:, 0], dimensions, matrices):
        image = ax.imshow(mat, aspect="auto", cmap=cmap_name, vmin=-vmax, vmax=vmax)
        ax.set_title(f"D={dim}")
        ax.set_yticks(np.arange(len(benchmark_order)), [b.capitalize() for b in benchmark_order])
        ax.set_xticks(np.arange(len(dt_values)), [f"{v:g}" for v in dt_values], rotation=35, ha="right")
        ax.set_xlabel(r"$\Delta t$")
        if ax is axes[0, 0]:
            ax.set_ylabel("Benchmark function")
        ax.set_xticks(np.arange(-0.5, len(dt_values), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(benchmark_order), 1), minor=True)
        ax.grid(which="minor", color=COLORS["white"], linewidth=0.7)
        ax.tick_params(which="minor", bottom=False, left=False)
        ax.add_patch(plt.Rectangle((ref_idx - 0.5, -0.5), 1.0, len(benchmark_order), fill=False, edgecolor=COLORS["ink"], linewidth=1.2, linestyle="--"))
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                if np.isfinite(mat[i, j]):
                    ax.text(j, i, f"{mat[i, j]:+.1f}", ha="center", va="center", fontsize=5.8, color=_cell_text_color(mat[i, j], -vmax, vmax, cmap_name))
    if image is not None:
        cb = fig.colorbar(image, ax=list(axes[:, 0]), shrink=0.96, pad=0.02)
        cb.set_label(r"Median paired $\Delta \log_{10}(f_{best})$ vs. reference $\Delta t=%.3g$" % reference_dt)
    fig.suptitle(r"CPSO time-step sensitivity (paired M2 screening)", y=1.01, weight="bold")
    fig.text(0.5, -0.01,
             "Heatmaps report median paired log-objective differences across five seeds; negative values favour the tested step. Dashed outline = reference step. Exploratory M2 screening.",
             ha="center", fontsize=6.4, color=COLORS["grey"])
    out = _save_named(fig, output, "fig_dt_sensitivity_summary.pdf")

    dt_rows = []
    for dim, mat, q1_mat, q3_mat in zip(dimensions, matrices, spread_q1, spread_q3):
        for i, benchmark in enumerate(benchmark_order):
            for j, dt in enumerate(dt_values):
                dt_rows.append({
                    "dimension": dim, "benchmark": benchmark, "dt": dt,
                    "reference_dt": reference_dt,
                    "median_paired_delta_log10": float(mat[i, j]),
                    "q25_paired_delta_log10": float(q1_mat[i, j]),
                    "q75_paired_delta_log10": float(q3_mat[i, j]),
                    "n_paired_seeds": len(expected_seeds),
                    "reference_is_optimal": "NOT_ESTABLISHED",
                })
    _write_csv(output / "dt_sensitivity_summary.csv", dt_rows)
    return out




def _paired_cell_effect(rows: list[dict[str, Any]], method: str, reference: str) -> dict[tuple[str, int], list[float]]:
    lookup: dict[tuple[str, int, int, str], float] = {}
    for r in rows:
        value = as_float(r, "best_value")
        key = (str(r.get("benchmark")), as_int(r, "dimension", -1), as_int(r, "seed", -1), str(r.get("method")))
        if key in lookup:
            raise ValueError(f"Duplicate paired objective record for {key}")
        if math.isfinite(value):
            lookup[key] = math.log10(max(value, LOG_FLOOR))
    effects: dict[tuple[str, int], list[float]] = defaultdict(list)
    for (benchmark, dim, seed, name), value in lookup.items():
        if name == method:
            ref = lookup.get((benchmark, dim, seed, reference), math.nan)
            if math.isfinite(ref):
                effects[(benchmark, dim)].append(value - ref)
    return effects


def _hierarchical_bootstrap(effects: dict[tuple[str, int], list[float]], *, seed: int, n_boot: int = 3000) -> tuple[float, float, float, list[tuple[str, int, float]]]:
    cells = [(key, vals) for key, vals in sorted(effects.items()) if vals]
    if not cells:
        return math.nan, math.nan, math.nan, []
    cell_medians = [(key[0], key[1], float(np.median(vals))) for key, vals in cells]
    estimate = float(np.median([v for _, _, v in cell_medians]))
    rng = np.random.default_rng(seed)
    samples = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        sampled_cells = []
        for _, vals in cells:
            arr = np.asarray(vals, dtype=float)
            sampled_cells.append(float(np.median(rng.choice(arr, size=len(arr), replace=True))))
        samples[b] = float(np.median(sampled_cells))
    return estimate, float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975)), cell_medians


def plot_internal_function_forest(dynamics: Path, topology: Path, output: Path) -> list[Path]:
    dyn = _campaign_records(dynamics)
    topo = _campaign_records(topology)
    if not dyn or not topo:
        raise FileNotFoundError("Dynamics and topology run records are both required for function-effect forest")
    definitions: list[tuple[str, dict[str, dict[tuple[str, int], list[float]]]]] = []
    async_specs: dict[str, dict[tuple[str, int], list[float]]] = {}
    async_rows = [r for r in dyn if "async" in str(r.get("study", ""))]
    for method in ("async_global_w2", "async_global_w4", "async_ring_w2", "async_ring_w4"):
        reference = "serial_ring" if "ring" in method else "serial_global"
        if any(str(r.get("method")) == method for r in async_rows):
            async_specs[f"{_label_method(method)} vs {_label_method(reference)}"] = _paired_cell_effect(async_rows, method, reference)
    definitions.append(("Asynchronous modes", async_specs))

    island_rows = [r for r in dyn if "island" in str(r.get("study", ""))]
    island_specs: dict[str, dict[tuple[str, int], list[float]]] = {}
    for method, reference in (
        ("island_global_m2_no_migration", "serial_global"),
        ("island_global_m4_ring_mig20", "serial_global"),
        ("island_global_m4_full_mig20", "serial_global"),
        ("island_ring_m4_ring_mig20", "serial_ring"),
    ):
        if any(str(r.get("method")) == method for r in island_rows):
            island_specs[f"{_label_method(method)} vs {_label_method(reference)}"] = _paired_cell_effect(island_rows, method, reference)
    definitions.append(("Island configurations", island_specs))

    topo_specs: dict[str, dict[tuple[str, int], list[float]]] = {}
    topology_labels = {
        "ring": "Ring", "grid_von_neumann": "Von Neumann", "wheel": "Wheel",
        "random": "Random", "small_world": "Watts–Strogatz", "scale_free": "Barabási–Albert",
    }
    for topology_name, label in topology_labels.items():
        method = f"serial_{topology_name}"
        if any(str(r.get("method")) == method for r in topo):
            topo_specs[f"{label} vs global-best"] = _paired_cell_effect(topo, method, "serial_global")
    definitions.append(("Social topologies", topo_specs))

    total_specs = sum(len(specs) for _, specs in definitions)
    if not total_specs:
        raise ValueError("No paired function/dimension effects could be reconstructed")
    configure_style()
    fig, axes = plt.subplots(3, 1, figsize=(9.1, max(7.4, 0.60 * total_specs + 2.2)), squeeze=False)
    ax_idx = 0
    for panel_title, specs in definitions:
        ax = axes[ax_idx, 0]
        ax_idx += 1
        if not specs:
            ax.set_axis_off()
            continue
        labels = list(specs)
        for i, label in enumerate(labels):
            estimate, low, high, cells = _hierarchical_bootstrap(specs[label], seed=20261108 + i + 71 * ax_idx)
            if not math.isfinite(estimate):
                continue
            color = PALETTE[i % len(PALETTE)]
            offsets = np.linspace(-0.11, 0.11, max(1, len(cells)))
            for offset, (benchmark, dim, value) in zip(offsets, cells):
                ax.scatter(value, i + offset, s=12, color=color, alpha=0.28, linewidths=0, zorder=2)
            ax.errorbar(estimate, i, xerr=[[max(0.0, estimate - low)], [max(0.0, high - estimate)]],
                        fmt="o", ms=4.1, capsize=2.4, color=color, zorder=4)
        ax.axvline(0.0, color=COLORS["grey"], linestyle="--", linewidth=0.8)
        ax.set_yticks(np.arange(len(labels)), labels, fontsize=6.1)
        ax.invert_yaxis()
        ax.set_title(panel_title, loc="left", fontsize=8.3, weight="bold")
        ax.set_xlabel(r"Median paired $\Delta\log_{10}(f_{best})$ vs. matched serial reference")
        ax.grid(axis="x")
    fig.suptitle("Paired objective effects by function and dimension", y=1.005, weight="bold")
    fig.text(0.5, 0.003,
             "Small points: fixed function×dimension cell medians. Large points/whiskers: median and 95% bootstrap interval resampling paired seeds within each fixed cell. Exploratory M2 pilots.",
             ha="center", fontsize=6.2, color=COLORS["grey"])
    fig.tight_layout(rect=(0, 0.025, 1, 0.985))
    return _save_named(fig, output, "fig_internal_function_forest.pdf")


def plot_migration_diagnostics(dynamics: Path, output: Path) -> list[Path]:
    rows = [r for r in _campaign_records(dynamics)
            if str(r.get("architecture")) == "islands" and "island" in str(r.get("study", ""))]
    if not rows:
        raise FileNotFoundError(f"No island run records found in {dynamics}")
    grouped: dict[str, list[tuple[int, int, float]]] = defaultdict(list)
    csv_rows: list[dict[str, Any]] = []
    for r in rows:
        iterations = as_int(r, "iterations", 0)
        interval = as_int(r, "migration_interval", 0)
        islands = as_int(r, "reported_island_count", as_int(r, "island_count", 0))
        accepted = as_int(r, "migrations_performed", 0)
        if iterations <= 0 or islands < 2:
            raise ValueError(f"Missing/invalid iteration or island metadata in run {r.get('run_key', r.get('method'))}")
        # The C++ implementation performs one destination replacement attempt
        # per island at each due epoch; the final generation does not migrate.
        epochs = sum(1 for step in range(interval, iterations, interval) if interval > 0 and step < iterations) if interval > 0 else 0
        attempts = epochs * islands
        if attempts < accepted:
            raise ValueError(f"Accepted migrations exceed attempts for run {r.get('run_key', r.get('method'))}")
        acceptance = accepted / attempts if attempts else math.nan
        method = str(r.get("method", "unknown"))
        grouped[method].append((attempts, accepted, acceptance))
        csv_rows.append({
            "method": method, "benchmark": r.get("benchmark"), "dimension": r.get("dimension"),
            "seed": r.get("seed"), "iterations": iterations, "island_count": islands,
            "migration_interval": interval, "migration_attempts": attempts,
            "migrations_accepted": accepted,
            "acceptance_rate": acceptance if math.isfinite(acceptance) else "",
        })
    methods = sorted(grouped)
    configure_style()
    fig, axes = plt.subplots(1, 2, figsize=(9.7, 3.7))
    x = np.arange(len(methods))
    attempts_med = [float(np.median([z[0] for z in grouped[m]])) for m in methods]
    accepted_med = [float(np.median([z[1] for z in grouped[m]])) for m in methods]
    width = 0.36
    axes[0].bar(x - width / 2, attempts_med, width=width, color=COLORS["light"], edgecolor=COLORS["grey"], label="Attempts")
    axes[0].bar(x + width / 2, accepted_med, width=width, color=COLORS["teal"], label="Accepted replacements")
    short = [m.replace("island_", "").replace("_", "\n") for m in methods]
    axes[0].set_xticks(x, short, fontsize=5.7)
    axes[0].set_ylabel("Median count per run")
    axes[0].set_title("Migration attempts and accepted replacements")
    axes[0].grid(axis="y")
    axes[0].legend(frameon=False, fontsize=6.1)
    rate_methods = [m for m in methods if any(math.isfinite(z[2]) for z in grouped[m])]
    rates = [[z[2] * 100.0 for z in grouped[m] if math.isfinite(z[2])] for m in rate_methods]
    positions = [methods.index(m) + 1 for m in rate_methods]
    if rates:
        bp = axes[1].boxplot(rates, positions=positions, patch_artist=True, showfliers=False, widths=0.6)
        for i, patch in enumerate(bp["boxes"]):
            patch.set_facecolor(PALETTE[i % len(PALETTE)])
            patch.set_alpha(0.55)
    axes[1].set_xticks(x + 1, short, fontsize=5.7)
    axes[1].set_ylabel("Accepted / attempted migrations (%)")
    axes[1].set_title("Migration acceptance (no-migration case excluded)")
    axes[1].set_ylim(0, 105)
    axes[1].grid(axis="y")
    fig.text(0.5, 0.004,
             "Attempts = due migration epochs × island count (one migrant evaluated per destination and epoch); accepted replacements are the C++ migrations_performed counter. Exploratory M2 pilot.",
             ha="center", fontsize=6.0, color=COLORS["grey"])
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    out = _save_named(fig, output, "fig_migration_diagnostics.pdf")
    _write_csv(output / "migration_diagnostics.csv", csv_rows)
    return out


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row.keys()))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def plot_runtime_decomposition(sync_campaign: Path, output: Path) -> list[Path]:
    source = sync_campaign / "campaign_runs.csv"
    required = (
        "benchmark", "method", "population_size", "dimension", "study", "seed",
        "elapsed_seconds", "initialization_seconds", "topology_setup_seconds",
        "evaluation_wall_seconds", "propagation_wall_seconds", "social_selection_seconds",
    )
    rows = [r for r in read_csv(source, required=required) if r.get("benchmark") == "sphere_work"]
    if not rows:
        raise ValueError(f"No sphere_work rows in {source}")

    # The campaign_runs.csv export does not always contain work_iterations.
    # The source study name is an explicit experimental identifier, e.g.
    # sphere_work_n64_cost512 or sphere_work_n64_cost4096. Recover ONLY from
    # that identifier and fail if it is missing or conflicts with recorded values.
    for row in rows:
        study = str(row.get("study", ""))
        match = re.search(r"(?:^|_)cost(\d+)(?:_|$)", study)
        from_study = int(match.group(1)) if match else None
        explicit = as_int(row, "work_iterations", -1)
        if explicit > 0 and from_study is not None and explicit != from_study:
            raise ValueError(f"Conflicting work iterations for study {study}: {explicit} vs. {from_study}")
        if explicit <= 0 and from_study is None:
            raise ValueError(f"Cannot recover work_iterations for sphere_work study {study!r}")
        row["work_iterations"] = str(explicit if explicit > 0 else from_study)

    combos = sorted({(as_int(r, "dimension"), as_int(r, "work_iterations"), as_int(r, "population_size")) for r in rows})
    expected_methods = ["serial_global"] + [f"ms_global_w{w}" for w in (1, 2, 4)] + [f"pp_global_w{w}" for w in (1, 2, 4)]
    components = [
        ("initialization_seconds", "Init."),
        ("topology_setup_seconds", "Topology"),
        ("propagation_wall_seconds", "Propagation"),
        ("social_selection_seconds", "Social selection"),
        ("evaluation_wall_seconds", "Objective eval."),
    ]
    colors = [COLORS["grey"], COLORS["gold"], COLORS["purple"], COLORS["teal"], COLORS["blue"], COLORS["light"]]
    all_rows: list[dict[str, Any]] = []
    configure_style()
    ncols = 2
    nrows = math.ceil(len(combos) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(11.0, 3.45 * nrows), squeeze=False)
    for ax, (dimension, work, population) in zip(axes.flat, combos):
        cell = [r for r in rows if (as_int(r, "dimension"), as_int(r, "work_iterations"), as_int(r, "population_size")) == (dimension, work, population)]
        methods_present = {r.get("method", "") for r in cell}
        missing = sorted(set(expected_methods) - methods_present)
        if missing:
            raise ValueError(f"Incomplete runtime cell D={dimension}, work={work}: missing methods {missing}")
        grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
        for r in cell:
            grouped[r["method"]].append(r)
        method_order = expected_methods
        stacked: dict[str, list[float]] = {}
        for method in method_order:
            method_rows = grouped[method]
            method_seeds = [as_int(row, "seed", -1) for row in method_rows]
            if len(method_rows) != 5 or len(set(method_seeds)) != 5:
                raise ValueError(
                    f"Expected five unique runs for {method}, D={dimension}, work={work}; "
                    f"got {len(method_rows)} rows and {len(set(method_seeds))} seeds"
                )
            per_run_components = []
            for row in method_rows:
                total = as_float(row, "elapsed_seconds")
                parts = [max(0.0, as_float(row, field, 0.0)) for field, _ in components]
                if not math.isfinite(total) or total <= 0:
                    raise ValueError(f"Invalid total runtime for {method}, D={dimension}, work={work}")
                residual = total - sum(parts)
                if residual < -max(1.0e-8, 0.01 * total):
                    raise ValueError(f"Instrumented runtime components exceed elapsed time for {method}, D={dimension}, work={work}")
                per_run_components.append(parts + [max(0.0, residual)])
            matrix = np.asarray(per_run_components, dtype=float)
            # Use arithmetic means for additive stacked decomposition: the mean
            # of per-run components sums exactly to the mean elapsed time.
            stacked[method] = list(np.mean(matrix, axis=0))
            for field_index, (field, label) in enumerate(components):
                all_rows.append({
                    "dimension": dimension, "population_size": population, "work_iterations": work,
                    "method": method, "worker_count": 1 if method == "serial_global" else int(method.rsplit("w", 1)[-1]),
                    "component": label, "mean_seconds": float(np.mean(matrix[:, field_index])),
                    "q25_seconds": float(np.quantile(matrix[:, field_index], 0.25)),
                    "q75_seconds": float(np.quantile(matrix[:, field_index], 0.75)),
                    "n_runs": len(method_rows), "residual_mean_seconds": "",
                })
            totals = np.asarray([as_float(record, "elapsed_seconds") for record in method_rows], dtype=float)
            residuals = matrix[:, -1]
            all_rows.append({
                "dimension": dimension, "population_size": population, "work_iterations": work,
                "method": method, "worker_count": 1 if method == "serial_global" else int(method.rsplit("w", 1)[-1]),
                "component": "Total elapsed", "mean_seconds": float(np.mean(totals)),
                "q25_seconds": float(np.quantile(totals, 0.25)),
                "q75_seconds": float(np.quantile(totals, 0.75)), "n_runs": len(method_rows),
                "residual_mean_seconds": float(np.mean(residuals)),
            })
        x = np.arange(len(method_order))
        bottom = np.zeros(len(method_order))
        labels = [label for _, label in components] + ["Residual"]
        for i, label in enumerate(labels):
            values = np.asarray([stacked[m][i] for m in method_order], dtype=float)
            ax.bar(x, values, bottom=bottom, width=0.72, color=colors[i], edgecolor="white", linewidth=0.35, label=label)
            bottom += values
        ax.set_xticks(x, ["Serial" if m == "serial_global" else ("MS" if m.startswith("ms_") else "PP") + f" {m.rsplit('w',1)[-1]}" for m in method_order], fontsize=6.2)
        ax.set_ylabel("Mean wall time (s)")
        ax.set_title(f"D={dimension}, N={population}, work={work:,}")
        ax.grid(axis="y")
    for ax in list(axes.flat)[len(combos):]:
        ax.set_axis_off()
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.015))
    fig.suptitle("Runtime-component decomposition on the synthetic sphere_work objective", y=1.005, weight="bold")
    fig.text(0.5, 0.012,
             "Bars are arithmetic means of per-run component times; the residual closes exactly to mean elapsed time. MacBook Air M2 pilot; synthetic objective-cost control only.",
             ha="center", fontsize=6.1, color=COLORS["grey"])
    fig.tight_layout(rect=(0, 0.055, 1, 0.97))
    out = _save_named(fig, output, "fig_runtime_decomposition.pdf")
    _write_csv(output / "runtime_component_summary.csv", all_rows)
    return out


def _function_id(problem_id: str) -> str:
    match = re.search(r"(?:^|[_-])f(?:unction)?[_-]?0*(\d+)(?:[_-]|$)", problem_id, flags=re.IGNORECASE)
    return f"F{int(match.group(1))}" if match else problem_id


def _average_ranks(values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(values.items(), key=lambda item: (item[1], item[0]))
    ranks: dict[str, float] = {}
    i = 0
    while i < len(ordered):
        j = i + 1
        while j < len(ordered) and math.isclose(ordered[j][1], ordered[i][1], rel_tol=1e-12, abs_tol=1e-14):
            j += 1
        average_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[ordered[k][0]] = average_rank
        i = j
    return ranks


def plot_external_function_heatmaps(bbob_campaign: Path, output: Path) -> list[Path]:
    records: list[dict[str, Any]] = []
    records_dir = bbob_campaign / "records"
    if records_dir.is_dir():
        for path in sorted(records_dir.glob("*.jsonl")):
            records.extend(_read_jsonl(path))
    if not records:
        raise FileNotFoundError(f"No BBOB JSONL records found in {records_dir}")
    methods = sorted({str(r.get("method")) for r in records if r.get("method")})
    problem_values: dict[tuple[str, int], dict[str, float]] = defaultdict(dict)
    for r in records:
        method = str(r.get("method", ""))
        problem_id = str(r.get("problem_id", ""))
        dimension = as_int(r, "dimension", -1)
        value = as_float(r, "best_value")
        if method and problem_id and dimension > 0 and math.isfinite(value):
            key = (problem_id, dimension)
            if method in problem_values[key]:
                raise ValueError(f"Duplicate BBOB method record for {problem_id}/{method}")
            problem_values[key][method] = value
    if len(methods) < 2 or not problem_values:
        raise ValueError("BBOB records do not contain at least two complete methods")
    instance_ranks: dict[tuple[str, int], dict[str, float]] = {}
    incomplete = []
    for key, values in problem_values.items():
        missing = sorted(set(methods) - set(values))
        if missing:
            incomplete.append((key, missing))
            continue
        instance_ranks[key] = _average_ranks(values)
    if not instance_ranks:
        raise ValueError("No BBOB problem instance has complete coverage across all methods")
    if incomplete:
        raise ValueError(f"BBOB heatmap withheld: {len(incomplete)} problem instances lack full method coverage")

    functions = sorted({_function_id(problem_id) for problem_id, _ in instance_ranks}, key=lambda x: int(x[1:]) if x.startswith("F") and x[1:].isdigit() else 9999)
    dimensions = sorted({dim for _, dim in instance_ranks})
    overall_rank = {method: float(np.median([ranks[method] for ranks in instance_ranks.values()])) for method in methods}
    methods = sorted(methods, key=lambda m: (overall_rank[m], _display_bbob_method(m)))

    rank_matrices = []
    coverage_counts = []
    for dim in dimensions:
        matrix = np.full((len(functions), len(methods)), np.nan)
        coverage: dict[str, int] = {}
        for i, function in enumerate(functions):
            cells = [ranks for (problem_id, d), ranks in instance_ranks.items() if d == dim and _function_id(problem_id) == function]
            coverage[function] = len(cells)
            if not cells:
                continue
            for j, method in enumerate(methods):
                matrix[i, j] = float(np.median([cell[method] for cell in cells]))
        rank_matrices.append(matrix)
        coverage_counts.append(coverage)

    configure_style()
    cmap_name = "viridis_r"
    vmin, vmax = 1, len(methods)
    pretty_methods = [_display_bbob_method(m) for m in methods]
    fig, axes = plt.subplots(len(dimensions), 1, figsize=(10.8, 1.35 + 1.15 * len(functions) * len(dimensions)), squeeze=False, constrained_layout=True)
    image = None
    for ax, dim, matrix, coverage in zip(axes[:, 0], dimensions, rank_matrices, coverage_counts):
        image = ax.imshow(matrix, aspect="auto", cmap=cmap_name, vmin=vmin, vmax=vmax)
        ax.set_title(f"D={dim}")
        ax.set_yticks(np.arange(len(functions)), [f"{fn}  (n={coverage.get(fn, 0)})" for fn in functions])
        ax.set_xticks(np.arange(len(methods)), pretty_methods, rotation=35, ha="right")
        if ax is axes[-1, 0]:
            ax.set_xlabel("Method")
        ax.set_xticks(np.arange(-0.5, len(methods), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(functions), 1), minor=True)
        ax.grid(which="minor", color=COLORS["white"], linewidth=0.7)
        ax.tick_params(which="minor", bottom=False, left=False)
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                if np.isfinite(matrix[i, j]):
                    val = float(matrix[i, j])
                    ax.text(j, i, f"{val:.1f}", ha="center", va="center", fontsize=6.0, color=_cell_text_color(val, vmin, vmax, cmap_name))
    axes[0, 0].set_ylabel("BBOB function family")
    if image is not None:
        cb = fig.colorbar(image, ax=list(axes[:, 0]), shrink=0.96, pad=0.02)
        cb.set_label("Median within-instance rank (1 = best)")
    fig.suptitle("BBOB stage-C pilot: median within-instance method ranks", y=1.01, weight="bold")
    fig.text(0.5, -0.01,
             f"Rows show medians across complete identical COCO problem instances; columns are ordered by overall median rank. Coverage: {len(instance_ranks)} fully covered instances. Exploratory pilot; not a COCO ECDF.",
             ha="center", fontsize=6.2, color=COLORS["grey"])
    out = _save_named(fig, output, "fig_external_function_heatmaps.pdf")

    rank_rows = []
    for dim, matrix, coverage in zip(dimensions, rank_matrices, coverage_counts):
        for i, function in enumerate(functions):
            for j, method in enumerate(methods):
                if np.isfinite(matrix[i, j]):
                    rank_rows.append({
                        "function": function,
                        "dimension": dim,
                        "method": method,
                        "median_rank": float(matrix[i, j]),
                        "complete_problem_instances": int(coverage.get(function, 0)),
                    })
    _write_csv(output / "bbob_function_median_ranks.csv", rank_rows)
    return out




def _observer_directory(args: argparse.Namespace) -> Path | None:
    if args.bbob_observer_data:
        return args.bbob_observer_data
    candidate = args.bbob_campaign / "coco_data"
    if candidate.is_dir():
        return candidate
    # The supplied machine inventory shows COCO files under exdata/Users/...;
    # discover that relocated tree without assuming the local account name.
    search_root = Path.cwd()
    pattern = "exdata/Users/*/Desktop/Progetti/Parallel-Continuous-Particle-Swarm-Optimization/results/external_benchmarks/bbob_stageC_pilot_m2/coco_data"
    matches = sorted(search_root.glob(pattern))
    if matches:
        return matches[0]
    # Allow a copied/renamed repository root while keeping the search anchored
    # to the exact campaign suffix; do not scan the entire home directory.
    matches = sorted(search_root.glob("**/results/external_benchmarks/bbob_stageC_pilot_m2/coco_data"))
    return matches[0] if matches else None


def _observer_coverage(args: argparse.Namespace) -> tuple[int, int, list[str], Path | None]:
    manifest_path = args.bbob_campaign / "campaign_manifest.json"
    manifest = read_json(manifest_path) if manifest_path.is_file() else {}
    methods = manifest.get("methods", [])
    expected = {str(m) for m in methods}
    if not expected:
        records_dir = args.bbob_campaign / "records"
        expected = {path.stem for path in records_dir.glob("*.jsonl")} if records_dir.is_dir() else set()
    root = _observer_directory(args)
    info = list(root.rglob("*.info")) if root and root.is_dir() else []
    represented = {p.parent.name.removesuffix("-0001") for p in info}
    if expected:
        missing = sorted(expected - represented)
        count = len(expected - set(missing))
        return count, len(expected), missing, root
    return len(represented), 0, sorted(expected - represented), root



def _first_hit_fe(row: dict[str, Any], target: float) -> int | None:
    history = _history(row)
    if history is None:
        return None
    x, y = history
    indices = np.flatnonzero(y <= target)
    return int(x[int(indices[0])]) if indices.size else None


def _average_rank_with_ties(values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(values.items(), key=lambda item: (item[1], item[0]))
    ranks: dict[str, float] = {}
    start = 0
    while start < len(ordered):
        stop = start + 1
        while stop < len(ordered) and math.isclose(
            ordered[stop][1], ordered[start][1], rel_tol=1.0e-12, abs_tol=1.0e-14
        ):
            stop += 1
        average_rank = (start + 1 + stop) / 2.0
        for index in range(start, stop):
            ranks[ordered[index][0]] = average_rank
        start = stop
    return ranks


def export_paper_table_data(
    sync_campaign: Path,
    dynamics_campaign: Path,
    topology_campaign: Path,
    output: Path,
) -> list[str]:
    """Export machine-readable table inputs; never edits TeX or fills TBD cells."""
    table_dir = output / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    created: list[str] = []

    # RQ1: exact recorded pair audit.
    eq_path = sync_campaign / "equivalence_audit.csv"
    if eq_path.is_file():
        eq_rows = read_csv(eq_path, required=("method", "workers", "verified_pairs", "expected_pairs"))
        export = []
        for row in eq_rows:
            expected = as_int(row, "expected_pairs", -1)
            verified = as_int(row, "verified_pairs", -1)
            if expected <= 0 or verified < 0 or verified > expected:
                raise ValueError(f"Invalid equivalence audit row: {row}")
            method = str(row.get("method", ""))
            backend = "Master--slave" if method.startswith("ms") else "Particle-parallel" if method.startswith("pp") else method
            export.append({
                "backend": backend,
                "workers": as_int(row, "workers"),
                "expected_pairs": expected,
                "identical_pairs": verified,
                "mismatches": expected - verified,
                "identity_percent": 100.0 * verified / expected,
            })
        _write_csv(table_dir / "table_equivalence.csv", export)
        created.append("tables/table_equivalence.csv")

    # RQ2: the p=4 summary uses paired serial/parallel wall times from the analyzer.
    scaling_path = sync_campaign / "strong_scaling_summary.csv"
    if scaling_path.is_file():
        scaling_rows = read_csv(scaling_path, required=(
            "method", "workers", "dimension", "population_size", "work_iterations",
            "median_parallel_runtime_seconds", "median_speedup", "bootstrap_95_low",
            "bootstrap_95_high", "parallel_efficiency",
        ))
        export = []
        for row in scaling_rows:
            if as_int(row, "workers", -1) != 4 or as_int(row, "work_iterations", 0) not in {512, 4096}:
                continue
            method = str(row["method"])
            export.append({
                "dimension": as_int(row, "dimension"),
                "population_size": as_int(row, "population_size"),
                "work_iterations": as_int(row, "work_iterations"),
                "function_evaluations_per_run": as_int(row, "population_size") * 81,
                "backend": "Master--slave" if method.startswith("ms_") else "Particle-parallel" if method.startswith("pp_") else method,
                "worker_count": 4,
                "median_parallel_runtime_seconds": as_float(row, "median_parallel_runtime_seconds"),
                "median_paired_speedup": as_float(row, "median_speedup"),
                "bootstrap_95_low": as_float(row, "bootstrap_95_low"),
                "bootstrap_95_high": as_float(row, "bootstrap_95_high"),
                "parallel_efficiency": as_float(row, "parallel_efficiency"),
                "n_paired_runs": as_int(row, "n_paired_runs", 0),
            })
        if export:
            _write_csv(table_dir / "table_strong_scaling_p4.csv", export)
            created.append("tables/table_strong_scaling_p4.csv")

    dyn = _campaign_records(dynamics_campaign)
    async_rows = [r for r in dyn if "async" in str(r.get("study", ""))
                  and str(r.get("architecture")) in {"serial", "asynchronous"}
                  and str(r.get("benchmark")) in EXPECTED_FUNCTIONS and _history(r) is not None]
    async_export = []
    for method in ("async_global_w2", "async_global_w4", "async_ring_w2", "async_ring_w4"):
        for dimension in sorted({as_int(r, "dimension", -1) for r in async_rows}):
            cell = [r for r in async_rows if str(r.get("method")) == method and as_int(r, "dimension", -1) == dimension]
            if not cell:
                continue
            by_function: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for row in cell:
                by_function[str(row["benchmark"])].append(row)
            if set(by_function) != set(EXPECTED_FUNCTIONS) or any(
                len(rows) != 5 or len({as_int(r, "seed", -1) for r in rows}) != 5
                for rows in by_function.values()
            ):
                raise ValueError(f"Incomplete async table cell {method}, D={dimension}")
            function_medians = []
            all_rows = [r for f in EXPECTED_FUNCTIONS for r in by_function[f]]
            success_target = as_float(all_rows[0], "success_target", 1.0e-8)
            hits = [_first_hit_fe(row, success_target) for row in all_rows]
            budgets = [int(_history(row)[0][-1]) for row in all_rows]
            successes = sum(hit is not None for hit in hits)
            ert = (
                (sum(int(hit) for hit in hits if hit is not None)
                 + sum(budget for hit, budget in zip(hits, budgets) if hit is None)) / successes
                if successes else math.inf
            )
            for function in EXPECTED_FUNCTIONS:
                function_medians.append(float(np.median([
                    math.log10(max(as_float(r, "best_value"), LOG_FLOOR)) for r in by_function[function]
                ])))
            async_export.append({
                "method": method, "dimension": dimension, "topology": "Global" if "global" in method else "Ring",
                "worker_count": 2 if method.endswith("w2") else 4,
                "n_runs": len(all_rows),
                "median_function_log10_best": float(np.median(function_medians)),
                "q25_function_log10_best": float(np.quantile(function_medians, 0.25)),
                "q75_function_log10_best": float(np.quantile(function_medians, 0.75)),
                "success_target": success_target, "successes": successes,
                "ert_function_evaluations": ert if math.isfinite(ert) else "",
                "function_evaluations_per_run": as_int(all_rows[0], "function_evaluations", 0),
            })
    if async_export:
        _write_csv(table_dir / "table_async_quality.csv", async_export)
        created.append("tables/table_async_quality.csv")

    island_rows = [r for r in dyn if "island" in str(r.get("study", ""))
                   and str(r.get("architecture")) == "islands"
                   and str(r.get("benchmark")) in EXPECTED_FUNCTIONS and _history(r) is not None]
    island_export = []
    for method in sorted({str(r.get("method")) for r in island_rows}):
        for dimension in sorted({as_int(r, "dimension", -1) for r in island_rows}):
            cell = [r for r in island_rows if str(r.get("method")) == method and as_int(r, "dimension", -1) == dimension]
            if not cell:
                continue
            functions = {str(r.get("benchmark")) for r in cell}
            if functions != set(EXPECTED_FUNCTIONS) or len(cell) != 30 or len({(r.get("benchmark"), as_int(r, "seed", -1)) for r in cell}) != 30:
                raise ValueError(f"Incomplete island table cell {method}, D={dimension}")
            attempts = accepted = 0
            for row in cell:
                iterations = as_int(row, "iterations", 0)
                interval = as_int(row, "migration_interval", 0)
                islands = as_int(row, "reported_island_count", as_int(row, "island_count", 1))
                epochs = len(range(interval, iterations, interval)) if interval > 0 else 0
                attempts += epochs * islands
                accepted += as_int(row, "migrations_performed", 0)
            target = as_float(cell[0], "success_target", 1.0e-8)
            successes = sum(as_float(row, "best_value") <= target for row in cell)
            first = cell[0]
            island_export.append({
                "method": method, "dimension": dimension,
                "island_count": as_int(first, "reported_island_count", as_int(first, "island_count", 1)),
                "local_population": as_int(first, "population_size", 0) // max(1, as_int(first, "reported_island_count", as_int(first, "island_count", 1))),
                "workers": as_int(first, "reported_worker_count", as_int(first, "workers", 1)),
                "intra_topology": first.get("topology", ""),
                "migration_topology": first.get("migration_topology", "none"),
                "migration_interval": as_int(first, "migration_interval", 0),
                "n_runs": len(cell), "success_target": target, "successes": successes,
                "migration_attempts": attempts, "migrations_accepted": accepted,
                "accepted_percent": 100.0 * accepted / attempts if attempts else "",
                "function_evaluations_per_run": as_int(first, "function_evaluations", 0),
            })
    if island_export:
        _write_csv(table_dir / "table_island_quality.csv", island_export)
        created.append("tables/table_island_quality.csv")

    # RQ4: paired topology effects and average within-cell ranks with tie handling.
    topo_candidates = sorted((topology_campaign / "01_serial_topology_quality").glob("*/runs.csv"))
    if topo_candidates:
        rows = read_csv(topo_candidates[0], required=("benchmark", "dimension", "seed", "topology", "best_value"))
        expected_topologies = {"global", "ring", "grid_von_neumann", "wheel", "random", "small_world", "scale_free"}
        topology_seeds = sorted({as_int(row, "seed", -1) for row in rows})
        topology_dimensions = sorted({as_int(row, "dimension", -1) for row in rows})
        if len(topology_seeds) != 5 or topology_seeds[0] < 0 or topology_dimensions != [30, 100]:
            raise ValueError(
                f"Expected five nonnegative paired seeds and dimensions [30, 100]; "
                f"found seeds={topology_seeds}, dimensions={topology_dimensions}"
            )
        expected_problem_cells = {
            (function, dimension, seed)
            for function in EXPECTED_FUNCTIONS
            for dimension in topology_dimensions
            for seed in topology_seeds
        }
        actual_topologies = {str(row["topology"]) for row in rows}
        if actual_topologies != expected_topologies:
            raise ValueError(
                f"Topology table requires all seven configured graphs; found={sorted(actual_topologies)}, "
                f"expected={sorted(expected_topologies)}"
            )
        observed_cells: dict[str, set[tuple[str, int, int]]] = defaultdict(set)
        for row in rows:
            observed_cells[str(row["topology"])].add(
                (str(row["benchmark"]), as_int(row, "dimension"), as_int(row, "seed"))
            )
        for topology_name in expected_topologies:
            if observed_cells[topology_name] != expected_problem_cells:
                missing = sorted(expected_problem_cells - observed_cells[topology_name])
                extra = sorted(observed_cells[topology_name] - expected_problem_cells)
                raise ValueError(
                    f"Incomplete topology coverage for {topology_name}; missing={missing[:5]}, extra={extra[:5]}"
                )
        if len(rows) != len(expected_topologies) * len(expected_problem_cells):
            raise ValueError("Topology runs.csv contains duplicate or unexpected rows")
        cells: dict[tuple[str, int, int], dict[str, float]] = defaultdict(dict)
        for row in rows:
            key = (str(row["benchmark"]), as_int(row, "dimension"), as_int(row, "seed"))
            topology_name = str(row["topology"])
            value = as_float(row, "best_value")
            if math.isfinite(value):
                cells[key][topology_name] = math.log10(max(value, LOG_FLOOR))
        rank_values: dict[str, list[float]] = defaultdict(list)
        paired_values: dict[str, dict[tuple[str, int], list[float]]] = defaultdict(lambda: defaultdict(list))
        for (benchmark, dim, seed), values in cells.items():
            if set(values) != expected_topologies:
                raise ValueError(
                    f"Incomplete topology comparison at {benchmark}/D={dim}/seed={seed}: "
                    f"found={sorted(values)}, expected={sorted(expected_topologies)}"
                )
            ranks = _average_rank_with_ties(values)
            for name, rank in ranks.items():
                rank_values[name].append(rank)
            reference = values["global"]
            for name, value in values.items():
                paired_values[name][(benchmark, dim)].append(value - reference)
        topology_export = []
        for name in ("global", "ring", "grid_von_neumann", "wheel", "random", "small_world", "scale_free"):
            function_effects = []
            for benchmark in EXPECTED_FUNCTIONS:
                dimension_values = [float(np.median(v)) for (function, _dim), v in paired_values[name].items()
                                    if function == benchmark and v]
                if dimension_values:
                    function_effects.append(float(np.median(dimension_values)))
            topology_export.append({
                "topology": name,
                "median_function_family_delta_log10_vs_global": float(np.median(function_effects)) if function_effects else (0.0 if name == "global" else ""),
                "mean_rank": float(np.mean(rank_values[name])) if rank_values[name] else "",
                "n_ranked_problem_cells": len(rank_values[name]),
                "n_function_families": len(function_effects),
                "interpretation": "rank 1 is best; lower delta_log10 favours topology; descriptive M2 pilot",
            })
        _write_csv(table_dir / "table_topology_quality.csv", topology_export)
        created.append("tables/table_topology_quality.csv")

    return created

def _write_catalogue(output: Path, args: argparse.Namespace, failures: list[str]) -> int:
    status_rows: list[dict[str, str]] = []
    info_methods, expected_methods, missing_methods, observer_root = _observer_coverage(args)
    for filename, section, description in PAPER_FIGURES:
        target = output / filename
        if filename in EXTERNAL_FIGURES:
            status = "EXTERNAL_PRESENT_UNVERIFIED" if target.is_file() else "WAITING_FOR_EXTERNAL_DATA"
        else:
            status = "GENERATED_FROM_PILOT" if target.is_file() else "NOT_GENERATED"
        note = (
            "Generated from recorded data; M2 source is exploratory and not confirmatory."
            if status == "GENERATED_FROM_PILOT" else
            "A file is present but was not generated or provenance-checked by this script."
            if status == "EXTERNAL_PRESENT_UNVERIFIED" else
            "Required evidence unavailable; no synthetic values inserted."
        )
        if filename == "fig_coco_bbob_ecdf.pdf":
            note = f"COCO Observer .info coverage: {info_methods}/{expected_methods} configured methods."
            if missing_methods:
                note += " Missing: " + ", ".join(missing_methods) + "."
            note += " Partial Observer data are insufficient for an all-method ECDF; complete the missing observers and use cocopp. JSONL ranks are not an ECDF."
        elif filename == "fig_coco_bbob_ls_ecdf.pdf":
            note = "No BBOB-LargeScale campaign records are present in the supplied inventory."
        elif filename in {"fig_cec2013_checkpoints.pdf", "fig_cec_function_breakdown.pdf"}:
            note = "No CEC 2013 LSGO campaign records are present in the supplied inventory; the CEC driver must first accept the documented 905-D F13/F14 cases."
        elif filename == "fig_external_function_heatmaps.pdf":
            note = "Descriptive median within-instance BBOB ranks from JSONL; common instance objective offsets cancel. Not an ECDF and exploratory only."
        elif filename == "fig_migration_diagnostics.pdf":
            note = "Attempts reconstructed from migration epochs × islands; accepted replacements are directly recorded by C++."
        elif filename == "fig_dt_sensitivity_summary.pdf":
            note = "Paired against recorded reference dt=30.0; this does not establish dt=30.0 as optimal."
        if filename == "fig_coco_bbob_ecdf.pdf" and observer_root:
            note += f" Observer directory: {observer_root}"
        status_rows.append({"figure": filename, "section": section, "status": status,
                            "description": description, "note": note})
    _write_csv(output / "figure_status.csv", status_rows)
    lines = [
        "# Paper figure catalogue", "",
        "Figures generated from the recorded M2 pilot outputs are exploratory; this script does not create confirmatory claims or synthesize missing data. External COCO/CEC figures are not generated by this script.", "",
        "| Figure | Section | Status | Note |", "|---|---|---|---|",
    ]
    for row in status_rows:
        lines.append(f"| `{row['figure']}` | {row['section']} | **{row['status']}** | {row['note']} |")
    if failures:
        lines += ["", "## Generation warnings", ""] + [f"- {warning}" for warning in failures]
    lines += [
        "", "## Methodological notes", "",
        "- Synchronous equivalence uses the run-level equality audit; worker counts in the supplied pilot are 1, 2 and 4 only.",
        "- Strong scaling is the synthetic `sphere_work` workload, with D=30/100, N=64, work_iterations=512/4096 and workers 1/2/4. These are M2-specific timings.",
        "- Anytime curves reduce the five seeds within each function first, then report the median and IQR across the six fixed functions. The axis is FE, not wall time.",
        "- The dt sensitivity plot shows paired log-objective differences against the reference dt=30.0; it does not identify an optimum by itself.",
        "- Runtime decomposition reports initialization, topology setup, propagation, social selection and objective evaluation separately; residual is elapsed time minus all instrumented components. Bars use arithmetic means so they are additive.",
        "- Migration attempts are one destination replacement attempt per island at each due migration epoch; the final generation does not migrate.",
        "- BBOB heatmaps show within-instance ranks, not log-error: instance-specific objective offsets cancel when methods are ranked on the same problem.",
        "- Graph descriptors are not present in the current run JSONL/CSV records and must not be inferred from topology labels alone.",
        "- No official BBOB ECDF is inferred from JSONL final values; complete COCO Observer .info coverage across methods is required.",
        "- No BBOB-LargeScale or CEC figure is created without the corresponding campaign output.",
        "",
    ]
    (output / "FIGURE_CATALOGUE.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Paper figure output: {output}")
    for row in status_rows:
        print(f"[{row['status']}] {row['figure']} — {row['note']}")
    if failures:
        print("\nPlotting warnings:")
        for warning in failures:
            print(" -", warning)
    generated_count = sum(row["status"] == "GENERATED_FROM_PILOT" for row in status_rows)
    print(f"\nGenerated {generated_count}/{len(status_rows)} manuscript figures from recorded data.")
    return generated_count


def build(args: argparse.Namespace) -> int:
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    staging = output / ".staging"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    configure_style()
    failures: list[str] = []

    # Remove stale files owned by this builder. Externally generated COCO/CEC
    # figures are preserved and merely detected in the catalogue.
    for filename, _, _ in PAPER_FIGURES:
        if filename in EXTERNAL_FIGURES:
            continue
        stem = Path(filename).stem
        for suffix in (".pdf", ".png", ".svg"):
            (output / f"{stem}{suffix}").unlink(missing_ok=True)

    tasks = [
        ("fig_sync_equivalence", lambda: internal_plots.plot_equivalence(args.sync_campaign, staging), "fig_sync_equivalence"),
        ("fig_strong_scaling", lambda: internal_plots.plot_scaling(args.sync_campaign, staging), "fig_strong_scaling"),
        ("fig_topology_quality", lambda: internal_plots.plot_topologies(args.topology_campaign, staging), "fig_topology_quality"),
    ]
    for name, function, target_stem in tasks:
        try:
            paths = function()
            candidates = [p for p in paths if p.suffix.lower() in {".pdf", ".png"}]
            if name == "fig_strong_scaling":
                candidates = [p for p in candidates if p.name.startswith("fig_02_strong_scaling")]
            if not candidates:
                raise ValueError("plotter returned no PDF/PNG files")
            _copy_named(candidates, output, target_stem)
        except Exception as exc:
            failures.append(f"{name}: {type(exc).__name__}: {exc}")

    optional_tasks = [
        ("fig_async_island_anytime", lambda: plot_async_island_anytime(args.dynamics_campaign, output)),
        ("fig_dt_sensitivity_summary", lambda: plot_dt_summary(args.dt_campaign, output)),
        ("fig_internal_function_forest", lambda: plot_internal_function_forest(args.dynamics_campaign, args.topology_campaign, output)),
        ("fig_migration_diagnostics", lambda: plot_migration_diagnostics(args.dynamics_campaign, output)),
        ("fig_runtime_decomposition", lambda: plot_runtime_decomposition(args.sync_campaign, output)),
        ("fig_external_function_heatmaps", lambda: plot_external_function_heatmaps(args.bbob_campaign, output)),
    ]
    for name, function in optional_tasks:
        try:
            function()
        except Exception as exc:
            failures.append(f"{name}: {type(exc).__name__}: {exc}")

    try:
        table_files = export_paper_table_data(
            args.sync_campaign, args.dynamics_campaign, args.topology_campaign, output
        )
        print("\nExported table data:")
        for table_file in table_files:
            print(f" - {table_file}")
    except Exception as exc:
        failures.append(f"table-data export: {type(exc).__name__}: {exc}")

    generated = _write_catalogue(output, args, failures)
    shutil.rmtree(staging, ignore_errors=True)
    return 0 if generated else 2


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sync-campaign", type=Path, default=Path("results/raw/cpso_serial_parallel_m2_pilot"))
    parser.add_argument("--dynamics-campaign", type=Path, default=Path("results/raw/cpso_dynamics_m2_pilot"))
    parser.add_argument("--topology-campaign", type=Path, default=Path("results/raw/cpso_topology_expansion_m2_pilot"))
    parser.add_argument("--dt-campaign", type=Path, default=Path("results/raw/cpso_dt_sensitivity_m2_screening"))
    parser.add_argument("--bbob-campaign", type=Path, default=Path("results/external_benchmarks/bbob_stageC_pilot_m2"))
    parser.add_argument("--bbob-observer-data", type=Path, default=None,
                        help="Path to the COCO Observer data tree, if stored outside the BBOB campaign directory.")
    parser.add_argument("--output", type=Path, default=Path("paper/figures/results"))
    args = parser.parse_args()
    for field in ("sync_campaign", "dynamics_campaign", "topology_campaign", "dt_campaign", "bbob_campaign"):
        setattr(args, field, getattr(args, field).expanduser().resolve())
    if args.bbob_observer_data is not None:
        args.bbob_observer_data = args.bbob_observer_data.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    return build(args)


if __name__ == "__main__":
    raise SystemExit(main())
