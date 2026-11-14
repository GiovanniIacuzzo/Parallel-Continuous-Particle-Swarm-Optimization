#!/usr/bin/env python3
"""Build a comprehensive, data-driven figure atlas for internal CPSO campaigns.

The script reads only completed JSONL run records produced by the experiment
runners (plus optional analysis CSVs). It never fabricates or imputes results.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plot_results.atlas_common import (
    COLORS, LOG_FLOOR, add_evidence_note, bootstrap_median, configure_style,
    discover_internal_records, evidence_label, family_cluster_bootstrap,
    finite_float, interpolate_step_history, method_label, read_json, safe_slug,
    save_figure, truthy,
)


def load_manifest(campaign_dir: Path) -> dict[str, Any]:
    return read_json(campaign_dir / "campaign_manifest.json")


def enrich_from_manifest(rows: list[dict[str, Any]], manifest: dict[str, Any]) -> list[dict[str, Any]]:
    campaign = manifest.get("campaign", {}) if isinstance(manifest, dict) else {}
    stage_map: dict[tuple[str, str], dict[str, Any]] = {}
    if isinstance(campaign, dict):
        for stage in campaign.get("stages", []):
            for study in stage.get("studies", []):
                stage_map[(str(stage.get("id", "")), str(study.get("id", "")))] = study.get("study", {})
    enriched = []
    for original in rows:
        row = dict(original)
        study = stage_map.get((str(row.get("stage", "")), str(row.get("study", ""))), {})
        if "work_iterations" not in row:
            row["work_iterations"] = study.get("work_iterations", 0)
        if "success_target_default" not in row:
            row["success_target_default"] = study.get("success_target_default", row.get("success_target", math.nan))
        if "population_size" not in row:
            row["population_size"] = study.get("population_size", math.nan)
        row.setdefault("study_title", study.get("title", row.get("study", "")))
        enriched.append(row)
    return enriched


def group_by_study(rows: list[dict[str, Any]]) -> dict[tuple[str, str], list[dict[str, Any]]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row.get("stage", "study")), str(row.get("study", "study")))].append(row)
    return dict(sorted(groups.items()))


def _quality_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for row in rows:
        if str(row.get("benchmark", "")) == "sphere_work":
            continue
        value = finite_float(row.get("best_value"))
        if math.isfinite(value):
            output.append(row)
    return output


def _methods(rows: list[dict[str, Any]]) -> list[str]:
    names = sorted({str(row.get("method", "unknown")) for row in rows})
    def key(name: str):
        row = next((r for r in rows if str(r.get("method")) == name), {})
        arch = str(row.get("architecture", ""))
        order = {"serial": 0, "master_slave": 1, "particle_parallel": 2, "asynchronous": 3, "islands": 4}
        return (order.get(arch, 9), int(finite_float(row.get("workers"), 1)), name)
    return sorted(names, key=key)


def _baseline_method(rows: list[dict[str, Any]], method: str) -> str | None:
    method_rows = [row for row in rows if str(row.get("method")) == method]
    if not method_rows:
        return None
    topology = str(method_rows[0].get("topology", "global"))
    candidates = [row for row in rows if str(row.get("architecture")) == "serial"
                  and str(row.get("topology", "global")) == topology]
    if candidates:
        return str(candidates[0].get("method"))
    candidates = [row for row in rows if str(row.get("architecture")) == "serial"]
    return str(candidates[0].get("method")) if candidates else None


def _save_panel_grid(
    rows: list[dict[str, Any]], output: Path, stem: str, title: str,
    draw_panel, *, max_cols: int = 2, figsize_per_panel: tuple[float, float] = (3.35, 2.45),
) -> list[Path] | None:
    pairs = sorted({(str(r.get("benchmark", "")), int(finite_float(r.get("dimension"), -1))) for r in rows})
    if not pairs:
        return None
    ncols = min(max_cols, len(pairs))
    nrows = math.ceil(len(pairs) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(figsize_per_panel[0] * ncols, figsize_per_panel[1] * nrows), squeeze=False)
    for ax, key in zip(axes.flat, pairs):
        panel = [r for r in rows if (str(r.get("benchmark", "")), int(finite_float(r.get("dimension"), -1))) == key]
        draw_panel(ax, panel, key)
    for ax in list(axes.flat)[len(pairs):]:
        ax.axis("off")
    fig.suptitle(title, y=1.005, fontweight="bold")
    fig.tight_layout()
    return save_figure(fig, output, stem)


def plot_final_distributions(rows: list[dict[str, Any]], output: Path, stem_prefix: str) -> list[Path] | None:
    rows = _quality_rows(rows)
    methods = _methods(rows)
    if len(methods) < 2:
        return None
    def draw(ax, panel, key):
        data = []
        for method in methods:
            vals = [math.log10(max(finite_float(r.get("best_value")), LOG_FLOOR))
                    for r in panel if str(r.get("method")) == method and math.isfinite(finite_float(r.get("best_value")))]
            data.append(vals if vals else [math.nan])
        bp = ax.boxplot(data, patch_artist=True, widths=0.62, showfliers=False,
                        medianprops={"color": "black", "linewidth": 1.0})
        for i, patch in enumerate(bp["boxes"]):
            patch.set_facecolor(COLORS[i % len(COLORS)])
            patch.set_alpha(0.45)
        rng = np.random.default_rng(90210 + key[1] + sum(map(ord, key[0])))
        for i, vals in enumerate(data, start=1):
            vals = np.asarray(vals, dtype=float)
            vals = vals[np.isfinite(vals)]
            if vals.size:
                ax.scatter(i + rng.uniform(-0.10, 0.10, vals.size), vals, s=12, alpha=0.68,
                           color=COLORS[(i - 1) % len(COLORS)], edgecolor="white", linewidth=0.25, zorder=3)
        ax.set_xticks(range(1, len(methods) + 1), [m.replace("_", "\n") for m in methods], rotation=0, fontsize=5.7)
        ax.set_title(f"{key[0].capitalize()} · D={key[1]}")
        ax.set_ylabel(r"$\log_{10}(f_{best})$")
        ax.grid(axis="y")
    result = _save_panel_grid(rows, output, f"{stem_prefix}_final_quality_distribution",
                              "Final solution quality: all seeds", draw)
    return result


def plot_convergence(rows: list[dict[str, Any]], output: Path, stem_prefix: str) -> list[Path] | None:
    rows = [r for r in _quality_rows(rows) if isinstance(r.get("best_history"), list) and isinstance(r.get("history_evaluations"), list)]
    methods = _methods(rows)
    if not rows or not methods:
        return None
    def draw(ax, panel, key):
        max_fe = min(int(finite_float(r.get("function_evaluations"), 0)) for r in panel if finite_float(r.get("function_evaluations"), 0) > 0)
        all_x = np.concatenate([np.asarray(r.get("history_evaluations", []), dtype=float) for r in panel if r.get("history_evaluations")])
        min_fe = max(1.0, float(np.min(all_x)))
        grid = np.linspace(min_fe, max_fe, 120)
        for i, method in enumerate(methods):
            method_rows = [r for r in panel if str(r.get("method")) == method]
            curves = []
            for row in method_rows:
                interp = interpolate_step_history(row, grid)
                if interp is not None:
                    curves.append(np.log10(np.maximum(interp, LOG_FLOOR)))
            if not curves:
                continue
            matrix = np.vstack(curves)
            median = np.nanmedian(matrix, axis=0)
            q1 = np.nanquantile(matrix, 0.25, axis=0)
            q3 = np.nanquantile(matrix, 0.75, axis=0)
            color = COLORS[i % len(COLORS)]
            ax.plot(grid, median, color=color, label=method.replace("_", " "), lw=1.25)
            ax.fill_between(grid, q1, q3, color=color, alpha=0.12, linewidth=0)
        ax.set_title(f"{key[0].capitalize()} · D={key[1]}")
        ax.set_xlabel("Function evaluations")
        ax.set_ylabel(r"Median $\log_{10}(f_{best})$ [IQR]")
        ax.grid(axis="y")
        ax.legend(frameon=False, fontsize=5.3, loc="best")
    result = _save_panel_grid(rows, output, f"{stem_prefix}_convergence_profiles",
                              "Best-so-far convergence profiles (median and IQR across seeds)", draw)
    if result:
        # replace default legends with one compact shared legend
        pass
    return result


def plot_quality_heatmap(rows: list[dict[str, Any]], output: Path, stem_prefix: str) -> list[Path] | None:
    rows = _quality_rows(rows)
    methods = _methods(rows)
    benchmarks = sorted({str(r.get("benchmark")) for r in rows})
    dimensions = sorted({int(finite_float(r.get("dimension"), -1)) for r in rows})
    if len(methods) < 2 or not benchmarks:
        return None
    generated = []
    for dimension in dimensions:
        matrix = np.full((len(benchmarks), len(methods)), np.nan)
        for i, benchmark in enumerate(benchmarks):
            for j, method in enumerate(methods):
                vals = [math.log10(max(finite_float(r.get("best_value")), LOG_FLOOR)) for r in rows
                        if str(r.get("benchmark")) == benchmark and int(finite_float(r.get("dimension"), -1)) == dimension
                        and str(r.get("method")) == method]
                if vals:
                    matrix[i, j] = float(np.median(vals))
        fig, ax = plt.subplots(figsize=(max(6.4, 0.72 * len(methods)), max(3.2, 0.48 * len(benchmarks))))
        image = ax.imshow(matrix, aspect="auto", cmap="viridis")
        ax.set_yticks(np.arange(len(benchmarks)), [b.capitalize() for b in benchmarks])
        ax.set_xticks(np.arange(len(methods)), [m.replace("_", "\n") for m in methods], fontsize=6)
        ax.set_title(fr"Median final objective by method and function · $D={dimension}$")
        cb = fig.colorbar(image, ax=ax, pad=0.015)
        cb.set_label(r"Median $\log_{10}(f_{best})$")
        fig.tight_layout()
        generated.extend(save_figure(fig, output, f"{stem_prefix}_quality_heatmap_d{dimension}"))
    return generated


def _paired_effects(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    lookup = {}
    for row in rows:
        try:
            key = (str(row["benchmark"]), int(row["dimension"]), int(row["seed"]), str(row["method"]))
            value = finite_float(row.get("best_value"))
        except (KeyError, TypeError, ValueError):
            continue
        if math.isfinite(value):
            lookup[key] = row
    methods = _methods(rows)
    output: dict[str, dict[str, Any]] = {}
    for method in methods:
        if method.startswith("serial_"):
            continue
        baseline = _baseline_method(rows, method)
        if not baseline:
            continue
        cells: dict[tuple[str, int], list[float]] = defaultdict(list)
        wins = losses = ties = 0
        for (benchmark, dimension, seed, candidate), row in lookup.items():
            if candidate != method:
                continue
            reference = lookup.get((benchmark, dimension, seed, baseline))
            if reference is None:
                continue
            delta = math.log10(max(finite_float(row.get("best_value")), LOG_FLOOR)) - math.log10(
                max(finite_float(reference.get("best_value")), LOG_FLOOR))
            cells[(benchmark, dimension)].append(delta)
            if delta < -1e-14:
                wins += 1
            elif delta > 1e-14:
                losses += 1
            else:
                ties += 1
        if not cells:
            continue
        family_values: dict[str, float] = {}
        per_family_dimensions: dict[str, list[float]] = defaultdict(list)
        for (family, dim), values in cells.items():
            per_family_dimensions[family].append(float(np.median(values)))
        for family, dimvals in per_family_dimensions.items():
            family_values[family] = float(np.median(dimvals))
        median = float(np.median(list(family_values.values())))
        low, high = family_cluster_bootstrap(family_values, seed=1801 + sum(map(ord, method)))
        output[method] = {
            "reference": baseline, "median": median, "low": low, "high": high,
            "family_values": family_values, "wins": wins, "losses": losses, "ties": ties,
        }
    return output


def plot_effects(rows: list[dict[str, Any]], output: Path, stem_prefix: str) -> list[Path] | None:
    effects = _paired_effects(_quality_rows(rows))
    if not effects:
        return None
    methods = list(effects)
    methods.sort(key=lambda m: effects[m]["median"])
    fig, ax = plt.subplots(figsize=(7.4, max(3.0, 0.42 * len(methods) + 1.2)))
    for y, method in enumerate(methods):
        item = effects[method]
        ax.errorbar(item["median"], y,
                    xerr=[[max(0, item["median"] - item["low"])], [max(0, item["high"] - item["median"])]],
                    fmt="o", color=COLORS[y % len(COLORS)], capsize=2.8, ms=4)
    ax.axvline(0.0, color="#555555", linestyle="--", linewidth=0.8)
    ax.set_yticks(range(len(methods)), [f"{m.replace('_', ' ')} vs {effects[m]['reference'].replace('_', ' ')}" for m in methods])
    ax.invert_yaxis()
    ax.set_xlabel(r"Function-family median paired $\Delta\log_{10}(f_{best})$ (negative favours method)")
    ax.set_title("Paired quality effects with function-family bootstrap intervals")
    ax.grid(axis="x")
    fig.text(0.5, 0.006, "Bootstrap resamples function families; dimensions and paired seeds are reduced within family. Descriptive for exploratory pilots.",
             ha="center", fontsize=6.0, color="#687481")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    return save_figure(fig, output, f"{stem_prefix}_paired_effect_forest")


def plot_win_loss(rows: list[dict[str, Any]], output: Path, stem_prefix: str) -> list[Path] | None:
    effects = _paired_effects(_quality_rows(rows))
    if not effects:
        return None
    methods = list(effects)
    totals = np.asarray([[effects[m]["wins"], effects[m]["ties"], effects[m]["losses"]] for m in methods], dtype=float)
    fig, ax = plt.subplots(figsize=(7.2, max(2.8, 0.36 * len(methods) + 1.0)))
    left = np.zeros(len(methods))
    labels = [("Wins", "#3A8D8C"), ("Ties", "#AAB3BC"), ("Losses", "#B65D58")]
    for j, (name, color) in enumerate(labels):
        ax.barh(np.arange(len(methods)), totals[:, j], left=left, color=color, edgecolor="white", linewidth=0.4, label=name)
        left += totals[:, j]
    ax.set_yticks(np.arange(len(methods)), [m.replace("_", " ") for m in methods])
    ax.invert_yaxis()
    ax.set_xlabel("Number of matched run comparisons")
    ax.set_title("Paired win / tie / loss counts (descriptive, not independent tests)")
    ax.grid(axis="x")
    ax.legend(frameon=False, ncol=3, loc="lower right")
    fig.tight_layout()
    return save_figure(fig, output, f"{stem_prefix}_paired_win_tie_loss")


def _success_and_ert(rows: list[dict[str, Any]]) -> tuple[list[str], list[str], np.ndarray, np.ndarray, np.ndarray]:
    methods = _methods(rows)
    cells = sorted({(str(r.get("benchmark")), int(finite_float(r.get("dimension"), -1))) for r in rows})
    success = np.full((len(cells), len(methods)), np.nan)
    ert = np.full_like(success, np.nan)
    nsucc = np.zeros_like(success)
    for i, (benchmark, dim) in enumerate(cells):
        for j, method in enumerate(methods):
            group = [r for r in rows if str(r.get("benchmark")) == benchmark and int(finite_float(r.get("dimension"), -1)) == dim and str(r.get("method")) == method]
            if not group:
                continue
            flags = []
            fe_costs = []
            for r in group:
                target = finite_float(r.get("success_target"), finite_float(r.get("success_target_default")))
                flag = truthy(r.get("success")) if "success" in r else (
                    math.isfinite(target) and finite_float(r.get("best_value")) <= target)
                flags.append(flag)
                budget = finite_float(r.get("function_evaluations"), math.nan)
                hit = finite_float(r.get("evaluations_to_target"), math.nan)
                fe_costs.append(hit if flag and math.isfinite(hit) else budget)
            success[i, j] = float(np.mean(flags))
            nsucc[i, j] = float(np.sum(flags))
            ert[i, j] = float(np.sum(fe_costs) / np.sum(flags)) if np.sum(flags) else math.inf
    labels = [f"{b} D={d}" for b, d in cells]
    return methods, labels, success, ert, nsucc


def _heatmap(matrix: np.ndarray, xlabels: list[str], ylabels: list[str], title: str, cbar: str,
             output: Path, stem: str, *, cmap: str = "viridis", vmin=None, vmax=None,
             annotate: bool = False) -> list[Path]:
    fig, ax = plt.subplots(figsize=(max(6.5, 0.72 * len(xlabels)), max(3.0, 0.38 * len(ylabels))))
    image = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(np.arange(len(xlabels)), [s.replace("_", "\n") for s in xlabels], fontsize=6)
    ax.set_yticks(np.arange(len(ylabels)), ylabels)
    ax.set_title(title)
    cb = fig.colorbar(image, ax=ax, pad=0.015)
    cb.set_label(cbar)
    if annotate and matrix.size < 160:
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                if math.isfinite(matrix[i, j]):
                    ax.text(j, i, f"{matrix[i,j]:.2g}", ha="center", va="center", fontsize=5.6,
                            color="white" if matrix[i, j] < np.nanmedian(matrix) else "black")
    fig.tight_layout()
    return save_figure(fig, output, stem)


def plot_success_ert(rows: list[dict[str, Any]], output: Path, stem_prefix: str) -> list[Path] | None:
    rows = _quality_rows(rows)
    if not rows:
        return None
    methods, labels, success, ert, nsucc = _success_and_ert(rows)
    if not np.isfinite(success).any():
        return None
    out = _heatmap(success, methods, labels, "Success rate by benchmark, dimension and method", "Empirical success rate",
                   output, f"{stem_prefix}_success_rate", vmin=0, vmax=1, annotate=True)
    log_ert = np.where(np.isfinite(ert) & (ert > 0), np.log10(ert), np.nan)
    if np.isfinite(log_ert).any():
        out += _heatmap(log_ert, methods, labels, "Expected running time (successful runs only; censored failures)",
                        r"$\log_{10}(\mathrm{ERT})$ [FE]", output, f"{stem_prefix}_ert_log10", annotate=True)
    return out


def plot_runtime_quality(rows: list[dict[str, Any]], output: Path, stem_prefix: str) -> list[Path] | None:
    rows = _quality_rows(rows)
    methods = _methods(rows)
    if len(methods) < 2 or not any(math.isfinite(finite_float(r.get("elapsed_seconds"))) for r in rows):
        return None
    # Runtime distributions by executor are descriptive: no speedup claim across different dynamics.
    groups = [[finite_float(r.get("elapsed_seconds")) for r in rows if str(r.get("method")) == m
               and math.isfinite(finite_float(r.get("elapsed_seconds")))] for m in methods]
    fig, ax = plt.subplots(figsize=(max(6.2, 0.58 * len(methods)), 3.4))
    bp = ax.boxplot(groups, patch_artist=True, showfliers=False, widths=0.62,
                    medianprops={"color": "black", "linewidth": 1.0})
    for i, patch in enumerate(bp["boxes"]):
        patch.set_facecolor(COLORS[i % len(COLORS)]); patch.set_alpha(0.5)
    ax.set_yscale("log")
    ax.set_xticks(np.arange(1, len(methods) + 1), [m.replace("_", "\n") for m in methods], fontsize=5.7)
    ax.set_ylabel("Elapsed wall time (s; log scale)")
    ax.set_title("Observed runtime distributions across quality runs")
    ax.grid(axis="y")
    fig.text(0.5, 0.006, "Descriptive only: asynchronous/island algorithms have different search dynamics; values are not interpreted as speedup.",
             ha="center", fontsize=6.0, color="#687481")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    out = save_figure(fig, output, f"{stem_prefix}_runtime_distribution")

    # Joint endpoint quality/runtime scatter; one point per run.
    fig, ax = plt.subplots(figsize=(6.6, 3.8))
    for i, method in enumerate(methods):
        sub = [r for r in rows if str(r.get("method")) == method and finite_float(r.get("best_value")) > 0
               and finite_float(r.get("elapsed_seconds")) > 0]
        if not sub: continue
        x = [finite_float(r.get("elapsed_seconds")) for r in sub]
        y = [math.log10(max(finite_float(r.get("best_value")), LOG_FLOOR)) for r in sub]
        ax.scatter(x, y, s=18, alpha=0.65, color=COLORS[i % len(COLORS)], label=method.replace("_", " "))
    ax.set_xscale("log")
    ax.set_xlabel("Elapsed wall time (s; log scale)")
    ax.set_ylabel(r"$\log_{10}(f_{best})$")
    ax.set_title("Observed quality–runtime pairs (one marker per run)")
    ax.grid(True); ax.legend(frameon=False, ncol=2)
    fig.text(0.5, 0.006, "No causal speedup interpretation: different search dynamics and machine-local elapsed times.", ha="center", fontsize=6.0, color="#687481")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    out += save_figure(fig, output, f"{stem_prefix}_quality_runtime_scatter")
    return out


def plot_ranks_and_inference(campaign_dir: Path, rows: list[dict[str, Any]], output: Path, stem_prefix: str) -> list[Path]:
    generated: list[Path] = []
    rank_path = campaign_dir / "method_average_ranks.csv"
    if rank_path.is_file():
        with rank_path.open("r", newline="", encoding="utf-8-sig") as f:
            rank_rows = list(csv.DictReader(f))
        if rank_rows:
            studies = sorted({(r.get("stage", ""), r.get("study", "")) for r in rank_rows})
            for stage, study in studies:
                sub = [r for r in rank_rows if r.get("stage") == stage and r.get("study") == study]
                sub.sort(key=lambda r: finite_float(r.get("average_rank"), math.inf))
                vals = [finite_float(r.get("average_rank")) for r in sub]
                fig, ax = plt.subplots(figsize=(6.8, max(2.8, 0.34 * len(sub) + 0.9)))
                ax.barh(np.arange(len(sub)), vals, color=[COLORS[i % len(COLORS)] for i in range(len(sub))])
                ax.set_yticks(np.arange(len(sub)), [r.get("method", "") for r in sub])
                ax.invert_yaxis(); ax.set_xlabel("Average rank (lower is better)")
                ax.set_title(f"Function-family average ranks · {study}"); ax.grid(axis="x")
                fig.tight_layout()
                generated += save_figure(fig, output, f"{stem_prefix}_{safe_slug(stage)}_{safe_slug(study)}_average_ranks")
    omnibus_path = campaign_dir / "omnibus_friedman_tests.csv"
    if omnibus_path.is_file():
        with omnibus_path.open("r", newline="", encoding="utf-8-sig") as f:
            tests = list(csv.DictReader(f))
        valid = [r for r in tests if math.isfinite(finite_float(r.get("friedman_p_block_permutation"))) and finite_float(r.get("friedman_p_block_permutation")) > 0]
        if valid:
            labels = [f"{r.get('stage','')} / {r.get('study','')}" for r in valid]
            pvals = [finite_float(r.get("friedman_p_block_permutation")) for r in valid]
            fig, ax = plt.subplots(figsize=(7.0, max(2.8, 0.36 * len(valid) + 0.9)))
            ax.barh(np.arange(len(valid)), -np.log10(pvals), color=COLORS[1])
            ax.axvline(-math.log10(0.05), color="#B65D58", linestyle="--", linewidth=0.8, label="0.05 reference")
            ax.set_yticks(np.arange(len(valid)), labels); ax.invert_yaxis()
            ax.set_xlabel(r"$-\log_{10}(p)$, blocked permutation omnibus")
            ax.set_title("Omnibus Friedman permutation diagnostics"); ax.grid(axis="x"); ax.legend(frameon=False)
            fig.tight_layout()
            generated += save_figure(fig, output, f"{stem_prefix}_omnibus_permutation_pvalues")
    return generated


def plot_scaling(rows: list[dict[str, Any]], output: Path, stem_prefix: str) -> list[Path]:
    """Paired synchronous scaling, stratified by study and topology.

    A data point is formed only from the serial/parallel pair with the same
    stage, study, objective workload, topology, D, N and seed. Distinct
    topologies or study protocols are never pooled into one speedup estimate.
    """
    perf = [r for r in rows if str(r.get("benchmark")) == "sphere_work"
            and str(r.get("architecture")) in {"serial", "master_slave", "particle_parallel"}]
    if not perf:
        return []

    lookup = {}
    for row in perf:
        key = (
            str(row.get("stage")), str(row.get("study")),
            int(finite_float(row.get("dimension"), -1)),
            int(finite_float(row.get("population_size"), -1)),
            int(finite_float(row.get("work_iterations"), 0)),
            str(row.get("topology", "global")),
            int(finite_float(row.get("seed"), -1)), str(row.get("method")),
        )
        lookup[key] = row

    paired: dict[tuple[str, str, str, int, int, int, str, int], list[dict[str, float]]] = defaultdict(list)
    for row in perf:
        arch = str(row.get("architecture"))
        if arch == "serial":
            continue
        workers = int(finite_float(row.get("workers"), 1))
        stage = str(row.get("stage")); study = str(row.get("study"))
        dim = int(finite_float(row.get("dimension"), -1))
        population = int(finite_float(row.get("population_size"), -1))
        workload = int(finite_float(row.get("work_iterations"), 0))
        topology = str(row.get("topology", "global"))
        seed = int(finite_float(row.get("seed"), -1))
        base = lookup.get((stage, study, dim, population, workload, topology, seed, f"serial_{topology}"))
        if base is None:
            base = next((candidate for candidate in perf
                         if str(candidate.get("stage")) == stage
                         and str(candidate.get("study")) == study
                         and int(finite_float(candidate.get("dimension"), -1)) == dim
                         and int(finite_float(candidate.get("population_size"), -1)) == population
                         and int(finite_float(candidate.get("work_iterations"), 0)) == workload
                         and str(candidate.get("topology", "global")) == topology
                         and int(finite_float(candidate.get("seed"), -1)) == seed
                         and str(candidate.get("architecture")) == "serial"), None)
        if base is None:
            continue
        t = finite_float(row.get("elapsed_seconds")); t0 = finite_float(base.get("elapsed_seconds"))
        if not (math.isfinite(t) and t > 0 and math.isfinite(t0) and t0 > 0):
            continue
        fe = finite_float(row.get("function_evaluations"))
        rate = fe / t if math.isfinite(fe) and fe > 0 else math.nan
        key = (stage, study, arch, dim, population, workload, topology, workers)
        paired[key].append({"speedup": t0 / t, "runtime": t, "efficiency": (t0 / t) / workers,
                            "throughput": rate})

    if not paired:
        return []
    generated: list[Path] = []
    experiment_groups = sorted({(key[0], key[1], key[5], key[6]) for key in paired})
    for stage, study, workload, topology in experiment_groups:
        selected_keys = [key for key in paired if key[0] == stage and key[1] == study
                         and key[5] == workload and key[6] == topology]
        fig, axes = plt.subplots(2, 2, figsize=(9.0, 6.0), squeeze=False)
        metric_specs = [
            ("runtime", "Median elapsed wall time (s)", "Absolute runtime"),
            ("speedup", "Median paired speedup", "End-to-end speedup"),
            ("efficiency", "Median parallel efficiency", "Parallel efficiency"),
            ("throughput", "Median function evaluations / second", "Throughput"),
        ]
        line_index = 0
        for arch in ("master_slave", "particle_parallel"):
            arch_keys = [key for key in selected_keys if key[2] == arch]
            combos = sorted({(key[3], key[4]) for key in arch_keys})
            for dim, population in combos:
                curve = sorted([key for key in arch_keys if key[3] == dim and key[4] == population], key=lambda k:k[7])
                if not curve:
                    continue
                label = f"{arch.replace('_', ' ')} · D={dim}, N={population}"
                color = COLORS[line_index % len(COLORS)]
                line_index += 1
                for ax, (metric, _, _) in zip(axes.flat, metric_specs):
                    xs=[]; med=[]; lows=[]; highs=[]
                    for key in curve:
                        vals=[entry[metric] for entry in paired[key] if math.isfinite(entry[metric])]
                        if not vals:
                            continue
                        lo,hi=bootstrap_median(vals,seed=20261107+key[7]+sum(map(ord,metric)))
                        xs.append(key[7]);med.append(float(np.median(vals)));lows.append(lo);highs.append(hi)
                    if not xs:
                        continue
                    if metric == "runtime":
                        ax.plot(xs,med,marker="o",color=color,label=label)
                    else:
                        ax.plot(xs,med,marker="o",color=color,label=label)
                        ax.fill_between(xs,lows,highs,color=color,alpha=.13,linewidth=0)
        # Add the actual serial baseline at one worker to runtime and throughput panels.
        baseline_dims = sorted({(key[3], key[4]) for key in selected_keys})
        for dim, population in baseline_dims:
            base_rows = [r for r in perf
                         if str(r.get("stage")) == stage and str(r.get("study")) == study
                         and int(finite_float(r.get("dimension"), -1)) == dim
                         and int(finite_float(r.get("population_size"), -1)) == population
                         and int(finite_float(r.get("work_iterations"), 0)) == workload
                         and str(r.get("topology", "global")) == topology
                         and str(r.get("architecture")) == "serial"]
            base_times = [finite_float(r.get("elapsed_seconds")) for r in base_rows
                          if finite_float(r.get("elapsed_seconds")) > 0]
            base_rates = [finite_float(r.get("function_evaluations")) / finite_float(r.get("elapsed_seconds"))
                          for r in base_rows if finite_float(r.get("function_evaluations")) > 0
                          and finite_float(r.get("elapsed_seconds")) > 0]
            if base_times:
                axes.flat[0].errorbar([1], [float(np.median(base_times))],
                                      yerr=[[float(np.median(base_times))-bootstrap_median(base_times, seed=8101)[0]],
                                            [bootstrap_median(base_times, seed=8101)[1]-float(np.median(base_times))]],
                                      fmt="s", ms=4, color="#222222", capsize=2,
                                      label=f"serial baseline · D={dim}, N={population}")
            if base_rates:
                axes.flat[3].errorbar([1], [float(np.median(base_rates))],
                                      yerr=[[float(np.median(base_rates))-bootstrap_median(base_rates, seed=8201)[0]],
                                            [bootstrap_median(base_rates, seed=8201)[1]-float(np.median(base_rates))]],
                                      fmt="s", ms=4, color="#222222", capsize=2,
                                      label=f"serial baseline · D={dim}, N={population}")
        for ax, (metric, ylabel, title) in zip(axes.flat, metric_specs):
            if metric == "speedup":
                max_workers=max(key[7] for key in selected_keys)
                ax.plot([1,max_workers],[1,max_workers],"k--",lw=.8,label="Ideal")
            elif metric == "efficiency":
                ax.axhline(1,color="#555555",ls="--",lw=.8,label="Ideal efficiency")
            ax.set_xlabel("Worker threads");ax.set_ylabel(ylabel);ax.set_title(title)
            x_ticks = {key[7] for key in selected_keys}
            if metric in {"runtime", "throughput"}:
                x_ticks.add(1)
            ax.set_xticks(sorted(x_ticks));ax.grid(axis="y");ax.legend(frameon=False,fontsize=6)
        fig.suptitle(f"Synchronous scaling · {stage}/{study} · topology={topology} · synthetic work={workload:,}",y=1.005)
        fig.text(.5,.004,"Pairs are matched by topology, D, N, workload and seed. sphere_work is a cost-control workload, not a solution-quality benchmark. Timing is machine-specific.",ha="center",fontsize=6,color="#687481")
        fig.tight_layout(rect=(0,.035,1,.965))
        generated += save_figure(fig,output,f"{stem_prefix}_scaling_{safe_slug(stage)}_{safe_slug(study)}_{safe_slug(topology)}_work{workload}")
    return generated


def plot_runtime_components(rows: list[dict[str, Any]], output: Path, stem_prefix: str) -> list[Path] | None:
    perf = [r for r in rows if str(r.get("benchmark")) == "sphere_work"]
    components = [
        ("initialization_seconds", "Initialization"),
        ("topology_setup_seconds", "Topology setup"),
        ("evaluation_wall_seconds", "Objective evaluation"),
        ("propagation_wall_seconds", "Propagation"),
        ("social_selection_seconds", "Social selection"),
        ("migration_seconds", "Migration"),
        ("async_update_wall_seconds", "Async update"),
    ]
    available = [(key, label) for key, label in components if any(math.isfinite(finite_float(r.get(key))) for r in perf)]
    if not perf or not available:
        return None
    # Split by study/D/N/workload. Never pool different synthetic costs or dimensions.
    combos = sorted({(str(r.get("study", "study")), int(finite_float(r.get("dimension"), -1)),
                      int(finite_float(r.get("population_size"), -1)), int(finite_float(r.get("work_iterations"), 0)))
                     for r in perf})
    generated: list[Path] = []
    for study, dimension, population, workload in combos:
        selected = [r for r in perf if str(r.get("study", "study")) == study
                    and int(finite_float(r.get("dimension"), -1)) == dimension
                    and int(finite_float(r.get("population_size"), -1)) == population
                    and int(finite_float(r.get("work_iterations"), 0)) == workload]
        methods = _methods(selected)
        if not methods:
            continue
        fig, ax = plt.subplots(figsize=(max(7.0, 0.78 * len(methods)), 4.0))
        x = np.arange(len(methods))
        width = 0.82 / len(available)
        for cidx, (field, label) in enumerate(available):
            vals = []
            for method in methods:
                observations = [finite_float(r.get(field)) for r in selected if str(r.get("method")) == method
                                and math.isfinite(finite_float(r.get(field)))]
                vals.append(float(np.median(observations)) if observations else 0.0)
            ax.bar(x - 0.41 + width * (cidx + 0.5), vals, width=width, label=label,
                   color=COLORS[cidx % len(COLORS)])
        totals = []
        for method in methods:
            observations = [finite_float(r.get("elapsed_seconds")) for r in selected if str(r.get("method")) == method
                            and math.isfinite(finite_float(r.get("elapsed_seconds")))]
            totals.append(float(np.median(observations)) if observations else math.nan)
        ax.plot(x, totals, "ko--", ms=3, lw=0.8, label="Total elapsed")
        ax.set_xticks(x, [m.replace("_", "\n") for m in methods], fontsize=6)
        ax.set_ylabel("Median measured phase wall time (s)")
        ax.set_title(f"Runtime components · D={dimension}, N={population}, work={workload:,}")
        ax.grid(axis="y"); ax.legend(frameon=False, ncol=2)
        fig.text(0.5, 0.006, "Phase measurements are shown independently; they are not assumed to be mutually exclusive or additive.",
                 ha="center", fontsize=6, color="#687481")
        fig.tight_layout(rect=(0, 0.04, 1, 1))
        generated += save_figure(fig, output, f"{stem_prefix}_runtime_components_d{dimension}_n{population}_work{workload}")
    return generated


def plot_island_diagnostics(rows: list[dict[str, Any]], output: Path, stem_prefix: str) -> list[Path]:
    island_rows = [r for r in _quality_rows(rows) if str(r.get("architecture")) == "islands"]
    if not island_rows:
        return []
    generated = []
    methods = _methods(island_rows)
    # Quality distribution stratified by island count and migration policy.
    xlabels = []
    values = []
    for method in methods:
        group = [r for r in island_rows if str(r.get("method")) == method]
        if not group: continue
        xlabels.append(f"{int(finite_float(group[0].get('island_count'), 1))} islands\n{group[0].get('migration_topology','none')}\n$\\tau_m$={int(finite_float(group[0].get('migration_interval'), 0))}")
        values.append([math.log10(max(finite_float(r.get("best_value")), LOG_FLOOR)) for r in group])
    if values:
        fig, ax = plt.subplots(figsize=(max(6.0, 0.9 * len(values)), 3.6))
        bp = ax.boxplot(values, patch_artist=True, showfliers=False, widths=0.62)
        for i, patch in enumerate(bp["boxes"]): patch.set_facecolor(COLORS[i % len(COLORS)]); patch.set_alpha(0.5)
        ax.set_xticks(np.arange(1, len(xlabels)+1), xlabels, fontsize=6)
        ax.set_ylabel(r"$\log_{10}(f_{best})$"); ax.set_title("Island-model final quality by configuration"); ax.grid(axis="y")
        fig.tight_layout(); generated += save_figure(fig, output, f"{stem_prefix}_island_configuration_quality")
    with_migrations = [r for r in island_rows if finite_float(r.get("migrations_performed"), math.nan) >= 0]
    if with_migrations:
        fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.1))
        axes[0].scatter([finite_float(r.get("migrations_performed")) for r in with_migrations],
                        [math.log10(max(finite_float(r.get("best_value")), LOG_FLOOR)) for r in with_migrations],
                        c=[COLORS[methods.index(str(r.get("method"))) % len(COLORS)] for r in with_migrations], s=22, alpha=0.7)
        axes[0].set_xlabel("Migrations performed"); axes[0].set_ylabel(r"$\log_{10}(f_{best})$"); axes[0].set_title("Migration count vs final quality")
        fraction = [finite_float(r.get("migration_seconds")) / finite_float(r.get("elapsed_seconds"))
                    for r in with_migrations if finite_float(r.get("elapsed_seconds")) > 0]
        if fraction:
            axes[1].hist(fraction, bins=min(12, max(4, len(fraction)//2)), color=COLORS[2], alpha=0.8)
        axes[1].set_xlabel("Migration time / elapsed time"); axes[1].set_ylabel("Run count"); axes[1].set_title("Migration timing fraction")
        for ax in axes: ax.grid(axis="y")
        fig.tight_layout(); generated += save_figure(fig, output, f"{stem_prefix}_island_migration_diagnostics")
    return generated


def plot_topology_diagnostics(rows: list[dict[str, Any]], output: Path, stem_prefix: str) -> list[Path]:
    quality = _quality_rows(rows)
    if not quality or len({str(r.get("topology")) for r in quality}) < 3:
        return []
    topologies = sorted({str(r.get("topology")) for r in quality})
    benchmarks = sorted({str(r.get("benchmark")) for r in quality})
    dims = sorted({int(finite_float(r.get("dimension"), -1)) for r in quality})
    # Median log-quality per topology and benchmark/dimension.
    generated = []
    for dim in dims:
        matrix = np.full((len(benchmarks), len(topologies)), np.nan)
        for i, bench in enumerate(benchmarks):
            for j, topo in enumerate(topologies):
                vals = [math.log10(max(finite_float(r.get("best_value")), LOG_FLOOR)) for r in quality
                        if r.get("benchmark") == bench and int(finite_float(r.get("dimension"), -1)) == dim and r.get("topology") == topo]
                if vals: matrix[i, j] = np.median(vals)
        generated += _heatmap(matrix, topologies, [b.capitalize() for b in benchmarks],
                              f"Topology quality by benchmark · D={dim}", r"Median $\log_{10}(f_{best})$",
                              output, f"{stem_prefix}_topology_quality_heatmap_d{dim}")
    # Function-family paired effects versus global topology.
    effects: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    lookup = {(r.get("benchmark"), int(finite_float(r.get("dimension"), -1)), int(finite_float(r.get("seed"), -1)), r.get("topology")): r for r in quality}
    for r in quality:
        topo = str(r.get("topology")); bench = str(r.get("benchmark")); dim = int(finite_float(r.get("dimension"), -1)); seed = int(finite_float(r.get("seed"), -1))
        if topo == "global": continue
        ref = lookup.get((bench, dim, seed, "global"))
        if ref is None: continue
        delta = math.log10(max(finite_float(r.get("best_value")), LOG_FLOOR)) - math.log10(max(finite_float(ref.get("best_value")), LOG_FLOOR))
        effects[topo][bench].append(delta)
    available = [t for t in topologies if t != "global" and effects[t]]
    if available:
        families = sorted(set.intersection(*(set(effects[t]) for t in available))) if len(available) > 1 else sorted(effects[available[0]])
        fig, ax = plt.subplots(figsize=(7.0, 3.6))
        data = [[np.median([np.median(effects[t][fam]) for fam in [fam]]) for fam in families] for t in available]
        bp = ax.boxplot(data, patch_artist=True, showfliers=False)
        for i, patch in enumerate(bp["boxes"]): patch.set_facecolor(COLORS[i % len(COLORS)]); patch.set_alpha(0.58)
        ax.axhline(0, color="#555555", linestyle="--", lw=0.8)
        ax.set_xticks(np.arange(1, len(available)+1), [t.replace("_", " ") for t in available])
        ax.set_ylabel(r"Function-family paired $\Delta\log_{10}(f_{best})$ vs global")
        ax.set_title("Relative topology effects (negative favours topology)"); ax.grid(axis="y")
        fig.text(0.5, 0.006, "Paired by benchmark, dimension and seed; each plotted family summary is descriptive.", ha="center", fontsize=6, color="#687481")
        fig.tight_layout(rect=(0,0.04,1,1)); generated += save_figure(fig, output, f"{stem_prefix}_topology_effects_vs_global")
    # Rank matrix across benchmark families and methods/topologies.
    rankmat = np.full((len(benchmarks), len(topologies)), np.nan)
    for i, bench in enumerate(benchmarks):
        family_scores = []
        for topo in topologies:
            vals = [math.log10(max(finite_float(r.get("best_value")), LOG_FLOOR)) for r in quality if r.get("benchmark") == bench and r.get("topology") == topo]
            family_scores.append(float(np.median(vals)) if vals else math.nan)
        valid = np.isfinite(family_scores)
        if np.any(valid):
            vals = np.asarray(family_scores)
            order = np.argsort(vals[valid], kind="stable")
            ranks = np.empty(np.sum(valid), dtype=float); ranks[order] = np.arange(1, np.sum(valid)+1)
            rankmat[i, valid] = ranks
    generated += _heatmap(rankmat, topologies, [b.capitalize() for b in benchmarks],
                          "Topology ranks by function family (lower is better)", "Average rank", output,
                          f"{stem_prefix}_topology_rank_heatmap", cmap="viridis_r", annotate=True)
    return generated


def plot_campaign_coverage(campaign_dir: Path, rows: list[dict[str, Any]], output: Path, stem_prefix: str) -> list[Path]:
    manifest = load_manifest(campaign_dir)
    campaign = manifest.get("campaign", {}) if isinstance(manifest, dict) else {}
    stages = campaign.get("stages", []) if isinstance(campaign, dict) else []
    if not stages:
        return []
    labels, completed, expected = [], [], []
    for stage in stages:
        for study in stage.get("studies", []):
            cfg = study.get("study", {})
            n_expected = len(cfg.get("benchmarks", [])) * len(cfg.get("dimensions", [])) * len(cfg.get("seeds", [])) * len(cfg.get("methods", []))
            actual = sum(1 for r in rows if r.get("stage") == stage.get("id") and r.get("study") == study.get("id"))
            if n_expected:
                labels.append(f"{stage.get('id')}\n{study.get('id')}"); completed.append(actual); expected.append(n_expected)
    if not labels: return []
    fig, ax = plt.subplots(figsize=(7.2, max(2.6, 0.48 * len(labels) + 0.8)))
    y = np.arange(len(labels)); ax.barh(y, expected, color="#E1E6EA", label="Expected")
    ax.barh(y, completed, color=COLORS[1], label="Completed")
    ax.set_yticks(y, labels); ax.invert_yaxis(); ax.set_xlabel("Optimization runs")
    ax.set_title("Campaign execution coverage"); ax.grid(axis="x"); ax.legend(frameon=False)
    for i,(c,e) in enumerate(zip(completed, expected)): ax.text(max(c,e)*1.01, i, f"{c}/{e} ({c/e:.0%})", va="center", fontsize=6)
    fig.tight_layout(); return save_figure(fig, output, f"{stem_prefix}_campaign_coverage")


def generate(campaign_dir: Path, output: Path) -> tuple[list[Path], list[str]]:
    campaign_dir = campaign_dir.expanduser().resolve()
    manifest = load_manifest(campaign_dir)
    rows = enrich_from_manifest(discover_internal_records(campaign_dir), manifest)
    if not rows:
        raise FileNotFoundError(f"No runs.jsonl or campaign_runs.csv found under {campaign_dir}")
    configure_style()
    output.mkdir(parents=True, exist_ok=True)
    generated: list[Path] = []
    missing: list[str] = []
    stem_prefix = safe_slug(str(manifest.get("campaign_id", campaign_dir.name)))

    groups = group_by_study(rows)
    for (stage, study), study_rows in groups.items():
        stem = f"{safe_slug(stage)}_{safe_slug(study)}"
        qrows = _quality_rows(study_rows)
        makers = [
            ("final quality distributions", lambda: plot_final_distributions(qrows, output, stem)),
            ("convergence profiles", lambda: plot_convergence(qrows, output, stem)),
            ("quality heatmaps", lambda: plot_quality_heatmap(qrows, output, stem)),
            ("paired effect forest", lambda: plot_effects(qrows, output, stem)),
            ("paired win/tie/loss", lambda: plot_win_loss(qrows, output, stem)),
            ("success and ERT", lambda: plot_success_ert(qrows, output, stem)),
            ("runtime / quality", lambda: plot_runtime_quality(qrows, output, stem)),
        ]
        study_paths = []
        # Performance-only sphere_work studies intentionally have no objective-quality plots.
        if qrows:
            for label, maker in makers:
                try:
                    result = maker()
                    if result: study_paths.extend(result)
                    else: missing.append(f"{stage}/{study}: {label} unavailable from recorded fields")
                except Exception as exc:
                    missing.append(f"{stage}/{study}: {label} failed: {type(exc).__name__}: {exc}")
        elif not any(str(r.get("benchmark")) != "sphere_work" for r in study_rows):
            pass
        else:
            missing.append(f"{stage}/{study}: objective-quality fields unavailable or invalid")
        generated.extend(study_paths)
        for label, maker in (
            ("island diagnostics", lambda: plot_island_diagnostics(study_rows, output, stem)),
            ("topology diagnostics", lambda: plot_topology_diagnostics(study_rows, output, stem)),
            ("scaling", lambda: plot_scaling(study_rows, output, stem)),
            ("runtime components", lambda: plot_runtime_components(study_rows, output, stem)),
        ):
            try:
                result = maker()
                if result: generated.extend(result)
            except Exception as exc:
                missing.append(f"{stage}/{study}: {label} failed: {type(exc).__name__}: {exc}")

    # Campaign-level post-analysis figures, when the analyzer has written its tables.
    try:
        generated.extend(plot_ranks_and_inference(campaign_dir, rows, output, stem_prefix))
    except Exception as exc:
        missing.append(f"campaign inference: {type(exc).__name__}: {exc}")
    try:
        generated.extend(plot_campaign_coverage(campaign_dir, rows, output, stem_prefix))
    except Exception as exc:
        missing.append(f"campaign coverage: {type(exc).__name__}: {exc}")

    catalogue = [
        f"# Figure atlas — {campaign_dir.name}", "",
        f"Evidence class: **{evidence_label(manifest)}**.",
        f"Completed records loaded: **{len(rows)}**.",
        "All figures are calculated from recorded run-level observations; missing groups are not synthesized.",
        "Timing figures are local to the hardware in the campaign manifest. Async/island wall times are descriptive, not speedups.",
        "",
        "## Generated files", "",
    ] + [f"- `{p.name}`" for p in generated if p.suffix == ".pdf"] + ["", "## Not generated / warnings", ""] + [f"- {x}" for x in missing]
    (output / "FIGURE_ATLAS.md").write_text("\n".join(catalogue) + "\n", encoding="utf-8")
    (output / "figure_atlas_manifest.json").write_text(json.dumps({
        "campaign_directory": str(campaign_dir), "evidence_class": evidence_label(manifest),
        "records_loaded": len(rows), "generated_pdf": [p.name for p in generated if p.suffix == ".pdf"],
        "generated_png": [p.name for p in generated if p.suffix == ".png"], "warnings": missing,
    }, indent=2), encoding="utf-8")
    return generated, missing


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True, help="Campaign output directory containing campaign_manifest.json and nested runs.jsonl")
    parser.add_argument("--output", type=Path, default=None, help="Figure output directory")
    args = parser.parse_args()
    campaign = args.campaign.expanduser().resolve()
    output = args.output.expanduser().resolve() if args.output else Path(__file__).resolve().parents[1] / "results/figures/parallel" / campaign.name
    from cpso_serial.output_policy import results_output
    output = results_output(output)
    generated, missing = generate(campaign, output)
    pdfs = [p for p in generated if p.suffix == ".pdf"]
    print(f"Generated {len(pdfs)} PDF figures ({len(generated)-len(pdfs)} PNG) from {campaign}")
    if missing:
        print("Warnings / not generated:")
        for item in missing: print(f"  - {item}")
    print(f"Catalogue: {output / 'FIGURE_ATLAS.md'}")
    return 0 if pdfs else 2


if __name__ == "__main__":
    raise SystemExit(main())
