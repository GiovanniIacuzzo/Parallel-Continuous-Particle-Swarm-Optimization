#!/usr/bin/env python3
"""Validate and analyze the current native CPSO paired-dt screening campaign.

This is a screening analysis, not a confirmatory hypothesis-testing pipeline.
Seeds are paired within each fixed benchmark x dimension cell. The bootstrap
interval is conditional on the selected benchmark functions and dimensions and
resamples paired seeds only. Objective values are compared on log10 scale
because the benchmark functions have different numerical scales.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(ROOT))

from analysis.plot_style import PALETTE, configure_style  # noqa: E402

LOG_FLOOR = 1.0e-16


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected an object at {path}:{line_number}")
            rows.append(row)
    return rows


def _log_objective(value: float) -> float:
    if not math.isfinite(value) or value < -1.0e-12:
        raise ValueError(f"Expected a finite non-negative objective, received {value!r}")
    return math.log10(max(value, LOG_FLOOR))


def load_and_validate(input_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    manifest_path = input_dir / "study_config.json"
    records_path = input_dir / "runs.jsonl"
    if not manifest_path.is_file() or not records_path.is_file():
        raise FileNotFoundError(
            f"Expected both {manifest_path.name} and {records_path.name} in {input_dir}"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    config = manifest.get("study_config")
    if not isinstance(config, dict):
        raise ValueError("study_config.json does not contain a study_config object")
    records = _read_jsonl(records_path)
    if not records:
        raise ValueError("No dt-sensitivity records were found")

    benchmarks = [str(value) for value in config["benchmarks"]]
    dimensions = [int(value) for value in config["dimensions"]]
    seeds = [int(value) for value in config["seeds"]]
    dts = [float(value) for value in config["dt_values"]]
    reference_dt = float(config["reference_dt"])
    population = int(config["population_size"])
    iterations = int(config["iterations"])
    expected_fe = population * (iterations + 1)
    expected_keys = {
        (benchmark, dimension, seed, dt)
        for benchmark in benchmarks for dimension in dimensions
        for seed in seeds for dt in dts
    }
    actual: dict[tuple[str, int, int, float], dict[str, Any]] = {}
    expected_history_axis = [population * (iteration + 1) for iteration in range(iterations + 1)]

    required = {
        "benchmark", "dimension", "seed", "dt", "topology", "population_size", "iterations",
        "function_evaluations", "best_value", "best_position", "history_evaluations",
        "best_history", "elapsed_seconds",
    }
    for index, row in enumerate(records, start=1):
        missing = required - set(row)
        if missing:
            raise ValueError(f"Run record {index} is missing fields: {sorted(missing)}")
        key = (str(row["benchmark"]), int(row["dimension"]), int(row["seed"]), float(row["dt"]))
        if key in actual:
            raise ValueError(f"Duplicate dt-sensitivity run: {key}")
        if key not in expected_keys:
            raise ValueError(f"Unexpected dt-sensitivity run: {key}")
        if row["topology"] != "global":
            raise ValueError(f"Run {key} changed the fixed global-best topology")
        if int(row["population_size"]) != population or int(row["iterations"]) != iterations:
            raise ValueError(f"Population/iteration mismatch in run {key}")
        if int(row["function_evaluations"]) != expected_fe:
            raise ValueError(f"FE mismatch in run {key}: expected {expected_fe}")
        axis = [int(value) for value in row["history_evaluations"]]
        history = [float(value) for value in row["best_history"]]
        if axis != expected_history_axis or len(history) != iterations + 1:
            raise ValueError(f"Convergence-history axis/length mismatch in run {key}")
        if any(not math.isfinite(value) for value in history):
            raise ValueError(f"Non-finite convergence history in run {key}")
        if any(next_value > value + 1.0e-12 for value, next_value in zip(history, history[1:])):
            raise ValueError(f"Best-so-far history increased in run {key}")
        best_value = float(row["best_value"])
        elapsed = float(row["elapsed_seconds"])
        if not math.isfinite(best_value) or abs(history[-1] - best_value) > 1.0e-12 * max(1.0, abs(best_value)):
            raise ValueError(f"Final objective/history mismatch in run {key}")
        _log_objective(best_value)
        if not math.isfinite(elapsed) or elapsed <= 0.0:
            raise ValueError(f"Non-positive/non-finite elapsed time in run {key}")
        actual[key] = row

    missing_keys = expected_keys - set(actual)
    if missing_keys:
        example = sorted(missing_keys)[:5]
        raise ValueError(f"Incomplete factorial dt design: {len(missing_keys)} runs missing, e.g. {example}")
    if reference_dt not in dts:
        raise ValueError("Predeclared reference_dt is not in dt_values")

    evidence_class = str(config.get("evidence_class", "exploratory"))
    if evidence_class not in {"confirmatory", "exploratory", "integration_only"}:
        raise ValueError(f"Unsupported dt evidence_class: {evidence_class!r}")
    validation = {
        "campaign_id": config.get("campaign_id"),
        "evidence_class": evidence_class,
        "status": "complete_factorial_design_validated",
        "n_runs": len(records),
        "n_benchmarks": len(benchmarks),
        "benchmarks": benchmarks,
        "dimensions": dimensions,
        "n_seeds": len(seeds),
        "seeds": seeds,
        "dt_values": dts,
        "reference_dt": reference_dt,
        "population_size": population,
        "iterations": iterations,
        "function_evaluations_per_run": expected_fe,
        "total_function_evaluations": len(records) * expected_fe,
        "pairing": "same benchmark, dimension and seed for every dt value",
        "analysis_scope": "exploratory_screening_conditional_on_selected_benchmarks_and_dimensions",
        "bootstrap_scope": "paired seeds resampled within each fixed benchmark x dimension cell; benchmark functions and dimensions are fixed design factors",
    }
    return config, records, validation


def _paired_deltas(records: list[dict[str, Any]], validation: dict[str, Any]) -> dict[tuple[str, int, int, float], float]:
    values = {
        (str(row["benchmark"]), int(row["dimension"]), int(row["seed"]), float(row["dt"])):
            _log_objective(float(row["best_value"]))
        for row in records
    }
    reference_dt = float(validation["reference_dt"])
    deltas: dict[tuple[str, int, int, float], float] = {}
    for benchmark in validation["benchmarks"]:
        for dimension in validation["dimensions"]:
            for seed in validation["seeds"]:
                reference = values[(benchmark, dimension, seed, reference_dt)]
                for dt in validation["dt_values"]:
                    deltas[(benchmark, dimension, seed, float(dt))] = (
                        values[(benchmark, dimension, seed, float(dt))] - reference
                    )
    return deltas


def summarize_problem_blocks(
    records: list[dict[str, Any]], validation: dict[str, Any],
    deltas: dict[tuple[str, int, int, float], float],
) -> list[dict[str, Any]]:
    lookup = {
        (str(row["benchmark"]), int(row["dimension"]), int(row["seed"]), float(row["dt"])): row
        for row in records
    }
    summary: list[dict[str, Any]] = []
    for benchmark in validation["benchmarks"]:
        for dimension in validation["dimensions"]:
            for dt in validation["dt_values"]:
                values = [float(lookup[(benchmark, dimension, seed, dt)]["best_value"])
                          for seed in validation["seeds"]]
                delta = [deltas[(benchmark, dimension, seed, dt)] for seed in validation["seeds"]]
                summary.append({
                    "benchmark": benchmark,
                    "dimension": dimension,
                    "dt": dt,
                    "n_paired_seeds": len(values),
                    "median_best_value": float(np.median(values)),
                    "q25_best_value": float(np.quantile(values, 0.25)),
                    "q75_best_value": float(np.quantile(values, 0.75)),
                    "median_delta_log10_vs_reference": float(np.median(delta)),
                    "q25_delta_log10_vs_reference": float(np.quantile(delta, 0.25)),
                    "q75_delta_log10_vs_reference": float(np.quantile(delta, 0.75)),
                })
    return summary


def build_family_effects(
    validation: dict[str, Any],
    deltas: dict[tuple[str, int, int, float], float],
) -> tuple[list[dict[str, Any]], dict[float, dict[str, dict[int, list[float]]]]]:
    nested: dict[float, dict[str, dict[int, list[float]]]] = {}
    family_rows: list[dict[str, Any]] = []
    for dt in validation["dt_values"]:
        nested[float(dt)] = {}
        for benchmark in validation["benchmarks"]:
            nested[float(dt)][benchmark] = {}
            dimension_effects: list[float] = []
            for dimension in validation["dimensions"]:
                paired = [deltas[(benchmark, dimension, seed, float(dt))] for seed in validation["seeds"]]
                nested[float(dt)][benchmark][dimension] = paired
                dimension_effects.append(float(np.median(paired)))
            family_effect = float(np.median(dimension_effects))
            family_rows.append({
                "benchmark_family": benchmark,
                "dt": float(dt),
                "family_median_delta_log10_vs_reference": family_effect,
                "dimension_effects": json.dumps(
                    {
                        str(dimension): float(np.median(nested[float(dt)][benchmark][dimension]))
                        for dimension in validation["dimensions"]
                    },
                    sort_keys=True,
                ),
            })
    return family_rows, nested


def hierarchical_bootstrap_profile(
    nested: dict[float, dict[str, dict[int, list[float]]]],
    validation: dict[str, Any], *, replicates: int = 5000, seed: int = 20261107,
) -> list[dict[str, Any]]:
    """Bootstrap paired seeds, conditioning on the selected problems/dimensions.

    The six benchmark functions are a deliberately selected screening set, not
    a random sample from a population of all optimization problems. They are
    therefore kept fixed in every replicate. For each fixed function x
    dimension cell, only its paired seeds are resampled; cell and family
    medians are then recomputed. The interval describes seed uncertainty
    conditional on this screening design, not generalization to all problems.
    """
    rng = np.random.default_rng(seed)
    families = list(validation["benchmarks"])
    dimensions = list(validation["dimensions"])
    output = []
    for dt in validation["dt_values"]:
        family_values = []
        for family in families:
            by_dimension = [
                float(np.median(nested[float(dt)][family][dimension])) for dimension in dimensions
            ]
            family_values.append(float(np.median(by_dimension)))
        samples = np.empty(replicates, dtype=float)
        for replicate in range(replicates):
            replicate_family_effects = []
            for family in families:  # fixed benchmark set; do not resample families
                dimension_effects = []
                for dimension in dimensions:  # fixed dimensions; resample paired seeds only
                    paired = np.asarray(nested[float(dt)][family][dimension], dtype=float)
                    sampled_seeds = rng.choice(paired, size=paired.size, replace=True)
                    dimension_effects.append(float(np.median(sampled_seeds)))
                replicate_family_effects.append(float(np.median(dimension_effects)))
            samples[replicate] = float(np.median(replicate_family_effects))
        output.append({
            "dt": float(dt),
            "n_fixed_benchmark_functions": len(families),
            "median_family_effect_log10": float(np.median(family_values)),
            "seed_bootstrap_95_low": float(np.quantile(samples, 0.025)),
            "seed_bootstrap_95_high": float(np.quantile(samples, 0.975)),
            "bootstrap_replicates": int(replicates),
            "interval_scope": "conditional_on_selected_benchmarks_and_dimensions",
        })
    return output


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write an empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_figures(
    summary: list[dict[str, Any]], profile: list[dict[str, Any]], validation: dict[str, Any],
    output: Path,
) -> list[Path]:
    configure_style()
    plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.size": 8.2})
    output.mkdir(parents=True, exist_ok=True)
    generated: list[Path] = []
    benchmarks = validation["benchmarks"]
    dimensions = validation["dimensions"]
    dts = validation["dt_values"]
    reference_dt = float(validation["reference_dt"])

    # Heatmap of paired log10 differences: rows are benchmark x dimension,
    # columns are dt, and zero is the predeclared current reference.
    row_labels: list[str] = []
    matrix: list[list[float]] = []
    indexed = {(r["benchmark"], int(r["dimension"]), float(r["dt"])): r for r in summary}
    for benchmark in benchmarks:
        for dimension in dimensions:
            row_labels.append(f"{benchmark}, D={dimension}")
            matrix.append([
                float(indexed[(benchmark, dimension, float(dt))]["median_delta_log10_vs_reference"])
                for dt in dts
            ])
    image = np.asarray(matrix, dtype=float)
    finite = np.abs(image[np.isfinite(image)])
    limit = max(0.05, float(np.quantile(finite, 0.95))) if finite.size else 1.0
    fig_height = max(4.0, 0.30 * len(row_labels) + 1.2)
    fig, ax = plt.subplots(figsize=(8.0, fig_height))
    plotted = ax.imshow(image, aspect="auto", cmap="coolwarm", vmin=-limit, vmax=limit)
    ax.set_yticks(np.arange(len(row_labels)), row_labels)
    ax.set_xticks(np.arange(len(dts)), [f"{value:g}" for value in dts])
    ax.set_xlabel(r"CPSO interval length, $\Delta t$")
    evidence_label = " — exploratory pilot" if validation.get("evidence_class") == "exploratory" else (
        " — integration only" if validation.get("evidence_class") == "integration_only" else ""
    )
    ax.set_title(r"Paired sensitivity relative to the reference $\Delta t$" + evidence_label, weight="bold")
    for index, dt in enumerate(dts):
        if float(dt) == reference_dt:
            ax.axvline(index, color="black", linewidth=1.0, linestyle="--")
    colorbar = fig.colorbar(plotted, ax=ax, pad=0.015)
    colorbar.set_label(r"Median paired $\Delta\log_{10}(f)$ (negative is better)")
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    path = output / "fig_dt_sensitivity_problem_heatmap.pdf"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    generated.append(path)

    # Aggregated profile across the fixed selected benchmark set.
    values = np.asarray([float(row["median_family_effect_log10"]) for row in profile])
    lows = np.asarray([float(row["seed_bootstrap_95_low"]) for row in profile])
    highs = np.asarray([float(row["seed_bootstrap_95_high"]) for row in profile])
    x = np.asarray([float(row["dt"]) for row in profile])
    fig, ax = plt.subplots(figsize=(6.8, 3.8))
    ax.fill_between(x, lows, highs, color=PALETTE["blue"], alpha=0.16, linewidth=0,
                    label="Paired-seed bootstrap 95% interval")
    ax.plot(x, values, marker="o", markersize=4.2, linewidth=1.6,
            color=PALETTE["navy"], label="Median across selected benchmark functions")
    ax.axhline(0.0, color=PALETTE["grey"], linewidth=0.9, linestyle="--")
    ax.axvline(reference_dt, color=PALETTE["red"], linewidth=0.9, linestyle=":",
               label=fr"Reference $\Delta t={reference_dt:g}$")
    ax.set_xscale("log")
    ax.set_xticks(x, [f"{value:g}" for value in x])
    ax.set_xlabel(r"CPSO interval length, $\Delta t$ (log scale)")
    ax.set_ylabel(r"Median paired $\Delta\log_{10}(f)$")
    ax.set_title("Seed uncertainty for the selected benchmark set" + evidence_label, weight="bold")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize=7.0)
    fig.tight_layout()
    path = output / "fig_dt_sensitivity_family_profile.pdf"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    generated.append(path)
    return generated


def analyze(input_dir: Path, output_dir: Path, figures_dir: Path, *, bootstrap_replicates: int = 5000) -> dict[str, Any]:
    config, records, validation = load_and_validate(input_dir)
    deltas = _paired_deltas(records, validation)
    problem_summary = summarize_problem_blocks(records, validation, deltas)
    family_rows, nested = build_family_effects(validation, deltas)
    profile = hierarchical_bootstrap_profile(
        nested, validation, replicates=bootstrap_replicates,
        seed=int(config.get("analysis_seed", 20261107)),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "dt_problem_summary.csv", problem_summary)
    _write_csv(output_dir / "dt_family_effects.csv", family_rows)
    _write_csv(output_dir / "dt_family_profile.csv", profile)
    validation["bootstrap_replicates"] = int(bootstrap_replicates)
    validation_path = output_dir / "validation.json"
    validation_path.write_text(json.dumps(validation, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    figures = make_figures(problem_summary, profile, validation, figures_dir)
    return {"validation": validation, "figures": [str(path) for path in figures]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path,
        required=True,
        help="Campaign directory containing study_config.json and runs.jsonl",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--figures-dir", type=Path, default=None)
    parser.add_argument("--bootstrap-replicates", type=int, default=5000)
    args = parser.parse_args()
    if args.bootstrap_replicates < 200:
        parser.error("--bootstrap-replicates must be at least 200")
    from cpso_serial.output_policy import results_output
    output_dir = results_output(args.output_dir or ROOT / "results/analysis/dt" / args.input.name)
    figures_dir = results_output(args.figures_dir or ROOT / "results/figures/dt" / args.input.name)
    result = analyze(args.input, output_dir, figures_dir, bootstrap_replicates=args.bootstrap_replicates)
    print(
        f"Validated {result['validation']['n_runs']} runs over "
        f"{len(result['validation']['dt_values'])} dt values; generated {len(result['figures'])} figures."
    )
    print(f"Analysis CSV/JSON: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
