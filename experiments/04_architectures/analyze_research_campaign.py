#!/usr/bin/env python3
"""Campaign-level paired inference, ERT and performance analysis."""
from __future__ import annotations

import argparse
import bisect
import csv
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

# Reuse the authoritative exact-equivalence audit for synchronous variants.
from run_architecture_study import verify_exact_equivalence
from checkpoint_io import read_jsonl_checkpoint

LOG_FLOOR = 1.0e-16
BOOTSTRAP_REPLICATES = 5000


def _boxplot_with_labels(ax, values, labels, **kwargs):
    """Use the boxplot label keyword supported by the installed Matplotlib version."""
    import matplotlib

    # The keyword was renamed in Matplotlib 3.9; it is accepted through
    # **kwargs in some releases, so signature introspection is insufficient.
    version = tuple(int(part.split(".")[0]) for part in matplotlib.__version__.split(".")[:2])
    label_keyword = "tick_labels" if version >= (3, 9) else "labels"
    return ax.boxplot(values, **{label_keyword: labels}, **kwargs)


def _quantile(values: list[float], probability: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    fraction = position - lower
    return float(ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction)


def _bootstrap_median(values: list[float], rng: random.Random, replicates: int = BOOTSTRAP_REPLICATES):
    if not values:
        return math.nan, math.nan
    if len(values) == 1:
        return values[0], values[0]
    n = len(values)
    medians = [statistics.median(values[rng.randrange(n)] for _ in range(n)) for _ in range(replicates)]
    return _quantile(medians, 0.025), _quantile(medians, 0.975)


def hierarchical_bootstrap_median(
    blocks: dict[tuple[str, int], list[float]], seed: int, replicates: int = BOOTSTRAP_REPLICATES
):
    """Resample problem instances, then paired seeds within each sampled instance."""
    keys = sorted(blocks)
    if not keys:
        return math.nan, math.nan
    rng = random.Random(seed)
    estimates: list[float] = []
    for _ in range(replicates):
        sampled_blocks: list[float] = []
        for _ in keys:
            key = keys[rng.randrange(len(keys))]
            values = blocks[key]
            sampled_blocks.extend(values[rng.randrange(len(values))] for _ in range(len(values)))
        estimates.append(statistics.median(sampled_blocks))
    return _quantile(estimates, 0.025), _quantile(estimates, 0.975)


def function_family_bootstrap_median(
    blocks: dict[tuple[str, int], list[float]],
    seed: int,
    replicates: int = BOOTSTRAP_REPLICATES,
) -> tuple[float, float]:
    """Nested bootstrap: function families -> dimensions -> paired seeds.

    The six analytic objective families are the highest-level sampling units.
    Dimensions are repeated measurements nested in a family, and seeds are
    paired replications nested in a family-dimension cell.
    """
    family_dimensions: dict[str, dict[int, list[float]]] = defaultdict(dict)
    for (family, dimension), values in blocks.items():
        if not values:
            continue
        family_dimensions[str(family)][int(dimension)] = list(values)
    families = sorted(family_dimensions)
    if not families:
        return math.nan, math.nan

    rng = random.Random(seed)
    estimates: list[float] = []
    for _ in range(replicates):
        sampled_values: list[float] = []
        for _ in families:
            family = families[rng.randrange(len(families))]
            dimensions = sorted(family_dimensions[family])
            for _ in dimensions:
                dimension = dimensions[rng.randrange(len(dimensions))]
                values = family_dimensions[family][dimension]
                sampled_values.extend(values[rng.randrange(len(values))] for _ in range(len(values)))
        estimates.append(statistics.median(sampled_values))
    return _quantile(estimates, 0.025), _quantile(estimates, 0.975)


def holm_adjust(p_values: list[float]) -> list[float]:
    """Holm step-down adjusted p-values, returned in original order."""
    count = len(p_values)
    if count == 0:
        return []
    order = sorted(range(count), key=lambda index: p_values[index])
    adjusted = [1.0] * count
    running = 0.0
    for rank, index in enumerate(order):
        candidate = min(1.0, (count - rank) * float(p_values[index]))
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing completed study file: {path}")
    rows, state = read_jsonl_checkpoint(path, repair=False)
    if state["discarded_partial_record"]:
        raise ValueError(
            f"Incomplete final JSONL record in {path}; resume the study once to repair the checkpoint, "
            "then rerun analysis."
        )
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _scipy_stats():
    try:
        from scipy import stats
        return stats
    except ImportError as exc:
        raise RuntimeError("Inferential analysis requires scipy; install the 'analysis' extra") from exc


def _wilcoxon(values: list[float]) -> tuple[float, float]:
    """Exact two-sided signed-rank sign-flip test on the independent family units.

    With six objective families, exhaustive sign enumeration is inexpensive
    (2**6 configurations) and avoids relying on asymptotic p-values at very
    small sample sizes. Zero differences are omitted from the rank set.
    """
    stats = _scipy_stats()
    nonzero = [float(value) for value in values if abs(float(value)) > 1.0e-15]
    if not nonzero:
        return 0.0, 1.0
    ranks = [float(v) for v in stats.rankdata([abs(value) for value in nonzero], method="average")]
    observed = abs(sum(math.copysign(rank, value) for value, rank in zip(nonzero, ranks)))
    positive_rank_sum = sum(rank for value, rank in zip(nonzero, ranks) if value > 0.0)
    count = len(nonzero)
    if count > 20:
        raise ValueError("Exact signed-rank enumeration is limited to 20 non-zero family units")
    extreme = 0
    total = 1 << count
    tolerance = 1.0e-12
    for mask in range(total):
        signed_sum = sum(
            rank if (mask >> index) & 1 else -rank
            for index, rank in enumerate(ranks)
        )
        if abs(signed_sum) >= observed - tolerance:
            extreme += 1
    return float(positive_rank_sum), float(extreme / total)


def _friedman_statistic(rank_matrix: list[list[float]]) -> float:
    """Friedman Q statistic for rows=blocks and columns=methods."""
    n_blocks = len(rank_matrix)
    n_methods = len(rank_matrix[0]) if rank_matrix else 0
    if n_blocks < 1 or n_methods < 2:
        return math.nan
    rank_sums = [sum(row[column] for row in rank_matrix) for column in range(n_methods)]
    return float(
        (12.0 / (n_blocks * n_methods * (n_methods + 1)))
        * sum(value * value for value in rank_sums)
        - 3.0 * n_blocks * (n_methods + 1)
    )


def _friedman_permutation_p(
    rank_matrix: list[list[float]], seed: int, replicates: int = 20000
) -> float:
    """Monte-Carlo blocked permutation p-value, shuffling method labels per family."""
    if not rank_matrix or replicates < 1:
        return math.nan
    observed = _friedman_statistic(rank_matrix)
    rng = random.Random(seed)
    extreme = 0
    for _ in range(replicates):
        permuted = []
        for row in rank_matrix:
            shuffled = list(row)
            rng.shuffle(shuffled)
            permuted.append(shuffled)
        if _friedman_statistic(permuted) >= observed - 1.0e-12:
            extreme += 1
    # Plus-one correction gives a valid Monte-Carlo randomization p-value.
    return float((extreme + 1) / (replicates + 1))


def validate_completed_study(study: dict[str, Any], records: list[dict[str, Any]], label: str) -> None:
    """Validate exact run coverage, FE accounting and result-history integrity."""
    expected_keys = {
        (benchmark, int(dimension), int(seed), method["name"])
        for benchmark in study["benchmarks"]
        for dimension in study["dimensions"]
        for seed in study["seeds"]
        for method in study["methods"]
    }
    if len(records) != len(expected_keys):
        raise RuntimeError(f"{label}: expected {len(expected_keys)} runs, found {len(records)}")

    seen_run_keys: set[str] = set()
    seen_experiments: set[tuple[str, int, int, str]] = set()
    expected_fe = int(study["population_size"]) * (int(study["iterations"]) + 1)
    for record in records:
        run_key = record.get("run_key")
        if not isinstance(run_key, str) or not run_key:
            raise RuntimeError(f"{label}: a record has no valid run_key")
        if run_key in seen_run_keys:
            raise RuntimeError(f"{label}: duplicate run_key {run_key}")
        seen_run_keys.add(run_key)
        experiment_key = (
            record.get("benchmark"), int(record.get("dimension", -1)),
            int(record.get("seed", -1)), record.get("method"),
        )
        if experiment_key not in expected_keys:
            raise RuntimeError(f"{label}: unexpected experiment record {experiment_key}")
        if experiment_key in seen_experiments:
            raise RuntimeError(f"{label}: duplicate experiment record {experiment_key}")
        seen_experiments.add(experiment_key)

        if int(record["function_evaluations"]) != expected_fe:
            raise RuntimeError(
                f"{label}: FE mismatch in {record['method']} seed={record['seed']}: "
                f"{record['function_evaluations']} != {expected_fe}"
            )
        history = record.get("best_history")
        axis = record.get("history_evaluations")
        if not isinstance(history, list) or not isinstance(axis, list) or len(history) != len(axis):
            raise RuntimeError(f"{label}: malformed or misaligned history for {record['method']}")
        if not axis or int(axis[0]) != int(study["population_size"]) or int(axis[-1]) != expected_fe:
            raise RuntimeError(f"{label}: invalid FE history axis for {record['method']}")
        if any(int(right) <= int(left) for left, right in zip(axis, axis[1:])):
            raise RuntimeError(f"{label}: non-increasing FE history axis for {record['method']}")
        try:
            best_value = float(record["best_value"])
            history_values = [float(value) for value in history]
            elapsed = float(record["elapsed_seconds"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(f"{label}: missing or invalid numeric result for {record['method']}") from exc
        if not math.isfinite(best_value) or not math.isfinite(elapsed) or elapsed < 0.0:
            raise RuntimeError(f"{label}: non-finite objective or invalid runtime for {record['method']}")
        if any(not math.isfinite(value) for value in history_values):
            raise RuntimeError(f"{label}: non-finite history for {record['method']}")
        if history_values[-1] != best_value:
            raise RuntimeError(f"{label}: final best value does not match history for {record['method']}")
        if any(right > left for left, right in zip(history_values, history_values[1:])):
            raise RuntimeError(f"{label}: best-so-far history increased for {record['method']}")

    missing = expected_keys - seen_experiments
    if missing:
        preview = sorted(missing)[:5]
        raise RuntimeError(f"{label}: missing {len(missing)} expected runs; examples: {preview}")


def _comparison_data(
    records: list[dict[str, Any]], comparison: dict[str, Any]
) -> tuple[dict[tuple[str, int], list[float]], list[tuple[float, float, int]], list[float]]:
    lookup = {
        (row["benchmark"], int(row["dimension"]), int(row["seed"]), row["method"]): row
        for row in records
    }
    method_name = comparison["method"]
    reference_name = comparison["reference"]
    metric = comparison.get("metric", "quality")
    blocks: dict[tuple[str, int], list[float]] = defaultdict(list)
    paired_runs: list[tuple[float, float, int]] = []
    speedups: list[float] = []
    for key, row in lookup.items():
        benchmark, dimension, seed, method = key
        if method != method_name:
            continue
        ref = lookup.get((benchmark, dimension, seed, reference_name))
        if ref is None:
            continue
        if metric == "runtime":
            ratio = float(ref["elapsed_seconds"]) / max(float(row["elapsed_seconds"]), 1.0e-15)
            speedups.append(ratio)
            paired_runs.append((ratio, 0.0, seed))
        else:
            delta = math.log10(max(float(row["best_value"]), LOG_FLOOR)) - math.log10(
                max(float(ref["best_value"]), LOG_FLOOR)
            )
            blocks[(benchmark, dimension)].append(delta)
            paired_runs.append((delta, 0.0, seed))
    return blocks, paired_runs, speedups


def _analyze_quality_study(
    stage: dict[str, Any], study_item: dict[str, Any], records: list[dict[str, Any]], output: Path
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    comparisons = study_item.get("comparisons", [])
    pair_rows: list[dict[str, Any]] = []
    raw_p: list[float] = []
    method_names = [method["name"] for method in study_item["study"]["methods"]]

    # The benchmark family is the independent top-level block.  Dimension
    # and seed are nested/repeated measurements and are not counted as
    # independent benchmark families for confirmatory inference.
    by_family_method: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in records:
        by_family_method[(row["benchmark"], row["method"])].append(
            math.log10(max(float(row["best_value"]), LOG_FLOOR))
        )
    function_families = sorted({str(r["benchmark"]) for r in records})
    problem_blocks = sorted({(r["benchmark"], int(r["dimension"])) for r in records})

    lookup = {
        (r["benchmark"], int(r["dimension"]), int(r["seed"]), r["method"]): r
        for r in records
    }
    for comparison in comparisons:
        if comparison.get("metric", "quality") != "quality":
            continue
        method_name = comparison["method"]
        reference_name = comparison["reference"]
        seed_deltas: dict[tuple[str, int], list[float]] = defaultdict(list)
        wins = losses = ties = 0
        for row in records:
            if row["method"] != method_name:
                continue
            key = (row["benchmark"], int(row["dimension"]), int(row["seed"]), reference_name)
            baseline = lookup.get(key)
            if baseline is None:
                raise RuntimeError(
                    f"{stage['id']}/{study_item['id']}: missing {reference_name} for "
                    f"{method_name} at {row['benchmark']} D={row['dimension']} seed={row['seed']}"
                )
            delta = math.log10(max(float(row["best_value"]), LOG_FLOOR)) - math.log10(
                max(float(baseline["best_value"]), LOG_FLOOR)
            )
            seed_deltas[(row["benchmark"], int(row["dimension"]))].append(delta)
            if delta < -1.0e-15:
                wins += 1
            elif delta > 1.0e-15:
                losses += 1
            else:
                ties += 1

        # First reduce paired seeds within each function-dimension block, then
        # reduce dimensions within each function family.  The latter six
        # family summaries are the confirmatory inferential units.
        block_medians = [statistics.median(values) for _, values in sorted(seed_deltas.items())]
        family_values: dict[str, list[float]] = defaultdict(list)
        for (family, _dimension), values in seed_deltas.items():
            family_values[family].append(statistics.median(values))
        family_medians = {
            family: statistics.median(values)
            for family, values in sorted(family_values.items())
        }
        family_delta_values = list(family_medians.values())
        statistic, p_value = _wilcoxon(family_delta_values)
        all_values = [value for values in seed_deltas.values() for value in values]
        ci_low, ci_high = function_family_bootstrap_median(
            seed_deltas,
            seed=20261105 + sum(ord(ch) for ch in stage["id"] + study_item["id"] + method_name),
        )
        median_delta = statistics.median(family_delta_values) if family_delta_values else math.nan
        raw_p.append(p_value)
        pair_rows.append({
            "stage": stage["id"],
            "study": study_item["id"],
            "method": method_name,
            "reference": reference_name,
            "n_problem_blocks_descriptive": len(block_medians),
            "n_independent_function_families": len(family_medians),
            "n_paired_runs": len(all_values),
            "median_problem_block_log10_delta": statistics.median(block_medians) if block_medians else math.nan,
            "median_function_family_log10_delta": median_delta,
            "median_objective_ratio": 10.0 ** median_delta if math.isfinite(median_delta) else math.nan,
            "hierarchical_bootstrap_95_low": ci_low,
            "hierarchical_bootstrap_95_high": ci_high,
            "wilcoxon_unit": "benchmark_function_family",
            "wilcoxon_statistic_families": statistic,
            "wilcoxon_p_raw": p_value,
            "wilcoxon_p_holm": math.nan,
            "paired_win_rate": wins / max(1, wins + losses + ties),
            "paired_wins": wins,
            "paired_losses": losses,
            "paired_ties": ties,
            "effect_direction": "method_better" if median_delta < 0 else "reference_better" if median_delta > 0 else "tie",
        })

    adjusted = holm_adjust(raw_p)
    for row, p_adjusted in zip(pair_rows, adjusted):
        row["wilcoxon_p_holm"] = p_adjusted

    rank_rows: list[dict[str, Any]] = []
    friedman_rows: list[dict[str, Any]] = []
    if len(method_names) >= 3 and len(function_families) >= 3:
        stats = _scipy_stats()
        score_matrix: list[list[float]] = []
        for family in function_families:
            family_scores = [
                statistics.median(by_family_method[(family, method)])
                for method in method_names
            ]
            score_matrix.append(family_scores)
        rank_matrix = [stats.rankdata(row, method="average").tolist() for row in score_matrix]
        for index, method_name in enumerate(method_names):
            average_rank = statistics.mean(row[index] for row in rank_matrix)
            rank_rows.append({
                "stage": stage["id"],
                "study": study_item["id"],
                "method": method_name,
                "n_independent_function_families": len(function_families),
                "average_rank": average_rank,
            })
        all_methods_tied = all(
            len(set(float(value) for value in row)) == 1 for row in score_matrix
        )
        status = "all_methods_tied_no_inferential_test" if all_methods_tied else "ok"
        if all_methods_tied:
            # An exact synchronous-equivalence study has no stochastic ranking
            # contrast to test. Preserve descriptive average ranks, but do not
            # manufacture a p-value from a degenerate Friedman statistic.
            q_stat = p_friedman = p_friedman_permutation = math.nan
            iman_davenport_f = iman_p = math.nan
        else:
            try:
                result = stats.friedmanchisquare(*[
                    [row[index] for row in score_matrix] for index in range(len(method_names))
                ])
                q_stat = _friedman_statistic(rank_matrix)
                p_friedman = float(result.pvalue)  # asymptotic reference only
                p_friedman_permutation = _friedman_permutation_p(
                    rank_matrix,
                    seed=20261105 + sum(ord(ch) for ch in stage["id"] + study_item["id"]),
                    replicates=20000,
                )
                n_blocks = len(function_families)
                n_methods = len(method_names)
                denominator = (n_blocks * (n_methods - 1)) - q_stat
                iman_davenport_f = (
                    ((n_blocks - 1) * q_stat) / denominator if denominator > 0 else math.inf
                )
                iman_p = float(stats.f.sf(iman_davenport_f, n_methods - 1, (n_methods - 1) * (n_blocks - 1)))
            except ValueError:
                q_stat = p_friedman = p_friedman_permutation = iman_davenport_f = iman_p = math.nan
                status = "test_not_computable"
        friedman_rows.append({
            "stage": stage["id"],
            "study": study_item["id"],
            "n_independent_function_families": len(function_families),
            "n_methods": len(method_names),
            "friedman_chi_square": q_stat,
            "friedman_p_asymptotic": p_friedman,
            "friedman_p_block_permutation": p_friedman_permutation,
            "friedman_permutation_replicates": 20000,
            "friedman_status": status,
            "iman_davenport_F": iman_davenport_f,
            "iman_davenport_p": iman_p,
        })
    return pair_rows, rank_rows, friedman_rows

def _analyze_performance_study(
    stage: dict[str, Any], study_item: dict[str, Any], records: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Compute paired wall-clock speedup only for validated synchronous pairs."""
    study = study_item["study"]
    if study.get("benchmarks") != ["sphere_work"] or study.get("check_equivalence", True) is not True:
        raise ValueError(
            f"{stage['id']}/{study_item['id']}: speedup requires sphere_work and exact-equivalence checks"
        )
    method_specs = {item["name"]: item for item in study["methods"]}
    lookup: dict[tuple[str, int, int, str], dict[str, Any]] = {}
    for row in records:
        key = (row["benchmark"], int(row["dimension"]), int(row["seed"]), row["method"])
        if key in lookup:
            raise RuntimeError(f"Duplicate record in performance study: {key}")
        lookup[key] = row

    output: list[dict[str, Any]] = []
    for comparison in study_item.get("comparisons", []):
        if comparison.get("metric") != "runtime":
            raise ValueError("Performance comparisons must all use metric='runtime'")
        method = comparison["method"]
        reference = comparison["reference"]
        method_spec = method_specs[method]
        reference_spec = method_specs[reference]
        if method_spec.get("architecture") not in {"master_slave", "particle_parallel"}:
            raise ValueError(f"{method} is not a synchronous parallel executor")
        if reference_spec.get("architecture") != "serial":
            raise ValueError(f"{reference} is not a serial reference")
        if method_spec.get("topology", "global") != reference_spec.get("topology", "global"):
            raise ValueError(f"{method}/{reference} use different social topologies")
        method_semantics = {
            key: value for key, value in method_spec.items()
            if key not in {"name", "architecture", "workers"}
        }
        reference_semantics = {
            key: value for key, value in reference_spec.items()
            if key not in {"name", "architecture", "workers"}
        }
        if method_semantics != reference_semantics:
            raise ValueError(f"{method}/{reference} differ in algorithmic settings")

        worker_count = int(method_spec["workers"])
        for dimension in study["dimensions"]:
            ratios: list[float] = []
            serial_times: list[float] = []
            parallel_times: list[float] = []
            for seed in study["seeds"]:
                parallel_row = lookup.get(("sphere_work", int(dimension), int(seed), method))
                serial_row = lookup.get(("sphere_work", int(dimension), int(seed), reference))
                if parallel_row is None or serial_row is None:
                    raise RuntimeError(
                        f"Missing paired speedup record for {method}/{reference}, "
                        f"D={dimension}, seed={seed}"
                    )
                parallel_time = float(parallel_row["elapsed_seconds"])
                serial_time = float(serial_row["elapsed_seconds"])
                if not math.isfinite(parallel_time) or parallel_time <= 0.0:
                    raise RuntimeError(
                        f"Invalid parallel elapsed_seconds for {method}, D={dimension}, seed={seed}"
                    )
                if not math.isfinite(serial_time) or serial_time <= 0.0:
                    raise RuntimeError(
                        f"Invalid serial elapsed_seconds for {reference}, D={dimension}, seed={seed}"
                    )
                if int(parallel_row.get("workers", worker_count)) != worker_count:
                    raise RuntimeError(f"Inconsistent worker count recorded for {method}")
                if int(parallel_row["function_evaluations"]) != int(serial_row["function_evaluations"]):
                    raise RuntimeError(f"FE budget mismatch in speedup pair {method}/{reference}")
                for field in ("population_size", "iterations"):
                    if field in parallel_row and field in serial_row:
                        if int(parallel_row[field]) != int(serial_row[field]):
                            raise RuntimeError(
                                f"{field} mismatch in speedup pair {method}/{reference}"
                            )
                # Exact synchronous incumbent-trace equivalence is verified by
                # analyze_campaign_directory before this function is called. Use
                # end-to-end native elapsed time; on parallel runs it includes
                # worker-pool startup/shutdown, which is part of executor overhead.
                ratios.append(serial_time / parallel_time)
                serial_times.append(serial_time)
                parallel_times.append(parallel_time)

            ci_low, ci_high = _bootstrap_median(
                ratios,
                random.Random(20261105 + int(dimension) + sum(ord(ch) for ch in method + study_item["id"])),
            )
            median_speedup = statistics.median(ratios)
            output.append({
                "stage": stage["id"],
                "study": study_item["id"],
                "population_size": int(study["population_size"]),
                "work_iterations": int(study.get("work_iterations", 0)),
                "dimension": int(dimension),
                "method": method,
                "reference": reference,
                "workers": worker_count,
                "n_paired_runs": len(ratios),
                "median_serial_runtime_seconds": statistics.median(serial_times),
                "median_parallel_runtime_seconds": statistics.median(parallel_times),
                "median_speedup": median_speedup,
                "bootstrap_95_low": ci_low,
                "bootstrap_95_high": ci_high,
                "parallel_efficiency": median_speedup / worker_count,
                "interpretation": "paired end-to-end speedup; verified synchronous incumbent trace and FE budget validated",
            })
    return output


def _campaign_figures(
    campaign: dict[str, Any], campaign_root: Path, study_records: dict[tuple[str, str], list[dict[str, Any]]],
    speedup_rows: list[dict[str, Any]],
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("Campaign figures require matplotlib; install the 'analysis' extra") from exc

    figure_root = campaign_root / "figures"
    figure_root.mkdir(parents=True, exist_ok=True)
    for stage in campaign["stages"]:
        for study_item in stage["studies"]:
            records = study_records.get((stage["id"], study_item["id"]), [])
            if not records or stage["kind"] != "quality":
                continue
            groups: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
            for row in records:
                groups[(row["benchmark"], int(row["dimension"]))].append(row)
            for (benchmark, dimension), rows in sorted(groups.items()):
                by_method: dict[str, list[dict[str, Any]]] = defaultdict(list)
                for row in rows:
                    by_method[row["method"]].append(row)
                methods = [m["name"] for m in study_item["study"]["methods"]]
                axes = sorted({int(fe) for row in rows for fe in row["history_evaluations"]})

                fig, ax = plt.subplots(figsize=(9, 5.2), constrained_layout=True)
                for method in methods:
                    method_rows = by_method[method]
                    histories = []
                    for row in method_rows:
                        axis = list(map(int, row["history_evaluations"]))
                        hist = list(map(float, row["best_history"]))
                        histories.append((axis, hist))
                    medians, q1, q3 = [], [], []
                    for fe in axes:
                        vals = []
                        for axis, hist in histories:
                            idx = bisect.bisect_right(axis, fe) - 1
                            if idx >= 0:
                                vals.append(max(hist[idx], LOG_FLOOR))
                        if vals:
                            medians.append(statistics.median(vals))
                            q1.append(_quantile(vals, 0.25))
                            q3.append(_quantile(vals, 0.75))
                    ax.plot(axes[:len(medians)], medians, label=method)
                    ax.fill_between(axes[:len(q1)], q1, q3, alpha=0.10)
                ax.set_yscale("log")
                ax.set_xlabel("Objective function evaluations")
                ax.set_ylabel("Median best-so-far objective")
                ax.set_title(f"{stage['id']} — {benchmark}, D={dimension}")
                ax.grid(True, which="both", alpha=0.25)
                ax.legend(fontsize=7)
                stem = f"{stage['id']}_{study_item['id']}_{benchmark}_d{dimension}_convergence"
                for ext in ("png", "pdf", "svg"):
                    fig.savefig(figure_root / f"{stem}.{ext}", dpi=240 if ext == "png" else None)
                plt.close(fig)

                fig, ax = plt.subplots(figsize=(9, 5.2), constrained_layout=True)
                data = [
                    [math.log10(max(float(r["best_value"]), LOG_FLOOR)) for r in by_method[m]]
                    for m in methods
                ]
                _boxplot_with_labels(ax, data, methods, showmeans=False)
                ax.set_ylabel("log10(final objective)")
                ax.set_title(f"{stage['id']} — final quality, {benchmark}, D={dimension}")
                ax.tick_params(axis="x", labelrotation=35)
                ax.grid(True, axis="y", alpha=0.25)
                stem = f"{stage['id']}_{study_item['id']}_{benchmark}_d{dimension}_final"
                for ext in ("png", "pdf", "svg"):
                    fig.savefig(figure_root / f"{stem}.{ext}", dpi=240 if ext == "png" else None)
                plt.close(fig)

    if speedup_rows:
        grouped: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
        for row in speedup_rows:
            grouped[(int(row["population_size"]), int(row["dimension"]))].append(row)
        for (population, dimension), rows in sorted(grouped.items()):
            fig, ax = plt.subplots(figsize=(8.5, 5.0), constrained_layout=True)
            workloads = sorted({int(row["work_iterations"]) for row in rows})
            for workload in workloads:
                for architecture, method_prefix in (("master_slave", "ms_"), ("particle_parallel", "pp_")):
                    subset = [
                        r for r in rows
                        if int(r["work_iterations"]) == workload and r["method"].startswith(method_prefix)
                    ]
                    subset.sort(key=lambda row: int(row["workers"]))
                    if subset:
                        ax.plot(
                            [int(r["workers"]) for r in subset],
                            [float(r["median_speedup"]) for r in subset],
                            marker="o", label=f"{architecture}, work={workload}",
                        )
            max_workers = max(int(row["workers"]) for row in rows)
            ax.plot([1, max_workers], [1, max_workers], linestyle="--", linewidth=1, label="ideal")
            ax.axhline(1.0, linestyle=":", linewidth=1)
            ax.set_xticks(sorted({int(r["workers"]) for r in rows}))
            ax.set_xlabel("Worker threads")
            ax.set_ylabel("Median paired speedup vs serial")
            ax.set_title(f"Strong scaling — N={population}, D={dimension}")
            ax.grid(True, alpha=0.25)
            ax.legend(fontsize=7)
            stem = f"strong_scaling_n{population}_d{dimension}"
            for ext in ("png", "pdf", "svg"):
                fig.savefig(figure_root / f"{stem}.{ext}", dpi=240 if ext == "png" else None)
            plt.close(fig)


def analyze_campaign_directory(
    campaign: dict[str, Any], campaign_root: Path, no_plots: bool = False,
    selected_stage_ids: set[str] | None = None,
) -> None:
    all_runs: list[dict[str, Any]] = []
    study_records: dict[tuple[str, str], list[dict[str, Any]]] = {}
    comparison_rows: list[dict[str, Any]] = []
    rank_rows: list[dict[str, Any]] = []
    friedman_rows: list[dict[str, Any]] = []
    speedup_rows: list[dict[str, Any]] = []
    equivalence_rows: list[dict[str, Any]] = []
    stage_status: list[dict[str, Any]] = []

    for stage in campaign["stages"]:
        if selected_stage_ids is not None and stage["id"] not in selected_stage_ids:
            continue
        for study_item in stage["studies"]:
            label = f"{stage['id']}/{study_item['id']}"
            path = campaign_root / stage["id"] / study_item["id"] / "runs.jsonl"
            records = _read_jsonl(path)
            validate_completed_study(study_item["study"], records, label)
            if study_item["study"].get("check_equivalence", True):
                # The verifier raises on the first mismatch. Only after it
                # returns do we materialize an auditable count of exact pairs.
                verify_exact_equivalence(records)
                expected_pairs = (
                    len(study_item["study"]["benchmarks"])
                    * len(study_item["study"]["dimensions"])
                    * len(study_item["study"]["seeds"])
                )
                methods_by_name = {
                    method["name"]: method for method in study_item["study"]["methods"]
                }
                for method_name, method in methods_by_name.items():
                    if method.get("architecture") not in {"master_slave", "particle_parallel"}:
                        continue
                    verified_pairs = sum(
                        1 for record in records if record.get("method") == method_name
                    )
                    equivalence_rows.append({
                        "stage": stage["id"],
                        "study": study_item["id"],
                        "method": method_name,
                        "architecture": method["architecture"],
                        "topology": method.get("topology", "global"),
                        "workers": int(method.get("workers", 1)),
                        "verified_pairs": verified_pairs,
                        "expected_pairs": expected_pairs,
                        "fraction_verified": verified_pairs / expected_pairs if expected_pairs else math.nan,
                        "status": "exact_match" if verified_pairs == expected_pairs else "incomplete",
                        "checked_fields": "best_value;best_position;best_history;history_evaluations;function_evaluations",
                    })
            study_records[(stage["id"], study_item["id"])] = records
            for record in records:
                all_runs.append({"stage": stage["id"], "study": study_item["id"], **record})
            stage_status.append({
                "stage": stage["id"],
                "study": study_item["id"],
                "expected_runs": len(study_item["study"]["benchmarks"])
                    * len(study_item["study"]["dimensions"])
                    * len(study_item["study"]["seeds"])
                    * len(study_item["study"]["methods"]),
                "completed_runs": len(records),
                "expected_fe_per_run": int(study_item["study"]["population_size"])
                    * (int(study_item["study"]["iterations"]) + 1),
                "status": "complete",
            })
            if stage["kind"] == "quality":
                pairs, ranks, omnibus = _analyze_quality_study(stage, study_item, records, campaign_root)
                comparison_rows.extend(pairs)
                rank_rows.extend(ranks)
                friedman_rows.extend(omnibus)
            else:
                speedup_rows.extend(_analyze_performance_study(stage, study_item, records))

    omnibus_p = [
        float(row["friedman_p_block_permutation"])
        for row in friedman_rows
        if math.isfinite(float(row["friedman_p_block_permutation"]))
    ]
    omnibus_adjusted = holm_adjust(omnibus_p)
    adjusted_index = 0
    for row in friedman_rows:
        value = float(row["friedman_p_block_permutation"])
        if math.isfinite(value):
            row["friedman_p_holm_campaign"] = omnibus_adjusted[adjusted_index]
            adjusted_index += 1
        else:
            row["friedman_p_holm_campaign"] = math.nan

    evidence_class = campaign.get("evidence_class", "exploratory")
    if evidence_class not in {"confirmatory", "exploratory", "integration_only"}:
        raise ValueError(f"Unsupported evidence_class: {evidence_class!r}")

    _write_csv(campaign_root / "campaign_runs.csv", all_runs)
    _write_csv(campaign_root / "campaign_validation.csv", stage_status)
    _write_csv(campaign_root / "pairwise_effects.csv", comparison_rows)
    # Keep the historical filename only for campaigns explicitly designated
    # confirmatory. Exploratory results must never acquire a confirmatory label.
    if evidence_class == "confirmatory":
        _write_csv(campaign_root / "confirmatory_pairwise_tests.csv", comparison_rows)
    elif (campaign_root / "confirmatory_pairwise_tests.csv").exists():
        (campaign_root / "confirmatory_pairwise_tests.csv").unlink()
    _write_csv(
        campaign_root / "equivalence_audit.csv",
        equivalence_rows,
        fields=["stage", "study", "method", "architecture", "topology", "workers",
                "verified_pairs", "expected_pairs", "fraction_verified", "status", "checked_fields"],
    )
    _write_csv(campaign_root / "method_average_ranks.csv", rank_rows)
    _write_csv(campaign_root / "omnibus_friedman_tests.csv", friedman_rows)
    _write_csv(campaign_root / "strong_scaling_summary.csv", speedup_rows)

    report = [
        f"# Research campaign analysis — {campaign['campaign_id']}",
        "",
        f"Evidence class: `{evidence_class}`",
        "",
        f"Completed optimization runs: {len(all_runs):,}",
        f"Completed studies: {len(stage_status)}",
        f"Stages: {', '.join(sorted({row['stage'] for row in stage_status}))}",
        "",
        "## Validation",
        "",
        "All completed studies were checked for exact run-key coverage, unique run keys, exact FE budget, finite outcomes, monotone best-so-far histories and valid FE-history endpoints.",
        "Synchronous master-slave and particle-parallel equivalence checks are enforced by the underlying study runner.",
        "",
        "## Statistical protocol",
        "",
        "- Seed pairing is performed within each benchmark-function × dimension block.",
        "- The six analytic benchmark functions are the independent top-level clusters. The two tested dimensions are repeated measurements nested within each function; seeds are paired replications nested within function × dimension.",
        "- In the definitive campaign, confirmatory pairwise inference uses an exact two-sided signed-rank sign-flip test on six function-family summaries (each family first reduces paired seeds within dimension, then dimensions). All 2^6 sign assignments are enumerated; zero differences are omitted from ranking. Holm correction is applied within each study's pre-specified comparison family. The 12 function × dimension summaries remain descriptive and are not counted as 12 independent functions.",
        "- Confidence intervals use a nested bootstrap: resample function families, then dimensions within selected families, then paired seeds within selected function × dimension cells. These intervals quantify uncertainty; they do not replace the corrected tests.",
        "- The Friedman omnibus statistic uses one median log10 objective per method and function family. Its confirmatory p-value is calibrated by 20,000 blocked permutations of method labels within function families. The conventional chi-square and Iman-Davenport p-values are retained as asymptotic diagnostics only; six independent families still imply limited power.",
        "- Strong-scaling speedup is reported only for synchronous algorithms whose trajectories are checked exactly against the serial reference. Efficiency is speedup divided by worker count.",
        "- Asynchronous and island runtime ratios are not called speedup because their search dynamics differ from the serial algorithm.",
        "- ERT is computed by the study runner using the predeclared objective target; unsuccessful runs are right-censored at their FE budget. If no seed succeeds, ERT is infinite.",
        "",
        "## Scope and limitations",
        "",
        "This campaign uses the repository's built-in analytic benchmark functions plus `sphere_work` strictly as a controlled execution-cost instrument. It does not yet constitute a COCO/BBOB, BBOB-Large-Scale or CEC 2013 benchmark submission. External-suite adapters, official instances, suite versions and their FE accounting must be integrated and validated before claims of state-of-the-art benchmark performance or broad generalization are made.",
        "The included campaign is intended to establish architecture effects, reproducible implementation behavior and local shared-memory scaling on the machine recorded in each run manifest. Timing results from a Linux development container are not substitutes for the user's target Mac measurements.",
        "",
        "## Outputs",
        "",
        "- `campaign_runs.csv`: all run records with stage/study identifiers.",
        "- `campaign_validation.csv`: run-count and FE-budget audit.",
        "- `pairwise_effects.csv`: paired effect summaries and associated inferential diagnostics; interpret them according to the campaign evidence class above.",
        "- `confirmatory_pairwise_tests.csv`: compatibility copy emitted only when evidence_class is explicitly `confirmatory`.",
        "- `equivalence_audit.csv`: exact synchronous trajectory-pair counts after the record-by-record verifier passed.",
        "- `omnibus_friedman_tests.csv` and `method_average_ranks.csv`: multi-method rank analysis.",
        "- `strong_scaling_summary.csv`: paired speedup, bootstrap interval and parallel efficiency.",
    ]
    (campaign_root / "CAMPAIGN_ANALYSIS.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    if not no_plots:
        figure_campaign = dict(campaign)
        figure_campaign["stages"] = [
            stage for stage in campaign["stages"]
            if selected_stage_ids is None or stage["id"] in selected_stage_ids
        ]
        _campaign_figures(figure_campaign, campaign_root, study_records, speedup_rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze a completed CPSO research campaign")
    parser.add_argument("campaign_config", type=Path)
    parser.add_argument("campaign_output", type=Path)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    campaign = json.loads(args.campaign_config.read_text(encoding="utf-8"))
    from cpso_serial.output_policy import results_output
    output = results_output(args.campaign_output)
    analyze_campaign_directory(campaign, output, no_plots=args.no_plots)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
