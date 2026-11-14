#!/usr/bin/env python3
"""Summarize completed COCO/CEC pilot runs without launching optimizations.

The script computes average within-problem ranks on *complete matched groups*.
For COCO, this is a descriptive supplement only: official runtime-to-target and
ECDF figures must be generated from Observer data with cocopp. For CEC, ranks
are reported at every configured FE checkpoint.

Bootstrap intervals resample function IDs as clusters (not individual runs),
so repeated dimensions/instances/seeds of one function are not treated as
independent top-level observations. The result is descriptive for small pilots
and is not a substitute for the confirmatory inferential protocol.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.plot_style import PALETTE, configure_style  # noqa: E402


def _slug(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    return value.strip("._-") or "campaign"


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _read_jsonl(path: Path) -> tuple[list[dict[str, Any]], int]:
    """Read durable JSONL records; tolerate only a malformed final partial line."""
    if not path.is_file():
        return [], 0
    raw = path.read_bytes()
    lines = raw.splitlines(keepends=True)
    records: list[dict[str, Any]] = []
    skipped_partial_tail = 0
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            if index == len(lines) - 1 and not line.endswith(b"\n"):
                skipped_partial_tail += 1
                break
            raise ValueError(f"Malformed complete JSONL record: {path}:{index + 1}") from exc
        if not isinstance(item, dict):
            raise ValueError(f"Expected JSON object in {path}:{index + 1}")
        records.append(item)
    return records, skipped_partial_tail


def _iter_campaign_files(campaign_dir: Path) -> list[Path]:
    records_dir = campaign_dir / "records"
    files: list[Path] = []
    if (campaign_dir / "records.jsonl").is_file():
        files.append(campaign_dir / "records.jsonl")
    if records_dir.is_dir():
        files.extend(sorted(records_dir.glob("*.jsonl")))
    return files


def _discover_campaigns(results_root: Path) -> list[tuple[Path, dict[str, Any]]]:
    campaigns: list[tuple[Path, dict[str, Any]]] = []
    for manifest_path in sorted(results_root.rglob("campaign_manifest.json")):
        manifest = _read_json(manifest_path)
        if manifest is None:
            continue
        suite = str(manifest.get("suite", ""))
        if suite.lower() not in {"bbob", "bbob-largescale", "cec2013-lsgo"}:
            continue
        if _iter_campaign_files(manifest_path.parent):
            campaigns.append((manifest_path.parent, manifest))
    return campaigns


def _average_ranks(values: dict[str, float], methods: list[str]) -> dict[str, float]:
    """Ascending average ranks; equal values receive average ranks."""
    ordered = sorted(((float(values[m]), m) for m in methods), key=lambda x: (x[0], x[1]))
    ranks: dict[str, float] = {}
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and math.isclose(ordered[end][0], ordered[index][0], rel_tol=1e-12, abs_tol=1e-15):
            end += 1
        average_rank = ((index + 1) + end) / 2.0
        for _, method in ordered[index:end]:
            ranks[method] = average_rank
        index = end
    return ranks


def _problem_function_id(group: dict[str, Any], problem_id: str) -> str:
    value = group.get("function_id")
    if value not in (None, ""):
        return str(value)
    match = re.search(r"(?:f|function)[_-]?(\d+)", problem_id, flags=re.IGNORECASE)
    if match:
        return match.group(1)
    # CEC problem IDs include their function label; keep a stable fallback.
    return problem_id.split("_")[0]


def _bootstrap_function_ci(
    per_function: dict[str, dict[str, list[float]]],
    methods: list[str],
    *,
    samples: int = 3000,
    seed: int = 20261106,
) -> dict[str, tuple[float, float]]:
    function_ids = sorted(per_function)
    if len(function_ids) < 2:
        return {method: (math.nan, math.nan) for method in methods}
    rng = np.random.default_rng(seed)
    output: dict[str, tuple[float, float]] = {}
    for method in methods:
        function_means = np.array([
            np.mean(per_function[function_id][method])
            for function_id in function_ids
            if per_function[function_id].get(method)
        ], dtype=float)
        if function_means.size < 2:
            output[method] = (math.nan, math.nan)
            continue
        indices = rng.integers(0, function_means.size, size=(samples, function_means.size))
        means = function_means[indices].mean(axis=1)
        output[method] = (float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975)))
    return output


def _rank_campaign(
    campaign_dir: Path,
    manifest: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    suite = str(manifest.get("suite", "unknown"))
    methods = [str(value) for value in manifest.get("methods", [])]
    if not methods:
        raise ValueError(f"Manifest has no method list: {campaign_dir / 'campaign_manifest.json'}")
    milestones = [int(x) for x in manifest.get("milestones", [])]
    metrics: list[tuple[str, str]] = []
    if suite.lower() == "cec2013-lsgo":
        metrics = [(f"best_at_{milestone}_FE", str(milestone)) for milestone in milestones]
    else:
        metrics = [("final_best_value", "best_value")]

    records: dict[tuple[str, int, str], dict[str, Any]] = {}
    source_count = 0
    partial_tail_count = 0
    for record_path in _iter_campaign_files(campaign_dir):
        file_records, skipped_tail = _read_jsonl(record_path)
        partial_tail_count += skipped_tail
        source_count += len(file_records)
        for record in file_records:
            if str(record.get("suite", suite)) != suite:
                continue
            try:
                key = (str(record["problem_id"]), int(record["seed"]), str(record["method"]))
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"Missing problem_id/seed/method in {record_path}") from exc
            previous = records.get(key)
            if previous is not None and previous != record:
                raise ValueError(f"Conflicting duplicate record {key} in {campaign_dir}")
            records[key] = record

    groups: dict[tuple[str, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    for (problem_id, seed, method), record in records.items():
        groups[(problem_id, seed)][method] = record

    manifest_expected = manifest.get("expected_run_count", manifest.get("expected_method_runs"))
    if manifest_expected is None:
        manifest_expected = len(groups) * len(methods)
    completeness = {
        "campaign": campaign_dir.name,
        "suite": suite,
        "expected_method_runs": int(manifest_expected),
        "records_loaded": source_count,
        "unique_method_runs": len(records),
        "complete_problem_seed_groups": 0,
        "incomplete_problem_seed_groups": 0,
        "methods_expected": len(methods),
        "methods_seen": len({key[2] for key in records}),
        "malformed_partial_tail_lines_skipped": partial_tail_count,
    }

    summary_rows: list[dict[str, Any]] = []
    for metric_name, metric_selector in metrics:
        group_rank_rows: list[tuple[str, str, dict[str, float]]] = []
        incomplete_groups = 0
        for (problem_id, seed), block in sorted(groups.items()):
            if any(method not in block for method in methods):
                incomplete_groups += 1
                continue
            values: dict[str, float] = {}
            valid = True
            for method in methods:
                record = block[method]
                if metric_selector == "best_value":
                    raw = record.get("best_value")
                else:
                    checkpoints = record.get("checkpoints", {})
                    raw = checkpoints.get(metric_selector) if isinstance(checkpoints, dict) else None
                try:
                    numeric = float(raw)
                except (TypeError, ValueError):
                    valid = False
                    break
                if not math.isfinite(numeric):
                    valid = False
                    break
                values[method] = numeric
            if not valid:
                incomplete_groups += 1
                continue
            function_id = str(next((block[method].get("function_id") for method in methods if block[method].get("function_id") not in (None, "")), ""))
            if not function_id:
                function_id = _problem_function_id({}, problem_id)
            group_rank_rows.append((problem_id, function_id, _average_ranks(values, methods)))

        completeness["complete_problem_seed_groups"] = max(
            int(completeness["complete_problem_seed_groups"]), len(group_rank_rows)
        )
        completeness["incomplete_problem_seed_groups"] = max(
            int(completeness["incomplete_problem_seed_groups"]), incomplete_groups
        )

        by_function: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
        for _problem_id, function_id, ranks in group_rank_rows:
            for method, rank in ranks.items():
                by_function[function_id][method].append(rank)

        confidence = _bootstrap_function_ci(by_function, methods, seed=20261106 + sum(ord(c) for c in metric_name))
        for method in methods:
            pooled = [rank for _, _, ranks in group_rank_rows if method in ranks for rank in [ranks[method]]]
            function_means = [float(np.mean(by_function[fid][method])) for fid in by_function if by_function[fid].get(method)]
            summary_rows.append({
                "campaign": campaign_dir.name,
                "suite": suite,
                "metric": metric_name,
                "method": method,
                "mean_rank_function_balanced": float(np.mean(function_means)) if function_means else math.nan,
                "mean_rank_all_groups": float(np.mean(pooled)) if pooled else math.nan,
                "bootstrap_95_low_function_cluster": confidence[method][0],
                "bootstrap_95_high_function_cluster": confidence[method][1],
                "complete_problem_seed_groups": len(group_rank_rows),
                "function_clusters": len(by_function),
            })

    return summary_rows, completeness


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _plot_summary(rows: list[dict[str, Any]], output: Path) -> list[Path]:
    configure_style()
    plot_paths: list[Path] = []
    buckets: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[(row["campaign"], row["suite"], row["metric"])].append(row)
    for (campaign, suite, metric), values in buckets.items():
        values = sorted(values, key=lambda row: (float(row["mean_rank_function_balanced"]), row["method"]))
        labels = [str(row["method"]) for row in values]
        means = np.array([float(row["mean_rank_function_balanced"]) for row in values])
        lows = np.array([float(row["bootstrap_95_low_function_cluster"]) for row in values])
        highs = np.array([float(row["bootstrap_95_high_function_cluster"]) for row in values])
        errors = np.vstack([np.maximum(0.0, means - lows), np.maximum(0.0, highs - means)])
        fig, ax = plt.subplots(figsize=(7.2, max(3.3, 0.31 * len(labels) + 0.8)))
        y = np.arange(len(labels))
        ax.barh(y, means, color=PALETTE["blue"], alpha=0.9, height=0.64)
        if np.all(np.isfinite(errors)):
            ax.errorbar(means, y, xerr=errors, fmt="none", ecolor=PALETTE["navy"], capsize=2, lw=0.9)
        ax.set_yticks(y, labels)
        ax.invert_yaxis()
        ax.set_xlabel("Average rank (lower is better)")
        ax.set_title(f"{suite}: matched final-performance ranks\n{campaign} · {metric}")
        ax.grid(axis="x")
        ax.set_axisbelow(True)
        fig.text(
            0.5, 0.01,
            "Means are balanced across function IDs; 95% percentile intervals resample function IDs as clusters. "
            "Descriptive pilot analysis, not confirmatory inference.",
            ha="center", va="bottom", fontsize=7, color=PALETTE["grey"],
        )
        target = output / f"rank_{_slug(campaign)}_{_slug(metric)}.pdf"
        target.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(target, format="pdf", metadata={
            "Title": f"Average ranks: {suite} {campaign} {metric}",
            "Creator": "CPSO external benchmark post-processing",
        })
        plt.close(fig)
        plot_paths.append(target)
    return plot_paths


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=ROOT / "results/external_benchmarks")
    parser.add_argument("--output", type=Path, default=ROOT / "results/figures/external_benchmarks")
    args = parser.parse_args()
    results_root = args.results_root.expanduser().resolve()
    from cpso_serial.output_policy import results_output
    output = results_output(args.output)
    if not results_root.is_dir():
        print(f"No external benchmark results directory: {results_root}")
        return 0

    campaigns = _discover_campaigns(results_root)
    if not campaigns:
        print(f"No COCO/CEC campaign manifests with records found under {results_root}")
        return 0

    all_summary: list[dict[str, Any]] = []
    completeness_rows: list[dict[str, Any]] = []
    for campaign_dir, manifest in campaigns:
        summary, completeness = _rank_campaign(campaign_dir, manifest)
        all_summary.extend(summary)
        completeness_rows.append(completeness)
        print(
            f"{campaign_dir.name}: {completeness['records_loaded']} records, "
            f"{completeness['complete_problem_seed_groups']} complete matched groups, "
            f"{completeness['incomplete_problem_seed_groups']} incomplete groups"
        )
    output.mkdir(parents=True, exist_ok=True)
    _write_csv(output / "external_rank_summary.csv", all_summary)
    _write_csv(output / "external_campaign_completeness.csv", completeness_rows)
    plots = _plot_summary(all_summary, output)
    print(f"Wrote {len(all_summary)} summary rows, {len(plots)} PDF figures and completeness report to {output}")
    print("For COCO campaigns, use cocopp Observer post-processing as the primary runtime-to-target/ECDF analysis.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
