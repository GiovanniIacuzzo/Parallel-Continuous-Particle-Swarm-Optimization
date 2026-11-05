#!/usr/bin/env python3
"""Execute a multi-stage, checkpointed CPSO research campaign.

All optimization runs are delegated to the single architecture-study engine.
This orchestration layer only defines stages, reproducible scheduling and
campaign-level provenance.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ROOT = Path(__file__).resolve().parent
if str(EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_ROOT))

from run_architecture_study import (  # noqa: E402
    execute_study,
    source_fingerprint,
    stable_hash,
    validate_study,
)
from environment_metadata import collect_environment_metadata  # noqa: E402
from checkpoint_io import atomic_write_json  # noqa: E402
from cpso_serial import native_build_info, native_version  # noqa: E402


def _load_campaign(path: Path) -> dict[str, Any]:
    try:
        campaign = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid campaign JSON: {exc}") from exc
    validate_campaign(campaign)
    return campaign


def validate_campaign(campaign: dict[str, Any]) -> None:
    required = {"campaign_id", "title", "schedule_seed", "stages"}
    missing = sorted(required - set(campaign))
    if missing:
        raise ValueError(f"Campaign is missing fields: {missing}")
    if not isinstance(campaign["campaign_id"], str) or not campaign["campaign_id"].strip():
        raise ValueError("campaign_id must be a non-empty string")
    if not isinstance(campaign["schedule_seed"], int) or isinstance(campaign["schedule_seed"], bool):
        raise ValueError("schedule_seed must be an integer")
    if not isinstance(campaign["stages"], list) or not campaign["stages"]:
        raise ValueError("stages must be a non-empty array")

    stage_ids: set[str] = set()
    study_ids: set[str] = set()
    for stage in campaign["stages"]:
        stage_id = stage.get("id")
        if not isinstance(stage_id, str) or not stage_id.strip() or stage_id in stage_ids:
            raise ValueError("Every campaign stage needs a unique non-empty id")
        stage_ids.add(stage_id)
        if stage.get("kind") not in {"quality", "performance"}:
            raise ValueError(f"Stage {stage_id}: kind must be quality or performance")
        studies = stage.get("studies")
        if not isinstance(studies, list) or not studies:
            raise ValueError(f"Stage {stage_id}: studies must be a non-empty array")
        for item in studies:
            study_id = item.get("id")
            if not isinstance(study_id, str) or not study_id.strip():
                raise ValueError(f"Stage {stage_id}: each study requires a non-empty id")
            full_id = f"{stage_id}/{study_id}"
            if full_id in study_ids:
                raise ValueError(f"Duplicate study id {full_id}")
            study_ids.add(full_id)
            study = item.get("study")
            if not isinstance(study, dict):
                raise ValueError(f"Study {full_id}: study must be an object")
            validate_study(study)
            method_names = {m["name"] for m in study["methods"]}
            for comparison in item.get("comparisons", []):
                method = comparison.get("method")
                reference = comparison.get("reference")
                if method not in method_names or reference not in method_names:
                    raise ValueError(
                        f"Study {full_id}: comparison references unknown methods {method!r}/{reference!r}"
                    )
                if method == reference:
                    raise ValueError(f"Study {full_id}: method cannot be compared with itself")
                if comparison.get("metric", "quality") not in {"quality", "runtime"}:
                    raise ValueError(f"Study {full_id}: comparison metric must be quality or runtime")


def flatten_studies(campaign: dict[str, Any], selected_stage_ids: set[str] | None = None):
    flattened = []
    for stage in campaign["stages"]:
        if selected_stage_ids is not None and stage["id"] not in selected_stage_ids:
            continue
        for study_item in stage["studies"]:
            flattened.append((stage, study_item))
    if not flattened:
        raise ValueError("No studies selected; check --stages")
    rng = random.Random(int(campaign["schedule_seed"]))
    rng.shuffle(flattened)
    return flattened


def count_runs(study: dict[str, Any]) -> int:
    return (
        len(study["benchmarks"])
        * len(study["dimensions"])
        * len(study["seeds"])
        * len(study["methods"])
    )


def campaign_manifest(campaign: dict[str, Any], code_hash: str, schedule: list[str]) -> dict[str, Any]:
    return {
        "schema_version": 2,
        "campaign_id": campaign["campaign_id"],
        "title": campaign["title"],
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "campaign_sha256": stable_hash(campaign),
        "source_sha256": code_hash,
        "schedule_seed": campaign["schedule_seed"],
        "execution_schedule": schedule,
        "host": {
            "platform": platform.platform(),
            "python": sys.version,
            "logical_cpu_count": os.cpu_count(),
        },
        "environment": collect_environment_metadata(
            native_version=native_version(),
            native_build_info=native_build_info(),
        ),
        "campaign": campaign,
    }


def execute_campaign(
    campaign: dict[str, Any],
    output: Path,
    *,
    selected_stage_ids: set[str] | None = None,
    dry_run: bool = False,
    no_plots: bool = False,
    analyze_only: bool = False,
) -> None:
    validate_campaign(campaign)
    flattened = flatten_studies(campaign, selected_stage_ids)
    full_flattened = flatten_studies(campaign, None)
    code_hash = source_fingerprint()
    full_schedule = [f"{stage['id']}/{study_item['id']}" for stage, study_item in full_flattened]
    selected_schedule = [f"{stage['id']}/{study_item['id']}" for stage, study_item in flattened]

    if dry_run:
        total = 0
        print(f"Campaign: {campaign['campaign_id']}")
        print(f"Execution plan ({len(flattened)} studies):")
        for stage, study_item in flattened:
            n_runs = count_runs(study_item["study"])
            total += n_runs
            print(f"  {stage['id']}/{study_item['id']}: {n_runs:,} runs")
        print(f"TOTAL: {total:,} optimization runs")
        return

    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "campaign_manifest.json"
    current_hash = stable_hash(campaign)
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("campaign_sha256") != current_hash:
            raise ValueError("Campaign configuration differs from existing output; choose a new directory")
        if existing.get("source_sha256") != code_hash:
            raise ValueError("Source code differs from existing campaign; use a new output directory")
        if existing.get("execution_schedule") != full_schedule:
            raise ValueError("Campaign execution schedule differs from the existing manifest")
    else:
        atomic_write_json(manifest_path, campaign_manifest(campaign, code_hash, full_schedule))

    # Log each invocation separately. This preserves the actual selected-stage
    # schedule when a long campaign is executed in several resumable batches.
    batch_record = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "selected_stages": sorted(selected_stage_ids) if selected_stage_ids is not None else "all",
        "study_schedule": selected_schedule,
        "analyze_only": bool(analyze_only),
        "no_plots": bool(no_plots),
    }
    with (output / "campaign_batches.jsonl").open("a", encoding="utf-8") as batch_stream:
        batch_stream.write(json.dumps(batch_record, sort_keys=True, separators=(",", ":")) + "\n")
        batch_stream.flush()
        os.fsync(batch_stream.fileno())

    if not analyze_only:
        for index, (stage, study_item) in enumerate(flattened, start=1):
            study_output = output / stage["id"] / study_item["id"]
            print(f"\n[{index}/{len(flattened)}] {stage['id']}/{study_item['id']}")
            execute_study(
                study_item["study"], study_output,
                no_plots=True,
                dry_run=False,
            )

    from analyze_research_campaign import analyze_campaign_directory
    analyze_campaign_directory(
        campaign, output, no_plots=no_plots, selected_stage_ids=selected_stage_ids
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the complete CPSO architecture research campaign")
    parser.add_argument("config", type=Path, help="campaign JSON configuration")
    parser.add_argument("--output", type=Path, default=None, help="campaign output directory")
    parser.add_argument("--stages", type=str, default=None, help="comma-separated stage IDs to execute")
    parser.add_argument("--dry-run", action="store_true", help="validate and count runs without executing")
    parser.add_argument("--no-plots", action="store_true", help="skip campaign-level figures")
    parser.add_argument("--analyze-only", action="store_true", help="reanalyze existing completed study runs")
    args = parser.parse_args()

    campaign = _load_campaign(args.config)
    selected = set(args.stages.split(",")) if args.stages else None
    unknown = selected - {stage["id"] for stage in campaign["stages"]} if selected else set()
    if unknown:
        raise ValueError(f"Unknown stage IDs: {sorted(unknown)}")
    output = args.output.expanduser().resolve() if args.output else (
        ROOT / "results" / "raw" / campaign["campaign_id"]
    )
    execute_campaign(
        campaign, output,
        selected_stage_ids=selected,
        dry_run=args.dry_run,
        no_plots=args.no_plots,
        analyze_only=args.analyze_only,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
