#!/usr/bin/env python3
"""Build PDF-only figures from existing campaign data; never execute optimizers.

Parallel campaigns now use the compact publication scaling post-processor.
Historical per-topology scaling/runtime PDFs are replaced in the atlas output;
other analysis modules (dt and external benchmarks) are unchanged.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from plot_results.atlas_common import read_json
from plot_results.plot_parallel_publication import generate as plot_parallel
from plot_results.plot_dt_atlas import generate as plot_dt
from plot_results.plot_external_atlas import generate as plot_external
from cpso_serial.output_policy import results_output, REPOSITORY_ROOT


def discover(root: Path) -> dict[str, list[Path]]:
    results = root / "results"
    found: dict[str, set[Path]] = {"parallel": set(), "dt": set(), "external": set()}
    for m in (results / "raw").rglob("study_config.json"):
        if (m.parent / "runs.jsonl").is_file():
            found["dt"].add(m.parent)
    for m in (results / "raw").rglob("campaign_manifest.json"):
        payload = read_json(m)
        if isinstance(payload.get("campaign"), dict) and payload["campaign"].get("stages"):
            found["parallel"].add(m.parent)
    for m in (results / "external_benchmarks").rglob("campaign_manifest.json"):
        payload = read_json(m)
        if str(payload.get("suite", "")).lower() in {"bbob", "bbob-largescale", "cec2013-lsgo"}:
            found["external"].add(m.parent)
    return {name: sorted(values) for name, values in found.items()}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, default=REPOSITORY_ROOT / "results/figures/atlas")
    ap.add_argument("--kind", choices=("all", "parallel", "dt", "external"), default="all")
    args = ap.parse_args()
    output = results_output(args.output)
    candidates = discover(REPOSITORY_ROOT)
    generators = {"parallel": plot_parallel, "dt": plot_dt, "external": plot_external}
    kinds = list(generators) if args.kind == "all" else [args.kind]
    summary = []
    for kind in kinds:
        for campaign in candidates[kind]:
            target = results_output(output / kind / campaign.name)
            try:
                generated, warnings = generators[kind](campaign, target)
                pdf = [p for p in generated if p.suffix.lower() == ".pdf"]
                status = "generated" if pdf else "no_figures"
            except Exception as exc:
                pdf = []
                warnings = [f"{type(exc).__name__}: {exc}"]
                status = "error"
            summary.append({
                "kind": kind, "source": str(campaign), "output": str(target),
                "n_pdf": len(pdf), "status": status, "warnings": warnings,
            })
            print(f"[{status}] {kind}/{campaign.name}: {len(pdf)} PDF")
            for warning in warnings:
                print(f"  {warning}")
    output.mkdir(parents=True, exist_ok=True)
    (output / "atlas_manifest.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Atlas: {output}")
    print("Only vector PDFs are generated; no experiment is executed and paper/ is unchanged.")
    return 0 if not any(x["status"] == "error" for x in summary) else 2


if __name__ == "__main__":
    raise SystemExit(main())
