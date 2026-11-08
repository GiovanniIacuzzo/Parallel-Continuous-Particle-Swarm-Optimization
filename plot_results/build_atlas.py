#!/usr/bin/env python3
"""Discover existing CPSO campaign outputs and build a complete figure atlas.

No optimizer is invoked. The script is read-only with respect to raw results;
all figures/catalogues are written below paper/figures/atlas by default.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from plot_results.atlas_common import read_json
from plot_results.plot_internal_atlas import generate as generate_internal
from plot_results.plot_dt_atlas import generate as generate_dt
from plot_results.plot_external_atlas import generate as generate_external
from plot_results.plot_legacy_atlas import generate as generate_legacy
from plot_results.plot_old_runners_atlas import generate as generate_old_runners
from plot_results.plot_internal import generate as generate_paper_internal
from plot_results.plot_dt_sensitivity import generate as generate_paper_dt
from plot_results.plot_external_ranks import generate as generate_paper_external


def discover(root: Path):
    results = root / "results"
    internal, dt, external, legacy = [], [], [], []
    raw = results / "raw"
    if raw.is_dir():
        # Legacy CSV-only studies are kept separate from campaign-manifest studies.
        for path in sorted(raw.rglob("scaling_runs.csv")):
            legacy.append(path.parent)
        for path in sorted(raw.rglob("master_slave_runs.csv")):
            legacy.append(path.parent)
        for path in sorted(raw.rglob("topology_runs.csv")):
            legacy.append(path.parent)
        for path in sorted(raw.rglob("runs.csv")):
            if (path.parent / "metadata.json").is_file() and (path.parent / "convergence.csv").is_file():
                legacy.append(path.parent)
        for manifest_path in sorted(raw.rglob("campaign_manifest.json")):
            campaign_dir = manifest_path.parent
            manifest = read_json(manifest_path)
            if isinstance(manifest.get("campaign"), dict) and manifest["campaign"].get("stages"):
                if any(campaign_dir.rglob("runs.jsonl")) or (campaign_dir / "campaign_runs.csv").is_file():
                    internal.append(campaign_dir)
            elif str(manifest.get("suite", "")).lower() in {"bbob", "bbob-largescale", "cec2013-lsgo"}:
                if (campaign_dir / "records.jsonl").is_file() or (campaign_dir / "records").is_dir():
                    external.append(campaign_dir)
        for config_path in sorted(raw.rglob("study_config.json")):
            campaign_dir = config_path.parent
            if (campaign_dir / "runs.jsonl").is_file():
                dt.append(campaign_dir)
    ext_root = results / "external_benchmarks"
    if ext_root.is_dir():
        for manifest_path in sorted(ext_root.rglob("campaign_manifest.json")):
            campaign_dir = manifest_path.parent
            manifest = read_json(manifest_path)
            if str(manifest.get("suite", "")).lower() in {"bbob", "bbob-largescale", "cec2013-lsgo"}:
                if campaign_dir not in external and ((campaign_dir / "records.jsonl").is_file() or (campaign_dir / "records").is_dir()):
                    external.append(campaign_dir)
    # A directory can have several old result files; only plot once.
    return sorted(set(internal)), sorted(set(dt)), sorted(set(external)), sorted(set(legacy))


def _resolve_many(values: list[str] | None, root: Path) -> list[Path] | None:
    if values is None:
        return None
    out=[]
    for value in values:
        p=Path(value).expanduser()
        out.append(p.resolve() if p.is_absolute() else (root/p).resolve())
    return out


def build(root: Path, output: Path, internal_dirs: list[Path], dt_dirs: list[Path], external_dirs: list[Path], legacy_dirs: list[Path]):
    output.mkdir(parents=True, exist_ok=True)
    summary: list[dict[str, Any]]=[]
    for kind, dirs, generator in (("internal",internal_dirs,generate_internal),("dt",dt_dirs,generate_dt),("external",external_dirs,generate_external),("legacy",legacy_dirs,generate_legacy)):
        for campaign in dirs:
            name=campaign.name
            target=output/kind/name
            try:
                files,warnings=generator(campaign,target)
                if kind == "legacy":
                    old_files, old_warnings = generate_old_runners(campaign,target)
                    files.extend(old_files)
                    warnings.extend(old_warnings)
                pdfs=[p for p in files if p.suffix==".pdf"]
                summary.append({"kind":kind,"campaign":str(campaign),"output":str(target),"pdf_count":len(pdfs),"png_count":len(files)-len(pdfs),"warnings":warnings,"status":"generated" if pdfs else "no_figures"})
            except Exception as exc:
                summary.append({"kind":kind,"campaign":str(campaign),"output":str(target),"pdf_count":0,"png_count":0,"warnings":[f"{type(exc).__name__}: {exc}"],"status":"failed"})
                print(f"[ERROR] {kind} {campaign}: {type(exc).__name__}: {exc}")
    payload={"root":str(root),"atlas_output":str(output),"campaigns":summary}
    (output/"atlas_manifest.json").write_text(json.dumps(payload,indent=2),encoding="utf-8")
    lines=["# CPSO complete figure atlas","",f"Repository: `{root}`","", "This atlas contains figures generated exclusively from recorded experiment outputs. No campaigns are launched and no observations are synthesized.","", "## Campaigns",""]
    if not summary: lines.append("No campaign output directories were discovered.")
    for item in summary:
        lines += [f"### {item['kind']} — {Path(item['campaign']).name}","",f"- Status: **{item['status']}**",f"- PDF figures: {item['pdf_count']}",f"- PNG figures: {item['png_count']}",f"- Output: `{item['output']}`"]
        if item["warnings"]: lines += ["- Warnings:"]+[f"  - {w}" for w in item["warnings"]]
        lines.append("")
    lines += ["## Interpretation rules","","- Internal quality plots show function values and best-so-far trajectories at matched FE budgets.","- Speedup/efficiency are only calculated from synchronous matched runs with identical topology, seed, population, dimension and synthetic objective cost.","- `sphere_work` is an execution-cost instrument, not an optimization-quality benchmark.","- Asynchronous and island elapsed times are descriptive; they are not called speedups against serial CPSO.","- Delta-t intervals condition on the selected functions and dimensions and resample paired seeds.","- BBOB custom ranks are descriptive supplements. Official ECDF/runtime-to-target figures must be generated by COCO `cocopp` from Observer data.","- CEC cross-function summaries use within-problem paired ranks or checkpoint summaries rather than pooling raw objective values across functions.",
        "- Legacy CSV-only master-slave/topology studies are plotted separately; timings remain hardware-specific.",""]
    (output/"FIGURE_ATLAS.md").write_text("\n".join(lines),encoding="utf-8")
    return summary



def build_paper_ready(root: Path):
    """Also populate the exact figure paths referenced by Section 4, when inputs exist."""
    results = []
    raw = root / "results" / "raw"
    sync = raw / "cpso_serial_parallel_m2_pilot"
    dynamics = raw / "cpso_dynamics_m2_pilot"
    topology = raw / "cpso_topology_expansion_m2_pilot"
    if any(p.exists() for p in (sync, dynamics, topology)):
        out = root / "paper" / "figures" / "internal_pilots"
        try:
            payload = generate_paper_internal(sync, dynamics, topology, out)
            paths = [path for value in payload.values() if isinstance(value, list) for path in value]
            results.append({"kind":"paper_ready_internal","output":str(out),"pdf_count":sum(p.suffix==".pdf" for p in paths),"warnings":[v for v in payload.values() if isinstance(v,str)]})
        except Exception as exc:
            results.append({"kind":"paper_ready_internal","output":str(out),"pdf_count":0,"warnings":[f"{type(exc).__name__}: {exc}"]})
    dt_analysis = root / "results" / "analysis" / "cpso_dt_sensitivity_m2_screening"
    if (dt_analysis / "dt_family_profile.csv").is_file():
        out = root / "paper" / "figures" / "dt_m2_screening"
        try:
            paths = generate_paper_dt(dt_analysis, out)
            results.append({"kind":"paper_ready_dt","output":str(out),"pdf_count":sum(p.suffix==".pdf" for p in paths),"warnings":[]})
        except Exception as exc:
            results.append({"kind":"paper_ready_dt","output":str(out),"pdf_count":0,"warnings":[f"{type(exc).__name__}: {exc}"]})
    ext_summary = root / "paper" / "figures" / "external_pilots" / "external_rank_summary.csv"
    if ext_summary.is_file():
        out = root / "paper" / "figures" / "external_pilots"
        try:
            payload = generate_paper_external(ext_summary, out)
            paths = [path for value in payload.values() if isinstance(value, list) for path in value]
            results.append({"kind":"paper_ready_external","output":str(out),"pdf_count":sum(p.suffix==".pdf" for p in paths),"warnings":[v for v in payload.values() if isinstance(v,str)]})
        except Exception as exc:
            results.append({"kind":"paper_ready_external","output":str(out),"pdf_count":0,"warnings":[f"{type(exc).__name__}: {exc}"]})
    return results

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root",type=Path,default=Path("."),help="Repository root")
    ap.add_argument("--output",type=Path,default=None,help="Atlas root (default: paper/figures/atlas)")
    ap.add_argument("--internal-campaign",action="append",default=None,help="Specific internal campaign output; repeatable")
    ap.add_argument("--dt-campaign",action="append",default=None,help="Specific dt campaign output; repeatable")
    ap.add_argument("--external-campaign",action="append",default=None,help="Specific BBOB/CEC campaign output; repeatable")
    ap.add_argument("--legacy-input",action="append",default=None,help="Specific legacy CSV study directory; repeatable")
    args=ap.parse_args();root=args.root.expanduser().resolve();out=args.output.expanduser().resolve() if args.output else root/"paper/figures/atlas"
    discovered=discover(root)
    internal=_resolve_many(args.internal_campaign,root) if args.internal_campaign is not None else discovered[0]
    dt=_resolve_many(args.dt_campaign,root) if args.dt_campaign is not None else discovered[1]
    external=_resolve_many(args.external_campaign,root) if args.external_campaign is not None else discovered[2]
    legacy=_resolve_many(args.legacy_input,root) if args.legacy_input is not None else discovered[3]
    print("Discovered figure sources:")
    for kind,dirs in (("internal",internal),("dt",dt),("external",external),("legacy",legacy)):
        print(f"  {kind}: {len(dirs)}")
        for d in dirs:print(f"    {d}")
    summary=build(root,out,internal,dt,external,legacy)
    paper_ready=build_paper_ready(root)
    if paper_ready:
        summary.extend(paper_ready)
        (out / "paper_ready_manifest.json").write_text(json.dumps(paper_ready,indent=2),encoding="utf-8")
        report = out / "FIGURE_ATLAS.md"
        with report.open("a", encoding="utf-8") as stream:
            stream.write("\n## Figure pronte per la Sezione 4 del manoscritto\n\n")
            for item in paper_ready:
                stream.write(f"- **{item['kind']}**: {item['pdf_count']} PDF in `{item['output']}`\n")
                for warning in item["warnings"]:
                    stream.write(f"  - {warning}\n")
    total=sum(x["pdf_count"] for x in summary)
    print(f"\nAtlas and paper-ready export complete: {total} PDF figures")
    print(f"Catalogue: {out/'FIGURE_ATLAS.md'}")
    return 0 if total else 2

if __name__=="__main__":raise SystemExit(main())
