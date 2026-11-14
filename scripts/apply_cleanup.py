#!/usr/bin/env python3
"""Safely apply the curated CPSO source snapshot to an existing checkout.

Default mode is preview only. Never touches the manuscript or experimental data.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

SOURCE = Path(__file__).resolve().parents[1]
SKIP_PARTS = {".git", "paper", "results", ".venv", "build", "build-m2", "build-linux", "__pycache__", ".pytest_cache"}


def source_files():
    for item in sorted(SOURCE.rglob("*")):
        if not item.is_file():
            continue
        rel = item.relative_to(SOURCE)
        if any(part in SKIP_PARTS or part.startswith("build-") for part in rel.parts):
            continue
        if item.name in {".DS_Store"} or item.suffix == ".pyc":
            continue
        yield rel, item


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--target", type=Path, required=True, help="Existing repository checkout")
    p.add_argument("--apply", action="store_true", help="Apply after reviewing a dry run")
    args = p.parse_args()
    target = args.target.expanduser().resolve()
    if not target.is_dir() or not (target / "CMakeLists.txt").is_file() or not (target / "cpp/src/optimizer.cpp").is_file():
        p.error("--target must point to an existing CPSO repository")
    if target == SOURCE or SOURCE in target.parents or target in SOURCE.parents:
        p.error("Source and target must be separate sibling directories")
    manifest=json.loads((SOURCE / "CLEANUP_MANIFEST.json").read_text())
    obsolete=[]
    for name in manifest["legacy_paths"]:
        f=target/name
        if f.exists() and f.is_file():
            obsolete.append(Path(name))
    replacements=[]; additions=[]
    for rel,source in source_files():
        dest=target/rel
        if dest.is_file() and dest.read_bytes()==source.read_bytes():
            continue
        (replacements if dest.exists() else additions).append(rel)
    print(f"Target: {target}")
    print(f"Legacy files to archive: {len(obsolete)}; changed files to back up and replace: {len(replacements)}; new files: {len(additions)}")
    for group,items in (("REMOVE",obsolete),("REPLACE",replacements),("ADD",additions)):
        for name in items:
            print(f"  {group:7} {name}")
    if not args.apply:
        print("PREVIEW ONLY — no changes. Rerun with --apply to perform safe migration.")
        return 0
    backup=target/'results/cleanup_backups'/datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')
    backup.mkdir(parents=True,exist_ok=False)
    for rel in obsolete:
        dest=target/rel
        old=backup/rel;old.parent.mkdir(parents=True,exist_ok=True)
        shutil.move(str(dest),str(old))
    for rel in replacements:
        dest=target/rel
        old=backup/rel;old.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(dest,old)
    for rel in [*replacements,*additions]:
        source=SOURCE/rel
        dest=target/rel;dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source,dest)
    for folder in ("experiments/02_master_slave","experiments/03_topologies","configs/experiments"):
        directory=target/folder
        if directory.is_dir() and not any(directory.iterdir()):
            directory.rmdir()
    # Archive only generated Python caches in the source tree. No results or paper paths.
    caches=[]
    for folder in ("analysis","plot_results","python","tests","experiments","scripts"):
        root=target/folder
        if not root.exists():continue
        for d in list(root.rglob("__pycache__")):
            if d.is_dir():
                shutil.rmtree(d);caches.append(str(d.relative_to(target)))
    (target/'results/README.md').parent.mkdir(parents=True,exist_ok=True)
    if not (target/'results/README.md').exists():
        shutil.copy2(SOURCE/'results/README.md',target/'results/README.md')
    for child in ("raw/dt","raw/parallel","analysis/dt","analysis/parallel","figures/dt","figures/parallel","figures/external","figures/atlas","external_benchmarks","logs","configs"):
        (target/'results'/child).mkdir(parents=True,exist_ok=True)
    summary={"timestamp_utc":datetime.now(timezone.utc).isoformat(),"source":str(SOURCE),"target":str(target),"archived":list(map(str,obsolete)),"replaced":list(map(str,replacements)),"added":list(map(str,additions)),"caches_removed":caches}
    (backup/'MIGRATION_SUMMARY.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(f"COMPLETE. Source backup at: {backup}")
    print("Existing paper/, results/ runs and .git/ were untouched. Review git diff and rerun tests.")
    return 0

if __name__=='__main__':
    raise SystemExit(main())
