#!/usr/bin/env python3
"""Publication-oriented, repeated-measures strong scaling for synchronous CPSO.

No modifications to the CPSO mathematical dynamics. Uses the repo's native
benchmark and synchronous serial/master-slave/particle-parallel APIs. Records
both serial-reference acceleration and backend-anchored strong scaling.

Execute from repository root. Designed to coexist with run_scaling_pilot.py.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
from cpso_serial import (  # noqa: E402
    CpsoConfig, native_build_info, run_master_slave, run_particle_parallel, run_serial,
)
from cpso_serial.native import _load_library  # noqa: E402

TOPOS = ("global", "ring", "grid_von_neumann", "wheel", "random_k4", "random_k8",
         "small_world_k4", "scale_free_k4")
ARCHS = ("master_slave", "particle_parallel")


def separated(value, parser=str):
    items = [parser(x.strip()) for x in value.split(",") if x.strip()]
    if not items or len(items) != len(set(items)):
        raise ValueError("List must contain unique, nonempty values: " + repr(value))
    return items


def now():
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False)


def hash_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        while part := stream.read(1 << 20):
            h.update(part)
    return h.hexdigest()


def configuration(d, seed, topo, n, it, dt):
    name = topo.split("_k")[0] if topo.endswith(("_k4", "_k8")) else topo
    kwargs = {}
    if name == "grid_von_neumann":
        div = [x for x in range(2, math.isqrt(n) + 1) if n % x == 0]
        if not div:
            raise ValueError("Grid topology requires a composite population size")
        rows = max(div)
        kwargs = {"grid_rows": rows, "grid_columns": n // rows}
    cfg = CpsoConfig(
        lower_bounds=(-2.0,) * d, upper_bounds=(2.0,) * d,
        population_size=n, iterations=it, seed=seed, dt=dt,
        mu_max=0.9, mu_min=0.4, cognitive=2.05, social=2.05,
        initial_velocity_fraction=0.1, boundary_restitution=0.5,
        topology=name, topology_degree=8 if topo == "random_k8" else 4,
        topology_rewiring_probability=0.15, **kwargs,
    )
    cfg.validate()
    return cfg


def run_method(cfg, work, arch, workers):
    if arch == "serial":
        return run_serial(cfg, "inverse_diffusion", work_iterations=work)
    if arch == "master_slave":
        return run_master_slave(cfg, "inverse_diffusion", workers=workers,
                                work_iterations=work)
    if arch == "particle_parallel":
        return run_particle_parallel(cfg, "inverse_diffusion", workers=workers,
                                     work_iterations=work)
    raise ValueError("Unrecognized architecture: " + arch)


def block_key(d, topo, seed, repeat):
    return (d, topo, seed, repeat)


def record_key(row):
    return (row["dimension"], row["topology"], row["seed"], row["repetition"],
            row["architecture"], row["workers"])


def record_from_result(d, topo, seed, rep, arch, workers, chunk, n, it, work, ordinal, value):
    return {
        "benchmark": "inverse_diffusion", "dimension": d, "topology": topo,
        "seed": seed, "repetition": rep, "architecture": arch, "workers": workers,
        "chunk": chunk, "work_iterations": work, "population": n, "iterations": it,
        "ordinal": ordinal, "utc_finished": now(),
        "elapsed_seconds": value.elapsed_seconds,
        "evaluation_wall_seconds": value.evaluation_wall_seconds,
        "propagation_seconds": value.propagation_seconds,
        "social_selection_seconds": value.social_selection_seconds,
        "topology_setup_seconds": value.topology_setup_seconds,
        "initialization_seconds": value.initialization_seconds,
        "evaluation_worker_seconds": value.evaluation_worker_seconds,
        "propagation_worker_seconds": value.propagation_worker_seconds,
        "function_evaluations": value.function_evaluations,
        "best_value": value.best_value,
        "best_position": list(value.best_position),
        "best_history": list(value.best_history),
        "history_evaluations": list(value.history_evaluations),
    }


def check_block(records, methods, n, it):
    if set(records) != set(methods):
        raise ValueError("Incomplete method set: " + repr(set(methods) - set(records)))
    serial = records[("serial", 1)]
    fe = n * (it + 1)
    if serial["function_evaluations"] != fe:
        raise ValueError("Serial FE count incorrect")
    for m, row in records.items():
        if row["function_evaluations"] != fe:
            raise ValueError(f"Function-evaluation mismatch {m}")
        for field in ("best_value", "best_position", "best_history", "history_evaluations"):
            if row[field] != serial[field]:
                raise ValueError(f"Exact numerical equivalence failed for {m}: {field}")
        elapsed = row["elapsed_seconds"]
        if not isinstance(elapsed, (int, float)) or not math.isfinite(elapsed) or elapsed <= 0:
            raise ValueError(f"Invalid time for {m}")


def percentile_sorted(arr, probability):
    return arr[min(len(arr) - 1, max(0, int((len(arr) - 1) * probability)))]


def seed_cluster_summary(data, identity, bootstraps):
    per_seed = {seed: statistics.median(values) for seed, values in data.items()}
    values = list(per_seed.values())
    mid = statistics.median(values)
    if len(values) < 2:
        return mid, mid, mid, len(values), sum(map(len, data.values()))
    salt = int(hashlib.sha256(canonical(identity).encode()).hexdigest()[:14], 16)
    rng = random.Random(salt)
    sample_medians = []
    n = len(values)
    for _ in range(bootstraps):
        sample_medians.append(statistics.median(values[rng.randrange(n)] for _ in range(n)))
    sample_medians.sort()
    return (mid, percentile_sorted(sample_medians, 0.025),
            percentile_sorted(sample_medians, 0.975), len(values),
            sum(map(len, data.values())))


def summarize(raw, settings):
    methods = [("serial", 1)] + [(a, w) for a in settings["architectures"]
                              for w in settings["workers"]]
    blocks = defaultdict(dict)
    for row in raw:
        b = block_key(row["dimension"], row["topology"], row["seed"], row["repetition"])
        m = (row["architecture"], row["workers"])
        if m in blocks[b]:
            raise ValueError(f"Duplicate record for block {b} method {m}")
        blocks[b][m] = row
    if len(blocks) != settings["n_blocks"]:
        raise ValueError("Cannot summarize an incomplete campaign")
    accum = defaultdict(lambda: defaultdict(list))
    paired = []
    for key, group in sorted(blocks.items()):
        check_block(group, methods, settings["population"], settings["iterations"])
        d, topo, seed, repeat = key
        ref_time = group[("serial", 1)]["elapsed_seconds"]
        for arch in settings["architectures"]:
            base1 = group[(arch, 1)]["elapsed_seconds"]
            for w in settings["workers"]:
                actual = group[(arch, w)]["elapsed_seconds"]
                metrics = {
                    "serial_speedup": ref_time / actual,
                    "serial_efficiency": ref_time / actual / w,
                    "strong_speedup": base1 / actual,
                    "strong_efficiency": base1 / actual / w,
                    "runtime_seconds": actual,
                }
                observation = {
                    "dimension": d, "topology": topo, "seed": seed, "repetition": repeat,
                    "architecture": arch, "workers": w,
                    "serial_seconds": ref_time,
                    "backend_one_worker_seconds": base1,
                    **metrics,
                }
                paired.append(observation)
                for metric, value in metrics.items():
                    accum[(d, topo, arch, w, metric)][seed].append(value)
    summary = []
    for (d, topo, arch, w, metric), seed_data in sorted(accum.items()):
        ident = [d, topo, arch, w, metric]
        mid, lo, hi, ns, nr = seed_cluster_summary(seed_data, ident, settings["bootstrap_draws"])
        summary.append({
            "benchmark": "inverse_diffusion", "dimension": d, "topology": topo,
            "work_iterations": settings["work_iterations"],
            "architecture": arch, "workers": w, "metric": metric,
            "median": mid, "ci95_low": lo, "ci95_high": hi,
            "n_seeds": ns, "n_observations": nr,
        })
    return paired, summary


def write_csv(path, rows):
    if not rows:
        raise ValueError("Refusing empty CSV")
    temp = path.with_suffix(".tmp")
    with temp.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(path)


def make_figures(summary, output, settings):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "serif", "font.size": 9, "axes.spines.top": False,
        "axes.spines.right": False, "pdf.fonttype": 42, "savefig.bbox": "tight",
    })
    styles = {
        "master_slave": ("#21618C", "o", "Master–Slave"),
        "particle_parallel": ("#BA5B29", "s", "Particle-Parallel"),
    }
    for d in settings["dimensions"]:
        for topo in settings["topologies"]:
            data = [r for r in summary if r["dimension"] == d and r["topology"] == topo]
            fig, axes = plt.subplots(2, 2, figsize=(10.2, 6.5), sharex="col")
            panels = [
                (0, 0, "strong_speedup", "Strong scaling — backend p=1", "Speedup"),
                (1, 0, "strong_efficiency", "", "Efficiency"),
                (0, 1, "serial_speedup", "Acceleration — serial baseline", "Speedup"),
                (1, 1, "serial_efficiency", "", "Efficiency"),
            ]
            for r, c, metric, title, ylabel in panels:
                ax = axes[r, c]
                for arch in settings["architectures"]:
                    color, mark, lab = styles[arch]
                    series = sorted((x for x in data if x["architecture"] == arch and
                                     x["metric"] == metric), key=lambda x: x["workers"])
                    if not series:
                        continue
                    xx = [x["workers"] for x in series]
                    yy = [x["median"] for x in series]
                    lows = [x["ci95_low"] for x in series]
                    highs = [x["ci95_high"] for x in series]
                    ax.plot(xx, yy, marker=mark, color=color, linewidth=1.8, markersize=5,
                            label=lab)
                    ax.fill_between(xx, lows, highs, color=color, alpha=.13, linewidth=0)
                xmax = max(settings["workers"])
                if r == 0:
                    ax.plot([1, xmax], [1, xmax], ls="--", lw=1., color="#6b7280",
                            label="Ideal")
                else:
                    ax.axhline(1, ls="--", lw=1., color="#6b7280")
                if title:
                    ax.set_title(title, fontsize=10)
                if c == 0:
                    ax.set_ylabel(ylabel)
                if r == 1:
                    ax.set_xlabel("Workers (p)")
                ax.set_xticks(settings["workers"])
                ax.set_xlim(.8, xmax + .3)
                ax.grid(alpha=.16)
            handles, names = axes[0, 0].get_legend_handles_labels()
            fig.legend(handles, names, loc="upper center", ncol=3, frameon=False,
                       bbox_to_anchor=(.5, .94))
            fig.suptitle(f"CPSO · inverse diffusion · D={d} · {topo.replace('_', ' ')}",
                         fontsize=13, y=.99)
            fig.text(.5, .025,
                     f"N={settings['population']}  |  {settings['iterations']} iterations  |  "
                     f"RK4 steps={settings['work_iterations']}  |  "
                     f"{len(settings['seeds'])} seeds × {settings['repetitions']} timing repeats  |  "
                     f"chunk={settings['chunk_size']}\n"
                     "Centres: medians of seed medians; 95% seed-cluster bootstrap intervals; "
                     "identical objective and CPSO trajectory",
                     ha="center", color="#555555", fontsize=8)
            fig.tight_layout(rect=(0, .075, 1, .895))
            name = f"fig_publication_scaling_inverse_diffusion_{topo}_d{d}.pdf"
            fig.savefig(output / name)
            plt.close(fig)
            print("Figure:", output / name)


def get_parser():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dimensions", default="30,100")
    ap.add_argument("--topologies", default="global,ring")
    ap.add_argument("--workers", default="1,2,4,8")
    ap.add_argument("--architectures", default="master_slave,particle_parallel")
    ap.add_argument("--seeds", default="95001,95002,95003,95004,95005,95006,95007,95008,95009,95010")
    ap.add_argument("--repetitions", type=int, default=5)
    ap.add_argument("--population", type=int, default=64)
    ap.add_argument("--iterations", type=int, default=64)
    ap.add_argument("--dt", type=float, default=.38125)
    ap.add_argument("--work", type=int, default=2048)
    ap.add_argument("--chunk-size", type=int, default=1)
    ap.add_argument("--bootstrap", type=int, default=3000)
    ap.add_argument("--schedule-seed", type=int, default=20261114)
    ap.add_argument("--output", type=Path, default=Path("results/raw/parallel/paper_P1_publication_scaling"))
    ap.add_argument("--max-blocks", type=int, default=0,
                    help="Only complete this many outstanding paired blocks; resume later without this flag")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-warmup", action="store_true")
    ap.add_argument("--no-figures", action="store_true")
    return ap


def settings_from_args(args):
    d = separated(args.dimensions, int)
    top = separated(args.topologies)
    w = separated(args.workers, int)
    arch = separated(args.architectures)
    seeds = separated(args.seeds, int)
    if any(x not in TOPOS for x in top):
        raise ValueError("Unknown topology in " + str(top))
    if any(x not in ARCHS for x in arch):
        raise ValueError("Unknown architecture in " + str(arch))
    if 1 not in w:
        raise ValueError("Workers must include p=1 for backend-anchored strong scaling")
    if any(x < 1 or x > 128 for x in w) or any(x < 1 for x in d):
        raise ValueError("Invalid workers or dimensions")
    if any(x < 0 for x in seeds) or args.repetitions < 2 or args.population < 2:
        raise ValueError("Invalid seed/repetitions/population")
    if args.iterations < 1 or not (math.isfinite(args.dt) and args.dt > 0):
        raise ValueError("Invalid iterations/dt")
    if not (16 <= args.work <= 100000 and args.work % 4 == 0):
        raise ValueError("inverse_diffusion work must be a multiple of 4 in [16,100000]")
    if args.chunk_size < 1 or args.chunk_size > 10000:
        raise ValueError("chunk-size must be positive and at most 10000")
    if args.bootstrap < 100 or args.max_blocks < 0:
        raise ValueError("Invalid bootstrap/max-blocks")
    if any(t == "grid_von_neumann" for t in top):
        if not [x for x in range(2, math.isqrt(args.population)+1) if args.population % x == 0]:
            raise ValueError("Grid requires composite population")
    count = len(d) * len(top) * len(seeds) * args.repetitions
    return {
        "benchmark": "inverse_diffusion", "dimensions": d, "topologies": top,
        "workers": w, "architectures": arch, "seeds": seeds,
        "repetitions": args.repetitions, "population": args.population,
        "iterations": args.iterations, "dt": args.dt, "work_iterations": args.work,
        "chunk_size": args.chunk_size, "bootstrap_draws": args.bootstrap,
        "schedule_seed": args.schedule_seed, "n_blocks": count,
    }


def main(argv=None):
    args = get_parser().parse_args(argv)
    settings = settings_from_args(args)
    methods = [("serial", 1)] + [(a, w) for a in settings["architectures"]
                               for w in settings["workers"]]
    blocks = [(d, t, seed, rep) for d in settings["dimensions"]
              for t in settings["topologies"]
              for seed in settings["seeds"]
              for rep in range(1, settings["repetitions"] + 1)]
    rng = random.Random(args.schedule_seed)
    rng.shuffle(blocks)
    out = args.output.expanduser().resolve()
    if out == ROOT or ROOT not in out.parents or "results" not in out.relative_to(ROOT).parts:
        raise ValueError("--output must be inside this repository's results/ directory")
    print(f"Campaign: {len(blocks)} repeated paired blocks × {len(methods)} methods = "
          f"{len(blocks) * len(methods)} native executions")
    print("Settings:", canonical(settings))
    print("Output:", out)
    if args.dry_run:
        return 0
    if args.chunk_size == 1:
        os.environ.pop("CPSO_THREADPOOL_CHUNK", None)
    else:
        os.environ["CPSO_THREADPOOL_CHUNK"] = str(args.chunk_size)
    native_info = native_build_info()
    native_path = Path(_load_library()._name)
    binary_sha = hash_file(native_path)
    if "Release" not in native_info:
        raise RuntimeError("CPSO native library must be built in Release mode")
    out.mkdir(parents=True, exist_ok=True)
    manifest = out / "publication_scaling_manifest.json"
    execution_fingerprint = {
        "settings": settings, "native_build_info": native_info,
        "native_binary_sha256": binary_sha,
    }
    if manifest.exists():
        previous = json.loads(manifest.read_text())
        if previous.get("execution_fingerprint") != execution_fingerprint:
            raise RuntimeError("Campaign settings/build changed since first execution; choose another --output")
    else:
        metadata = {
            "execution_fingerprint": execution_fingerprint,
            "platform": platform.platform(), "machine": platform.machine(),
            "python_version": sys.version,
            "cpu_logical_count": os.cpu_count(),
            "started_utc": now(),
            "notes": "Repeated independent timings; one paired serial reference per seed x repetition; randomized block and method order",
        }
        manifest.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
    raw_path = out / "publication_scaling_runs.jsonl"
    completed = {}
    if raw_path.exists():
        with raw_path.open(encoding="utf-8") as source:
            for line in source:
                if not line.strip():
                    continue
                item = json.loads(line)
                key = record_key(item)
                if key in completed:
                    raise RuntimeError("Duplicate cached run " + repr(key))
                completed[key] = item
    if not args.no_warmup:
        d, t, seed, rep = blocks[0]
        cfg = configuration(d, seed, t, settings["population"], 2, settings["dt"])
        for a, w in methods:
            run_method(cfg, settings["work_iterations"], a, w)
        print("Warm-up completed (not recorded)")
    completed_blocks = 0
    ordinal = max((row.get("ordinal", 0) for row in completed.values()), default=0)
    for idx, block in enumerate(blocks, 1):
        d, topo, seed, rep = block
        group = {}
        for a, w in methods:
            key = (d, topo, seed, rep, a, w)
            if key in completed:
                group[(a, w)] = completed[key]
        if len(group) == len(methods):
            check_block(group, methods, settings["population"], settings["iterations"])
            continue
        cfg = configuration(d, seed, topo, settings["population"], settings["iterations"], settings["dt"])
        # Deterministic per-block shuffling: interruption/resume cannot change run order.
        salt = int(hashlib.sha256(canonical(block).encode()).hexdigest()[:14], 16)
        order = methods[:]
        random.Random(salt ^ args.schedule_seed).shuffle(order)
        for a, w in order:
            if (a, w) in group:
                continue
            value = run_method(cfg, settings["work_iterations"], a, w)
            ordinal += 1
            row = record_from_result(d, topo, seed, rep, a, w, settings["chunk_size"],
                                     settings["population"], settings["iterations"],
                                     settings["work_iterations"], ordinal, value)
            with raw_path.open("a", encoding="utf-8") as stream:
                stream.write(canonical(row) + "\n")
                stream.flush()
            group[(a, w)] = row
            completed[(d, topo, seed, rep, a, w)] = row
        check_block(group, methods, settings["population"], settings["iterations"])
        completed_blocks += 1
        print(f"[{idx}/{len(blocks)}] D={d} {topo} seed={seed} rep={rep}: "
              f"exact equivalence PASS (new blocks this call: {completed_blocks})", flush=True)
        if args.max_blocks and completed_blocks >= args.max_blocks:
            break
    expected_n = len(blocks) * len(methods)
    if len(completed) != expected_n:
        print(f"Partial checkpoint: {len(completed)}/{expected_n} runs saved. "
              "Repeat identical command without --max-blocks to resume.")
        return 0
    paired, summary = summarize(list(completed.values()), settings)
    write_csv(out / "publication_scaling_paired.csv", paired)
    write_csv(out / "publication_scaling_summary.csv", summary)
    if not args.no_figures:
        make_figures(summary, out, settings)
    print(f"COMPLETE: {len(completed)} native executions, {len(paired)} paired comparison rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
