#!/usr/bin/env python3
"""Run a small, strictly paired CPSO strong-scaling pilot (PDF + CSV only).

Native objectives: sphere_work (synthetic cost) and inverse_diffusion (RK4
inverse-source solver).  Same benchmark, seed, topology, dt, FE budget, and
initial swarm for all synchronous executors. Strict trajectory checks.

No changes to the prior campaign or paper; output MUST be under results/.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import random
import statistics
import sys
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[2]
PYTHON_ROOT = ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))
from cpso_serial import CpsoConfig, native_build_info, run_master_slave, run_particle_parallel, run_serial  # noqa: E402
from cpso_serial.output_policy import results_output  # noqa: E402

ARCHITECTURES = ("serial", "master_slave", "particle_parallel")
TOPOLOGIES = ("global", "ring", "grid_von_neumann", "wheel", "random_k4", "random_k8",
              "small_world_k4", "scale_free_k4")
BENCHMARKS = ("sphere_work", "inverse_diffusion")
FIELDS = ("benchmark", "dimension", "population", "iterations", "seed", "topology",
          "work_iterations", "architecture", "workers", "chunk", "elapsed_seconds",
          "evaluation_wall_seconds", "propagation_seconds", "social_selection_seconds",
          "function_evaluations", "best_value", "best_position", "best_history",
          "history_evaluations")


def parse_csv(value: str, type_fn=str) -> list:
    result = [type_fn(v.strip()) for v in value.split(",") if v.strip()]
    if not result or len(set(result)) != len(result):
        raise ValueError(f"Expected a non-empty comma-separated set of unique values, got {value!r}")
    return result


def config_for(benchmark: str, dimension: int, seed: int, topo: str, n: int, it: int, dt: float):
    lo, hi = (-2., 2.) if benchmark == "inverse_diffusion" else (-100., 100.)
    degree = 8 if topo == "random_k8" else 4
    base = topo.split("_k")[0] if topo.endswith(("_k4", "_k8")) else topo
    base = {"random": "random", "small_world": "small_world", "scale_free": "scale_free"}.get(base, base)
    kwargs = {"grid_rows": 8, "grid_columns": 8} if base == "grid_von_neumann" and n == 64 else {}
    if base == "grid_von_neumann" and n != 64:
        rows = max((r for r in range(2, math.isqrt(n) + 1) if n % r == 0), default=0)
        if rows == 0:
            raise ValueError("Grid requires nontrivial population factorization")
        kwargs = {"grid_rows": rows, "grid_columns": n // rows}
    return CpsoConfig(lower_bounds=(lo,)*dimension, upper_bounds=(hi,)*dimension,
                      population_size=n, iterations=it, seed=seed, dt=dt,
                      mu_max=.9, mu_min=.4, cognitive=2.05, social=2.05,
                      initial_velocity_fraction=.1, boundary_restitution=.5,
                      topology=base, topology_degree=degree,
                      topology_rewiring_probability=.15, **kwargs)


def job_key(row: dict) -> tuple:
    return tuple(row[field] for field in ("benchmark", "dimension", "seed", "topology",
                                           "work_iterations", "architecture", "workers"))


def run_one(conf: CpsoConfig, benchmark: str, cost: int, architecture: str, workers: int):
    if architecture == "serial":
        return run_serial(conf, benchmark, work_iterations=cost)
    if architecture == "master_slave":
        return run_master_slave(conf, benchmark, workers=workers, work_iterations=cost)
    if architecture == "particle_parallel":
        return run_particle_parallel(conf, benchmark, workers=workers, work_iterations=cost)
    raise ValueError(architecture)


def verify_group(rows: list[dict], population: int, iterations: int) -> None:
    serial = [r for r in rows if r["architecture"] == "serial"]
    if len(serial) != 1:
        raise RuntimeError("A paired group must contain exactly one serial record")
    reference = serial[0]
    expected_fe = population * (iterations+1)
    for row in rows:
        if int(row["function_evaluations"]) != expected_fe:
            raise RuntimeError(f"FE mismatch: {row['architecture']}, {row['workers']}")
        for key in ("best_value", "best_position", "best_history", "history_evaluations"):
            if row[key] != reference[key]:
                raise RuntimeError(f"Synchronous equivalence failed: {key} on {row['architecture']} w{row['workers']}")
        if not math.isfinite(float(row["elapsed_seconds"])) or float(row["elapsed_seconds"]) <= 0.:
            raise RuntimeError("Elapsed wall time is invalid")


def row_from_result(benchmark, d, n, it, seed, topo, work, arch, workers, chunk, result):
    return dict(benchmark=benchmark, dimension=d, population=n, iterations=it, seed=seed,
                topology=topo, work_iterations=work, architecture=arch,
                workers=workers, chunk=chunk, elapsed_seconds=result.elapsed_seconds,
                evaluation_wall_seconds=result.evaluation_wall_seconds,
                propagation_seconds=result.propagation_seconds,
                social_selection_seconds=result.social_selection_seconds,
                function_evaluations=result.function_evaluations, best_value=result.best_value,
                best_position=list(result.best_position), best_history=list(result.best_history),
                history_evaluations=list(result.history_evaluations))


def csv_rows(filename: Path, rows: list[dict]):
    if not rows:
        return
    with filename.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def bootstrap_median(numbers: list[float], seed: int, b: int = 1500):
    centre = statistics.median(numbers)
    if len(numbers) < 2:
        return centre, centre, centre
    rng = random.Random(seed)
    medians = sorted(statistics.median(numbers[rng.randrange(len(numbers))]
                                      for _ in range(len(numbers))) for _ in range(b))
    return centre, medians[int(0.025 * (b-1))], medians[int(0.975 * (b-1))]


def aggregate(rows: list[dict]):
    by_group = defaultdict(dict)
    for row in rows:
        case = tuple(row[k] for k in ("benchmark", "dimension", "seed", "topology", "work_iterations"))
        marker = (row["architecture"], row["workers"])
        if marker in by_group[case]:
            raise RuntimeError(f"Duplicate results for {case}, {marker}")
        by_group[case][marker] = row
    paired = defaultdict(list)
    for case, methods in by_group.items():
        ref = methods.get(("serial", 1))
        if ref is None:
            raise RuntimeError(f"Missing serial reference for {case}")
        for (architecture, workers), row in methods.items():
            if architecture == "serial":
                continue
            if any(row[k] != ref[k] for k in ("best_value", "best_position", "best_history", "history_evaluations")):
                raise RuntimeError(f"Equivalence not satisfied for {case}, {architecture}, {workers}")
            t_ref, t = float(ref["elapsed_seconds"]), float(row["elapsed_seconds"])
            if t_ref <= 0 or t <= 0: raise RuntimeError("Non-positive elapsed time")
            paired[(case[0], case[1], case[3], case[4], architecture, workers)].append(t_ref / t)
    summary = []
    for key, ratios in sorted(paired.items()):
        ident = "|".join(map(str, key))
        bootstrap_seed = int(hashlib.sha256(ident.encode()).hexdigest()[:12], 16)
        median, low, high = bootstrap_median(ratios, bootstrap_seed)
        benchmark, dim, topo, work, arch, workers = key
        summary.append(dict(benchmark=benchmark, dimension=dim, topology=topo,
                            work_iterations=work, architecture=arch, workers=workers,
                            n_paired_seeds=len(ratios), speedup_median=median,
                            speedup_ci95_low=low, speedup_ci95_high=high,
                            efficiency_median=median/workers,
                            efficiency_ci95_low=low/workers,
                            efficiency_ci95_high=high/workers))
    return summary


def make_plots(summary: list[dict], out: Path, chunk: int):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family":"serif", "font.size":9,
                         "axes.spines.top":False, "axes.spines.right":False,
                         "pdf.fonttype":42, "savefig.bbox":"tight"})
    bds = sorted({(r["benchmark"], r["dimension"], r["topology"]) for r in summary})
    for benchmark, d, topo in bds:
        data = [r for r in summary if (r["benchmark"],r["dimension"],r["topology"]) == (benchmark,d,topo)]
        loads = sorted({r["work_iterations"] for r in data})
        cols = len(loads)
        fig, axes = plt.subplots(2, cols, figsize=(4.0*cols,6.0), squeeze=False, sharex="col")
        styles = {"master_slave":("#21618c","o","Master-Slave"),
                  "particle_parallel":("#bf6424","s","Particle-Parallel")}
        for col, work in enumerate(loads):
            subset = [r for r in data if r["work_iterations"] == work]
            for row_idx, (metric, lower, upper) in enumerate([
                ("speedup_median","speedup_ci95_low","speedup_ci95_high"),
                ("efficiency_median","efficiency_ci95_low","efficiency_ci95_high")]):
                ax = axes[row_idx][col]
                for arch, (color,marker,label) in styles.items():
                    series = sorted((r for r in subset if r["architecture"]==arch),key=lambda r:r["workers"])
                    if not series: continue
                    x = [r["workers"] for r in series]; y = [r[metric] for r in series]
                    yerr = [[max(0.0,r[metric]-r[lower]) for r in series],
                            [max(0.0,r[upper]-r[metric]) for r in series]]
                    ax.errorbar(x,y,yerr=yerr,fmt=marker+"-",lw=1.8,markersize=5,
                                color=color,capsize=2.5,label=label if (row_idx==0 and col==0) else None)
                workers = sorted({r["workers"] for r in subset})
                if row_idx == 0:
                    ax.plot([1,max(workers)],[1,max(workers)],ls="--",c="#777777",lw=.9,label="Ideal" if col==0 else None)
                else:
                    ax.axhline(1.,ls="--",c="#777777",lw=.9)
                ax.set_xticks(workers)
                ax.set_xlim(0.8, max(workers)+.3)
                ax.grid(alpha=.2)
                if col==0: ax.set_ylabel("Speedup S(p)" if row_idx==0 else "Efficiency E(p)")
                if row_idx==0: ax.set_title(f"Work = {work:,} steps/iterations")
                if row_idx==1: ax.set_xlabel("Workers p")
        fig.suptitle(f"{benchmark.replace('_',' ').title()}  |  D={d}  |  {topo.replace('_',' ')}", y=.99,fontsize=13)
        handles,labels=axes[0][0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False,
                   bbox_to_anchor=(.5,.94))
        fig.text(.5,.016,f"Matched-seed median + 95% bootstrap interval; end-to-end native runtime; chunk={chunk}; descriptive pilot",
                 ha="center",fontsize=7,color="#555555")
        fig.tight_layout(rect=(0,.045,1,.9))
        name=f"fig_scaling_{benchmark}_{topo}_d{d}.pdf"
        fig.savefig(out/name)
        plt.close(fig)
        print(f"  PDF: {out/name}")


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--benchmarks",default="sphere_work,inverse_diffusion")
    ap.add_argument("--topologies",default="global,ring")
    ap.add_argument("--dimensions",default="30")
    ap.add_argument("--work",default="128,512")
    ap.add_argument("--workers",default="1,2,4")
    ap.add_argument("--seeds",default="91001,91002,91003")
    ap.add_argument("--population",type=int,default=64)
    ap.add_argument("--iterations",type=int,default=32)
    ap.add_argument("--dt",type=float,default=0.38125)
    ap.add_argument("--chunk-size",type=int,default=1)
    ap.add_argument("--output",type=Path,default=None)
    ap.add_argument("--dry-run",action="store_true")
    ap.add_argument("--no-warmup",action="store_true")
    args=ap.parse_args()
    bs=parse_csv(args.benchmarks)
    topologies=parse_csv(args.topologies)
    dims=parse_csv(args.dimensions,int)
    costs=parse_csv(args.work,int)
    workers=parse_csv(args.workers,int)
    seeds=parse_csv(args.seeds,int)
    if set(bs)-set(BENCHMARKS) or set(topologies)-set(TOPOLOGIES):
        raise ValueError("Unknown benchmark/topology")
    if min(dims)<1 or min(costs)<0 or min(workers)<1 or min(seeds)<0:
        raise ValueError("Dimensions/work/seeds must be nonnegative, workers must be positive")
    if args.chunk_size<1 or args.chunk_size>10000 or args.population<2 or args.iterations<1 or args.dt<=0:
        raise ValueError("Invalid chunk/population/iterations/dt")
    if "inverse_diffusion" in bs and any(w<16 or w>100000 or w%4 for w in costs):
        raise ValueError("inverse_diffusion requires integration steps divisible by 4 in [16,100000]")
    if "sphere_work" in bs and max(costs)>100_000_000: raise ValueError("Excessive synthetic work")
    if "grid_von_neumann" in topologies and args.population<4: raise ValueError("Grid population too small")
    key = f"chunk{args.chunk_size}"
    out=results_output(args.output or ROOT/"results"/"raw"/"parallel"/f"strong_scaling_pilot_{key}")
    groups=[(b,d,s,t,w) for b in bs for d in dims for s in seeds for t in topologies for w in costs]
    methods=[("serial",1)]+[(a,p) for a in ARCHITECTURES[1:] for p in workers]
    print(f"Pilot: {len(groups)} matched cases; {len(methods)} methods each; {len(groups)*len(methods)} total runs")
    print(f"  output = {out}")
    if args.dry_run: return 0
    if args.chunk_size == 1:
        os.environ.pop("CPSO_THREADPOOL_CHUNK", None)
    else:
        os.environ["CPSO_THREADPOOL_CHUNK"] = str(args.chunk_size)
    config_payload={"benchmarks":bs,"topologies":topologies,"dimensions":dims,"work":costs,
                    "workers":workers,"seeds":seeds,"population":args.population,
                    "iterations":args.iterations,"dt":args.dt,"chunk_size":args.chunk_size,
                    "native_build_info":native_build_info()}
    out.mkdir(parents=True,exist_ok=True)
    manifest=out/"scaling_pilot_manifest.json"
    if manifest.exists():
        current=json.loads(manifest.read_text(encoding="utf-8"))
        if current != config_payload: raise RuntimeError("Existing output was created with different settings or native build. Choose another --output")
    else:
        manifest.write_text(json.dumps(config_payload,indent=2)+"\n",encoding="utf-8")
    records_file=out/"scaling_pilot_runs.jsonl"
    completed={}
    if records_file.exists():
        for line in records_file.read_text(encoding="utf-8").splitlines():
            row=json.loads(line)
            key=job_key(row)
            if key in completed: raise RuntimeError(f"Duplicate run {key}")
            completed[key]=row
    rng=random.Random(20261114)
    rng.shuffle(groups)
    if not args.no_warmup:
        b,d,s,t,w=groups[0]
        c=config_for(b,d,s,t,args.population,2,args.dt)
        for a,nw in methods:
            run_one(c,b,w,a,nw)
        print("Warm-up completed (not recorded)")
    for idx,(b,d,s,t,w) in enumerate(groups,1):
        conf=config_for(b,d,s,t,args.population,args.iterations,args.dt)
        shuffled=list(methods)
        rng.shuffle(shuffled)
        for a,p in shuffled:
            ident=(b,d,s,t,w,a,p)
            if ident in completed: continue
            result=run_one(conf,b,w,a,p)
            row=row_from_result(b,d,args.population,args.iterations,s,t,w,a,p,args.chunk_size,result)
            with records_file.open("a",encoding="utf-8") as stream:
                stream.write(json.dumps(row,separators=(",",":"))+"\n")
                stream.flush()
            completed[ident]=row
        these=[completed[(b,d,s,t,w,a,p)] for a,p in methods]
        verify_group(these,args.population,args.iterations)
        print(f"[{idx}/{len(groups)}] {b} D={d} {t} work={w} seed={s}: exact-equivalence PASS")
    summary=aggregate(list(completed.values()))
    csv_rows(out/"scaling_pilot_summary.csv",summary)
    make_plots(summary,out,args.chunk_size)
    print(f"Complete: {len(completed)} runs; {len(summary)} paired summary rows")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
