#!/usr/bin/env python3
"""Study III: screen, ablate, and compare the parallel CPSO."""
from __future__ import annotations

import argparse
import fcntl
import importlib.metadata
import itertools
import math
import os
from pathlib import Path

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(variable, "1")

from publication_tools.common import cached_run, fingerprint, freeze, prepare, read, shuffled
from publication_tools.study3_analysis import ADAPTATION_GAIN, GAP_FLOOR, PRACTICAL_TIE, TARGETS, analyze
from publication_tools.study3_execution import (
    BBOB_HOLDOUT, BBOB_PILOT, NATIVE_FUNCTIONS, POPULATION, method_plan,
    planned_jobs, run_job, source_manifest,
)

STAGES = ("screen", "ablation", "fe", "time")
PROFILES = ("smoke", "pilot", "confirm")


def _stage_dir(root, stage, suite):
    return root / ("fe_bbob" if suite == "bbob" else stage)


def _previous(root, stage):
    folder = _stage_dir(root, stage, "native")
    files = [folder / name for name in ("complete.json", "protocol.json", "selected.json")]
    if not all(path.is_file() for path in files):
        raise ValueError(f"Completare prima '{stage}' in {root}")
    complete, protocol, selected = [read(path) for path in files]
    if (protocol["stage"] != stage or complete["protocol_sha256"] != fingerprint(protocol)
            or complete["decision_sha256"] != fingerprint(selected)):
        raise ValueError(f"Protocollo o ricetta incoerente in {folder}")
    return selected["recipe"], dict(stage=stage, protocol=complete["protocol_sha256"],
                                    decision=complete["decision_sha256"])


def _design(args):
    stage, suite, profile = args.stage, args.suite, args.profile
    if suite == "bbob" and stage != "fe":
        raise ValueError("BBOB è disponibile per lo stage fe")
    if stage in {"screen", "ablation"} and profile == "confirm":
        raise ValueError("Screen e ablation sono campagne di sviluppo: usare il profilo pilot")
    timed = stage == "time"
    if (timed and args.budgets) or (not timed and args.seconds):
        raise ValueError("Usare --seconds con time e --budgets con gli altri stage")
    if (suite == "bbob" and args.functions) or (suite == "native" and (args.bbob_functions or args.instances)):
        raise ValueError("Usare le opzioni per la suite scelta")
    if not 2 <= args.workers <= 4096:
        raise ValueError("--workers deve essere compreso fra 2 e 4096")
    if not 16 <= args.rk4_steps <= 100_000 or args.rk4_steps % 4:
        raise ValueError("--rk4-steps deve essere multiplo di quattro fra 16 e 100000")

    if timed:
        defaults = (0.15,) if profile == "smoke" else ((5.0, 15.0) if profile == "pilot" else (15.0, 60.0))
        budgets = args.seconds or defaults
        names = args.functions or ("rastrigin", "inverse_diffusion")
    else:
        if profile == "smoke":
            defaults = (200, 2000) if stage in {"screen", "ablation"} else (2000,)
        else:
            defaults = (10_000, 50_000) if suite == "bbob" else (50_000, 500_000)
        budgets = args.budgets or defaults
        names = args.functions or (("rastrigin", "rosenbrock", "ackley") if profile != "confirm" else NATIVE_FUNCTIONS)
        if profile == "smoke" and args.functions is None:
            names = ("rastrigin", "rosenbrock")

    default_dimensions = (20,) if suite == "bbob" else (20, 100)
    if suite == "bbob" and profile == "confirm":
        default_dimensions = (20, 40)
    if profile == "smoke":
        default_dimensions = (5,)
    dimensions = args.dimensions or default_dimensions
    if any(d <= 0 for d in dimensions) or (suite == "bbob" and any(
        d not in (2, 3, 5, 10, 20, 40) for d in dimensions
    )):
        raise ValueError("Dimensione non valida per la suite scelta")
    if not budgets or any(not math.isfinite(b) or b <= 0 for b in budgets):
        raise ValueError("Budget non valido")
    if not timed and any(b < POPULATION or b % POPULATION for b in budgets):
        raise ValueError(f"I budget FE devono essere multipli di {POPULATION}")
    if timed and (args.schedule_iterations < 2 or args.time_cap_fe < 2 * POPULATION
                  or args.time_cap_fe % POPULATION):
        raise ValueError("Orizzonte dello schedule o tetto FE non valido")

    if suite == "bbob":
        functions = args.bbob_functions or ((3,) if profile == "smoke" else (
            BBOB_PILOT if profile == "pilot" else BBOB_HOLDOUT
        ))
        instances = args.instances or ((6,) if profile == "smoke" else (
            (6, 7) if profile == "pilot" else (11, 12, 13, 14, 15)
        ))
        if any(f < 1 or f > 24 for f in functions) or any(i < 1 for i in instances):
            raise ValueError("Usare funzioni BBOB da 1 a 24 e istanze positive")
        cases = [(f"f{f:02d}", i) for f, i in itertools.product(sorted(set(functions)), sorted(set(instances)))]
    else:
        cases = [(name, 0) for name in dict.fromkeys(names)]

    n = args.seeds
    if n is None:
        if profile == "smoke" or suite == "bbob":
            n = 1
        elif profile == "confirm":
            n = 20
        else:
            n = 5 if stage in {"screen", "ablation", "time"} else 3
    default_start = {"screen": 830_001, "ablation": 840_001, "fe": 850_001, "time": 860_001}
    start = args.seed_start if args.seed_start is not None else default_start[stage]
    if profile == "confirm" and args.seed_start is None:
        start += 100_000
    if n < 1 or start < 0 or start + n > 2**64:
        raise ValueError("Numero di semi o intervallo non valido")

    selected = source = None
    if stage == "ablation":
        selected, source = _previous(args.development, "screen")
    elif stage in {"fe", "time"}:
        selected, source = _previous(args.development, "ablation")

    protocol = dict(
        schema=1, stage=stage, suite=suite, profile=profile, population=POPULATION,
        budgets=sorted(set(budgets)), dimensions=sorted(set(dimensions)), cases=cases,
        seeds=list(range(start, start + n)), methods=method_plan(stage, args.workers, selected),
        recipe_source=source, rk4_steps=args.rk4_steps,
        schedule_iterations=args.schedule_iterations if timed else None,
        time_cap_fe=args.time_cap_fe if timed else None,
        comparison_scope=("same C++ native objective; Python/NumPy PSO updates"
                          if suite == "native" else "COCO Observer in serial objective batches; FE only"),
        score=dict(gap_floor=GAP_FLOOR, practical_tie_log10=PRACTICAL_TIE,
                   adaptation_gain_log10=ADAPTATION_GAIN, targets=TARGETS),
        coco_version=importlib.metadata.version("coco-experiment") if suite == "bbob" else None,
    )
    return protocol


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=STAGES)
    parser.add_argument("--suite", choices=("native", "bbob"), default="native")
    parser.add_argument("--profile", choices=PROFILES, default="pilot")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--development", type=Path, help="Root della campagna screen/ablation")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seeds", type=int)
    parser.add_argument("--seed-start", type=int)
    parser.add_argument("--dimensions", nargs="+", type=int)
    parser.add_argument("--functions", nargs="+", choices=(*NATIVE_FUNCTIONS, "inverse_diffusion"))
    parser.add_argument("--bbob-functions", nargs="+", type=int)
    parser.add_argument("--instances", nargs="+", type=int)
    parser.add_argument("--budgets", nargs="+", type=int)
    parser.add_argument("--seconds", nargs="+", type=float)
    parser.add_argument("--rk4-steps", type=int, default=512)
    parser.add_argument("--schedule-iterations", type=int, default=1000)
    parser.add_argument("--time-cap-fe", type=int, default=200_000_000)
    args = parser.parse_args()
    root = (args.output or Path(f"results/study3_{args.profile}")).resolve()
    default_development = Path("results/study3_pilot") if args.profile == "confirm" else root
    args.development = (args.development or default_development).resolve()
    folder = _stage_dir(root, args.stage, args.suite)

    try:
        protocol = _design(args)
    except (ValueError, importlib.metadata.PackageNotFoundError) as error:
        parser.error(str(error))
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / ".study3.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error(f"Campagna già in esecuzione: {folder}")
        run_campaign(folder, protocol)


def run_campaign(folder, protocol):
    prepare(folder, protocol)
    freeze(folder / "study3_sources.json", source_manifest())
    jobs = shuffled(planned_jobs(protocol), 310_871)
    print(f"Studio III | {protocol['stage']} | {protocol['suite']} | {len(jobs)} run | {folder}", flush=True)
    records = []
    for i, key in enumerate(jobs, 1):
        result = cached_run(folder, key, lambda: run_job(folder, protocol, key))
        records.append((key, result))
        if i == 1 or i % 10 == 0 or i == len(jobs):
            print(f"{i}/{len(jobs)} · {key['method']} · {key['function']} D={key['dimension']} "
                  f"· {result['fe']:,} FE · {result['elapsed']:.3f} s", flush=True)
    analyze(folder, protocol, records)
    selected = folder / "selected.json"
    freeze(folder / "complete.json", dict(
        runs=len(records), methods=len(protocol["methods"]), protocol_sha256=fingerprint(protocol),
        decision_sha256=fingerprint(read(selected)) if selected.exists() else None,
    ))
    print(f"Completato: {folder}", flush=True)


if __name__ == "__main__":
    main()
