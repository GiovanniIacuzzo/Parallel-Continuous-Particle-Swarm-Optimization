#!/usr/bin/env python3
"""Screen CPSO update batches and compare one frozen candidate on each suite."""
from __future__ import annotations

import argparse
import fcntl
from importlib.metadata import PackageNotFoundError, version
import itertools
import math
import os
from pathlib import Path
import subprocess
import sys

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                 "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(variable, "1")

from publication_tools.common import ROOT, cached_run, fingerprint, freeze, prepare, read, sha, shuffled
from publication_tools.study3_competition_analysis import summarize
from publication_tools.study3_execution import BASELINES, NATIVE_FUNCTIONS, run_job, source_manifest
from publication_tools.study3_transforms import VARIANTS

CANDIDATES = {"cpso_100_100": (100, 100), "cpso_50_50": (50, 50), "cpso_100_50": (100, 50)}
RECIPE = dict(topology="random_k8", coordinate=True, dt=1.0, adaptation="fixed")
REFINEMENTS = {
    "cpso_100_50": RECIPE,
    "cpso_local": RECIPE | {"local_search": True},
    "cpso_restart": RECIPE | {"partial_restart": True},
    "cpso_local_restart": RECIPE | {"local_search": True, "partial_restart": True},
}
OPPONENTS = ("cso", "slpso", "spso2011")
PAPER_SUITES = ("inverse", "bbob", "native", "cec")
PAPER_RECIPE = REFINEMENTS["cpso_local_restart"] | {"quadratic_local": True}
STAGES = ("screen", "refine", *PAPER_SUITES, "paper")
DEFAULT_SEED = dict(screen=1_100_001, native=1_200_001, bbob=1_300_001,
                    cec=1_400_001, inverse=1_500_001, refine=1_700_001)


def _selection(folder, profile, refined=False):
    protocol, decision, complete = (read(folder / name) for name in
                                    ("protocol.json", "selected.json", "complete.json"))
    recipes = REFINEMENTS if refined else dict.fromkeys(CANDIDATES, RECIPE)
    if (protocol.get("stage") != ("refine" if refined else "screen") or protocol.get("profile") != profile
            or complete.get("protocol_sha256") != fingerprint(protocol)
            or complete.get("decision_sha256") != fingerprint(decision)
            or decision.get("candidate") not in recipes):
        raise ValueError(f"Screening incompleto o incompatibile in {folder}")
    candidate = decision["candidate"]
    chosen = protocol["methods"][candidate]
    if ((chosen["population"], chosen["batch"]) != ((100, 50) if refined else CANDIDATES[candidate])
            or chosen["recipe"] != recipes[candidate]):
        raise ValueError("La configurazione selezionata non corrisponde al protocollo")
    return candidate, decision, protocol


def _cases(args, smoke):
    stage = args.stage
    if stage == "refine":
        return [dict(function="inverse_diffusion", dimension=10 if smoke else 100, instance=0),
                *(dict(function=f, dimension=10 if smoke else 40, instance=501, variant="rotated")
                  for f in ("rastrigin", "rosenbrock", "ackley"))]
    if stage == "cec":
        ids = args.function_ids or ((1,) if smoke else range(1, 16))
        if any(not 1 <= f <= 15 for f in ids):
            raise ValueError("CEC2013 LSGO usa le funzioni da 1 a 15")
        return [dict(function=f"f{f:02d}", dimension=905 if f in (13, 14) else 1000, instance=0)
                for f in sorted(set(ids))]
    dimensions = sorted(set(args.dimensions or ((5,) if smoke else
                            (40,) if stage == "bbob" else (20, 100))))
    if stage == "bbob":
        ids = args.function_ids or ((3,) if smoke else range(1, 25))
        instances = args.instances or ((31,) if smoke and args.paper else
                                        (31, 32, 33, 34, 35) if args.paper else
                                        (26,) if smoke and args.refinement else (21,) if smoke else
                                        (26, 27, 28, 29, 30) if args.refinement else (21, 22, 23, 24, 25))
        if (any(not 1 <= f <= 24 for f in ids)
                or any(d not in (2, 3, 5, 10, 20, 40) for d in dimensions)):
            raise ValueError("Funzione o dimensione BBOB non valida")
        return [dict(function=f"f{f:02d}", dimension=d, instance=i)
                for f, d, i in itertools.product(sorted(set(ids)), dimensions, sorted(set(instances)))]
    if stage == "inverse":
        return [dict(function="inverse_diffusion", dimension=d, instance=0) for d in dimensions]
    names = args.functions or (("rastrigin", "rosenbrock") if smoke else NATIVE_FUNCTIONS)
    instances = args.instances or ((101,) if stage == "screen" and smoke else
                                   (401,) if smoke and args.paper else
                                   (401, 402, 403) if args.paper else
                                   (301,) if smoke and args.refinement else
                                   (301, 302, 303) if args.refinement else
                                   (201,) if smoke else (101, 102, 103) if stage == "screen"
                                   else (201, 202, 203))
    return [dict(function=f, dimension=d, instance=i, variant=v)
            for f, d, v in itertools.product(dict.fromkeys(names), dimensions, VARIANTS)
            for i in ((0,) if v == "original" else sorted(set(instances)))]


def design(args, development):
    stage = args.stage
    smoke = args.profile == "smoke"
    if stage in {"screen", "refine"} and args.refinement:
        raise ValueError("--refinement si usa solo per le campagne di conferma")
    if stage == "refine" and (args.dimensions or args.instances):
        raise ValueError("I casi del pilot refine sono predefiniti")
    if args.workers < 2 or args.workers > 4096:
        raise ValueError("--workers deve essere compreso fra 2 e 4096")
    if args.dimensions and (stage == "cec" or min(args.dimensions) < 2):
        raise ValueError("Le dimensioni CEC sono fissate. Nelle altre suite occorre D >= 2.")
    if args.instances and (stage in {"cec", "inverse"} or min(args.instances) < 1):
        raise ValueError("--instances richiede interi positivi e si usa con screen, native o bbob")
    if ((args.functions and stage not in {"screen", "native"})
            or (args.function_ids and stage not in {"bbob", "cec"})
            or (args.seconds and stage != "inverse") or (args.budgets and stage == "inverse")):
        raise ValueError("Usare le opzioni di funzione e budget previste per lo stage scelto")
    if stage == "inverse":
        budgets = args.seconds or ((0.25,) if smoke else (15.0,) if args.paper else (15.0, 60.0))
        if any(not math.isfinite(value) or value <= 0.0 for value in budgets):
            raise ValueError("Le durate devono essere finite e positive")
    else:
        defaults = {"screen": (10_000,), "native": (50_000,), "bbob": (50_000,),
                    "cec": (120_000,), "refine": (5_000, 10_000, 20_000)}
        smoke_budgets = (1_500, 3_000) if stage == "refine" else (300,)
        budgets = args.budgets or (smoke_budgets if smoke else defaults[stage])
        if any(b < 100 or b % 100 for b in budgets):
            raise ValueError("I budget devono essere multipli di 100 FE")

    count = args.seeds if args.seeds is not None else (1 if smoke or stage == "bbob" else
                                                       20 if stage == "inverse" else 5)
    offset = 4_000_000 if args.paper else 2_000_000 if args.refinement else 0
    start = args.seed_start if args.seed_start is not None else DEFAULT_SEED[stage] + offset
    if count < 1 or start < 0 or start + count > 2**64:
        raise ValueError("Intervallo di semi non valido")
    seeds = list(range(start, start + count))
    cases = _cases(args, smoke)
    source = None
    if args.paper:
        methods = {"cpso_selected": dict(recipe=PAPER_RECIPE, population=100, batch=50,
                                         workers=args.workers)}
        source = {"preset": "cpso_local_restart_quadratic"}
    elif stage == "refine":
        methods = {name: dict(recipe=recipe, population=100, batch=50, workers=args.workers)
                   for name, recipe in REFINEMENTS.items()}
    elif stage == "screen":
        methods = {name: dict(recipe=RECIPE, population=population, batch=batch, workers=args.workers)
                   for name, (population, batch) in CANDIDATES.items()}
    else:
        candidate, decision, screen = _selection(args.refinement or development, args.profile,
                                                refined=args.refinement is not None)
        source = dict(decision=decision, protocol_sha256=fingerprint(screen))
        if stage == "native" and (set(seeds) & set(screen["seeds"]) or
                {case["instance"] for case in cases if case["instance"]} &
                {case["instance"] for case in screen["cases"] if case["instance"]}):
            raise ValueError("Per la conferma nativa usare semi e istanze trasformate nuovi")
        methods = {"cpso_selected": screen["methods"][candidate] | {"workers": args.workers}}
        if args.refinement and candidate != "cpso_100_50":
            methods["cpso_control_100_50"] = dict(recipe=RECIPE, population=100, batch=50,
                                                  workers=args.workers)
    if stage != "refine":
        opponents = BASELINES if args.paper else OPPONENTS
        methods.update({name: dict(recipe=None, population=100, workers=args.workers) for name in opponents})
    if stage == "inverse" and not args.paper:
        methods["cpso_one_worker"] = methods["cpso_selected"] | {"workers": 1}
    package = {"bbob": "coco-experiment", "cec": "cec2013lsgo"}.get(stage)
    return dict(stage="time" if stage == "inverse" else stage, profile=args.profile,
                suite=stage if stage in {"bbob", "cec"} else "native",
                cases=cases, budgets=sorted(set(budgets)), seeds=seeds, methods=methods,
                recipe_source=source, package_version=version(package) if package else None,
                rk4_steps=16 if smoke else 512, schedule_iterations=1000,
                time_cap_fe=200_000_000, schema=1)


def _jobs(protocol):
    budgets = protocol["budgets"]
    if protocol["suite"] == "cec" or protocol["stage"] == "refine":
        budgets = [max(budgets)]
    return (case | dict(seed=seed, budget=budget, method=method)
            for case, seed, budget, method in itertools.product(
                protocol["cases"], protocol["seeds"], budgets, protocol["methods"]
            ))


def _checkpoints(protocol, key, result):
    if protocol["suite"] == "cec" or protocol["stage"] == "refine":
        for budget in protocol["budgets"][:-1]:
            trace = [row for row in result["trace"] if row[0] <= budget]
            # Intervention counters are totals for the full run only.
            snapshot = {k: v for k, v in result.items() if k not in
                        {"local_trials", "local_improvements", "restart_trials", "restart_improvements"}}
            yield (key | {"budget": budget},
                   snapshot | dict(fe=budget, value=trace[-1][2], trace=trace, elapsed=None))
    yield key, result


def run(folder, protocol):
    prepare(folder, protocol)
    paths = [ROOT / "study3_competition.py", *sorted((ROOT / "python/cpso_serial").rglob("*.py")),
             ROOT / "experiments/05_external_benchmarks/probe_cec905.py"]
    freeze(folder / "sources.json", source_manifest() | {
        str(path.relative_to(ROOT)): sha(path) for path in paths
    })
    if protocol["suite"] == "cec" and not (folder / "cec_905_check.json").exists():
        probes = {}
        for case in protocol["cases"]:
            if case["dimension"] == 905:
                function = case["function"]
                cmd = [sys.executable, str(paths[-1]), str(int(function[1:]))]
                probes[function] = subprocess.check_output(cmd, text=True).strip()
        freeze(folder / "cec_905_check.json", probes)
    jobs = shuffled(_jobs(protocol), 728_031)
    print(f"{protocol['stage']} · {len(jobs)} run · {folder}", flush=True)
    records = []
    for index, key in enumerate(jobs, 1):
        result = cached_run(folder, key, lambda: run_job(folder, protocol, key))
        records.extend(_checkpoints(protocol, key, result))
        if index == 1 or index % 20 == 0 or index == len(jobs):
            print(f"{index}/{len(jobs)} · {key['method']} · {key['function']} "
                  f"D={key['dimension']} · {result['fe']:,} FE", flush=True)
    decision = summarize(folder, protocol, records)
    if decision is not None:
        freeze(folder / "selected.json", decision)
    freeze(folder / "complete.json", dict(runs=len(jobs), observations=len(records),
           protocol_sha256=fingerprint(protocol), decision_sha256=fingerprint(decision) if decision else None))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=STAGES)
    parser.add_argument("--profile", choices=("main", "smoke"), default="main")
    parser.add_argument("--development", type=Path, help="Directory dello screening già completato")
    parser.add_argument("--refinement", type=Path, help="Directory del pilot refine completato")
    parser.add_argument("--paper-suite", choices=PAPER_SUITES, nargs="+",
                        help="Esegui solo queste suite del protocollo paper")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--budgets", nargs="+", type=int)
    parser.add_argument("--seconds", nargs="+", type=float)
    parser.add_argument("--dimensions", nargs="+", type=int)
    parser.add_argument("--functions", nargs="+", choices=NATIVE_FUNCTIONS)
    parser.add_argument("--function-ids", nargs="+", type=int)
    parser.add_argument("--instances", nargs="+", type=int)
    parser.add_argument("--seeds", type=int)
    parser.add_argument("--seed-start", type=int)
    args = parser.parse_args()
    args.paper = args.stage == "paper"
    if args.paper_suite and not args.paper:
        parser.error("--paper-suite richiede lo stage paper")
    if args.paper and any((args.refinement, args.development, args.budgets, args.seconds,
                           args.dimensions, args.functions, args.function_ids, args.instances,
                           args.seeds, args.seed_start is not None)):
        parser.error("Lo stage paper usa un protocollo fisso; --paper-suite limita le suite")
    root = Path(f"results/study3_{'refinement' if args.refinement or args.stage == 'refine' else 'competition'}_{args.profile}")
    development = (args.development or root / "screen").resolve()
    try:
        if args.paper:
            paper_root = (args.output or Path(f"results/study3_paper_{args.profile}")).resolve()
            selected_suites = args.paper_suite or PAPER_SUITES
            campaigns = [(paper_root / stage, design(argparse.Namespace(**(vars(args) | {"stage": stage})),
                                                    development))
                         for stage in PAPER_SUITES if stage in selected_suites]
        else:
            campaigns = [((args.output or root / args.stage).resolve(), design(args, development))]
        for folder, protocol in campaigns:
            folder.mkdir(parents=True, exist_ok=True)
            with (folder / ".competition.lock").open("a") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                run(folder, protocol)
    except (ValueError, FileNotFoundError, BlockingIOError, PackageNotFoundError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
