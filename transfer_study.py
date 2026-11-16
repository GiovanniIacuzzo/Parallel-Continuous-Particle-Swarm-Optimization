"""Paired step-size ablation, independent BBOB validation and CEC transfer."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

import pandas as pd
from scipy.stats import binomtest

from study import campaign_lock, pin

ROOT = Path(__file__).resolve().parent
EXT = ROOT / "experiments/05_external_benchmarks"
sys.path[:0] = [str(ROOT / "python"), str(EXT)]
from common import METHODS, checkpoint_values, collect_external_provenance, deterministic_problem_seed, run_method
from cpso_serial.native import _load_library

PRIMARY = "cpso_random_k8"
GRID = [1., 0.35, 3.6, 30.]
SEED = 20261116


def problem_stream(stage, design):
    if stage == "cec":
        from cpso_serial.benchmarks.cec2013_lsgo import CEC2013LSGOAdapter
        adapter = CEC2013LSGOAdapter()
        for f in design["functions"]:
            problem = adapter.get_problem(f)
            for i in design["replicates"]:
                yield f, i, problem.dimension, problem.objective, problem.lower_bounds, problem.upper_bounds
    else:
        import cocoex
        for d in design["dimensions"]:
            for i in design["replicates"]:
                ids = ",".join(map(str, design["functions"]))
                suite = cocoex.Suite("bbob", f"instances: {i}", f"dimensions: {d} function_indices: {ids}")
                for p in suite:
                    yield p.id_function, i, d, p, p.lower_bounds, p.upper_bounds


def run_one(objective, lo, hi, method, dt, budget, seed, milestones):
    count = 0
    def measured(x):
        nonlocal count
        value = float(objective(x))
        if not math.isfinite(value):
            raise ValueError("Nonfinite objective")
        count += 1
        return value
    result = run_method(method, measured, lo, hi, population_size=64, max_evaluations=budget,
                        seed=seed, cpso_dt=dt)
    if count != budget or result.function_evaluations != budget:
        raise ValueError("Evaluation count mismatch")
    values = checkpoint_values(result, milestones)
    if any(not math.isfinite(x) for x in values.values()) or list(values.values()) != sorted(values.values(), reverse=True):
        raise ValueError("Invalid checkpoint trajectory")
    return values


def campaign(stage, smoke):
    root = ROOT / "results/transfer_v2" / ("smoke" if smoke else "study")
    folder = root / stage
    calibrate = stage == "calibrate"
    cec = stage == "cec"
    configs = {f"dt{dt:g}": (PRIMARY, dt) for dt in GRID} if calibrate else {
        m: (m, 1.) for m in [PRIMARY, *METHODS]}
    if not calibrate:
        selection = json.loads((root/"calibrate/selection.json").read_text())
        chosen = selection["dt"]
        if chosen != 1.:
            configs["cpso_random_k8_selected"] = (PRIMARY, chosen)
        if stage == "validate" and any(v != chosen for v in selection["by_dimension"].values()):
            configs["cpso_random_k8_by_dimension"] = (PRIMARY, selection["by_dimension"])
    funcs = ([1] if smoke else list(range(1, 16 if cec else 25)))
    dims = ([] if cec else ([20] if smoke else [5, 20]))
    reps = ([1] if smoke else list(range(1, 6))) if cec else (
        ([3] if smoke else [3, 4, 5]) if calibrate else ([11] if smoke else list(range(11, 16))))
    design = dict(stage=stage, functions=funcs, dimensions=dims, replicates=reps, methods=configs,
                  population=64, seed=SEED, budgets=([64,128] if smoke else [512,2048,4096]) if cec
                  else ([64,128] if smoke else [64,192,384]), budget_unit="FE" if cec else "FE/D")
    # Convert tuples to JSON lists for reproducible equality on resume.
    file = pin(folder/"design.json", json.loads(json.dumps(design)))
    prov = collect_external_provenance(file)
    prov["launcher"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    prov["native_binary"] = hashlib.sha256(Path(_load_library()._name).read_bytes()).hexdigest()
    pin(folder/"provenance.json", prov)
    if cec and not smoke:
        for f in (13, 14):
            subprocess.run([sys.executable, str(EXT/"probe_cec905.py"), str(f)], check=True)
    rows = []
    for f, i, d, objective, lo, hi in problem_stream(stage, design):
        seed = deterministic_problem_seed(f"{stage}-{f}-{i}-{d}", SEED)
        milestones = [b*(1 if cec else d) for b in design["budgets"]]
        for name, (method, rule) in configs.items():
            dt = rule[str(d)] if isinstance(rule, dict) else rule
            path = folder/"runs"/f"f{f}_d{d}_i{i}_{name}.json"
            if not path.exists():
                values = run_one(objective, lo, hi, method, dt, milestones[-1], seed, milestones)
                pin(path, dict(function=f, replicate=i, dimension=d, method=name, seed=seed, checkpoints=values))
            run = json.loads(path.read_text())
            for fe in milestones:
                rows.append(dict(function=f, replicate=i, dimension=d, method=name, fe=fe,
                                 final=fe==milestones[-1], value=run["checkpoints"][str(fe)]))
        print(f"Completed F{f} D{d} replicate {i}", flush=True)
    a = pd.DataFrame(rows)
    expected = len(funcs)*max(1,len(dims))*len(reps)*len(configs)*len(design["budgets"])
    keys = ["function","dimension","replicate","fe"]
    if len(a)!=expected or a.duplicated(keys+["method"]).any():
        raise ValueError("Incomplete or duplicate design")
    def identity(row):
        method, rule = configs[row.method]
        dt = rule[str(row.dimension)] if isinstance(rule, dict) else rule
        return f"{method}@{dt}"

    # A selected rule can coincide with the control in one dimension.
    # Keep both labels for paired tests, and rank the unique configuration once.
    a["configuration"] = [identity(row) for row in a.itertuples()]
    groups = keys + ["configuration"]
    if a.groupby(groups).value.nunique().max() != 1:
        raise ValueError("Identical seeded configurations disagree")
    unique = a.drop_duplicates(groups).copy()
    unique["rank"] = unique.groupby(keys).value.rank()
    a = a.merge(unique[groups + ["rank"]], on=groups, validate="many_to_one")
    a.to_csv(folder/"checkpoints.csv", index=False)
    final = a[a.final]
    rank = final.groupby(["dimension","method"])["rank"].mean()
    rank.to_csv(folder/"mean_ranks.csv")
    if calibrate:
        score = rank.groupby("method").mean()
        best = min(configs, key=lambda name: (score[name], list(configs).index(name)))
        by_dim = {}
        for d in dims:
            scores = rank.loc[d]
            winner = min(configs, key=lambda name: (scores[name], list(configs).index(name)))
            by_dim[str(d)] = configs[winner][1]
        pin(folder/"selection.json", {"dt": configs[best][1], "by_dimension": by_dim, "mean_ranks": score.to_dict()})
    else:
        primary = "cpso_random_k8_selected" if chosen != 1. else PRIMARY
        if "cpso_random_k8_by_dimension" in configs:
            primary = "cpso_random_k8_by_dimension"
        contrasts = []
        for dim, group in ([("all",final)] if cec else final.groupby("dimension")):
            paired = group.pivot(index=["function","replicate"], columns="method", values="value")
            for name in configs:
                if name == primary:
                    continue
                effect = (paired[primary]-paired[name]).groupby(level=0).median()
                w, l = int((effect<0).sum()), int((effect>0).sum())
                contrasts.append(dict(dimension=dim, primary=primary, opponent=name, wins=w,
                                      losses=l, ties=len(effect)-w-l, p=binomtest(w,w+l).pvalue if w+l else 1.))
        frame = pd.DataFrame(contrasts).sort_values("p")
        frame["p_holm"] = (frame.p * list(range(len(frame),0,-1))).cummax().clip(upper=1)
        frame.to_csv(folder/"contrasts.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["calibrate","validate","cec"])
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    with campaign_lock(ROOT/"results"):
        campaign(args.stage, args.smoke)
