"""Execution and reproducibility for the synchronous Study III campaigns."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
import itertools
import math
import time

import numpy as np

from .coco_logging import archive_interrupted_observer, observer_optimum, short_coco_observer
from .common import BOUNDS, ROOT, fingerprint, native_config, native_version, sha
from cpso_serial.native import native_batch_evaluator, run_synchronous_study
from cpso_serial.pso_baselines import PSOConfig, available_pso_methods, minimize_pso

POPULATION = 100
BASELINES = available_pso_methods()
TIMED_BASELINES = ("spso2011", "cso", "clpso", "slpso")
NATIVE_FUNCTIONS = ("rastrigin", "rosenbrock", "ackley", "sphere", "griewank", "salomon")
BBOB_PILOT = (3, 6, 10, 12, 16, 23)
BBOB_HOLDOUT = tuple(f for f in range(1, 25) if f not in (1, 8, 15, 21, *BBOB_PILOT))
ADAPTATIONS = {
    "fixed": (False, False),
    "step": (True, False),
    "recovery": (False, True),
    "step_recovery": (True, True),
}
REFERENCE = dict(topology="global", coordinate=False, dt=0.38125, adaptation="fixed")


def screening_recipes():
    for coordinate, topology, dt in itertools.product(
        (False, True), ("global", "random_k8"), (0.38125, 1.0)
    ):
        randomness = "coordinate" if coordinate else "scalar"
        yield f"{randomness}_{topology}_{dt:g}", dict(
            topology=topology, coordinate=coordinate, dt=dt, adaptation="fixed"
        )


def method_plan(stage, workers, selected):
    if stage == "screen":
        return {name: dict(recipe=recipe, workers=workers) for name, recipe in screening_recipes()}
    if stage == "ablation":
        return {name: dict(recipe=selected | {"adaptation": name}, workers=workers)
                for name in ADAPTATIONS}

    methods = {"cpso_selected": dict(recipe=selected, workers=workers)}
    if stage == "time":
        methods["cpso_one_worker"] = dict(recipe=selected, workers=1)
        opponents = TIMED_BASELINES
    else:
        if selected != REFERENCE:
            methods["cpso_fixed_reference"] = dict(recipe=REFERENCE, workers=workers)
        if selected["adaptation"] != "fixed" and (selected | {"adaptation": "fixed"}) != REFERENCE:
            methods["cpso_matched_fixed"] = dict(
                recipe=selected | {"adaptation": "fixed"}, workers=workers
            )
        methods["pso_matched_discrete"] = dict(
            recipe=selected | {"adaptation": "fixed"}, workers=workers, discrete=True
        )
        opponents = BASELINES
    for name in opponents:
        methods[name] = dict(recipe=None, workers=workers)
    return methods


def planned_jobs(protocol):
    for (function, instance), dimension, seed, budget, method in itertools.product(
        protocol["cases"], protocol["dimensions"], protocol["seeds"],
        protocol["budgets"], protocol["methods"],
    ):
        yield dict(function=function, instance=instance, dimension=dimension, seed=seed,
                   budget=budget, method=method)


class DeadlineReached(Exception):
    """A new synchronous batch cannot be published within the time budget."""


class BatchTrace:
    def __init__(self, started, deadline_seconds=0.0):
        self.started = started
        self.deadline_seconds = deadline_seconds
        self.fe = 0
        self.discarded = 0
        self.best = math.inf
        self.rows = []

    def evaluate(self, evaluator, points):
        if self.deadline_seconds and time.perf_counter() - self.started >= self.deadline_seconds:
            raise DeadlineReached
        values = evaluator(points)
        elapsed = time.perf_counter() - self.started
        if self.deadline_seconds and elapsed >= self.deadline_seconds:
            self.discarded = len(points)
            raise DeadlineReached
        self.fe += len(points)
        self.best = min(self.best, float(np.min(values)))
        self.rows.append([self.fe, elapsed, self.best])
        return values


def compress_trace(rows):
    """Keep the initial point, all improvements, and the last completed batch."""
    if not rows:
        return []
    kept = [rows[0]]
    for fe, seconds, value in rows[1:]:
        if value < kept[-1][2]:
            kept.append([fe, seconds, value])
    if rows[-1][0] != kept[-1][0]:
        kept.append(rows[-1])
    return kept


def _unavailable_scalar(_):
    raise AssertionError("Study III native baseline must use batch evaluation")


def _baseline(key, settings, bounds, budget, work, schedule, seconds, objective):
    config = PSOConfig(*bounds, population_size=POPULATION, max_evaluations=budget,
                       seed=key["seed"], schedule_iterations=schedule or None)
    trace = BatchTrace(time.perf_counter(), seconds)
    if objective is not None:
        result = minimize_pso(objective, config, method=key["method"])
        return dict(
            rows=[[int(fe), 0.0, float(best)] for fe, best in
                  zip(result.history_evaluations, result.best_history)],
            elapsed=time.perf_counter() - trace.started, fe=result.function_evaluations,
            value=result.best_value, discarded_evaluations=0,
            implementation="numpy_coco_observer",
        )

    with native_batch_evaluator(key["function"], settings["workers"], work) as evaluate:
        try:
            result = minimize_pso(
                _unavailable_scalar, config, method=key["method"],
                batch_objective=lambda points: trace.evaluate(evaluate, points),
            )
        except DeadlineReached:
            result = None
        if result is not None and (result.function_evaluations != trace.fe
                                   or result.best_value != trace.best):
            raise RuntimeError("PSO state and shared evaluator disagree")
    return dict(rows=trace.rows, elapsed=time.perf_counter() - trace.started,
                fe=trace.fe, value=trace.best if trace.fe else None,
                discarded_evaluations=trace.discarded,
                implementation="numpy_update_native_evaluator")


def _cpso(key, settings, bounds, budget, work, schedule, seconds, objective):
    recipe = settings["recipe"]
    config = native_config(
        key["function"] if objective is None else "sphere", key["dimension"], key["seed"],
        budget // POPULATION - 1, recipe["dt"], recipe["topology"], POPULATION,
    )
    config = replace(config, lower_bounds=bounds[0], upper_bounds=bounds[1])
    adapt_step, recovery = ADAPTATIONS[recipe["adaptation"]]
    result, details = run_synchronous_study(
        config, key["function"], workers=settings["workers"], graph_seed=key["seed"],
        coordinate=recipe["coordinate"], discrete=settings.get("discrete", False),
        adapt_step=adapt_step, recovery=recovery, seconds=seconds,
        schedule_iterations=schedule, work_iterations=work, objective=objective,
    )
    timestamps = details.pop("batch_seconds")
    if objective is not None:
        timestamps = [0.0] * len(timestamps)
    rows = [list(row) for row in zip(
        result.history_evaluations, timestamps, result.best_history, strict=True
    )]
    return dict(rows=rows, elapsed=result.elapsed_seconds, fe=result.function_evaluations,
                value=result.best_value if result.function_evaluations else None,
                implementation="cpp_coco_observer" if objective is not None else "cpp_native",
                **details)


@contextmanager
def _problem(folder, key):
    if key["instance"] == 0:
        function = key["function"]
        lo, hi = (-2.0, 2.0) if function == "inverse_diffusion" else BOUNDS[function]
        yield ((lo,) * key["dimension"], (hi,) * key["dimension"]), None, None
        return

    import cocoex

    function_id = int(key["function"][1:])
    instance = key["instance"]
    suite = cocoex.Suite("bbob", f"instances: {instance}",
                        f'dimensions: {key["dimension"]} function_indices: {function_id}')
    problem = suite[0]
    log = folder / "observer" / fingerprint(key)
    archive_interrupted_observer(log)
    bounds = (tuple(problem.lower_bounds), tuple(problem.upper_bounds))
    with short_coco_observer(problem, log, key["method"]):
        yield bounds, problem, log


def run_job(folder, protocol, key):
    settings = protocol["methods"][key["method"]]
    seconds = key["budget"] if protocol["stage"] == "time" else 0.0
    budget = protocol["time_cap_fe"] if seconds else key["budget"]
    schedule = protocol["schedule_iterations"] if seconds else 0
    work = protocol["rk4_steps"] if key["function"] == "inverse_diffusion" else 0
    execute = _baseline if settings["recipe"] is None else _cpso
    with _problem(folder, key) as (bounds, problem, log):
        objective = None if problem is None else lambda point: float(problem(np.asarray(point)))
        result = execute(key, settings, bounds, budget, work, schedule, seconds, objective)
        if problem is not None and int(problem.evaluations) != result["fe"]:
            raise RuntimeError(f"COCO and solver FE counts disagree for {key}")
    optimum = 0.0 if log is None else observer_optimum(log)
    if seconds and result["fe"] >= budget:
        raise RuntimeError(f"FE ceiling reached before a valid time-budget result: {key}. "
                           "Increase --time-cap-fe in a fresh output directory.")
    if not seconds and result["fe"] != budget:
        raise RuntimeError(f"Expected exactly {budget} FE: {key}, got {result['fe']}")
    rows = result.pop("rows")
    if bool(rows) != bool(result["fe"]):
        raise RuntimeError(f"FE and trace disagree: {key}")
    if rows and (rows[-1][0] != result["fe"] or rows[-1][2] != result["value"]
                 or rows[-1][1] > result["elapsed"]):
        raise RuntimeError(f"Final state and trace disagree: {key}")
    actual_seconds = result["elapsed"]
    if seconds and (actual_seconds < seconds or (rows and rows[-1][1] >= seconds)):
        raise RuntimeError(f"Invalid deadline accounting: {key}")
    if rows:
        trace = np.asarray(rows, dtype=float)
        if (not np.all(np.isfinite(trace)) or np.any(np.diff(trace[:, 0]) <= 0)
                or np.any(np.diff(trace[:, 1]) < 0) or np.any(np.diff(trace[:, 2]) > 0)
                or result["value"] < optimum - 1e-7):
            raise RuntimeError(f"Invalid best-so-far trace: {key}")
    return result | dict(trace=compress_trace(rows), optimum=optimum)


def source_manifest():
    paths = [ROOT / "study3.py", *sorted((ROOT / "cpp/src").glob("*.cpp")),
             *sorted((ROOT / "cpp/include/cpso").glob("*.h*")),
             ROOT / "cpp/src/optimizer_internal.hpp"]
    return {str(path.relative_to(ROOT)): sha(path) for path in paths} | {
        "native_api": native_version(),
    }
