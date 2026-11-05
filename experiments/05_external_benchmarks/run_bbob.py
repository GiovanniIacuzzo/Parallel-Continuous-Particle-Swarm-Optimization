from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "python") not in sys.path:
    sys.path.insert(0, str(ROOT / "python"))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from cpso_serial.benchmarks.coco_bbob import (  # noqa: E402
    load_coco_experiment,
    make_bbob_observer,
    make_bbob_suite,
    wrap_coco_problem,
)
from common import (  # noqa: E402
    METHODS,
    append_jsonl_durable,
    build_run_record,
    deterministic_problem_seed,
    estimate_bbob_campaign,
    load_existing_run_keys,
    record_with_key,
    run_method,
    write_campaign_manifest,
)


def execute(config_path: Path, output: Path, *, dry_run: bool = False) -> int:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    suite_name = str(config.get("suite", "bbob"))
    if suite_name not in {"bbob", "bbob-largescale"}:
        raise ValueError("This driver supports COCO 'bbob' and 'bbob-largescale' suites")
    methods = config.get("methods", list(METHODS))
    unknown = sorted(set(methods) - set(METHODS))
    if unknown:
        raise ValueError(f"Unknown methods in campaign: {unknown}")
    population = int(config.get("population_size", 40))
    budget_multiplier = int(config.get("budget_multiplier", 10_000))
    if population < 2 or budget_multiplier < 1:
        raise ValueError("population_size must be >=2 and budget_multiplier must be positive")

    counts = estimate_bbob_campaign(
        suite_name,
        config.get("instances", "instances: 1-15"),
        config.get("suite_options", ""),
        budget_multiplier=budget_multiplier,
        population_size=population,
        method_count=len(methods),
    )
    plan = {
        "campaign_id": config.get("campaign_id", config_path.stem),
        "suite": suite_name,
        "suite_instances": config.get("instances", "instances: 1-15"),
        "suite_options": config.get("suite_options", ""),
        "population_size": population,
        "budget_multiplier": budget_multiplier,
        "methods": methods,
        "budget_rule": "max_evaluations = budget_multiplier * dimension; exact and divisible by population",
        "seed_rule": "same deterministic seed derived from COCO problem ID for every method",
        **counts,
        "output_directory": str(output),
    }
    if dry_run:
        print(json.dumps(plan, indent=2))
        return 0

    cocoex = load_coco_experiment()
    output.mkdir(parents=True, exist_ok=True)
    write_campaign_manifest(output / "campaign_manifest.json", plan, config_path)
    for method in methods:
        method_directory = output / "coco_data" / method
        observer = make_bbob_observer(
            cocoex,
            output_directory=str(method_directory.resolve()),
            algorithm_name=method,
            suite_name=suite_name,
        )
        suite = make_bbob_suite(
            cocoex,
            suite_name=suite_name,
            instances=plan["suite_instances"],
            options=plan["suite_options"],
        )
        result_path = output / "records" / f"{method}.jsonl"
        completed = load_existing_run_keys(result_path)
        for raw_problem in suite:
            problem = wrap_coco_problem(raw_problem)
            raw_problem.observe_with(observer)
            seed = deterministic_problem_seed(problem.problem_id, int(config.get("seed", 20261105)))
            max_evaluations = budget_multiplier * problem.dimension
            if max_evaluations % population != 0:
                raise ValueError(
                    f"FE budget {max_evaluations} for {problem.problem_id} is not divisible by population {population}"
                )
            record_key = "|".join((suite_name, problem.problem_id, method, str(seed)))
            if record_key in completed:
                continue
            print(f"{method}: {problem.problem_id} (D={problem.dimension}, FE={max_evaluations})")
            result = run_method(
                method,
                problem.objective,
                problem.lower_bounds,
                problem.upper_bounds,
                population_size=population,
                max_evaluations=max_evaluations,
                seed=seed,
            )
            coco_evaluations = int(getattr(raw_problem, "evaluations", result.function_evaluations))
            if coco_evaluations != result.function_evaluations:
                raise RuntimeError(
                    f"COCO FE counter mismatch on {problem.problem_id}: "
                    f"COCO={coco_evaluations}, optimizer={result.function_evaluations}"
                )
            record = build_run_record(
                suite=suite_name,
                problem_id=problem.problem_id,
                problem_name=problem.name,
                function_id=None,
                dimension=problem.dimension,
                method=method,
                seed=seed,
                max_evaluations=max_evaluations,
                result=result,
            )
            record["coco_evaluations"] = coco_evaluations
            append_jsonl_durable(result_path, record_with_key(record))
            completed.add(record_key)
        del suite
        del observer
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Run CPSO and PSO baselines on COCO BBOB or BBOB-LargeScale")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/benchmarks/bbob_smoke.json")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    output = args.output or ROOT / "results" / "external_benchmarks" / args.config.stem
    return execute(args.config, output, dry_run=args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
