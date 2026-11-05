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

from cpso_serial.benchmarks.cec2013_lsgo import CEC2013LSGOAdapter  # noqa: E402
from common import (  # noqa: E402
    METHODS,
    append_jsonl_durable,
    build_run_record,
    load_existing_run_keys,
    record_with_key,
    run_method,
    write_campaign_manifest,
)


def execute(config_path: Path, output: Path, *, dry_run: bool = False) -> int:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    function_ids = [int(v) for v in config.get("function_ids", range(1, 16))]
    seeds = [int(v) for v in config.get("seeds", range(1, 26))]
    methods = config.get("methods", list(METHODS))
    max_evaluations = int(config.get("max_evaluations", 3_000_000))
    population = int(config.get("population_size", 40))
    milestones = [int(v) for v in config.get("milestones", [120_000, 600_000, 3_000_000])]
    if not function_ids or any(v < 1 or v > 15 for v in function_ids):
        raise ValueError("function_ids must be selected from [1,15]")
    if not seeds or any(v < 0 or v >= 2**64 for v in seeds):
        raise ValueError("seeds must be unsigned 64-bit integers")
    if sorted(set(methods) - set(METHODS)):
        raise ValueError(f"Unknown methods: {sorted(set(methods) - set(METHODS))}")
    if population < 2 or max_evaluations < population or max_evaluations % population:
        raise ValueError("max_evaluations must be >= population_size and exactly divisible by it")
    if any(v <= 0 or v > max_evaluations or v % population for v in milestones):
        raise ValueError("Each milestone must be a positive multiple of population_size within the FE budget")

    plan = {
        "campaign_id": config.get("campaign_id", config_path.stem),
        "suite": "CEC2013-LSGO",
        "function_ids": function_ids,
        "seeds": seeds,
        "methods": methods,
        "population_size": population,
        "max_evaluations": max_evaluations,
        "milestones": milestones,
        "expected_run_count": len(function_ids) * len(seeds) * len(methods),
        "benchmark_reference_protocol": {
            "official_dimension": 1000,
            "official_runs_per_function": 25,
            "official_max_evaluations": 3_000_000,
            "official_checkpoints": [120_000, 600_000, 3_000_000],
        },
        "output_directory": str(output),
    }
    if dry_run:
        print(json.dumps(plan, indent=2))
        return 0

    adapter = CEC2013LSGOAdapter()
    output.mkdir(parents=True, exist_ok=True)
    write_campaign_manifest(output / "campaign_manifest.json", plan, config_path)
    result_path = output / "records.jsonl"
    completed = load_existing_run_keys(result_path)
    for function_id in function_ids:
        problem = adapter.get_problem(function_id)
        if problem.dimension != 1000:
            raise RuntimeError(
                f"CEC 2013 LSGO function F{function_id} reported dimension {problem.dimension}; "
                "the official LSGO protocol expects D=1000"
            )
        for seed in seeds:
            for method in methods:
                record_key = "|".join(("CEC2013-LSGO", problem.name, method, str(seed)))
                if record_key in completed:
                    continue
                print(
                    f"{method}: {problem.name} seed={seed} "
                    f"(D={problem.dimension}, FE={max_evaluations})"
                )
                result = run_method(
                    method,
                    problem.objective,
                    problem.lower_bounds,
                    problem.upper_bounds,
                    population_size=population,
                    max_evaluations=max_evaluations,
                    seed=seed,
                )
                record = build_run_record(
                    suite="CEC2013-LSGO",
                    problem_id=problem.name,
                    problem_name=problem.name,
                    function_id=function_id,
                    dimension=problem.dimension,
                    method=method,
                    seed=seed,
                    max_evaluations=max_evaluations,
                    result=result,
                    milestones=milestones,
                )
                append_jsonl_durable(result_path, record_with_key(record))
                completed.add(record_key)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Run CPSO and PSO baselines on CEC 2013 LSGO")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/benchmarks/cec2013_smoke.json")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    output = args.output or ROOT / "results" / "external_benchmarks" / args.config.stem
    return execute(args.config, output, dry_run=args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
