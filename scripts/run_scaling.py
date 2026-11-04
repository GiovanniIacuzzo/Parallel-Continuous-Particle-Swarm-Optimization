#!/usr/bin/env python3
"""Run repeated worker/topology sweeps with a fixed global population and FE budget."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from cpso_hpc.config import load_config
from cpso_hpc.engine import CpsoEngine


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="TOML file with [problem], [algorithm], [cpso], [sweep].")
    parser.add_argument("--output-dir", default="results/scaling")
    args = parser.parse_args()

    config = load_config(args.config)
    # The [sweep] section is not needed by the core config validator.
    import tomllib
    with Path(args.config).open("rb") as stream:
        raw = tomllib.load(stream)
    sweep = raw.get("sweep", {})
    workers_list = [int(x) for x in sweep.get("workers", [1, 2, 4])]
    topologies = [str(x).lower() for x in sweep.get("topologies", ["star"])]
    repetitions = int(sweep.get("repetitions", 10))
    seed_start = int(sweep.get("seed_start", config["algorithm"]["seed"]))
    if repetitions < 1:
        raise ValueError("sweep.repetitions must be >= 1")
    if any(t not in {"independent", "star", "ring"} for t in topologies):
        raise ValueError("Unknown topology in [sweep].")
    if any(p < 1 or p > config["algorithm"]["population_size"] for p in workers_list):
        raise ValueError("Every sweep worker count must be in [1, population_size].")

    engine = CpsoEngine()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []

    for topology in topologies:
        for workers in workers_list:
            for rep in range(repetitions):
                run_cfg = {
                    "problem": config["problem"],
                    "algorithm": dict(config["algorithm"]),
                    "cpso": config["cpso"],
                }
                run_cfg["algorithm"].update({
                    "topology": topology,
                    "workers": workers,
                    "seed": seed_start + rep,
                })
                result = engine.run(run_cfg)
                row = {
                    "topology": topology,
                    "workers": workers,
                    "repetition": rep,
                    "seed": seed_start + rep,
                    "benchmark": result.benchmark,
                    "dimension": result.dimension,
                    "population_size": result.population_size,
                    "iterations": result.iterations,
                    "function_evaluations": result.function_evaluations,
                    "best_value": result.best_value,
                    "elapsed_seconds": result.elapsed_seconds,
                    "compute_seconds": result.compute_seconds,
                    "communication_seconds": result.communication_seconds,
                    "openmp_enabled": result.openmp_enabled,
                    "mu_max": run_cfg["cpso"]["mu_max"],
                    "mu_min": run_cfg["cpso"]["mu_min"],
                    "cognitive": run_cfg["cpso"]["cognitive"],
                    "social": run_cfg["cpso"]["social"],
                    "dt": run_cfg["cpso"]["dt"],
                    "boundary_restitution": run_cfg["cpso"]["boundary_restitution"],
                    "sync_interval": run_cfg["algorithm"]["sync_interval"],
                }
                rows.append(row)
                stem = f"{topology}_p{workers}_run{rep:02d}"
                result.save_json(output_dir / f"{stem}.json")
                print(
                    f"{topology:11s} P={workers:3d} run={rep + 1:02d}/{repetitions} "
                    f"f={result.best_value:.6e} time={result.elapsed_seconds:.4f}s"
                )

    if rows:
        (output_dir / "experiment_configuration.json").write_text(
            json.dumps({"problem": config["problem"], "algorithm": config["algorithm"],
                        "cpso": config["cpso"], "sweep": sweep}, indent=2), encoding="utf-8"
        )
        with (output_dir / "scaling_runs.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        (output_dir / "scaling_runs.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"Saved {len(rows)} runs under {output_dir}")


if __name__ == "__main__":
    main()
