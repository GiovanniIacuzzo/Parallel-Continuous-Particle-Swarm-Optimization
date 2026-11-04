from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import load_config
from .engine import CpsoEngine, environment_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the C++ CPSO engine with a TOML configuration.")
    parser.add_argument("--config", required=True, help="Path to experiment TOML file.")
    parser.add_argument("--output", default="results/run", help="Output prefix (without extension).")
    args = parser.parse_args()

    config = load_config(args.config)
    engine = CpsoEngine()
    result = engine.run(config)
    prefix = Path(args.output)
    result.save_json(prefix.with_suffix(".json"))
    result.save_history_csv(prefix.with_name(prefix.name + "_history.csv"))
    meta_path = prefix.with_name(prefix.name + "_environment.json")
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(environment_report(engine), indent=2), encoding="utf-8")

    print(f"CPSO finished: f_best={result.best_value:.12e}")
    print(f"FEs={result.function_evaluations} | elapsed={result.elapsed_seconds:.6f}s | workers={result.workers}")
    print(f"Results: {prefix.with_suffix('.json')}")


if __name__ == "__main__":
    main()
