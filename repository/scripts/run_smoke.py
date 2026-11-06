from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from cpso_serial import CpsoConfig, run_serial  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a deterministic serial CPSO smoke experiment")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "smoke.json")
    args = parser.parse_args()
    payload = json.loads(args.config.read_text(encoding="utf-8"))
    benchmark = payload.pop("benchmark")
    config = CpsoConfig(**payload)
    result = run_serial(config, benchmark=benchmark)
    output = {
        "benchmark": benchmark,
        "best_value": result.best_value,
        "best_position": list(result.best_position),
        "function_evaluations": result.function_evaluations,
        "iterations_completed": result.iterations_completed,
        "elapsed_seconds": result.elapsed_seconds,
        "best_history": list(result.best_history),
    }
    print(json.dumps(output, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
