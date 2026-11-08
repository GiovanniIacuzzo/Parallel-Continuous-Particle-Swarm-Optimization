#!/usr/bin/env python3
"""Discover all recorded experiment outputs and generate the complete figure atlas."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from plot_results.build_atlas import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
