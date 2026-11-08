from __future__ import annotations

import ast
import csv
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

LOG_FLOOR = 1.0e-16
COLORS = ["#173F5F", "#2F6B9A", "#3A8D8C", "#C28A2C", "#76558F", "#B65D58", "#687481",
          "#4D908E", "#A26769", "#8A817C", "#577590", "#BC6C25"]


def configure_style() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["STIX Two Text", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 8.0,
        "axes.titlesize": 9.0,
        "axes.labelsize": 8.0,
        "xtick.labelsize": 7.0,
        "ytick.labelsize": 7.0,
        "legend.fontsize": 6.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.7,
        "grid.alpha": 0.22,
        "grid.linewidth": 0.45,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.04,
    })


def save_figure(fig: plt.Figure, output: Path, stem: str, *, dpi: int = 320) -> list[Path]:
    output.mkdir(parents=True, exist_ok=True)
    pdf = output / f"{stem}.pdf"
    png = output / f"{stem}.png"
    fig.savefig(pdf, format="pdf", metadata={"Creator": "CPSO plot_results atlas"})
    fig.savefig(png, format="png", dpi=dpi)
    plt.close(fig)
    return [pdf, png]


def safe_slug(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value).strip())
    return value.strip("._-") or "unknown"


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return obj


def read_jsonl(path: Path, *, tolerate_partial_tail: bool = False) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    raw = path.read_bytes()
    lines = raw.splitlines(keepends=True)
    rows: list[dict[str, Any]] = []
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            is_last_partial = index == len(lines) - 1 and not line.endswith(b"\n")
            if tolerate_partial_tail and is_last_partial:
                break
            raise ValueError(f"Malformed JSONL record: {path}:{index + 1}") from exc
        if not isinstance(item, dict):
            raise ValueError(f"Expected JSON object in {path}:{index + 1}")
        rows.append(item)
    return rows


def finite_float(value: Any, default: float = math.nan) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def method_label(row: dict[str, Any]) -> str:
    name = str(row.get("method", "unknown"))
    if name.startswith("serial_"):
        return "Serial · " + name[7:].replace("_", " ")
    return name.replace("_", " ")


def evidence_label(manifest: dict[str, Any]) -> str:
    campaign = manifest.get("campaign", {}) if isinstance(manifest, dict) else {}
    evidence = campaign.get("evidence_class") if isinstance(campaign, dict) else None
    evidence = evidence or manifest.get("evidence_class") or manifest.get("plan", {}).get("evidence_class")
    return str(evidence or "not specified")


def bootstrap_median(values: Iterable[float], *, seed: int = 20261107, n_boot: int = 2000) -> tuple[float, float]:
    data = np.asarray([float(v) for v in values if math.isfinite(float(v))], dtype=float)
    if data.size == 0:
        return math.nan, math.nan
    if data.size == 1:
        return float(data[0]), float(data[0])
    rng = np.random.default_rng(seed)
    samples = rng.choice(data, size=(n_boot, data.size), replace=True)
    stats = np.median(samples, axis=1)
    return float(np.quantile(stats, 0.025)), float(np.quantile(stats, 0.975))


def family_cluster_bootstrap(
    family_values: dict[str, float], *, seed: int = 20261107, n_boot: int = 3000
) -> tuple[float, float]:
    vals = np.asarray([v for v in family_values.values() if math.isfinite(v)], dtype=float)
    if vals.size == 0:
        return math.nan, math.nan
    if vals.size == 1:
        return float(vals[0]), float(vals[0])
    rng = np.random.default_rng(seed)
    samples = rng.choice(vals, size=(n_boot, vals.size), replace=True)
    medians = np.median(samples, axis=1)
    return float(np.quantile(medians, 0.025)), float(np.quantile(medians, 0.975))


def history_arrays(row: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    fe = row.get("history_evaluations", [])
    values = row.get("best_history", [])
    if not isinstance(fe, list) or not isinstance(values, list) or not fe or len(fe) != len(values):
        return np.asarray([], dtype=float), np.asarray([], dtype=float)
    x = np.asarray(fe, dtype=float)
    y = np.asarray(values, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    if x.size == 0:
        return x, y
    order = np.argsort(x)
    return x[order], np.minimum.accumulate(y[order])


def interpolate_step_history(row: dict[str, Any], grid: np.ndarray) -> np.ndarray | None:
    x, y = history_arrays(row)
    if x.size == 0:
        return None
    idx = np.searchsorted(x, grid, side="right") - 1
    result = np.full(grid.shape, np.nan, dtype=float)
    valid = idx >= 0
    result[valid] = y[idx[valid]]
    if x[0] > grid[0]:
        result[grid < x[0]] = y[0]
    result[grid > x[-1]] = y[-1]
    return result


def discover_internal_records(campaign_dir: Path) -> list[dict[str, Any]]:
    paths = sorted(campaign_dir.rglob("runs.jsonl"))
    if not paths and (campaign_dir / "campaign_runs.csv").is_file():
        with (campaign_dir / "campaign_runs.csv").open("r", newline="", encoding="utf-8-sig") as stream:
            rows = list(csv.DictReader(stream))
        # CSV serialization turns run-level history lists into their textual
        # representation. Restore only these known array fields, without eval.
        for row in rows:
            for field in ("best_history", "history_evaluations", "best_position"):
                raw = row.get(field)
                if not isinstance(raw, str) or not raw.strip():
                    continue
                try:
                    row[field] = json.loads(raw)
                except json.JSONDecodeError:
                    try:
                        row[field] = ast.literal_eval(raw)
                    except (ValueError, SyntaxError):
                        row[field] = []
        return rows
    rows: list[dict[str, Any]] = []
    for path in paths:
        relative = path.parent.relative_to(campaign_dir)
        data = read_jsonl(path, tolerate_partial_tail=False)
        for row in data:
            copy = dict(row)
            if len(relative.parts) >= 2:
                copy.setdefault("stage", relative.parts[0])
                copy.setdefault("study", relative.parts[1])
            else:
                copy.setdefault("stage", relative.parts[0] if relative.parts else "study")
                copy.setdefault("study", path.parent.name)
            rows.append(copy)
    return rows


def load_external_records(campaign_dir: Path) -> list[dict[str, Any]]:
    files = []
    if (campaign_dir / "records.jsonl").is_file():
        files.append(campaign_dir / "records.jsonl")
    records_dir = campaign_dir / "records"
    if records_dir.is_dir():
        files.extend(sorted(records_dir.glob("*.jsonl")))
    rows: list[dict[str, Any]] = []
    for path in files:
        rows.extend(read_jsonl(path, tolerate_partial_tail=True))
    return rows


def add_evidence_note(fig: plt.Figure, evidence: str, extra: str = "") -> None:
    pieces = [f"Evidence class: {evidence.replace('_', ' ')}"]
    if extra:
        pieces.append(extra)
    fig.text(0.5, 0.006, " · ".join(pieces), ha="center", va="bottom", fontsize=6.0, color="#687481")
