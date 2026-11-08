from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Shared publication style. The palette is deliberately color-blind aware and
# remains distinguishable in grayscale through markers/linestyles where useful.
COLORS = {
    "navy": "#173F5F",
    "blue": "#2F6B9A",
    "teal": "#3A8D8C",
    "gold": "#C28A2C",
    "red": "#B65D58",
    "purple": "#76558F",
    "grey": "#687481",
    "light": "#DCE4EA",
    "ink": "#17212B",
    "white": "#FFFFFF",
}
PALETTE = [COLORS[k] for k in ("navy", "blue", "teal", "gold", "purple", "red", "grey")]


def configure_style() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["STIX Two Text", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 8.2,
        "axes.titlesize": 9.2,
        "axes.labelsize": 8.2,
        "xtick.labelsize": 7.2,
        "ytick.labelsize": 7.2,
        "legend.fontsize": 7.0,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.75,
        "lines.linewidth": 1.45,
        "grid.linewidth": 0.5,
        "grid.alpha": 0.25,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.035,
    })


def read_csv(path: Path, required: Iterable[str] = ()) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"File CSV non trovato: {path}")
    with path.open("r", newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        fields = set(reader.fieldnames or [])
        missing = sorted(set(required) - fields)
        if missing:
            raise ValueError(f"Colonne mancanti in {path}: {', '.join(missing)}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"CSV vuoto: {path}")
    return rows


def read_json(path: Path, *, required: Iterable[str] = ()) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"JSON non valido: {path}: {exc}") from exc
    if not isinstance(obj, dict):
        raise ValueError(f"Il file JSON deve contenere un oggetto: {path}")
    missing = sorted(set(required) - set(obj))
    if missing:
        raise ValueError(f"Chiavi mancanti in {path}: {', '.join(missing)}")
    return obj


def as_float(row: dict[str, Any], key: str, default: float = math.nan) -> float:
    value = row.get(key, default)
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def as_int(row: dict[str, Any], key: str, default: int = 0) -> int:
    value = row.get(key, default)
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def save_figure(fig: plt.Figure, output_dir: Path, filename: str, *, dpi: int = 320) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / f"{filename}.pdf"
    png_path = output_dir / f"{filename}.png"
    fig.savefig(pdf_path, format="pdf", metadata={"Creator": "CPSO plot_results"})
    fig.savefig(png_path, format="png", dpi=dpi)
    plt.close(fig)
    return pdf_path, png_path


def evidence_note(evidence_class: str | None) -> str:
    if evidence_class == "confirmatory":
        return "Confirmatory campaign"
    if evidence_class == "exploratory":
        return "Exploratory pilot; descriptive evidence only"
    if evidence_class == "integration_only":
        return "Integration test; not scientific evidence"
    return "Evidence class not specified"


def manifest_evidence(campaign_dir: Path) -> str | None:
    for name in ("campaign_manifest.json", "manifest.json"):
        path = campaign_dir / name
        if path.is_file():
            payload = read_json(path)
            campaign = payload.get("campaign", {})
            value = campaign.get("evidence_class") if isinstance(campaign, dict) else None
            return str(value) if value else str(payload.get("evidence_class")) if payload.get("evidence_class") else None
    return None


def safe_stem(value: str) -> str:
    cleaned = "".join(ch.lower() if ch.isalnum() else "_" for ch in value)
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    return cleaned.strip("_") or "unknown"


def require_finite(rows: list[dict[str, str]], columns: Iterable[str], source: Path) -> None:
    for col in columns:
        vals = [as_float(row, col) for row in rows]
        if not any(math.isfinite(value) for value in vals):
            raise ValueError(f"Nessun valore numerico finito per '{col}' in {source}")


def print_saved(paths: tuple[Path, Path]) -> None:
    print(f"  creati: {paths[0]}")
    print(f"          {paths[1]}")
