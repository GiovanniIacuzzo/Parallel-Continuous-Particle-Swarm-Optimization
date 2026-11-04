from __future__ import annotations

import matplotlib.pyplot as plt

PALETTE = {
    "navy": "#173F5F",
    "blue": "#2F6B9A",
    "teal": "#3A8D8C",
    "gold": "#D39A32",
    "red": "#B84A4A",
    "grey": "#6B7280",
    "light": "#DCE5EC",
    "independent": "#6B7280",
    "star": "#2F6B9A",
    "ring": "#D39A32",
}


def configure_style() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["STIX Two Text", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.6,
        "grid.linewidth": 0.55,
        "grid.alpha": 0.28,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.04,
    })
