"""Reproduce the paired final-budget comparison from the M2 transfer campaign."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

NAMES = {
    "pso_classic_gbest": "Classic gbest",
    "pso_inertia_gbest": "Inertia gbest",
    "pso_inertia_lbest_ring": "Inertia lbest (ring)",
    "pso_constriction_gbest": "Constriction gbest",
    "pso_fips": "FIPS (ring)",
    "pso_clpso": "CLPSO",
    "pso_spso2011": "SPSO-2011",
    "pso_dms": "DMS-PSO",
    "pso_slpso": "SL-PSO",
    "pso_cso": "CSO",
}


def summarize(folder, dimension):
    cec = dimension is None
    a = pd.read_csv(folder / "checkpoints.csv")
    budget = 4096 if cec else 384 * dimension
    primary = "cpso_random_k8" if cec else "cpso_random_k8_by_dimension"
    part = a.loc[(a.fe == budget) & a.method.isin([primary, *NAMES])]
    if not cec:
        part = part.loc[part.dimension == dimension]
    values = part.pivot(index=["function", "replicate"], columns="method", values="value")
    functions = range(1, 16 if cec else 25)
    reps = range(1, 6) if cec else range(11, 16)
    expected = pd.MultiIndex.from_product([functions, reps], names=values.index.names)
    if (not values.index.equals(expected) or set(values) != {primary, *NAMES}
            or not np.isfinite(values.to_numpy()).all()):
        raise ValueError(f"Incomplete paired design in {folder}, D={dimension}")
    difference = values[list(NAMES)].rsub(values[primary], axis=0).groupby(level="function").median()
    return pd.DataFrame({"wins": (difference < 0).sum(), "ties": (difference == 0).sum(),
                         "losses": (difference > 0).sum()}).loc[list(NAMES)]


def main(base):
    panels = [("validate", 5, "BBOB · D = 5", "dt = 1"),
              ("validate", 20, "BBOB · D = 20", "dt = 0.35"),
              ("cec", None, "CEC 2013 LSGO", "dt = 1")]
    plt.rcParams.update({"pdf.fonttype": 42, "font.size": 9})
    fig, axes = plt.subplots(1, 3, sharey=True, figsize=(9.3, 4.4))
    for ax, (stage, d, title, step) in zip(axes, panels):
        counts = summarize(base / stage, d)
        total = counts.iloc[0].sum()
        y = np.arange(len(counts))
        offsets = np.zeros(len(counts))
        for key, color, label in [("wins", "#17665f", "CPSO wins"),
                                  ("ties", "#c5c9ca", "Tie"),
                                  ("losses", "#995235", "Baseline wins")]:
            widths = counts[key].to_numpy()
            ax.barh(y, widths, left=offsets, color=color, height=.66, label=label)
            for row, width in enumerate(widths):
                if width >= 3:
                    ax.text(offsets[row]+width/2, row, str(width), ha="center", va="center",
                            color="black" if key == "ties" else "white", fontsize=8)
            offsets += widths
        ax.set(xlim=(0,total), xticks=[0, total//3, 2*total//3, total],
               xlabel="Number of functions", title=title+"\n"+step,
               yticks=y, yticklabels=list(NAMES.values()))
        ax.spines[["top","right","left"]].set_visible(False)
        ax.tick_params(axis="y", length=0)
    axes[0].invert_yaxis()
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0,.08,1,1))
    dest = base / "analysis" / "fig_cpso_vs_10_pso.pdf"
    dest.parent.mkdir(exist_ok=True)
    fig.savefig(dest)
    plt.close(fig)
    print(dest)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, default=Path("results/transfer_v2/study"))
    main(parser.parse_args().study)
