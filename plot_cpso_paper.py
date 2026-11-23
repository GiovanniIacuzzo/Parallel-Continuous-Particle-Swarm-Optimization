#!/usr/bin/env python3
"""Generate Study III publication PDFs from completed results (directory or ZIP).

Examples:
  python plot_cpso_paper.py risultati.zip
  python plot_cpso_paper.py results/study3_paper_main --with-traces
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from io import BytesIO
import json
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd


# Consistent ordering and line identities across every panel and legend.
METHODS = {
    "cpso_selected": ("CPSO", "#B33E44", "-"),
    "cso": ("CSO", "#173F5F", "--"),
    "slpso": ("SL-PSO", "#247D76", "-."),
    "spso2011": ("SPSO-2011", "#77598C", ":"),
    "constriction_gbest": ("Constriction gbest", "#657789", (0, (5, 2))),
    "inertia_gbest": ("Inertia gbest", "#507D99", (0, (3, 2, 1, 2))),
    "inertia_lbest_ring": ("Inertia lbest ring", "#518B81", (0, (2, 1.5))),
    "classic_gbest": ("Classic gbest", "#7A6A59", (0, (1, 2))),
    "clpso": ("CLPSO", "#9D733B", (0, (6, 2, 2, 2))),
    "fips_ring": ("FIPS ring", "#6D8058", (0, (4, 1.5, 1, 1.5))),
    "dms_pso": ("DMS-PSO", "#717179", (0, (7, 3))),
}
PRIMARY = tuple(METHODS)[:4]
TARGET = 1e-8
SUITES = ("inverse", "bbob", "cec", "native")
WIDTH = 7.10


class Results:
    """Read summaries and optional run files directly from either source."""

    def __init__(self, directory: Path, archive: ZipFile | None):
        self.directory = directory
        self.archive = archive
        self.prefixes = {}
        if archive is not None:
            for name in archive.namelist():
                path = PurePosixPath(name)
                if "__MACOSX" in path.parts or path.name != "protocol.json":
                    continue
                if path.parent.name in SUITES:
                    suite = path.parent.name
                    if suite in self.prefixes:
                        raise ValueError(f"ZIP contains more than one {suite} campaign")
                    self.prefixes[suite] = path.parent

    def read(self, suite, name):
        if self.archive is None:
            return (self.directory / suite / name).read_bytes()
        if suite not in self.prefixes:
            raise FileNotFoundError(f"No {suite}/protocol.json found in ZIP")
        try:
            return self.archive.read(str(self.prefixes[suite] / name))
        except KeyError as error:
            raise FileNotFoundError(f"Missing {suite}/{name} in ZIP") from error

    def csv(self, suite, name):
        return pd.read_csv(BytesIO(self.read(suite, f"analysis/{name}.csv")))

    def runs(self, suite):
        if self.archive is None:
            for path in sorted((self.directory / suite / "runs").glob("*.json")):
                yield json.loads(path.read_bytes())
        else:
            folder = self.prefixes[suite] / "runs"
            for name in sorted(self.archive.namelist()):
                path = PurePosixPath(name)
                if path.parent == folder and path.suffix == ".json":
                    yield json.loads(self.archive.read(name))


def configure_style():
    plt.rcParams.update({
        "font.family": "STIXGeneral",
        "mathtext.fontset": "stix",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.65,
        "axes.axisbelow": True,
        "pdf.fonttype": 42,
    })


def line_style(method):
    _, color, linestyle = METHODS[method]
    return dict(color=color, linestyle=linestyle,
                lw=2.25 if method == "cpso_selected" else 1.30 if method in PRIMARY else 0.95,
                zorder=5 if method == "cpso_selected" else 3 if method in PRIMARY else 2,
                alpha=1.0 if method in PRIMARY else 0.78)


def legend(fig, methods=METHODS):
    handles = [Line2D([], [], label=METHODS[method][0], **line_style(method)) for method in methods]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.016),
               ncol=min(4, len(handles)), frameon=False, handlelength=2.65, columnspacing=1.7)


def save(fig, output, filename):
    path = output / filename
    fig.savefig(path, facecolor="white", metadata={"Title": path.stem.replace("_", " ")})
    plt.close(fig)
    print(path)


def target_data(results, suite):
    protocol = json.loads(results.read(suite, "protocol.json"))
    completed = json.loads(results.read(suite, "complete.json"))
    frame = results.csv(suite, "target_hits")
    budgets = protocol["budgets"]
    axis = "hit_seconds" if suite == "inverse" else "hit_fe"
    censor = "censor_seconds" if suite == "inverse" else "censor_fe"
    keys = ["method", "function", "dimension", "seed", "instance", "budget", "target"]
    if "variant" in frame:
        keys.append("variant")
    expected = len(protocol["cases"]) * len(protocol["seeds"]) * len(METHODS) * len(budgets)
    target_count = frame.target.nunique()
    hits = frame.dropna(subset=[axis])
    if (set(protocol["methods"]) != set(METHODS) or set(frame.method) != set(METHODS)
            or completed["observations"] != expected
            or not np.isfinite(frame.target).all() or target_count < 1
            or frame.duplicated(keys).any() or len(frame) != expected * target_count
            or not frame.groupby([key for key in keys if key != "target"], dropna=False)
                        .size().eq(target_count).all()
            or not np.isfinite(hits[axis]).all() or (hits[axis] <= 0).any()
            or not np.isfinite(frame[censor]).all() or (hits[axis] > hits[censor]).any()
            or (hits[axis] > hits.budget).any()):
        raise ValueError(f"{suite}: incomplete, duplicated or invalid target records")
    return frame[frame.budget == max(budgets)].copy(), max(budgets)


def ecdf(ax, frame, variable, lower, budget, equal_variants=False):
    if frame.empty:
        raise ValueError("No observations for an ECDF panel")
    for method, rows in frame.groupby("method", sort=False):
        if equal_variants:
            # Give each function × variant equal weight despite unequal instance counts.
            size = rows.groupby(["function", "variant"])[variable].transform("size")
            weights = 1.0 / size.to_numpy()
        else:
            weights = np.ones(len(rows))
        times = rows[variable].to_numpy()
        hit = np.isfinite(times)
        order = np.argsort(times[hit], kind="stable")
        x = times[hit][order]
        y = np.cumsum(weights[hit][order]) / weights.sum()
        ax.step(np.r_[lower, x, budget], np.r_[0.0, y, y[-1] if len(y) else 0.0],
                where="post", **line_style(method))
    ax.set(xscale="log", xlim=(lower, budget), ylim=(-0.024, 1.035),
           yticks=np.linspace(0, 1, 6))
    ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.grid(axis="y", color="#DDE2E7", linewidth=0.55)


def inverse_targets(frame, budget, output):
    frame = frame[np.isclose(frame.target, TARGET, rtol=1e-10, atol=0)]
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 3.08), sharey=True)
    fig.subplots_adjust(left=0.105, right=0.98, top=0.88, bottom=0.34, wspace=0.18)
    for ax, dimension, letter in zip(axes, sorted(frame.dimension.unique()), ("a", "b"), strict=True):
        panel = frame[frame.dimension == dimension]
        lower = min(float(panel.hit_seconds.min()) / 1.25, budget / 100)
        if not np.isfinite(lower):
            lower = budget / 100
        ecdf(ax, panel, "hit_seconds", lower, budget)
        ax.set(title=f"({letter})  Dimension {dimension}", xlabel="Wall-clock time (s)")
    axes[0].set_ylabel(r"Runs reaching gap $leq 10^{-8}$")
    legend(fig)
    save(fig, output, "fig_study3_inverse_time_to_target.pdf")


def fe_targets(panels, output, filename, *, equal_variants=False):
    fig, axes = plt.subplots(1, len(panels), figsize=(WIDTH, 3.13), sharey=True)
    fig.subplots_adjust(left=0.105, right=0.98, top=0.88, bottom=0.34, wspace=0.19)
    for ax, (title, rows, budget), letter in zip(axes, panels, "ab", strict=True):
        lower = max(1.0, min(float(rows.hit_fe.min()) / 1.25, budget / 100))
        if not np.isfinite(lower):
            lower = budget / 100
        ecdf(ax, rows, "hit_fe", lower, budget, equal_variants)
        ax.set(title=f"({letter})  {title}", xlabel="Function evaluations")
    axes[0].set_ylabel("Run–target pairs reached")
    legend(fig)
    save(fig, output, filename)


def external_performance(results, bbob, budget, output):
    """Show BBOB attainment and CEC final-gap ratios on their own scales."""
    fig, (left, right) = plt.subplots(1, 2, figsize=(WIDTH, 3.23))
    fig.subplots_adjust(left=0.105, right=0.98, top=0.89, bottom=0.33, wspace=0.35)
    lower = max(1.0, min(float(bbob.hit_fe.min()) / 1.25, budget / 100))
    ecdf(left, bbob, "hit_fe", lower, budget)
    highest = bbob.groupby("method").hit_fe.agg(lambda col: col.notna().mean()).max()
    if 0 < highest <= 0.5:
        ceiling = np.ceil(20 * highest) / 20
        left.set(ylim=(-0.02 * ceiling, 1.05 * ceiling), yticks=np.arange(0, ceiling + 0.001, 0.05))
    left.set(title=r"(a)  BBOB, $D = 40$", xlabel="Function evaluations", ylabel="Run–target pairs reached")

    cec = results.csv("cec", "measurements")
    cec = cec[cec.budget == cec.budget.max()].copy()
    keys = ["function", "dimension", "instance", "seed"]
    if (cec.duplicated(keys + ["method"]).any() or set(cec.method) != set(METHODS)
            or not cec.groupby(keys).size().eq(len(METHODS)).all()
            or not np.isfinite(cec.gap).all() or (cec.gap <= 0).any()):
        raise ValueError("CEC: invalid paired final gaps")
    cec["ratio"] = cec.gap / cec.groupby(keys).gap.transform("min")
    cap = max(10.0, 10.0 ** np.ceil(np.log10(cec.ratio.max())))
    for method, rows in cec.groupby("method"):
        ratio = np.sort(rows.ratio.to_numpy())
        y = np.arange(1, len(ratio) + 1) / len(ratio)
        right.step(np.r_[1.0, ratio, cap], np.r_[0, y, 1], where="post", **line_style(method))
    right.set(title="(b)  CEC2013 LSGO", xscale="log", xlim=(1, cap),
              xlabel="Final gap / best final gap in matched block",
              ylabel="Matched blocks reached", ylim=(-0.024, 1.035),
              yticks=np.linspace(0, 1, 6))
    right.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    right.grid(axis="y", color="#DDE2E7", linewidth=0.55)
    legend(fig)
    save(fig, output, "fig_study3_external_performance.pdf")


def benchmark_effects(results, output):
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 5.6))
    fig.subplots_adjust(left=0.09, right=0.97, bottom=0.19, top=0.92, wspace=0.30)
    effects = [results.csv(suite, "outcomes_by_function") for suite in ("bbob", "cec")]
    bound = max(1, max(frame.delta_log_gap.abs().max() for frame in effects)) * 1.14
    groups = {
        "first_alone": ("CPSO lower gap", "#247D76", "o"),
        "tied_first": ("Practical tie", "#6B7280", "D"),
        "loss": ("Baseline lower gap", "#B33E44", "s"),
    }
    for ax, rows, title in zip(axes, effects, ("(a)  COCO BBOB", "(b)  CEC2013 LSGO"), strict=True):
        rows = rows.sort_values("function").reset_index(drop=True)
        ax.axvspan(-0.01, 0.01, color="#E4E8ED", linewidth=0)
        ax.axvline(0, color="#626C76", lw=0.65)
        for key, (_, color, marker) in groups.items():
            selected = rows[rows.outcome == key]
            ax.hlines(selected.index, 0, selected.delta_log_gap, color=color, alpha=0.74, lw=0.85)
            ax.scatter(selected.delta_log_gap, selected.index, color=color, marker=marker, s=18, zorder=4)
        if not rows.outcome.isin(groups).all() or not np.isfinite(rows.delta_log_gap).all():
            raise ValueError(f"{title}: undefined gap comparison")
        ax.set_xscale("symlog", linthresh=0.01)
        ax.set(xlim=(-bound, bound), ylim=(len(rows) - 0.4, -0.6), title=title)
        ax.set_xticks([-1, -0.1, -0.01, 0, 0.01, 0.1, 1],
                      ["−1", "−0.1", "−0.01", "0", "0.01", "0.1", "1"])
        ax.set_yticks(rows.index, rows.function.str.upper())
        ax.tick_params(axis="y", length=0, pad=3)
        ax.grid(axis="y", color="#E6E9ED", linewidth=0.45)
    fig.supxlabel(r"Mean $\log_{10}$ gap difference (CPSO − best baseline per function)", y=0.095)
    handles = [Line2D([], [], label=label, marker=marker, color=color, linestyle="none", markersize=4.6)
               for label, color, marker in groups.values()]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.016),
               frameon=False, ncol=3)
    save(fig, output, "fig_study3_external_function_effects.pdf")


def inverse_convergence(results, targets, budget, output):
    runs = {(item["key"]["dimension"], item["key"]["method"], item["key"]["seed"]): item["result"]
            for item in results.runs("inverse")
            if item["key"]["budget"] == budget and item["key"]["method"] in PRIMARY}
    expected = set(targets[targets.method.isin(PRIMARY)][["dimension", "method", "seed"]]
                   .itertuples(index=False, name=None))
    if runs.keys() != expected:
        raise ValueError("--with-traces requires the complete inverse/runs/ files for the displayed methods")
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 3.04), sharey=True)
    fig.subplots_adjust(left=0.12, right=0.98, bottom=0.27, top=0.87, wspace=0.18)
    for ax, dimension, letter in zip(axes, sorted(targets.dimension.unique()), ("a", "b"), strict=True):
        panel = {key: val for key, val in runs.items() if key[0] == dimension}
        start = max(val["trace"][0][1] for val in panel.values())
        grid = np.geomspace(start, budget, 320)
        for method in PRIMARY:
            samples = []
            for (_, name, _), val in panel.items():
                if name != method:
                    continue
                trace = np.asarray(val["trace"], dtype=float)
                if (not np.isfinite(trace).all() or np.any(np.diff(trace[:, 1]) < 0)
                        or np.any(np.diff(trace[:, 2]) > 0) or trace[-1, 1] > budget):
                    raise ValueError("Invalid inverse convergence trace")
                pos = np.searchsorted(trace[:, 1], grid, side="right") - 1
                samples.append(np.maximum(trace[pos, 2] - val["optimum"], 1e-12))
            lo, mid, hi = np.quantile(samples, [0.25, 0.50, 0.75], axis=0)
            ax.step(grid, mid, where="post", **line_style(method))
            if method == "cpso_selected":
                ax.fill_between(grid, lo, hi, step="post", color=METHODS[method][1], alpha=0.15, lw=0)
        ax.set(title=f"({letter})  Dimension {dimension}", xlabel="Wall-clock time (s)",
               xscale="log", yscale="log", xlim=(start, budget))
        ax.grid(axis="y", color="#DDE2E7", linewidth=0.55)
    axes[0].set_ylabel(r"Median best-so-far gap (floor $10^{-12}$)")
    legend(fig, PRIMARY)
    save(fig, output, "fig_study3_inverse_convergence.pdf")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", nargs="?", type=Path, default=Path("results/study3_paper_main"))
    parser.add_argument("--output", type=Path, default=Path("figures/study3_paper"))
    parser.add_argument("--with-traces", action="store_true", help="Add the inverse convergence PDF")
    args = parser.parse_args()
    configure_style()
    try:
        opener = nullcontext() if args.source.is_dir() else ZipFile(args.source)
        with opener as archive:
            results = Results(args.source, archive)
            frames = {suite: target_data(results, suite) for suite in SUITES}
            args.output.mkdir(parents=True, exist_ok=True)
            inverse_targets(*frames["inverse"], args.output)
            external_performance(results, *frames["bbob"], args.output)
            rows, cap = frames["native"]
            native = [(f"Dimension {dim}", group, cap) for dim, group in rows.groupby("dimension")]
            fe_targets(native, args.output, "fig_study3_native_target_ecdf.pdf", equal_variants=True)
            benchmark_effects(results, args.output)
            if args.with_traces:
                inverse_convergence(results, *frames["inverse"], args.output)
    except (OSError, ValueError, KeyError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
