#!/usr/bin/env python3
"""Build vector figures from completed Study III runs, without rerunning optimization."""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
import numpy as np


SHORT = {
    "cpso_selected": "CPSO", "cpso_one_worker": "CPSO · 1 worker",
    "cpso_fixed_reference": "CPSO · reference", "cpso_matched_fixed": "CPSO · matched fixed",
    "pso_matched_discrete": "PSO · matched discrete", "classic_gbest": "Classic gbest",
    "inertia_gbest": "Inertia gbest", "inertia_lbest_ring": "Inertia lbest ring",
    "constriction_gbest": "Constriction gbest", "clpso": "CLPSO", "slpso": "SL-PSO",
    "spso2011": "SPSO-2011", "fips_ring": "FIPS ring", "cso": "CSO", "dms_pso": "DMS-PSO",
    "fixed": "Fixed step", "step": "Adaptive step", "recovery": "Recovery",
    "step_recovery": "Adaptive step + recovery",
}
PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#614995", "#9B7300",
           "#157B82", "#333333", "#94403D", "#31689A", "#626A1A", "#7A4B71",
           "#CE580D", "#4F4F4F"]
DASHES = ["-", "--", "-.", ":", (0, (6, 2, 1, 2)), (0, (3, 2, 1, 2, 1, 2))]


def load_json(path):
    return json.loads(path.read_text())


def label(method, protocol):
    if protocol["stage"] == "screen":
        recipe = protocol["methods"][method]["recipe"]
        randomness = "Coordinate" if recipe["coordinate"] else "Scalar"
        graph = "Random-8" if recipe["topology"] == "random_k8" else "Global"
        return f"{randomness}, {graph}, {recipe['dt']:g}"
    if method == "cpso_selected" and protocol["stage"] == "time":
        return f"CPSO · {protocol['methods'][method]['workers']} workers"
    return SHORT.get(method, method)


def function_label(function):
    return function.upper() if function.startswith("f") and function[1:].isdigit() else function.replace("_", " ").capitalize()


def budget_label(budget, timed):
    return f"{budget:g} s" if timed else f"{budget:,} FE"


def budget_slug(budget, timed):
    return (f"{budget:g}".replace(".", "p") + "s") if timed else f"{budget}fe"


def rank_data(folder, protocol):
    ranks = {}
    with (folder / "analysis/rank_summary.csv").open(newline="") as stream:
        for row in csv.DictReader(stream):
            key = float(row["budget"]), int(row["dimension"]), row["function"], row["method"]
            if key in ranks:
                raise ValueError(f"Duplicate rank cell: {key}")
            ranks[key] = float(row["mean_rank"])
    functions = list(dict.fromkeys(function for function, _ in protocol["cases"]))
    methods = [method for method in protocol["methods"] if method != "cpso_one_worker"]
    for budget in protocol["budgets"]:
        for dimension in protocol["dimensions"]:
            for function in functions:
                for method in methods:
                    if (budget, dimension, function, method) not in ranks:
                        raise ValueError(f"Incomplete rank matrix in {folder}")
    order = sorted(methods, key=lambda m: (np.mean([r for k, r in ranks.items() if k[-1] == m]), m))
    return ranks, functions, order


def heatmap(path, protocol, budget, ranks, functions, methods):
    dimensions = protocol["dimensions"]
    width = max(6.4, 3.2 + 0.44 * (len(functions) + 1))
    height = len(dimensions) * max(1.8, 0.30 * len(methods) + 0.72)
    fig, axes = plt.subplots(len(dimensions), 1, figsize=(width, height + 0.35),
                             layout="constrained", squeeze=False)
    norm = Normalize(1, len(methods))
    for dimension, ax in zip(dimensions, axes[:, 0]):
        cells = np.asarray([[ranks[budget, dimension, f, m] for f in functions] for m in methods])
        values = np.column_stack([cells, cells.mean(axis=1)])
        image = ax.imshow(values, cmap="YlGnBu_r", norm=norm, aspect="auto")
        ax.set_xticks(range(values.shape[1]), [*map(function_label, functions), "Mean"],
                      rotation=30 if len(functions) > 8 else 0, ha="right" if len(functions) > 8 else "center")
        ax.set_yticks(range(len(methods)), [label(m, protocol) for m in methods])
        ax.tick_params(length=0)
        ax.set_title(f"D = {dimension} · {budget_label(budget, protocol['stage'] == 'time')}", loc="left")
        ax.axvline(len(functions) - 0.5, color="white", linewidth=2)
        best = np.isclose(values, values.min(axis=0), atol=1e-10, rtol=0)
        for i, row in enumerate(values):
            for j, value in enumerate(row):
                if best[i, j]:
                    ax.add_patch(Rectangle((j - 0.48, i - 0.48), 0.96, 0.96,
                                           fill=False, edgecolor="#D55E00", linewidth=1.75))
                rgb = image.cmap(norm(value))[:3]
                luminance = np.dot(rgb, [0.2126, 0.7152, 0.0722])
                ax.text(j, i, f"{value:.1f}", color="white" if luminance < 0.51 else "#151515",
                        ha="center", va="center", fontweight="bold" if best[i, j] else "normal",
                        fontsize=8 if len(functions) > 8 else 9)
        for spine in ax.spines.values():
            spine.set_visible(False)
    bar = fig.colorbar(image, ax=axes[:, 0].tolist(), fraction=0.040, pad=0.03)
    bar.set_label("Mean rank (lower is better)")
    fig.savefig(path)
    plt.close(fig)


def convergence(path, protocol, budget, traces, functions, methods, chosen):
    timed = protocol["stage"] == "time"
    dimensions = protocol["dimensions"]
    x = np.geomspace(budget / 500 if timed else protocol["population"], budget, 180)
    style = {}
    order = [chosen, *(m for m in methods if m != chosen)]
    for i, method in enumerate(order):
        style[method] = dict(color=PALETTE[i % len(PALETTE)],
                             linestyle=DASHES[i % len(DASHES)],
                             linewidth=2.3 if i == 0 else 1.25,
                             zorder=5 if i == 0 else 2)
    fig, axes = plt.subplots(len(dimensions), len(functions), squeeze=False,
                             figsize=(max(6.7, len(functions) * 3.3), 2.65 * len(dimensions) + 1.25),
                             sharex=True)
    for i, dimension in enumerate(dimensions):
        for j, function in enumerate(functions):
            ax = axes[i, j]
            for method in order:
                curves = traces.get((dimension, function, method), [])
                if not curves:
                    continue
                data = np.ma.masked_invalid(curves)
                present = np.ma.count(data, axis=0) >= (len(data) // 2 + 1)
                median = np.ma.median(data, axis=0).filled(np.nan)
                if np.any(present):
                    ax.step(x[present], median[present], where="post", **style[method])
            ax.set(xscale="log", yscale="log", xlim=(x[0], x[-1]),
                   title=f"{function_label(function)} · D = {dimension}")
            if j == 0:
                ax.set_ylabel("Median objective gap")
            if i == len(dimensions) - 1:
                ax.set_xlabel("Wall-clock time (s)" if timed else "Function evaluations")
            ax.grid(which="major", color="#D7DCE2", linewidth=0.6)
    handles = [Line2D([0], [0], label=label(m, protocol), **style[m]) for m in order]
    columns = 2 if protocol["stage"] == "screen" else min(len(handles), 4)
    legend_rows = (len(order) + columns - 1) // columns
    fig.legend(handles=handles, loc="lower center", ncols=columns, frameon=False, fontsize=8.5)
    fig.tight_layout(rect=(0, (0.24 * legend_rows + 0.16) / fig.get_figheight(), 1, 1))
    fig.savefig(path)
    plt.close(fig)


def collect_traces(folder, protocol, budget, count):
    timed = protocol["stage"] == "time"
    x = np.geomspace(budget / 500 if timed else protocol["population"], budget, 180)
    floor = protocol["score"]["gap_floor"]
    curves = defaultdict(list)
    seen = set()
    for path in (folder / "runs").glob("*.json"):
        record = load_json(path)
        key, result = record["key"], record["result"]
        if key["budget"] != budget:
            continue
        identity = tuple(key[field] for field in ("dimension", "function", "method", "instance", "seed"))
        if identity in seen:
            raise ValueError(f"Duplicate trace: {identity}")
        seen.add(identity)
        y = np.full(len(x), np.nan)
        trace = np.asarray(result["trace"], dtype=float)
        if len(trace):
            axis = trace[:, int(timed)]
            index = np.searchsorted(axis, x, side="right") - 1
            ok = index >= 0
            y[ok] = np.maximum(floor, trace[index[ok], 2] - result["optimum"])
        curves[identity[:3]].append(y)
    if len(seen) != count:
        raise ValueError(f"Expected {count} runs at budget {budget}, found {len(seen)} in {folder}")
    return curves


def ablation_effects(path, folder, protocol):
    variants = ["step", "recovery", "step_recovery"]
    rows = {}
    with (folder / "analysis/measurements.csv").open(newline="") as stream:
        for row in csv.DictReader(stream):
            block = int(row["budget"]), int(row["dimension"]), row["function"], int(row["seed"])
            rows.setdefault(block, {})[row["method"]] = float(row["log_gap"])
    cells = [(None, None), *((b, d) for b in protocol["budgets"] for d in protocol["dimensions"])]
    fig, ax = plt.subplots(figsize=(7.4, 3.25 + 0.11 * len(cells)))
    rng = np.random.default_rng(412)
    for i, variant in enumerate(variants):
        offsets = np.arange(len(cells), dtype=float) + (i - 1) * 0.20
        centers, limits = [], []
        for budget, dimension in cells:
            by_seed = defaultdict(list)
            for (b, d, _, seed), scores in rows.items():
                if budget is None or (b, d) == (budget, dimension):
                    by_seed[seed].append(scores[variant] - scores["fixed"])
            effects = np.asarray([np.mean(by_seed[seed]) for seed in sorted(by_seed)])
            center = effects.mean()
            replicates = rng.choice(effects, (10000, len(effects)), replace=True).mean(axis=1)
            lo, hi = np.quantile(replicates, [0.025, 0.975])
            centers.append(center)
            limits.append([max(0., center - lo), max(0., hi - center)])
        ax.errorbar(centers, offsets, xerr=np.asarray(limits).T, color=PALETTE[i],
                    fmt=["o", "s", "D"][i], markersize=4.5, capsize=2.5, linewidth=1.2,
                    label=label(variant, protocol))
    ax.axvline(0, color="#454545", linewidth=1, linestyle="--")
    ax.set_yticks(range(len(cells)), ["Pooled", *(f"{b:,} FE · D = {d}" for b, d in cells[1:])])
    ax.invert_yaxis()
    ax.set_xlabel("Paired change in log₁₀ gap vs fixed step (negative favors adaptation)")
    ax.grid(axis="x", color="#D7DCE2", linewidth=0.6)
    fig.legend(loc="lower center", ncols=3, frameon=False)
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    fig.savefig(path)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage_dir", type=Path, help="Directory screen, ablation, fe, fe_bbob or time")
    parser.add_argument("--output", type=Path, help="Defaults to stage_dir/publication_figures")
    parser.add_argument("--budget", type=float, help="One planned budget, in FE or seconds")
    args = parser.parse_args()
    folder = args.stage_dir
    protocol = load_json(folder / "protocol.json")
    complete = load_json(folder / "complete.json")
    expected = (len(protocol["budgets"]) * len(protocol["cases"]) * len(protocol["dimensions"])
                * len(protocol["seeds"]) * len(protocol["methods"]))
    if complete["runs"] != expected:
        parser.error(f"Expected {expected} completed runs")
    budgets = [b for b in protocol["budgets"] if args.budget is None or b == args.budget]
    if not budgets:
        parser.error("Requested budget is absent from the protocol")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9.5, "pdf.fonttype": 42,
                         "mathtext.fontset": "dejavusans", "axes.spines.top": False,
                         "axes.spines.right": False, "axes.titleweight": "semibold"})
    out = args.output or folder / "publication_figures"
    out.mkdir(parents=True, exist_ok=True)
    ranks, functions, methods = rank_data(folder, protocol)
    stage = protocol["stage"]
    timed = stage == "time"
    chosen = load_json(folder / "selected.json")["method"] if stage in {"screen", "ablation"} else "cpso_selected"
    for budget in budgets:
        suffix = budget_slug(budget, timed)
        heatmap(out / f"{stage}_rank_heatmap_{suffix}.pdf", protocol, budget, ranks, functions, methods)
        if protocol["suite"] == "bbob":
            continue
        curves = collect_traces(folder, protocol, budget, expected // len(protocol["budgets"]))
        plot_methods = methods + (["cpso_one_worker"] if timed else [])
        convergence(out / f"{stage}_convergence_{suffix}.pdf",
                    protocol, budget, curves, functions, plot_methods, chosen)
    if stage == "ablation":
        ablation_effects(out / "ablation_paired_effects.pdf", folder, protocol)
    print(f"Vector PDFs: {out}")


if __name__ == "__main__":
    main()
