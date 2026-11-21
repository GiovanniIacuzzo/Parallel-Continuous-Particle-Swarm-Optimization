"""Paired comparisons, censored target statistics and vector figures."""
from __future__ import annotations

import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest

from .common import csv, freeze, holm
from .study3_execution import BASELINES, REFERENCE, screening_recipes

GAP_FLOOR = 1e-12
PRACTICAL_TIE = 0.01  # log10 units
ADAPTATION_GAIN = 0.05
TARGETS = (1e2, 1.0, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-8)
BLOCK = ["budget", "function", "dimension", "instance", "seed"]
CELL = BLOCK[:3]


def gap(value, optimum):
    return max(GAP_FLOOR, value - optimum)


def _measurements(protocol, records):
    rows = []
    for key, result in records:
        row = key | {k: v for k, v in result.items() if k != "trace"}
        row["gap"] = gap(result["value"], result["optimum"]) if result["fe"] else np.nan
        row["log_gap"] = math.log10(row["gap"]) if result["fe"] else np.nan
        row["workers"] = protocol["methods"][key["method"]]["workers"]
        row["seconds_overrun"] = max(0.0, result["elapsed"] - key["budget"]) if protocol["stage"] == "time" else 0
        rows.append(row)
    return pd.DataFrame(rows).sort_values(BLOCK + ["method"])


def _rank(frame):
    ranked = frame.copy()
    # Use a common floor for near-optimal runs and a transitive numerical tie rule.
    ranked["rank"] = ranked["log_gap"].round(8).fillna(np.inf).groupby(
        [ranked[column] for column in BLOCK]
    ).rank(method="average")
    return ranked


def _decision(stage, ranked, protocol):
    rank_by_method = ranked.groupby("method")["rank"].mean()
    if stage == "screen":
        reference = next(name for name, recipe in screening_recipes() if recipe == REFERENCE)
        picked = min(rank_by_method.index, key=lambda name: (
            rank_by_method[name], name != reference, name
        ))
        return dict(method=picked, recipe=protocol["methods"][picked]["recipe"],
                    mean_rank=float(rank_by_method[picked]), criterion="minimum mean block rank")
    logs = ranked.pivot(index=BLOCK, columns="method", values="log_gap")
    effects = (logs.sub(logs.fixed, axis=0)).mean()
    picked = min(effects.index, key=lambda name: (effects[name], name.count("_"), name))
    if effects[picked] > -ADAPTATION_GAIN:
        picked = "fixed"
    return dict(method=picked, recipe=protocol["methods"][picked]["recipe"],
                mean_log10_effect_vs_fixed=float(effects[picked]),
                minimum_gain_log10=ADAPTATION_GAIN)


def _paired(frame, reference, opponents, suite, timing_comparable, out):
    rows = []
    rng = np.random.default_rng(73845)
    for cell, group in frame.groupby(CELL):
        left = group[group.method == reference].set_index(["instance", "seed"])
        for opponent in opponents:
            right = group[group.method == opponent].set_index(["instance", "seed"])
            joined = left.join(right, lsuffix="_a", rsuffix="_b", how="inner")
            effect = joined.log_gap_a - joined.log_gap_b
            labels = np.sign(effect.where(effect.abs() > PRACTICAL_TIE, 0.0))
            a_missing = joined.log_gap_a.isna()
            b_missing = joined.log_gap_b.isna()
            labels = labels.mask(a_missing & ~b_missing, 1).mask(~a_missing & b_missing, -1)
            unit = "instance" if suite == "bbob" else "seed"
            per_unit = effect.groupby(level=unit).mean().dropna().to_numpy()
            lo = hi = None
            if len(per_unit) >= 2:
                sample = rng.choice(per_unit, size=(2000, len(per_unit)), replace=True)
                lo, hi = np.quantile(sample.mean(axis=1), [0.025, 0.975])
            wins = int((labels < 0).sum())
            losses = int((labels > 0).sum())
            paired_effect = float(per_unit.mean()) if len(per_unit) else None
            vote = np.sign(paired_effect) if paired_effect is not None and abs(paired_effect) > PRACTICAL_TIE else 0
            if a_missing.sum() != b_missing.sum():
                vote = int(np.sign(int(a_missing.sum()) - int(b_missing.sum())))
            rows.append(dict(zip(CELL, cell)) | dict(
                reference=reference, opponent=opponent, paired_runs=len(joined),
                wins=wins, ties=len(joined) - wins - losses, losses=losses,
                reference_no_result=int(a_missing.sum()), opponent_no_result=int(b_missing.sum()),
                function_vote=int(vote), effect_log10=paired_effect, ci_low=lo, ci_high=hi,
                seconds_ratio_opponent_over_reference=(
                    float(np.exp(np.log(joined.elapsed_b / joined.elapsed_a).mean()))
                    if timing_comparable else None
                ),
                median_fe_reference=float(joined.fe_a.median()),
                median_fe_opponent=float(joined.fe_b.median()),
            ))
    paired = pd.DataFrame(rows)
    paired.to_csv(out / f"paired_{reference}.csv", index=False)
    counts = []
    for (budget, dimension, opponent), group in paired.groupby(["budget", "dimension", "opponent"]):
        votes = group.function_vote
        w, l = int((votes < 0).sum()), int((votes > 0).sum())
        counts.append(dict(budget=budget, dimension=dimension, reference=reference, opponent=opponent,
                           wins=w, losses=l, ties=len(votes) - w - l,
                           p_sign=binomtest(w, w + l).pvalue if w + l else 1.0))
    summary = pd.DataFrame(counts)
    summary["p_holm"] = holm(summary.p_sign)
    summary.to_csv(out / f"function_wins_{reference}.csv", index=False)


def _against_best_baseline(frame, out):
    rows = []
    for cell, group in frame.groupby(CELL):
        scores = group.groupby("method").log_gap.mean()
        missing = group.groupby("method").log_gap.apply(lambda g: int(g.isna().sum()))
        contenders = sorted(set(scores.index) & set(BASELINES))
        best = min(contenders, key=lambda name: (missing[name], scores[name] if pd.notna(scores[name]) else math.inf, name))
        delta = scores["cpso_selected"] - scores[best]
        no_result_diff = int(missing["cpso_selected"]) - int(missing[best])
        vote = int(np.sign(no_result_diff or (delta if abs(delta) > PRACTICAL_TIE else 0)))
        rows.append(dict(zip(CELL, cell)) | dict(
            best_baseline=best, effect_log10=delta, vote=vote,
            cpso_no_result=missing["cpso_selected"], baseline_no_result=missing[best],
        ))
    frame = pd.DataFrame(rows)
    frame.to_csv(out / "best_baseline_by_function.csv", index=False)
    counts = (frame.groupby(["budget", "dimension"]).vote.value_counts().unstack(fill_value=0)
              .reindex(columns=[-1, 0, 1], fill_value=0))
    counts.columns = ["wins", "ties", "losses"]
    counts.to_csv(out / "best_baseline_function_wins.csv")


def _parallel_prefix(records, out, selected_workers):
    pairs = {}
    for key, result in records:
        if key["method"] in {"cpso_selected", "cpso_one_worker"}:
            block = tuple(key[column] for column in BLOCK)
            pairs.setdefault(block, {})[key["method"]] = result
    rows = []
    for block, pair in sorted(pairs.items()):
        one, many = pair["cpso_one_worker"], pair["cpso_selected"]
        common_fe = min(one["fe"], many["fe"])

        def improvements(result):
            trace = result["trace"]
            return [(fe, value) for i, (fe, _, value) in enumerate(trace)
                    if fe <= common_fe and (i == 0 or value < trace[i - 1][2])]

        equal = improvements(one) == improvements(many)
        rows.append(dict(zip(BLOCK, block)) | dict(
            common_fe=common_fe, exact_best_so_far_prefix=equal,
            fe_one_worker=one["fe"], fe_selected_workers=many["fe"],
            selected_workers=selected_workers,
        ))
        if not equal:
            raise RuntimeError(f"Synchronous best-so-far FE prefixes disagree: {block}")
    csv(out / "parallel_prefix_checks.csv", rows)


def _target_rows(protocol, records):
    time_axis = protocol["stage"] == "time"
    for key, result in records:
        trace = result["trace"]
        for target in TARGETS:
            hit = next(((fe, seconds) for fe, seconds, value in trace
                        if value - result["optimum"] <= target), None)
            yield key | dict(target=target, hit_fe=hit[0] if hit else None,
                             hit_seconds=hit[1] if hit and protocol["suite"] == "native" else None,
                             censor_fe=result["fe"],
                             censor_seconds=key["budget"] if time_axis else result["elapsed"])


def _ert(targets, out, timing_comparable):
    summary = []
    for cell, group in targets.groupby(CELL + ["method", "target"]):
        successes = int(group.hit_fe.notna().sum())
        ert_fe = group.hit_fe.fillna(group.censor_fe).sum() / successes if successes else None
        ert_seconds = None
        if timing_comparable and successes:
            ert_seconds = group.hit_seconds.fillna(group.censor_seconds).sum() / successes
        summary.append(dict(zip(CELL + ["method", "target"], cell)) | dict(
            successes=successes, runs=len(group), ert_fe=ert_fe, ert_seconds=ert_seconds
        ))
    csv(out / "ert.csv", summary)


def _label(method, protocol):
    if method == "cpso_selected":
        return f"CPSO ({protocol['methods'][method]['workers']} workers)"
    if method == "cpso_one_worker":
        return "CPSO (1 worker)"
    return method.replace("_", " ")


def _plots(folder, protocol, records, ranked, targets):
    out = folder / "figures"
    out.mkdir(exist_ok=True)
    stage = protocol["stage"]
    if stage in {"screen", "ablation"}:
        ranks = ranked.groupby(["budget", "method"])["rank"].mean().unstack("budget")
        ranks = ranks.loc[ranks.mean(axis=1).sort_values().index]
        fig, ax = plt.subplots(figsize=(7.0, 3.9))
        for budget, column in ranks.items():
            ax.scatter(column, ranks.index, label=f"{budget:,} FE")
        ax.invert_yaxis()
        ax.set_xlabel("Mean paired rank")
        ax.legend(frameon=False)
        ax.grid(axis="x", alpha=0.25)
        fig.savefig(out / f"{stage}_ranks.pdf", bbox_inches="tight")
        plt.close(fig)
        return

    timed = stage == "time"
    x_axis = 1 if timed else 0
    axis_name = "seconds" if timed else "fe"
    max_budget = max(protocol["budgets"])
    # Keep the convergence panels legible. All methods remain available in CSV and ECDF.
    shown = ["cpso_selected", "cpso_one_worker"] if timed else ["cpso_selected"]
    shown += [m for m in ("spso2011", "cso", "clpso", "slpso") if m in protocol["methods"]]
    cmap = dict(zip(shown, plt.get_cmap("tab10").colors))
    for dimension in protocol["dimensions"]:
        functions = [function for function, _ in protocol["cases"]]
        functions = list(dict.fromkeys(functions))
        if protocol["suite"] == "native":
            cols = min(3, len(functions))
            rows = (len(functions) + cols - 1) // cols
            fig, axes = plt.subplots(rows, cols, figsize=(3.4 * cols, 3 * rows + 0.55),
                                     squeeze=False)
            x = np.geomspace(max_budget / 400, max_budget, 120) if timed else np.geomspace(100, max_budget, 120)
            for function, ax in zip(functions, axes.flat):
                for method in shown:
                    curves = []
                    for key, result in records:
                        if (key["function"], key["dimension"], key["budget"], key["method"]) != (
                            function, dimension, max_budget, method
                        ):
                            continue
                        trace = result["trace"]
                        y = np.full(len(x), np.nan)
                        if trace:
                            idx = np.searchsorted([p[x_axis] for p in trace], x, side="right") - 1
                            valid = idx >= 0
                            y[valid] = [gap(trace[i][2], result["optimum"]) for i in idx[valid]]
                        curves.append(y)
                    if curves:
                        values = np.ma.masked_invalid(curves)
                        enough = np.ma.count(values, axis=0) >= (len(curves) // 2 + 1)
                        median = np.ma.median(values, axis=0).filled(np.nan)
                        ax.plot(x[enough], median[enough], color=cmap[method], label=_label(method, protocol))
                ax.set(xscale="log", yscale="log", title=function.replace("_", " ").capitalize(),
                       xlabel="Wall-clock time (s)" if timed else "Function evaluations",
                       ylabel="Median objective gap")
                ax.grid(alpha=0.2)
            for ax in list(axes.flat)[len(functions):]:
                ax.set_visible(False)
            handles, labels = axes[0, 0].get_legend_handles_labels()
            if handles:
                fig.legend(handles, labels, loc="lower center", ncols=min(3, len(handles)), frameon=False)
            fig.tight_layout(rect=(0, 0.15, 1, 1))
            fig.savefig(out / f"convergence_{axis_name}_d{dimension}.pdf", bbox_inches="tight")
            plt.close(fig)

        fig, ax = plt.subplots(figsize=(7.3, 4.5))
        part = targets[(targets.dimension == dimension) & (targets.budget == max_budget)]
        for method, group in part.groupby("method"):
            hit = np.sort(group[f"hit_{axis_name}"].dropna().to_numpy(dtype=float))
            x_start = max_budget / 1000 if timed else 1
            if len(hit):
                xx = np.r_[min(x_start, hit[0]), hit, max_budget]
                yy = np.r_[0, np.arange(1, len(hit) + 1) / len(group), len(hit) / len(group)]
            else:
                xx, yy = [x_start, max_budget], [0, 0]
            ax.step(xx, yy, where="post", label=_label(method, protocol))
        ax.set(xscale="log", xlabel="Wall-clock time (s)" if timed else "Function evaluations",
               ylabel="Fraction of run–target pairs reached", ylim=(-0.015, 1.015))
        ax.grid(alpha=0.2)
        ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), frameon=False, fontsize=8)
        fig.savefig(out / f"target_ecdf_{axis_name}_d{dimension}.pdf", bbox_inches="tight")
        plt.close(fig)


def analyze(folder, protocol, records):
    stage = protocol["stage"]
    out = folder / "analysis"
    out.mkdir(exist_ok=True)
    frame = _measurements(protocol, records)
    primary = "cpso_selected"
    ranked = _rank(frame[frame.method != "cpso_one_worker"])
    frame.to_csv(out / "measurements.csv", index=False)
    ranked.to_csv(out / "ranked_results.csv", index=False)
    rank_summary = ranked.groupby(CELL + ["method"], as_index=False).agg(
        mean_rank=("rank", "mean"), median_gap=("gap", "median"),
        median_elapsed_seconds=("elapsed", "median"), median_fe=("fe", "median"),
    )
    rank_summary.to_csv(out / "rank_summary.csv", index=False)
    leaderboard = rank_summary.groupby(["budget", "dimension", "method"], as_index=False).agg(
        mean_rank=("mean_rank", "mean"), functions=("function", "size")
    )
    leaderboard.sort_values(["budget", "dimension", "mean_rank"]).to_csv(
        out / "leaderboard.csv", index=False
    )

    winner = None
    if stage in {"screen", "ablation"}:
        winner = _decision(stage, ranked, protocol)
        primary = ("fixed" if stage == "ablation" else
                   next(name for name, recipe in screening_recipes() if recipe == REFERENCE))

    timing_comparable = protocol["suite"] == "native"
    _paired(frame, primary, [m for m in protocol["methods"] if m != primary],
            protocol["suite"], timing_comparable and stage != "time", out)
    if stage in {"fe", "time"}:
        _against_best_baseline(frame, out)
    if stage == "fe":
        for control in ("cpso_matched_fixed", "cpso_fixed_reference"):
            if control in protocol["methods"] and (
                protocol["methods"][control]["recipe"]
                == protocol["methods"]["pso_matched_discrete"]["recipe"]
            ):
                _paired(frame, control, ["pso_matched_discrete"], protocol["suite"],
                        timing_comparable, out)
    targets = pd.DataFrame(_target_rows(protocol, records))
    targets.to_csv(out / "target_hits.csv", index=False)
    _ert(targets, out, timing_comparable)

    if stage == "time":
        _parallel_prefix(records, out, protocol["methods"]["cpso_selected"]["workers"])
    _plots(folder, protocol, records, ranked, targets)
    if winner is not None:
        freeze(folder / "selected.json", winner)
