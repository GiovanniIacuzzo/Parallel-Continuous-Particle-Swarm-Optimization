"""Results by original benchmark function and selection on development data."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .study3_analysis import PRACTICAL_TIE, _measurements, _parallel_prefix, _plots, _target_rows


def _function_scores(frame):
    # Give equal weight to seeds within instance, instances within variant,
    # and variants/dimensions within function.
    keys = ["budget", "function", "dimension", "variant", "instance", "method"]
    cells = frame.groupby(keys, as_index=False).agg(
        mean_log_gap=("log_gap", "mean"), mean_rank=("rank", "mean"),
        no_result_fraction=("no_result", "mean"),
    )
    metrics = ["mean_log_gap", "mean_rank", "no_result_fraction"]
    by_variant = cells.groupby(keys[:4] + ["method"], as_index=False)[metrics].mean()
    by_dimension = by_variant.groupby(keys[:3] + ["method"], as_index=False)[metrics].mean()
    by_function = by_dimension.groupby(keys[:2] + ["method"], as_index=False)[metrics].mean()
    return by_variant, by_dimension, by_function


def _outcomes(scores):
    rows = []
    for cell, group in scores.groupby(["budget", "function"]):
        indexed = group.set_index("method")
        candidate = indexed.loc["cpso_selected"]
        rivals = indexed.drop(index="cpso_selected")

        def score(row):
            return (row.no_result_fraction,
                    float(row.mean_log_gap) if pd.notna(row.mean_log_gap) else math.inf)

        best = min(rivals.index, key=lambda method: (*score(rivals.loc[method]), method))
        cand_missing, cand_gap = score(candidate)
        best_missing, best_gap = score(rivals.loc[best])
        delta = cand_gap - best_gap if math.isfinite(cand_gap) and math.isfinite(best_gap) else None
        if cand_missing != best_missing:
            outcome = "first_alone" if cand_missing < best_missing else "loss"
        elif delta is None:
            outcome = "no_result"
        elif abs(delta) <= PRACTICAL_TIE:
            outcome = "tied_first"
        else:
            outcome = "first_alone" if delta < 0.0 else "loss"
        rows.append(dict(budget=cell[0], function=cell[1], best_baseline=best,
                         delta_log_gap=delta, outcome=outcome,
                         cpso_no_result_fraction=cand_missing,
                         baseline_no_result_fraction=best_missing))
    return pd.DataFrame(rows)


def summarize(folder, protocol, records):
    out = folder / "analysis"
    out.mkdir(exist_ok=True)
    frame = _measurements(protocol, records)
    frame.to_csv(out / "measurements.csv", index=False)
    competitive = frame[~frame.method.isin(("cpso_one_worker", "cpso_control_100_50"))].copy()
    if "variant" not in competitive:
        competitive["variant"] = "official" if protocol["suite"] in {"bbob", "cec"} else "original"
    else:
        competitive["variant"] = competitive["variant"].fillna("original")
    blocks = ["budget", "function", "dimension", "variant", "instance", "seed"]
    if (competitive.duplicated(blocks + ["method"]).any()
            or not competitive.groupby(blocks).size().eq(competitive.method.nunique()).all()):
        raise RuntimeError("Incomplete paired competition results")
    competitive["no_result"] = competitive.log_gap.isna()
    competitive["rank"] = competitive.log_gap.round(8).fillna(np.inf).groupby(
        [competitive[column] for column in blocks]
    ).rank(method="average")
    by_variant, by_dimension, by_function = _function_scores(competitive)
    by_variant.to_csv(out / "scores_by_variant.csv", index=False)
    by_dimension.to_csv(out / "scores_by_dimension.csv", index=False)
    by_function.to_csv(out / "scores_by_function.csv", index=False)
    leaderboard = (by_function.groupby(["budget", "method"], as_index=False)
                   .agg(mean_rank=("mean_rank", "mean"), functions=("function", "size"))
                   .sort_values(["budget", "mean_rank", "method"]))
    leaderboard.to_csv(out / "leaderboard.csv", index=False)
    stage = protocol["stage"]
    if stage == "refine":
        targets = pd.DataFrame(_target_rows(protocol, records))
        targets.drop(columns=["hit_seconds", "censor_seconds"]).to_csv(out / "target_hits.csv", index=False)
        largest_budget = leaderboard.budget.max()
        final = leaderboard[leaderboard.budget == largest_budget]
        best = final[final.mean_rank == final.mean_rank.min()]
        preference = ("cpso_100_50", "cpso_local", "cpso_restart", "cpso_local_restart")
        candidate = next(method for method in preference if method in set(best.method))
        return dict(candidate=candidate, mean_rank=float(best.mean_rank.min()),
                    budget=int(largest_budget), criterion="minimum mean rank at the final checkpoint")
    if stage == "screen":
        ranks = leaderboard.groupby("method").mean_rank.mean()
        candidate = min((name for name in ranks.index if name.startswith("cpso_")),
                        key=lambda name: (ranks[name], name != "cpso_100_100", name))
        return dict(candidate=candidate, mean_rank=float(ranks[candidate]),
                    criterion="minimum equally weighted mean rank among CPSO candidates")

    if "cpso_control_100_50" in protocol["methods"]:
        keys = ["budget", "function", "dimension", "instance", "seed"]
        if "variant" in frame:
            keys.append("variant")
        control = frame[frame.method == "cpso_control_100_50"].set_index(keys)
        chosen = frame[frame.method == "cpso_selected"].set_index(keys)
        paired = chosen[["log_gap", "fe"]].join(control[["log_gap", "fe"]],
                                                  rsuffix="_control", validate="one_to_one")
        paired["delta_log_gap"] = paired.log_gap - paired.log_gap_control
        paired.to_csv(out / "paired_vs_100_50.csv")

    outcomes = _outcomes(by_function)
    outcomes.to_csv(out / "outcomes_by_function.csv", index=False)
    counts = (outcomes.groupby("budget").outcome.value_counts().unstack(fill_value=0)
              .reindex(columns=["first_alone", "tied_first", "loss"], fill_value=0))
    if outcomes.outcome.eq("no_result").any():
        counts["no_result"] = outcomes.groupby("budget").outcome.agg(
            lambda group: group.eq("no_result").sum()
        )
    counts.to_csv(out / "outcome_counts.csv")
    timed = stage == "time"
    targets = pd.DataFrame(_target_rows(protocol, records))
    if not timed:
        targets = targets.drop(columns=["hit_seconds", "censor_seconds"])
    targets.to_csv(out / "target_hits.csv", index=False)
    if timed:
        if "cpso_one_worker" in protocol["methods"]:
            _parallel_prefix(records, out, protocol["methods"]["cpso_selected"]["workers"])
        (targets[targets.target == 1e-8].groupby(["budget", "dimension", "method"], as_index=False)
         .agg(runs=("hit_seconds", "size"), successes=("hit_seconds", "count"),
              median_hit_seconds=("hit_seconds", "median"), median_hit_fe=("hit_fe", "median"))
         .to_csv(out / "time_to_target.csv", index=False))
    if timed or (protocol.get("recipe_source") or {}).get("preset"):
        plot_protocol = protocol | dict(dimensions=sorted(frame.dimension.unique()),
                                        cases=[(case["function"], 0) for case in protocol["cases"]])
        _plots(folder, plot_protocol, records, competitive, targets)
    return None
