"""
Role-level cluster bootstrap for Bias Intensity (BI) shifts across conditions.

Implements the analysis described in the paper's Appendix E.1: BI is a pooled statistic
(computed from label counts pooled across roles, exactly as build_fig2_data.py's
mean_intensity is -- see compute_overall_prop_all_with_loo()'s prop_context), not an
average of independent per-role BI estimates. The cluster bootstrap therefore resamples
*roles* with replacement, and for each replicate re-pools the resampled roles' raw label
counts (duplicating a role's counts every time it's drawn) before recomputing BI once from
that pooled distribution -- matching "recomputing pooled BI within each replicate" in the
paper's own description of the method.

BI is always pooled across roles *within one model* first (this is what fig2_data.csv's
mean_intensity already is: prop_context pools occupation counts per model). Cross-model
numbers (Table 9, the headline figure) are then the mean of the four per-model bootstrap
trajectories, taken replicate-by-replicate so the resulting CI reflects the same
"average of four already-verified per-model BI series" construction fig2_data.csv uses,
rather than re-pooling all four models' raw counts into one undifferentiated pool.

Produces:
  - table10_cluster_bootstrap_per_model.csv: per-model BI differences between each pair of
    conditions, per cohort (paper Table 10).
  - table9_cluster_bootstrap.csv: the same, averaged across the four models (paper Table 9).
  - camera_dimension_cluster_bootstrap.csv: Camera cohort decomposed into
    depth_of_field/framing/perspective, averaged across models.
  - headline_bi_cluster_bootstrap.csv: the all-cohorts, all-models CA-U vs CF shift.

Run after bias_quantification_pipeline.py; reuses its data-loading and BI helpers directly
so the numbers are guaranteed consistent with fig2_data.csv.
"""

import os
import sys

import numpy as np
import pandas as pd

from bias_quantification_pipeline import (
    OUT_DIR,
    PLOT_MODEL_MAP,
    build_overall_prop_all,
    context_shortener,
    dimension_to_family,
    bias_severity_score,
    get_preferred_prompt_group,
)

N_REPLICATES = int(os.environ.get("BIAS_BOOTSTRAP_REPLICATES", "1000"))
RANDOM_SEED = int(os.environ.get("BIAS_BOOTSTRAP_SEED", "0"))
CI_LOW, CI_HIGH = 2.5, 97.5

CAMERA_DIMENSIONS = ["depth_of_field", "framing", "perspective"]
CONDITION_PAIRS = [("CA-U", "CF"), ("CA-R", "CF"), ("CA-U", "CA-R")]
CONDITIONS = ["CF", "CA-R", "CA-U"]


def build_role_label_counts(overall_prop_all: pd.DataFrame, fig_prompt_group: str) -> pd.DataFrame:
    """One row per (Model_plot, Context_short, dimension_family, dimension, cohort,
    occupation, label): the raw label count for that role, ready to be pooled (summed)
    across a set of roles -- either all of them (point estimate) or a bootstrap resample."""

    d = overall_prop_all[
        (overall_prop_all["prompt_group"] == fig_prompt_group)
        & (~overall_prop_all["cohort"].isin(["totals", "safety"]))
        & overall_prop_all["label"].notna()
        & (overall_prop_all["label"] != "unknown")
    ].copy()

    d["Model_plot"] = d["Model"].map(PLOT_MODEL_MAP).fillna(d["Model"])
    d["Context_short"] = context_shortener(d["Context"])
    d = d[d["Context_short"].isin(["CF", "CA-R", "CA-U"])]
    d["dimension_family"] = dimension_to_family(d["cohort"])

    return d[
        ["Model_plot", "Context_short", "dimension_family", "dimension", "cohort", "occupation", "label", "count"]
    ].drop_duplicates(subset=["Model_plot", "Context_short", "dimension", "occupation", "label"])


def pooled_bi_for_weights(counts_df: pd.DataFrame, dim_col: str, role_weights: pd.Series) -> pd.Series:
    """Sum `count` weighted by role_weights (an occupation -> multiplicity Series) within
    each dim_col group, per label, then compute BI once from the pooled per-label totals.
    Returns a Series indexed by dim_col value. counts_df must already be filtered to one
    model and one context.
    """
    weighted = counts_df.assign(weighted_count=counts_df["count"] * counts_df["occupation"].map(role_weights).fillna(0))
    weighted = weighted[weighted["weighted_count"] > 0]
    if weighted.empty:
        return pd.Series(dtype=float)

    pooled = weighted.groupby([dim_col, "label"], as_index=False, sort=False)["weighted_count"].sum()
    return pooled.groupby(dim_col, sort=False)["weighted_count"].agg(lambda s: bias_severity_score(s.to_numpy()))


def per_model_dimension_trajectories(
    counts_df: pd.DataFrame, model: str, roles: np.ndarray, rng: np.random.Generator
) -> dict:
    """{context: {dimension: array of length N_REPLICATES+1}} for one model, pooling roles
    *within a single dimension's own label space* (never mixing label spaces across
    dimensions -- see module docstring). Index 0 of each array is the point estimate (all
    roles, weight 1); indices 1..N are the bootstrap replicates.
    """
    model_counts = counts_df[counts_df["Model_plot"] == model]
    n_roles = len(roles)
    out = {ctx: {} for ctx in CONDITIONS}

    for ctx in CONDITIONS:
        sub = model_counts[model_counts["Context_short"] == ctx]
        point = pooled_bi_for_weights(sub, "dimension", pd.Series(1.0, index=roles))
        for dim, value in point.items():
            out[ctx][dim] = np.full(N_REPLICATES + 1, np.nan)
            out[ctx][dim][0] = value

        for rep in range(N_REPLICATES):
            draw = rng.integers(0, n_roles, size=n_roles)
            weights = pd.Series(np.bincount(draw, minlength=n_roles), index=roles).astype(float)
            bi = pooled_bi_for_weights(sub, "dimension", weights)
            for dim, value in bi.items():
                if dim not in out[ctx]:
                    out[ctx][dim] = np.full(N_REPLICATES + 1, np.nan)
                out[ctx][dim][rep + 1] = value

    return out


def average_trajectories(trajectory_sets: list[dict]) -> dict:
    """Average a list of {context: {key: array}} dicts elementwise (replicate-by-replicate),
    keeping only keys present in every set. Used both to average dimensions up to a family
    and to average per-model results across models -- same operation either way."""
    out = {ctx: {} for ctx in CONDITIONS}
    all_keys = set()
    for ts in trajectory_sets:
        for ctx in CONDITIONS:
            all_keys.update(ts[ctx].keys())

    for ctx in CONDITIONS:
        for key in all_keys:
            arrays = [ts[ctx][key] for ts in trajectory_sets if key in ts[ctx]]
            if len(arrays) != len(trajectory_sets):
                continue
            out[ctx][key] = np.nanmean(np.vstack(arrays), axis=0)
    return out


def family_trajectories(dim_trajectories: dict, dim_to_family: dict[str, str]) -> dict:
    """{context: {dimension: array}} -> {context: {family: array}}, averaging the
    dimension-level trajectories that belong to each family (mean of per-dimension BI,
    matching fig2_data.csv's own aggregation -- never a raw-count pool across dimensions)."""
    families = sorted(set(dim_to_family.values()))
    out = {ctx: {} for ctx in CONDITIONS}
    for ctx in CONDITIONS:
        for family in families:
            dims_in_family = [d for d, f in dim_to_family.items() if f == family and d in dim_trajectories[ctx]]
            if not dims_in_family:
                continue
            arrays = [dim_trajectories[ctx][d] for d in dims_in_family]
            out[ctx][family] = np.nanmean(np.vstack(arrays), axis=0)
    return out


def diffs_table(trajectories: dict, extra_cols: dict | None = None) -> pd.DataFrame:
    """trajectories: {context: {dim_value: array}}. Row 0 of each array is the point
    estimate, rows 1: are bootstrap replicates."""
    extra_cols = extra_cols or {}
    dim_values = sorted(set().union(*[set(trajectories[ctx].keys()) for ctx in CONDITIONS]))

    rows = []
    for dim_value in dim_values:
        for hi, lo in CONDITION_PAIRS:
            if dim_value not in trajectories[hi] or dim_value not in trajectories[lo]:
                continue
            arr_hi, arr_lo = trajectories[hi][dim_value], trajectories[lo][dim_value]
            diffs_all = arr_hi - arr_lo
            point_diff = diffs_all[0]
            boot_diffs = diffs_all[1:]
            boot_diffs = boot_diffs[np.isfinite(boot_diffs)]
            if len(boot_diffs) == 0:
                continue
            rows.append(
                {
                    **extra_cols,
                    "dimension_family": dim_value,
                    "comparison": f"{hi} - {lo}",
                    "point_estimate": float(point_diff),
                    "bootstrap_mean": float(np.mean(boot_diffs)),
                    "ci_low": float(np.percentile(boot_diffs, CI_LOW)),
                    "ci_high": float(np.percentile(boot_diffs, CI_HIGH)),
                    "n_replicates": len(boot_diffs),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    rng = np.random.default_rng(RANDOM_SEED)

    overall_prop_all = build_overall_prop_all()
    fig_prompt_group = get_preferred_prompt_group(overall_prop_all)
    print(f"preferred_prompt_group: {fig_prompt_group}", file=sys.stderr)

    counts = build_role_label_counts(overall_prop_all, fig_prompt_group)
    roles = np.sort(counts["occupation"].unique())
    models = sorted(counts["Model_plot"].unique())
    print(f"role_label_counts: {len(counts)} rows across {len(roles)} roles, models={models}", file=sys.stderr)

    dim_to_family = counts.drop_duplicates("dimension").set_index("dimension")["dimension_family"].to_dict()

    # One dimension-level bootstrap pass per model; every other table is a mean of these
    # trajectories (up to family, and/or across models), never a re-pool of raw counts.
    per_model_dims = {m: per_model_dimension_trajectories(counts, m, roles, rng) for m in models}

    # --- Table 10: per-model, cohort (family) level ---
    per_model_families = {m: family_trajectories(per_model_dims[m], dim_to_family) for m in models}
    table10_rows = [diffs_table(per_model_families[m], extra_cols={"Model_plot": m}) for m in models]
    table10 = pd.concat(table10_rows, ignore_index=True)
    table10.to_csv(os.path.join(OUT_DIR, "table10_cluster_bootstrap_per_model.csv"), index=False)
    print(f"table10_cluster_bootstrap_per_model: {len(table10)} rows", file=sys.stderr)

    # --- Table 9: cohort level, averaged across models ---
    pooled_families = average_trajectories(list(per_model_families.values()))
    table9 = diffs_table(pooled_families)
    table9.to_csv(os.path.join(OUT_DIR, "table9_cluster_bootstrap.csv"), index=False)
    print(f"table9_cluster_bootstrap:\n{table9.to_string()}", file=sys.stderr)

    # --- Camera dimension-level decomposition, averaged across models ---
    pooled_dims = average_trajectories(list(per_model_dims.values()))
    camera_dims_only = {ctx: {d: v for d, v in pooled_dims[ctx].items() if d in CAMERA_DIMENSIONS} for ctx in CONDITIONS}
    camera_table = diffs_table(camera_dims_only)
    camera_table = camera_table.rename(columns={"dimension_family": "dimension"})
    camera_table.to_csv(os.path.join(OUT_DIR, "camera_dimension_cluster_bootstrap.csv"), index=False)
    print(f"camera_dimension_cluster_bootstrap:\n{camera_table.to_string()}", file=sys.stderr)

    # --- headline: mean over the four cohorts (equal weight per family, matching how
    # fig2_data.csv's own mean_intensity is averaged), already averaged across models ---
    headline_ready = {ctx: {"ALL": np.nanmean(np.vstack(list(pooled_families[ctx].values())), axis=0)} for ctx in CONDITIONS}
    headline = diffs_table(headline_ready)
    headline = headline[headline["comparison"] == "CA-U - CF"]
    if not headline.empty:
        row = headline.iloc[0]
        print(
            f"Headline (CA-U - CF, all cohorts, averaged across models): "
            f"point={row['point_estimate']:.3f}, bootstrap_mean={row['bootstrap_mean']:.3f}, "
            f"95% CI=[{row['ci_low']:.3f}, {row['ci_high']:.3f}]",
            file=sys.stderr,
        )
    headline.to_csv(os.path.join(OUT_DIR, "headline_bi_cluster_bootstrap.csv"), index=False)

    print("DONE", file=sys.stderr)


if __name__ == "__main__":
    main()
