"""
Python port of tb3.R -- the richer Table 3 (label persistence) computation.

Run this after bias_quantification_pipeline.py: it reads that script's
occ_table_all.csv output and reuses its helper functions (safe_div_na,
chisq_3ctx, fmt_p, MODEL_KEY_MAP). Unlike bias_quantification_pipeline.py's
own table3_label_persistence.csv, this version adds per-model consistency
flags and a cross-context pp-range column, writing:
    table3_base_long.csv
    table3_model_summary.csv
    table3_label_persistence_wide.csv
    table3_top20_most_consistent_labels.csv
"""

import os
import numpy as np
import pandas as pd

from bias_quantification_pipeline import (
    OUT_DIR,
    MODEL_KEY_MAP,
    chisq_3ctx,
    safe_div_na,
    fmt_p,
)

ALPHA = 0.05
MAX_PCT_RANGE = 5
MIN_TOTAL_LABEL_COUNT = 5
TABLE3_N_ROWS = 20000
TABLE3_MODELS = ["XL", "M35", "Flx", "Qwen"]


def get_preferred_prompt_group_from_occ_table(occ_table_all: pd.DataFrame) -> str:
    uniq = occ_table_all["prompt_group"].dropna().unique().tolist()
    if "ALL_TYPES" in uniq:
        return "ALL_TYPES"
    return uniq[0]


def build_table3_base_long(occ_table_all: pd.DataFrame, preferred_prompt_group: str) -> pd.DataFrame:
    d = occ_table_all[
        (occ_table_all["Model"] != "ALL")
        & (occ_table_all["prompt_group"] == preferred_prompt_group)
        & (~occ_table_all["cohort"].isin(["totals", "safety"]))
        & occ_table_all["label"].notna()
        & (occ_table_all["label"] != "unknown")
    ].copy()

    d["Model_key"] = d["Model"].map(MODEL_KEY_MAP).fillna(d["Model"])
    d = d[d["Model_key"].isin(TABLE3_MODELS)]

    key = ["Model_key", "occupation", "label"]
    g = d.groupby(key, as_index=False, sort=True).agg(
        x_CF=("x_CF", "sum"), n_CF=("n_CF", "sum"),
        x_CA_Unrel=("x_CA_Unrel", "sum"), n_CA_Unrel=("n_CA_Unrel", "sum"),
        x_CA_Rel=("x_CA_Rel", "sum"), n_CA_Rel=("n_CA_Rel", "sum"),
    )

    g["pct_CF"] = 100 * safe_div_na(g["x_CF"].to_numpy(), g["n_CF"].to_numpy())
    g["pct_CA_Unrel"] = 100 * safe_div_na(g["x_CA_Unrel"].to_numpy(), g["n_CA_Unrel"].to_numpy())
    g["pct_CA_Rel"] = 100 * safe_div_na(g["x_CA_Rel"].to_numpy(), g["n_CA_Rel"].to_numpy())

    g["x_all"] = g["x_CF"] + g["x_CA_Unrel"] + g["x_CA_Rel"]
    g["n_all"] = g["n_CF"] + g["n_CA_Unrel"] + g["n_CA_Rel"]
    g["pct_all"] = 100 * safe_div_na(g["x_all"].to_numpy(), g["n_all"].to_numpy())

    pct_cols = g[["pct_CF", "pct_CA_Unrel", "pct_CA_Rel"]].to_numpy()
    g["pct_min"] = np.nanmin(pct_cols, axis=1)
    g["pct_max"] = np.nanmax(pct_cols, axis=1)
    g["pct_range"] = g["pct_max"] - g["pct_min"]

    g["present_all_three_contexts"] = (g["x_CF"] > 0) & (g["x_CA_Unrel"] > 0) & (g["x_CA_Rel"] > 0)
    g["enough_label_count"] = g["x_all"] >= MIN_TOTAL_LABEL_COUNT

    test = chisq_3ctx(g["x_CF"], g["n_CF"], g["x_CA_Unrel"], g["n_CA_Unrel"], g["x_CA_Rel"], g["n_CA_Rel"])
    g = pd.concat([g.reset_index(drop=True), test.reset_index(drop=True)], axis=1)

    g["Model_key"] = pd.Categorical(g["Model_key"], categories=TABLE3_MODELS, ordered=True)

    g["consistent_between_contexts"] = (
        g["enough_label_count"] & g["p_value"].notna() & (g["p_value"] > ALPHA) & (g["pct_range"] <= MAX_PCT_RANGE)
    )

    conditions = [
        g["consistent_between_contexts"] & g["present_all_three_contexts"],
        g["consistent_between_contexts"] & ~g["present_all_three_contexts"],
        ~g["consistent_between_contexts"],
    ]
    choices = [
        "Consistent and present in all 3 contexts",
        "Consistent proportions but not present in all 3 contexts",
        "Context-sensitive / inconsistent",
    ]
    g["consistency_label"] = np.select(conditions, choices, default="Unclear")

    g["consistency_score"] = np.where(
        g["present_all_three_contexts"] & g["enough_label_count"],
        g["pct_all"] / (1 + g["pct_range"]),
        np.nan,
    )

    g["pct_round"] = np.round(g["pct_all"].to_numpy())
    g["range_round"] = np.round(g["pct_range"].to_numpy(), 1)
    g["chi_round"] = np.round(g["chisq_stat"].to_numpy(), 2)
    g["p_print"] = fmt_p(g["p_value"].to_numpy())

    return g


def build_table3_model_summary(base_long: pd.DataFrame) -> pd.DataFrame:
    g = base_long.groupby("Model_key", as_index=False, sort=True, observed=True).agg(
        n_labels_tested=("consistent_between_contexts", "size"),
        n_consistent=("consistent_between_contexts", "sum"),
        mean_pct_range=("pct_range", "mean"),
        median_pct_range=("pct_range", "median"),
    )
    g["pct_consistent"] = 100 * safe_div_na(g["n_consistent"].to_numpy(), g["n_labels_tested"].to_numpy())
    g = g.sort_values(["pct_consistent", "mean_pct_range"], ascending=[False, True], kind="stable").reset_index(drop=True)
    return g[["Model_key", "n_labels_tested", "n_consistent", "pct_consistent", "mean_pct_range", "median_pct_range"]]


def build_table3_rank(base_long: pd.DataFrame) -> pd.DataFrame:
    g = base_long.groupby(["occupation", "label"], as_index=False, sort=True).agg(
        n_models=("Model_key", "nunique"),
        mean_pct=("pct_all", "mean"),
        min_pct=("pct_all", "min"),
        mean_range=("pct_range", "mean"),
        max_range=("pct_range", "max"),
        mean_p=("p_value", "mean"),
        min_p=("p_value", "min"),
        n_consistent_models=("consistent_between_contexts", "sum"),
        all_models_consistent=("consistent_between_contexts", "min"),  # all() -> min of bool
        mean_consistency_score=("consistency_score", "mean"),
    )
    g["all_models_consistent"] = g["all_models_consistent"].astype(bool)
    g = g[g["n_models"] == len(TABLE3_MODELS)]
    g = g.sort_values(
        ["all_models_consistent", "n_consistent_models", "mean_range", "max_range", "mean_pct", "min_pct", "mean_p"],
        ascending=[False, False, True, True, False, False, False],
        kind="stable",
    ).head(TABLE3_N_ROWS)
    return g


def build_table3_label_persistence_wide(base_long: pd.DataFrame, rank: pd.DataFrame) -> pd.DataFrame:
    sel = base_long[
        base_long.set_index(["occupation", "label"]).index.isin(rank.set_index(["occupation", "label"]).index)
    ]
    sel = sel[
        ["occupation", "label", "Model_key", "pct_round", "range_round", "chi_round", "p_print", "consistent_between_contexts"]
    ]

    wide = sel.pivot_table(
        index=["occupation", "label"], columns="Model_key",
        values=["pct_round", "range_round", "chi_round", "p_print", "consistent_between_contexts"],
        aggfunc="first", observed=True,
    ).reset_index()

    out = pd.DataFrame({"occupation": wide["occupation"], "label": wide["label"]})
    for mk in TABLE3_MODELS:
        for vc in ["pct_round", "range_round", "chi_round", "p_print", "consistent_between_contexts"]:
            colname = f"{mk}_{vc}"
            out[colname] = wide[(vc, mk)].to_numpy() if (vc, mk) in wide.columns else np.nan

    out = out.merge(rank, on=["occupation", "label"], how="left")
    out = out.sort_values(
        ["all_models_consistent", "n_consistent_models", "mean_range", "max_range", "mean_pct", "min_pct", "mean_p"],
        ascending=[False, False, True, True, False, False, False],
        kind="stable",
    ).reset_index(drop=True)

    rename = {
        "occupation": "Occupation", "label": "Label",
        "XL_pct_round": "XL_%", "M35_pct_round": "3.5_%", "Flx_pct_round": "Flx_%", "Qwen_pct_round": "Qwen_%",
        "XL_range_round": "XL_range", "M35_range_round": "3.5_range", "Flx_range_round": "Flx_range", "Qwen_range_round": "Qwen_range",
        "XL_chi_round": "XL_chi2", "M35_chi_round": "3.5_chi2", "Flx_chi_round": "Flx_chi2", "Qwen_chi_round": "Qwen_chi2",
        "XL_p_print": "XL_p", "M35_p_print": "3.5_p", "Flx_p_print": "Flx_p", "Qwen_p_print": "Qwen_p",
        "XL_consistent_between_contexts": "XL_consistent", "M35_consistent_between_contexts": "3.5_consistent",
        "Flx_consistent_between_contexts": "Flx_consistent", "Qwen_consistent_between_contexts": "Qwen_consistent",
    }
    keep = [
        "occupation", "label",
        "XL_pct_round", "M35_pct_round", "Flx_pct_round", "Qwen_pct_round",
        "XL_range_round", "M35_range_round", "Flx_range_round", "Qwen_range_round",
        "XL_chi_round", "M35_chi_round", "Flx_chi_round", "Qwen_chi_round",
        "XL_p_print", "M35_p_print", "Flx_p_print", "Qwen_p_print",
        "XL_consistent_between_contexts", "M35_consistent_between_contexts",
        "Flx_consistent_between_contexts", "Qwen_consistent_between_contexts",
        "mean_pct", "min_pct", "mean_range", "max_range", "mean_p", "min_p",
        "n_consistent_models", "all_models_consistent", "mean_consistency_score",
    ]
    for c in keep:
        if c not in out.columns:
            out[c] = np.nan
    out = out[keep].rename(columns=rename)
    return out


def main():
    occ_table_all = pd.read_csv(os.path.join(OUT_DIR, "occ_table_all.csv"))
    preferred_prompt_group = get_preferred_prompt_group_from_occ_table(occ_table_all)

    base_long = build_table3_base_long(occ_table_all, preferred_prompt_group)
    base_long.to_csv(os.path.join(OUT_DIR, "table3_base_long.csv"), index=False)

    model_summary = build_table3_model_summary(base_long)
    model_summary.to_csv(os.path.join(OUT_DIR, "table3_model_summary.csv"), index=False)

    rank = build_table3_rank(base_long)
    wide = build_table3_label_persistence_wide(base_long, rank)
    wide.to_csv(os.path.join(OUT_DIR, "table3_label_persistence_wide.csv"), index=False)

    top20 = wide.head(20)
    top20.to_csv(os.path.join(OUT_DIR, "table3_top20_most_consistent_labels.csv"), index=False)

    print(f"table3_base_long: {len(base_long)} rows")
    print(f"table3_model_summary: {len(model_summary)} rows")
    print(f"table3_label_persistence_wide: {len(wide)} rows")


if __name__ == "__main__":
    main()
