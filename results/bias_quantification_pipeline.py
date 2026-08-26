"""
Python port of final_full_pipeline_updated.R.

Reproduces the bias-quantification tables (overall_prop_all, bias_occ_loo,
occ_table_all[_wide], table2_*, table3_label_persistence, fig2/fig3 data)
from the raw per-image frequencies.csv files.

Expected raw-data layout, per model:
    <RAW_DATA_ROOT>/<model_dir>/<Context>/<occupation>/[<type>/]<prompt>/frequency/frequencies.csv

All paths are configurable via environment variables (see below) so the
script does not depend on any one machine's directory layout.

Notes on faithfulness to the R original: out.csv (cohort/dimension/label/
canonical_label) is optional -- if absent, labels are normalized but not
canonicalized. The "Sence" entry in TABLE2_DIMS is a typo carried over
from the R source; dimension_to_family_table2() never returns "Sence", so
it is dead filter text and Table 2 has no Scene column, matching the R
output exactly.
"""

import os
import re
import sys
import math
import numpy as np
import pandas as pd
from scipy import stats as sstats

pd.set_option("future.no_silent_downcasting", True)

# Root containing one subfolder per model (e.g. `<prefix>_SDXL`, `<prefix>_Flux`, ...),
# each holding the Context/occupation/[type/]prompt/frequency/frequencies.csv tree.
RAW_DATA_ROOT = os.environ.get(
    "BIAS_RAW_DATA_ROOT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "92_job_description_AllModels")
)

# Where output CSVs are written.
OUT_DIR = os.environ.get("BIAS_OUT_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "output"))
os.makedirs(OUT_DIR, exist_ok=True)

MODEL_SUFFIXES = ["SDXL", "3.5", "Flux", "Qwen"]
MODEL_PREFIX = os.environ.get("BIAS_MODEL_PREFIX", "92_job_description")
MODELS = [f"{MODEL_PREFIX}_{s}" for s in MODEL_SUFFIXES]

# Optional extra raw-data roots per model, for prompt-variant folders that live outside
# RAW_DATA_ROOT/<model_dir> (e.g. a second data drop). Unioned with the primary root at
# read time -- no files are copied or mutated in place. Set via BIAS_EXTRA_ROOTS_<SUFFIX>
# as a colon-separated list of directories, e.g.:
#   BIAS_EXTRA_ROOTS_SDXL="/path/to/extra1:/path/to/extra2"
EXTRA_ROOTS = {}
for _suffix in MODEL_SUFFIXES:
    _env_key = f"BIAS_EXTRA_ROOTS_{_suffix.upper().replace('.', '')}"
    _val = os.environ.get(_env_key, "")
    if _val:
        EXTRA_ROOTS[f"{MODEL_PREFIX}_{_suffix}"] = [p for p in _val.split(":") if p]

FREQ_FOLDER = "frequency"
MAX_PER_PROMPT = 10
CAP_PROMPT_COUNTS = True
INCLUDE_ALL_TYPES_POOLED_ROW = True
TABLE3_N_ROWS = 100000
ALPHA = 0.05

_KEY_SUFFIX_MAP = {"SDXL": "XL", "3.5": "M35", "Flux": "Flx", "Qwen": "Qwen"}
_PLOT_SUFFIX_MAP = {"SDXL": "SDXL", "3.5": "SD3.5", "Flux": "Flux", "Qwen": "Qwen"}

MODEL_KEY_MAP = {f"{MODEL_PREFIX}_{s}": _KEY_SUFFIX_MAP[s] for s in MODEL_SUFFIXES}
MODEL_KEY_MAP["ALL"] = "ALL"

PLOT_MODEL_MAP = {f"{MODEL_PREFIX}_{s}": _PLOT_SUFFIX_MAP[s] for s in MODEL_SUFFIXES}
PLOT_MODEL_MAP["ALL"] = "ALL"

# =========================================================
# HELPERS
# =========================================================


def safe_div_na(num, den):
    num = np.asarray(num, dtype="float64")
    den = np.asarray(den, dtype="float64")
    out = np.full(num.shape, np.nan, dtype="float64")
    ok = ~np.isnan(num) & ~np.isnan(den) & (den > 0)
    out[ok] = num[ok] / den[ok]
    return out


def normalize_label(x: pd.Series) -> pd.Series:
    s = x.astype("string")
    s = s.str.strip().str.lower()
    s = s.fillna("")
    s = s.str.replace(r"[\s\-]+", "_", regex=True)
    s = s.str.replace(r"_+", "_", regex=True)
    s = s.str.replace(r"^_|_$", "", regex=True)
    return s.astype(object).where(s.notna(), "")


_CF_SET = {"Context-free_CF", "CF", "Context_Free_CF"}
_CAU_SET = {"Context-aware_Unrelated_CA-U", "Context-aware_Unrelated_CA-Unrel", "CA-U", "CA_Unrel"}
_CAR_SET = {"Context-aware_Related_CA-R", "CA-R", "CA_Rel"}


def context_shortener(x: pd.Series) -> pd.Series:
    out = x.copy()
    out = out.where(~x.isin(_CF_SET), "CF")
    out = out.where(~x.isin(_CAU_SET), "CA-U")
    out = out.where(~x.isin(_CAR_SET), "CA-R")
    return out


def _regex_family(x0: pd.Series, order):
    """order: list of (regex, label) applied in sequence via case_when semantics"""
    result = pd.Series([None] * len(x0), index=x0.index, dtype=object)
    remaining = pd.Series(True, index=x0.index)
    for pattern, label in order:
        hit = remaining & x0.str.contains(pattern, regex=True, na=False)
        result[hit] = label
        remaining &= ~hit
    return result, remaining


def dimension_to_family(x: pd.Series) -> pd.Series:
    x0 = x.astype(str).str.lower()
    order = [
        (r"scene|location|background|setting|environment|scene_appearance", "Scene"),
        (r"camera|angle|view|shot|framing|composition", "Camera"),
        (r"object|item|tool|accessor|accessories|clothing|prop", "Objects"),
        (r"people|person|gender|age|race|ethnic|body|hair|face|skin", "People"),
    ]
    result, remaining = _regex_family(x0, order)
    # TRUE ~ str_to_title(x)  (title-case the ORIGINAL x, not x0)
    orig = x.astype(str)
    titled = orig[remaining].str.title()
    result[remaining] = titled
    return result


def dimension_to_family_table2(x: pd.Series) -> pd.Series:
    x0 = x.astype(str).str.lower()
    order = [
        (r"people|person|gender|age|race|ethnic|body|hair|face|skin", "People"),
        (r"object|item|tool|accessor|accessories|clothing|prop", "Objects"),
        (r"camera|angle|view|shot|framing|composition", "Camera"),
        (r"scene|location|background|setting|environment|scene_appearance", "Scene"),
    ]
    result, remaining = _regex_family(x0, order)
    result[remaining] = np.nan
    return result


def activity_to_family(x: pd.Series) -> pd.Series:
    x0 = x.astype(str).str.lower()
    result = pd.Series([None] * len(x0), index=x0.index, dtype=object)
    remaining = pd.Series(True, index=x0.index)

    hit = remaining & x0.str.contains(r"loc|location|place|background|setting", regex=True, na=False)
    result[hit] = "Location substitution"
    remaining &= ~hit

    hit = remaining & x0.str.contains(r"activity|semantic|rephrase|paraphrase|original", regex=True, na=False)
    result[hit] = "Semantic rephrasing"
    remaining &= ~hit

    hit = remaining & (x0 == "all_types")
    result[hit] = "All activity types"
    remaining &= ~hit

    result[remaining] = x.astype(str)[remaining]
    return result


def chisq_contrib(obs, exp):
    obs = np.asarray(obs, dtype="float64")
    exp = np.asarray(exp, dtype="float64")
    out = np.full(obs.shape, np.nan, dtype="float64")
    na = np.isnan(obs) | np.isnan(exp)
    pos = (~na) & (exp > 0)
    out[pos] = (obs[pos] - exp[pos]) ** 2 / exp[pos]
    zero = (~na) & (exp == 0) & (obs == 0)
    out[zero] = 0.0
    return out


def bias_severity_score(p):
    p = np.asarray(p, dtype="float64")
    p = p[np.isfinite(p)]
    if len(p) == 0 or np.nansum(p) <= 0:
        return np.nan
    p = p / np.sum(p)
    n_classes = len(p)
    if n_classes <= 1:
        return 0.0
    p_nonzero = p[p > 0]
    normalized_entropy = np.sum(p_nonzero * np.log(p_nonzero)) / math.log(n_classes)
    return 1 + normalized_entropy


def fmt_p(p):
    p = np.asarray(p, dtype="float64")
    out = np.empty(p.shape, dtype=object)
    for i, v in enumerate(p):
        if np.isnan(v):
            out[i] = ""
        elif v < 0.005:
            out[i] = ".00"
        else:
            s = f"{v:.2f}"
            out[i] = re.sub(r"^0", "", s)
    return out


def fmt_cell_prompt_robustness(pct_inv, mean_p):
    pct_inv = np.asarray(pct_inv, dtype="float64")
    mean_p = np.asarray(mean_p, dtype="float64")
    out = np.empty(pct_inv.shape, dtype=object)
    for i in range(len(pct_inv)):
        p_txt = re.sub(r"^0", "", f"{mean_p[i]:.2f}") if not np.isnan(mean_p[i]) else "nan"
        out[i] = f"{pct_inv[i]:.1f}/{p_txt}"
    return out


def fmt_cell_no_type_prevalence(mean_pct):
    mean_pct = np.asarray(mean_pct, dtype="float64")
    out = np.empty(mean_pct.shape, dtype=object)
    for i, v in enumerate(mean_pct):
        out[i] = "" if np.isnan(v) else f"{v:.1f}"
    return out


def chisq_3ctx(x_cf, n_cf, x_u, n_u, x_r, n_r):
    X = np.column_stack([x_cf, x_u, x_r]).astype("float64")
    N = np.column_stack([n_cf, n_u, n_r]).astype("float64")

    X[~np.isfinite(X)] = np.nan
    N[~np.isfinite(N)] = np.nan

    keep = (N > 0) & np.isfinite(N) & np.isfinite(X) & (X >= 0) & (X <= N)
    k_eff = keep.sum(axis=1)

    Xz = np.where(keep, X, 0.0)
    Nz = np.where(keep, N, 0.0)
    x_sum = np.nansum(Xz, axis=1)
    n_sum = np.nansum(Nz, axis=1)

    p_hat = safe_div_na(x_sum, n_sum)

    chisq = np.zeros(X.shape[0])
    pval = np.ones(X.shape[0])
    df = np.maximum(k_eff - 1, 0)

    ok = (k_eff >= 2) & np.isfinite(p_hat) & (p_hat > 0) & (p_hat < 1)

    if ok.any():
        Nh = N[ok]
        Xh = X[ok]
        Kh = keep[ok]
        ph = p_hat[ok]

        numer = (Xh - Nh * ph[:, None]) ** 2
        denom = Nh * (ph * (1 - ph))[:, None]
        with np.errstate(divide="ignore", invalid="ignore"):
            term = np.where(Kh & (denom > 0), numer / denom, 0.0)
        term = np.nan_to_num(term, nan=0.0)
        chisq_ok = term.sum(axis=1)
        chisq[ok] = chisq_ok


        dfok = df[ok]
        pv = sstats.chi2.sf(chisq_ok, dfok)
        pval[ok] = pv

    return pd.DataFrame({"chisq_stat": chisq, "df": df, "p_value": pval})


def bh_adjust(p: pd.Series) -> pd.Series:
    """Replicates R's p.adjust(p, method='BH') exactly, NA-preserving."""
    out = pd.Series(np.nan, index=p.index, dtype="float64")
    mask = p.notna() & np.isfinite(p.astype("float64"))
    vals = p[mask].astype("float64").to_numpy()
    n = len(vals)
    if n == 0:
        return out
    order_desc = np.argsort(-vals, kind="stable")  # order(p, decreasing=TRUE)
    ranks_desc = np.arange(n, 0, -1)  # i <- n:1
    p_sorted_desc = vals[order_desc]
    raw = (n / ranks_desc) * p_sorted_desc
    cm = np.minimum.accumulate(raw)
    cm = np.minimum(cm, 1.0)
    # ro <- order(o)  -> inverse permutation of order_desc
    result_desc_order = cm
    final = np.empty(n)
    final[order_desc] = result_desc_order
    out.loc[mask.index[mask]] = final
    return out


# =========================================================
# CANONICAL LABELS
# out.csv must have: cohort, dimension, label, canonical_label
# =========================================================

OUT_CSV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out.csv")
OUT_CSV_EXISTS = os.path.exists(OUT_CSV_PATH)

if OUT_CSV_EXISTS:
    _canon_raw = pd.read_csv(OUT_CSV_PATH)
    _canon_raw["label_norm"] = normalize_label(_canon_raw["label"])
    _canon_raw["canonical_label"] = normalize_label(_canon_raw["canonical_label"])
    _canon_raw = _canon_raw[
        (_canon_raw["label_norm"].astype(str).str.len() > 0)
        & (_canon_raw["canonical_label"].astype(str).str.len() > 0)
    ]
    # distinct(cohort, dimension, label_norm, .keep_all = TRUE) -> keep first occurrence
    CANON_MAP = _canon_raw.drop_duplicates(
        subset=["cohort", "dimension", "label_norm"], keep="first"
    )[["cohort", "dimension", "label_norm", "canonical_label"]].copy()
else:
    CANON_MAP = pd.DataFrame(columns=["cohort", "dimension", "label_norm", "canonical_label"])


def apply_canonical_labels(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if len(CANON_MAP) == 0:
        df["label"] = normalize_label(df["label"])
        return df

    df["label_norm"] = normalize_label(df["label"])
    merged = df.merge(
        CANON_MAP, how="left", on=["cohort", "dimension", "label_norm"]
    )
    merged["label"] = merged["canonical_label"].where(
        merged["canonical_label"].notna(), merged["label_norm"]
    )
    merged = merged.drop(columns=["label_norm", "canonical_label"])
    return merged


# =========================================================
# READER
# =========================================================


def _is_hidden(name: str) -> bool:
    return name.startswith(".")


def list_frequencies_files(model_root: str, freq_folder: str):
    """Replicates R list.files(path, pattern='^frequencies\\.csv$', recursive=TRUE,
    full.names=TRUE) with default all.files=FALSE (hidden dirs/files excluded from
    the recursive walk), then filtered to those under a /freq_folder/ path segment.
    """
    matches = []
    for root, dirs, files in os.walk(model_root):
        # prune hidden directories in-place so os.walk does not descend into them
        dirs[:] = sorted(d for d in dirs if not _is_hidden(d))
        for f in files:
            if f == "frequencies.csv":
                matches.append(os.path.join(root, f))
    pattern = re.compile(r"([/\\])" + re.escape(freq_folder) + r"([/\\])")
    matches = [m for m in matches if pattern.search(m)]
    return sorted(matches)


def read_one_model(model_name: str, freq_folder: str = "frequency") -> pd.DataFrame:
    model_root = os.path.join(RAW_DATA_ROOT, model_name)
    # Union the primary raw-data root with any extra roots supplied for this model (additional
    # prompt-variant folders that live outside RAW_DATA_ROOT -- see EXTRA_ROOTS above). Each
    # (root, file) pair keeps its own root for relpath purposes so the Context/occupation/
    # prompt/[type]/freq_folder depth logic below is computed correctly per source tree.
    roots = [model_root] + EXTRA_ROOTS.get(model_name, [])
    root_file_pairs = []
    for root in roots:
        for f in list_frequencies_files(root, freq_folder):
            root_file_pairs.append((root, f))

    if len(root_file_pairs) == 0:
        return pd.DataFrame()

    frames = []
    for root, f in root_file_pairs:
        x = pd.read_csv(f)
        if "count" not in x.columns:
            x["count"] = 1
        x["count"] = pd.to_numeric(x["count"], errors="coerce")
        x["count"] = x["count"].fillna(0)

        rel = os.path.relpath(f, root)
        rel_parts = rel.split(os.sep)  # 0-based python list

        # Replicate R's 1-based rel_parts / idx logic
        idxs = [i + 1 for i, p in enumerate(rel_parts) if p == freq_folder]  # 1-based
        idx = idxs[-1] if len(idxs) > 0 else None

        context_val = rel_parts[0] if len(rel_parts) >= 1 else np.nan
        occupation_val = rel_parts[1] if len(rel_parts) >= 2 else np.nan
        if idx is not None and idx > 1:
            prompt_val = rel_parts[idx - 2]  # rel_parts[idx-1] in 1-based -> idx-2 in 0-based
        else:
            prompt_val = os.path.splitext(os.path.basename(f))[0]

        if idx is not None and idx >= 5:
            type_val = rel_parts[idx - 3]  # rel_parts[idx-2] in 1-based -> idx-3 in 0-based
        else:
            type_val = np.nan
        if pd.isna(type_val) or type_val == "":
            type_val = "no_type"

        x["Model"] = model_name
        x["Context"] = context_val
        x["occupation"] = occupation_val
        x["type"] = type_val
        x["prompt"] = prompt_val
        frames.append(x)

    return pd.concat(frames, ignore_index=True, sort=False)


# =========================================================
# CORE: LOO occupation-level table
# =========================================================


def compute_overall_prop_all_with_loo(
    merged: pd.DataFrame,
    max_per_prompt: int = 10,
    cap_prompt_counts: bool = True,
    include_all_types_pooled_row: bool = True,
) -> pd.DataFrame:

    df_raw = merged.copy()
    df_raw["type"] = df_raw["type"].where(~(df_raw["type"].isna() | (df_raw["type"] == "")), "no_type")
    df_raw["label"] = normalize_label(df_raw["label"])
    df_raw = df_raw[df_raw["label"].notna() & (df_raw["label"] != "unknown") & (df_raw["cohort"] != "totals")]

    if len(df_raw) == 0:
        return pd.DataFrame()

    df_type = df_raw.copy()
    df_type["prompt_group"] = df_type["type"]
    df_type["prompt_id"] = df_type["type"].astype(str) + "||" + df_type["prompt"].astype(str)

    df_all = df_raw.copy()
    df_all["prompt_group"] = "ALL_TYPES"
    df_all["prompt_id"] = df_all["type"].astype(str) + "||" + df_all["prompt"].astype(str)

    df0 = pd.concat([df_type, df_all], ignore_index=True) if include_all_types_pooled_row else df_type

    key0 = ["Model", "dimension", "cohort", "prompt_group", "Context", "occupation", "prompt_id", "label"]
    df0 = df0.groupby(key0, as_index=False, sort=True)["count"].sum()

    if cap_prompt_counts:
        df0["count"] = np.minimum(df0["count"], max_per_prompt)

    occ_grid = df0[["Model", "dimension", "cohort", "prompt_group", "Context", "occupation"]].drop_duplicates()
    label_grid = df0[["Model", "dimension", "cohort", "prompt_group", "label"]].drop_duplicates()

    prompts_occ = df0[
        ["Model", "dimension", "cohort", "prompt_group", "Context", "occupation", "prompt_id"]
    ].drop_duplicates()

    prompts_ctx = prompts_occ.copy()
    prompts_ctx["ctx_pid"] = prompts_ctx["occupation"].astype(str) + "||" + prompts_ctx["prompt_id"].astype(str)
    prompts_ctx = prompts_ctx[["Model", "dimension", "cohort", "prompt_group", "Context", "ctx_pid"]].drop_duplicates()

    prompts_all = prompts_occ.copy()
    prompts_all["all_pid"] = (
        prompts_all["Context"].astype(str) + "||" + prompts_all["occupation"].astype(str) + "||" + prompts_all["prompt_id"].astype(str)
    )
    prompts_all = prompts_all[["Model", "dimension", "cohort", "prompt_group", "all_pid"]].drop_duplicates()

    den_occ = (
        prompts_occ.groupby(["Model", "dimension", "cohort", "prompt_group", "Context", "occupation"], as_index=False, sort=True)
        .agg(n_prompts_occ=("prompt_id", "nunique"))
    )
    den_occ["n_occ"] = den_occ["n_prompts_occ"] * max_per_prompt

    den_ctx = (
        prompts_ctx.groupby(["Model", "dimension", "cohort", "prompt_group", "Context"], as_index=False, sort=True)
        .agg(n_prompts_ctx=("ctx_pid", "nunique"))
    )
    den_ctx["n_context"] = den_ctx["n_prompts_ctx"] * max_per_prompt

    den_all = (
        prompts_all.groupby(["Model", "dimension", "cohort", "prompt_group"], as_index=False, sort=True)
        .agg(n_prompts_all=("all_pid", "nunique"))
    )
    den_all["n_all"] = den_all["n_prompts_all"] * max_per_prompt

    obs_occ_raw = (
        df0.groupby(["Model", "dimension", "cohort", "prompt_group", "Context", "occupation", "label"], as_index=False, sort=True)
        .agg(obs_occ=("count", "sum"))
    )

    grp_key = ["Model", "dimension", "cohort", "prompt_group"]
    obs_occ = occ_grid.merge(label_grid, on=grp_key, how="left")
    obs_occ = obs_occ.merge(
        obs_occ_raw,
        on=["Model", "dimension", "cohort", "prompt_group", "Context", "occupation", "label"],
        how="left",
    )
    obs_occ["obs_occ"] = obs_occ["obs_occ"].fillna(0)

    obs_context = (
        obs_occ.groupby(["Model", "dimension", "cohort", "prompt_group", "Context", "label"], as_index=False, sort=True)
        .agg(obs_context=("obs_occ", "sum"))
    )

    obs_whole = (
        obs_context.groupby(["Model", "dimension", "cohort", "prompt_group", "label"], as_index=False, sort=True)
        .agg(**{"obs.whole": ("obs_context", "sum")})
    )

    bins_df = (
        label_grid.groupby(["Model", "dimension", "cohort", "prompt_group"], as_index=False, sort=True)
        .agg(bins=("label", "nunique"))
    )
    bins_df["uniform.prop"] = np.where(bins_df["bins"] > 0, 1.0 / bins_df["bins"], 0.0)
    bins_df["df"] = np.maximum(bins_df["bins"] - 1, 0)

    out = obs_occ.merge(den_occ, on=["Model", "dimension", "cohort", "prompt_group", "Context", "occupation"], how="left")
    out = out.merge(obs_context, on=["Model", "dimension", "cohort", "prompt_group", "Context", "label"], how="left")
    out = out.merge(den_ctx, on=["Model", "dimension", "cohort", "prompt_group", "Context"], how="left")
    out = out.merge(obs_whole, on=["Model", "dimension", "cohort", "prompt_group", "label"], how="left")
    out = out.merge(den_all, on=["Model", "dimension", "cohort", "prompt_group"], how="left")
    out = out.merge(bins_df, on=["Model", "dimension", "cohort", "prompt_group"], how="left")

    out["prop_overall"] = safe_div_na(out["obs.whole"], out["n_all"])
    out["prop_context"] = safe_div_na(out["obs_context"], out["n_context"])
    out["prop_occ"] = safe_div_na(out["obs_occ"], out["n_occ"])

    out["exp.whole"] = out["n_all"] * out["uniform.prop"]

    out["rest_n_context"] = out["n_all"] - out["n_context"]
    out["rest_label_context"] = out["obs.whole"] - out["obs_context"]
    out["prop_context_loo_baseline"] = safe_div_na(out["rest_label_context"], out["rest_n_context"])
    out["exp_context_loo"] = out["n_context"] * out["prop_context_loo_baseline"]
    out["ch_v_context_loo"] = chisq_contrib(out["obs_context"], out["exp_context_loo"])

    out["rest_n_occ"] = out["n_all"] - out["n_occ"]
    out["rest_label_occ"] = out["obs.whole"] - out["obs_occ"]
    out["prop_occ_loo_baseline"] = safe_div_na(out["rest_label_occ"], out["rest_n_occ"])
    out["exp_occ_loo"] = out["n_occ"] * out["prop_occ_loo_baseline"]
    out["ch_v_occ_loo"] = chisq_contrib(out["obs_occ"], out["exp_occ_loo"])

    out["count"] = out["obs_occ"]
    out["n.with.coh"] = out["n_all"]
    out["n.with.coh.Cont"] = out["n_context"]
    out["n.with.coh.Cont.occ"] = out["n_occ"]

    out["exp.whole.context"] = out["exp_context_loo"]
    out["obs.whole.context"] = out["obs_context"]
    out["exp.whole.context.occ"] = out["exp_occ_loo"]
    out["obs.whole.context.occ"] = out["obs_occ"]

    out["exp_occ"] = out["exp_occ_loo"]
    out["ch_v_occ"] = out["ch_v_occ_loo"]
    out["ch.v"] = out["ch_v_occ"]

    final_cols = [
        "Model", "dimension", "cohort", "prompt_group", "Context", "occupation", "label",
        "count", "bins", "uniform.prop",
        "n.with.coh", "prop_overall",
        "n.with.coh.Cont", "prop_context",
        "n.with.coh.Cont.occ", "prop_occ",
        "exp_occ", "ch_v_occ", "ch.v", "df",
        "exp.whole", "obs.whole",
        "exp.whole.context", "obs.whole.context",
        "exp.whole.context.occ", "obs.whole.context.occ",
        "prop_context_loo_baseline", "prop_occ_loo_baseline",
    ]
    return out[final_cols]


# =========================================================
# BUILD overall_prop_all
# =========================================================


def build_overall_prop_all():
    parts = []
    for model_name in MODELS:
        print(f"Reading: {model_name}", file=sys.stderr)
        merged = read_one_model(model_name, freq_folder=FREQ_FOLDER)
        if len(merged) == 0:
            continue

        if model_name.endswith("SDXL") and "label" in merged.columns:
            bad = merged["label"].astype(str).str.contains(r"\|", regex=True, na=False)
            merged = merged[~bad]

        if "dimension" in merged.columns:
            merged = merged[merged["dimension"] != "role_hint"]

        merged = apply_canonical_labels(merged)

        part = compute_overall_prop_all_with_loo(
            merged,
            max_per_prompt=MAX_PER_PROMPT,
            cap_prompt_counts=CAP_PROMPT_COUNTS,
            include_all_types_pooled_row=INCLUDE_ALL_TYPES_POOLED_ROW,
        )
        if len(part) > 0:
            parts.append(part)

    overall_prop_all = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    return overall_prop_all


# =========================================================
# BIAS OCCUPATION LOO SUMMARY
# =========================================================


def build_bias_occ_loo(overall_prop_all: pd.DataFrame) -> pd.DataFrame:
    d = overall_prop_all[
        (~overall_prop_all["cohort"].isin(["totals", "safety"])) & overall_prop_all["prompt_group"].notna()
    ].copy()

    key = ["Model", "prompt_group", "dimension", "cohort", "Context", "occupation"]
    g = d.groupby(key, as_index=False, sort=True).agg(
        **{"chi.value": ("ch_v_occ", "sum"), "df": ("df", "max")}
    )
    finite_chi = np.isfinite(g["chi.value"])
    finite_df = np.isfinite(g["df"])
    ok = finite_chi & finite_df & (g["df"] > 0)

    g["p_value"] = np.nan
    g.loc[ok, "p_value"] = sstats.chi2.sf(g.loc[ok, "chi.value"].to_numpy(), g.loc[ok, "df"].to_numpy())

    g["p_adj"] = np.nan
    for _, idx in g.groupby(["Model", "prompt_group"], sort=True).groups.items():
        sub = g.loc[idx, "p_value"]
        g.loc[idx, "p_adj"] = bh_adjust(sub)

    return g


# =========================================================
# OCC TABLE
# =========================================================


def build_occ_table_all(overall_prop_all: pd.DataFrame):
    d = overall_prop_all[
        (~overall_prop_all["cohort"].isin(["totals", "safety"])) & overall_prop_all["prompt_group"].notna()
    ].copy()
    d["Context_short"] = context_shortener(d["Context"])
    d = d[d["Context_short"].isin(["CF", "CA-U", "CA-R"])]

    key = ["Model", "dimension", "cohort", "prompt_group", "occupation", "label", "Context_short"]
    g = d.groupby(key, as_index=False, sort=True).agg(
        x=("count", "sum"), n=("n.with.coh.Cont.occ", "max")
    )
    g["pct"] = np.where(g["n"] > 0, 100 * g["x"] / g["n"], np.nan)

    id_cols = ["Model", "dimension", "cohort", "prompt_group", "occupation", "label"]
    piv_x = g.pivot_table(index=id_cols, columns="Context_short", values="x", aggfunc="first", fill_value=0)
    piv_n = g.pivot_table(index=id_cols, columns="Context_short", values="n", aggfunc="first", fill_value=0)
    piv_pct = g.pivot_table(index=id_cols, columns="Context_short", values="pct", aggfunc="first")

    for ctx in ["CF", "CA-U", "CA-R"]:
        if ctx not in piv_x.columns:
            piv_x[ctx] = 0
        if ctx not in piv_n.columns:
            piv_n[ctx] = 0
        if ctx not in piv_pct.columns:
            piv_pct[ctx] = np.nan

    occ_table = pd.DataFrame(index=piv_x.index).reset_index()
    occ_table["x_CF"] = piv_x["CF"].to_numpy()
    occ_table["n_CF"] = piv_n["CF"].to_numpy()
    occ_table["pct_CF"] = piv_pct["CF"].to_numpy()
    occ_table["x_CA_Unrel"] = piv_x["CA-U"].to_numpy()
    occ_table["n_CA_Unrel"] = piv_n["CA-U"].to_numpy()
    occ_table["pct_CA_Unrel"] = piv_pct["CA-U"].to_numpy()
    occ_table["x_CA_Rel"] = piv_x["CA-R"].to_numpy()
    occ_table["n_CA_Rel"] = piv_n["CA-R"].to_numpy()
    occ_table["pct_CA_Rel"] = piv_pct["CA-R"].to_numpy()

    occ_table["x_all"] = occ_table["x_CF"] + occ_table["x_CA_Unrel"] + occ_table["x_CA_Rel"]
    occ_table["n_all"] = occ_table["n_CF"] + occ_table["n_CA_Unrel"] + occ_table["n_CA_Rel"]
    occ_table["pct_all"] = np.where(occ_table["n_all"] > 0, 100 * occ_table["x_all"] / occ_table["n_all"], np.nan)

    occ_table = occ_table.sort_values(id_cols, kind="stable").reset_index(drop=True)

    test3 = chisq_3ctx(
        occ_table["x_CF"], occ_table["n_CF"],
        occ_table["x_CA_Unrel"], occ_table["n_CA_Unrel"],
        occ_table["x_CA_Rel"], occ_table["n_CA_Rel"],
    )
    occ_table = pd.concat([occ_table.reset_index(drop=True), test3.reset_index(drop=True)], axis=1)

    occ_table["p_adj"] = np.nan
    for _, idx in occ_table.groupby(["Model", "prompt_group"], sort=True).groups.items():
        sub = occ_table.loc[idx, "p_value"]
        occ_table.loc[idx, "p_adj"] = bh_adjust(sub)

    # occ_table_all_models: aggregate across Model
    key2 = ["dimension", "cohort", "prompt_group", "occupation", "label"]
    occ_table_all_models = occ_table.groupby(key2, as_index=False, sort=True).agg(
        x_CF=("x_CF", "sum"), n_CF=("n_CF", "sum"),
        x_CA_Unrel=("x_CA_Unrel", "sum"), n_CA_Unrel=("n_CA_Unrel", "sum"),
        x_CA_Rel=("x_CA_Rel", "sum"), n_CA_Rel=("n_CA_Rel", "sum"),
    )
    occ_table_all_models["pct_CF"] = np.where(
        occ_table_all_models["n_CF"] > 0, 100 * occ_table_all_models["x_CF"] / occ_table_all_models["n_CF"], np.nan
    )
    occ_table_all_models["pct_CA_Unrel"] = np.where(
        occ_table_all_models["n_CA_Unrel"] > 0,
        100 * occ_table_all_models["x_CA_Unrel"] / occ_table_all_models["n_CA_Unrel"],
        np.nan,
    )
    occ_table_all_models["pct_CA_Rel"] = np.where(
        occ_table_all_models["n_CA_Rel"] > 0,
        100 * occ_table_all_models["x_CA_Rel"] / occ_table_all_models["n_CA_Rel"],
        np.nan,
    )
    occ_table_all_models["x_all"] = (
        occ_table_all_models["x_CF"] + occ_table_all_models["x_CA_Unrel"] + occ_table_all_models["x_CA_Rel"]
    )
    occ_table_all_models["n_all"] = (
        occ_table_all_models["n_CF"] + occ_table_all_models["n_CA_Unrel"] + occ_table_all_models["n_CA_Rel"]
    )
    occ_table_all_models["pct_all"] = np.where(
        occ_table_all_models["n_all"] > 0, 100 * occ_table_all_models["x_all"] / occ_table_all_models["n_all"], np.nan
    )
    occ_table_all_models["Model"] = "ALL"

    col_order = [
        "Model", "dimension", "cohort", "prompt_group", "occupation", "label",
        "x_CF", "n_CF", "pct_CF",
        "x_CA_Unrel", "n_CA_Unrel", "pct_CA_Unrel",
        "x_CA_Rel", "n_CA_Rel", "pct_CA_Rel",
        "x_all", "n_all", "pct_all",
    ]
    occ_table_all_models = occ_table_all_models[col_order]

    test3_all = chisq_3ctx(
        occ_table_all_models["x_CF"], occ_table_all_models["n_CF"],
        occ_table_all_models["x_CA_Unrel"], occ_table_all_models["n_CA_Unrel"],
        occ_table_all_models["x_CA_Rel"], occ_table_all_models["n_CA_Rel"],
    )
    occ_table_all_models = pd.concat(
        [occ_table_all_models.reset_index(drop=True), test3_all.reset_index(drop=True)], axis=1
    )
    occ_table_all_models["p_adj"] = np.nan

    occ_table_final_cols = list(occ_table_all_models.columns)
    occ_table_all = pd.concat(
        [occ_table_all_models, occ_table[occ_table_final_cols]], ignore_index=True
    )

    return occ_table_all


def build_occ_table_all_wide(occ_table_all: pd.DataFrame) -> pd.DataFrame:
    d = occ_table_all.copy()
    d["Model_key"] = d["Model"].map(MODEL_KEY_MAP).fillna(d["Model"])
    d.loc[d["Model"] == "ALL", "Model_key"] = "ALL"

    id_cols = ["dimension", "cohort", "prompt_group", "occupation", "label"]
    value_cols = ["pct_CF", "pct_CA_Unrel", "pct_CA_Rel", "pct_all", "chisq_stat", "df", "p_value", "p_adj"]

    # column order should follow first-appearance of Model_key in the (sorted) data
    d_sorted = d.sort_values(["Model"] + id_cols, kind="stable")
    model_key_order = list(dict.fromkeys(d_sorted["Model_key"].tolist()))
    # occ_table_all_models (Model=="ALL") block comes first in the row-bind, so "ALL" leads
    model_key_order = ["ALL"] + [m for m in model_key_order if m != "ALL"]

    wide = d.pivot_table(index=id_cols, columns="Model_key", values=value_cols, aggfunc="first")
    wide = wide.reset_index()

    out_cols = list(id_cols)
    flat = {}
    for c in id_cols:
        flat[c] = wide[c]
    new_df = pd.DataFrame(flat)
    for vc in value_cols:
        for mk in model_key_order:
            colname = f"{mk}_{vc}"
            if (vc, mk) in wide.columns:
                new_df[colname] = wide[(vc, mk)].to_numpy()
            else:
                new_df[colname] = np.nan
    return new_df


# =========================================================
# TABLE 2 + TABLE 3 BACKBONE
# =========================================================


def build_label_persistence_loo(overall_prop_all: pd.DataFrame) -> pd.DataFrame:
    d = overall_prop_all[
        (~overall_prop_all["cohort"].isin(["totals", "safety"]))
        & overall_prop_all["prompt_group"].notna()
        & overall_prop_all["label"].notna()
        & (overall_prop_all["label"] != "unknown")
    ].copy()

    d["Model_key"] = d["Model"].map(MODEL_KEY_MAP).fillna(d["Model"])
    d["Context_short"] = context_shortener(d["Context"])
    d["dimension_family"] = dimension_to_family(d["dimension"])
    d["activity_family"] = activity_to_family(d["prompt_group"])

    key = [
        "Model", "Model_key", "prompt_group", "activity_family",
        "dimension", "dimension_family", "cohort", "occupation", "label",
    ]

    finite_chv = np.isfinite(d["ch_v_occ"])
    d["_finite_chv"] = finite_chv

    g = d.groupby(key, as_index=False, sort=True).agg(
        label_n=("count", "sum"),
        total_n=("n.with.coh.Cont.occ", "sum"),
        chi_sq_loo=("ch_v_occ", "sum"),
    )
    g["pct"] = 100 * safe_div_na(g["label_n"].to_numpy(), g["total_n"].to_numpy())

    nctx = (
        d[d["_finite_chv"]]
        .groupby(key, as_index=False, sort=True)
        .agg(n_contexts=("Context", "nunique"))
    )
    g = g.merge(nctx, on=key, how="left")
    g["n_contexts"] = g["n_contexts"].fillna(0).astype(int)
    g["df_loo"] = np.maximum(g["n_contexts"] - 1, 0)


    p_value_loo = np.ones(len(g))
    ok = (g["df_loo"] > 0) & np.isfinite(g["chi_sq_loo"])
    p_value_loo[ok.to_numpy()] = sstats.chi2.sf(
        g.loc[ok, "chi_sq_loo"].to_numpy(), g.loc[ok, "df_loo"].to_numpy()
    )
    g["p_value_loo"] = p_value_loo

    g["invariant"] = g["p_value_loo"].notna() & (g["p_value_loo"] > ALPHA)
    return g


TABLE2_DIMS = ["People", "Objects", "Camera", "Sence"]  # "Sence" typo intentionally kept (see report)
TABLE2_MODEL_LEVELS = ["XL", "3.5", "Flux", "Qwen", "Total"]


def get_preferred_prompt_group(overall_prop_all: pd.DataFrame) -> str:
    uniq = overall_prop_all["prompt_group"].dropna().unique().tolist()
    if "ALL_TYPES" in uniq:
        return "ALL_TYPES"
    return uniq[0]


def build_table2_source(overall_prop_all: pd.DataFrame, preferred_prompt_group: str) -> pd.DataFrame:
    d = overall_prop_all[
        (~overall_prop_all["cohort"].isin(["totals", "safety"]))
        & overall_prop_all["prompt_group"].notna()
        & overall_prop_all["label"].notna()
        & (overall_prop_all["label"] != "unknown")
    ].copy()

    d["Context_short"] = context_shortener(d["Context"])

    d["prompt_type_category"] = np.select(
        [d["Context_short"] == "CF", d["Context_short"].isin(["CA-R", "CA-U"])],
        ["No type", "Other type"],
        default=None,
    )

    d["dimension_family"] = dimension_to_family_table2(d["cohort"])

    d["Model_key"] = d["Model"].map(MODEL_KEY_MAP).fillna(d["Model"])
    pretty_map = {"M35": "3.5", "Flx": "Flux", "XL": "XL", "Qwen": "Qwen"}
    d["Model_pretty"] = d["Model_key"].map(pretty_map).fillna(d["Model_key"])

    d = d[
        d["Context_short"].isin(["CF", "CA-R", "CA-U"])
        & d["prompt_type_category"].notna()
        & d["dimension_family"].notna()
        & d["dimension_family"].isin(TABLE2_DIMS)
        & d["Model_pretty"].notna()
    ]

    d_no_all = d[d["prompt_group"] != "ALL_TYPES"]
    if len(d_no_all) > 0:
        d = d_no_all

    key = [
        "Model_pretty", "Model_key", "prompt_type_category",
        "dimension", "dimension_family", "cohort", "occupation", "label",
    ]

    g = d.groupby(key, as_index=False, sort=True).agg(
        label_n=("count", "sum"),
        total_n=("n.with.coh.Cont.occ", "sum"),
        chi_sq_loo=("ch_v_occ", "sum"),
    )
    g["pct"] = 100 * safe_div_na(g["label_n"].to_numpy(), g["total_n"].to_numpy())

    d["_finite_chv"] = np.isfinite(d["ch_v_occ"])
    nctx = (
        d[d["_finite_chv"]]
        .groupby(key, as_index=False, sort=True)
        .agg(n_contexts=("Context_short", "nunique"))
    )
    g = g.merge(nctx, on=key, how="left")
    g["n_contexts"] = g["n_contexts"].fillna(0).astype(int)
    g["df_loo"] = np.maximum(g["n_contexts"] - 1, 0)


    p_value_loo = np.ones(len(g))
    ok = (g["df_loo"] > 0) & np.isfinite(g["chi_sq_loo"])
    p_value_loo[ok.to_numpy()] = sstats.chi2.sf(
        g.loc[ok, "chi_sq_loo"].to_numpy(), g.loc[ok, "df_loo"].to_numpy()
    )
    g["p_value_loo"] = p_value_loo
    g["invariant"] = g["p_value_loo"].notna() & (g["p_value_loo"] > ALPHA)

    return g


def _table2_no_type(table2_source: pd.DataFrame):
    src = table2_source[table2_source["prompt_type_category"] == "No type"]

    by_dim = src.groupby(["Model_pretty", "dimension_family"], as_index=False, sort=True).agg(
        n_labels_tested=("pct", "size"), mean_pct=("pct", "mean"), median_pct=("pct", "median")
    )

    by_all = src.groupby(["Model_pretty"], as_index=False, sort=True).agg(
        n_labels_tested=("pct", "size"), mean_pct=("pct", "mean"), median_pct=("pct", "median")
    )
    by_all["dimension_family"] = "All"

    by_total_dim = src.groupby(["dimension_family"], as_index=False, sort=True).agg(
        n_labels_tested=("pct", "size"), mean_pct=("pct", "mean"), median_pct=("pct", "median")
    )
    by_total_dim["Model_pretty"] = "Total"

    total_all = pd.DataFrame(
        {
            "Model_pretty": ["Total"],
            "dimension_family": ["All"],
            "n_labels_tested": [len(src)],
            "mean_pct": [src["pct"].mean()],
            "median_pct": [src["pct"].median()],
        }
    )

    long = pd.concat([by_dim, by_all, by_total_dim, total_all], ignore_index=True)
    long = long[
        long["Model_pretty"].notna()
        & long["dimension_family"].notna()
        & long["dimension_family"].isin(TABLE2_DIMS + ["All"])
    ]
    long["Model"] = pd.Categorical(long["Model_pretty"], categories=TABLE2_MODEL_LEVELS, ordered=True)
    long["dimension_family"] = pd.Categorical(
        long["dimension_family"], categories=TABLE2_DIMS + ["All"], ordered=True
    )
    long = long[long["Model"].notna() & long["dimension_family"].notna()]
    long["cell"] = fmt_cell_no_type_prevalence(long["mean_pct"].to_numpy())

    long = long.groupby(["Model", "dimension_family"], as_index=False, sort=True, observed=True).agg(
        n_labels_tested=("n_labels_tested", "first"),
        mean_pct=("mean_pct", "first"),
        median_pct=("median_pct", "first"),
        cell=("cell", "first"),
    )
    long = long.sort_values(["Model", "dimension_family"], kind="stable").reset_index(drop=True)

    wide = long.pivot_table(index="Model", columns="dimension_family", values="cell", aggfunc="first", observed=True)
    wide = wide.reindex(columns=TABLE2_DIMS + ["All"])
    wide = wide.fillna("")
    wide = wide.reset_index()
    wide = wide.sort_values("Model", kind="stable").reset_index(drop=True)
    return long, wide


def _table2_other(table2_source: pd.DataFrame):
    src = table2_source[table2_source["prompt_type_category"] == "Other type"]

    def agg(g):
        return pd.Series(
            {
                "n_labels_tested": len(g),
                "pct_invariant": 100 * g["invariant"].mean(skipna=True),
                "mean_p": g["p_value_loo"].mean(skipna=True),
                "median_p": g["p_value_loo"].median(skipna=True),
            }
        )

    by_dim = src.groupby(["Model_pretty", "dimension_family"], sort=True).apply(agg, include_groups=False).reset_index()
    by_all = src.groupby(["Model_pretty"], sort=True).apply(agg, include_groups=False).reset_index()
    by_all["dimension_family"] = "All"
    by_total_dim = src.groupby(["dimension_family"], sort=True).apply(agg, include_groups=False).reset_index()
    by_total_dim["Model_pretty"] = "Total"

    total_all_vals = agg(src)
    total_all = pd.DataFrame(
        {
            "Model_pretty": ["Total"],
            "dimension_family": ["All"],
            "n_labels_tested": [total_all_vals["n_labels_tested"]],
            "pct_invariant": [total_all_vals["pct_invariant"]],
            "mean_p": [total_all_vals["mean_p"]],
            "median_p": [total_all_vals["median_p"]],
        }
    )

    long = pd.concat([by_dim, by_all, by_total_dim, total_all], ignore_index=True)
    long = long[
        long["Model_pretty"].notna()
        & long["dimension_family"].notna()
        & long["dimension_family"].isin(TABLE2_DIMS + ["All"])
    ]
    long["Model"] = pd.Categorical(long["Model_pretty"], categories=TABLE2_MODEL_LEVELS, ordered=True)
    long["dimension_family"] = pd.Categorical(
        long["dimension_family"], categories=TABLE2_DIMS + ["All"], ordered=True
    )
    long = long[long["Model"].notna() & long["dimension_family"].notna()]
    long["cell"] = fmt_cell_prompt_robustness(long["pct_invariant"].to_numpy(), long["mean_p"].to_numpy())

    long = long.groupby(["Model", "dimension_family"], as_index=False, sort=True, observed=True).agg(
        n_labels_tested=("n_labels_tested", "first"),
        pct_invariant=("pct_invariant", "first"),
        mean_p=("mean_p", "first"),
        median_p=("median_p", "first"),
        cell=("cell", "first"),
    )
    long = long.sort_values(["Model", "dimension_family"], kind="stable").reset_index(drop=True)

    wide = long.pivot_table(index="Model", columns="dimension_family", values="cell", aggfunc="first", observed=True)
    wide = wide.reindex(columns=TABLE2_DIMS + ["All"])
    wide = wide.fillna("")
    wide = wide.reset_index()
    wide = wide.sort_values("Model", kind="stable").reset_index(drop=True)
    return long, wide


def build_table2(overall_prop_all: pd.DataFrame, preferred_prompt_group: str):
    table2_source = build_table2_source(overall_prop_all, preferred_prompt_group)

    table2_no_type_isolated_long, table2_no_type_isolated = _table2_no_type(table2_source)
    table2_other_type_robustness_long, table2_other_type_robustness = _table2_other(table2_source)

    # drop the never-matched "Sence" column (dimension_to_family_table2 never returns "Sence")
    for tbl in (table2_no_type_isolated, table2_other_type_robustness):
        if "Sence" in tbl.columns:
            tbl.drop(columns=["Sence"], inplace=True)

    no_type_part = table2_no_type_isolated_long.copy()
    no_type_part["prompt_type_category"] = "No type"
    no_type_part["table_metric"] = "mean baseline prevalence"
    no_type_part["pct_invariant"] = np.nan
    no_type_part["mean_p"] = np.nan
    no_type_part["median_p"] = np.nan
    no_type_part = no_type_part[
        [
            "prompt_type_category", "table_metric", "Model", "dimension_family",
            "n_labels_tested", "mean_pct", "median_pct", "pct_invariant", "mean_p", "median_p", "cell",
        ]
    ]

    other_part = table2_other_type_robustness_long.copy()
    other_part["prompt_type_category"] = "Other type"
    other_part["table_metric"] = "% invariant / mean p-value"
    other_part["mean_pct"] = np.nan
    other_part["median_pct"] = np.nan
    other_part = other_part[
        [
            "prompt_type_category", "table_metric", "Model", "dimension_family",
            "n_labels_tested", "mean_pct", "median_pct", "pct_invariant", "mean_p", "median_p", "cell",
        ]
    ]

    table2_prompt_robustness_long = pd.concat([no_type_part, other_part], ignore_index=True)
    table2_prompt_robustness = table2_other_type_robustness.copy()

    return {
        "table2_no_type_isolated": table2_no_type_isolated,
        "table2_no_type_isolated_long": table2_no_type_isolated_long,
        "table2_other_type_robustness": table2_other_type_robustness,
        "table2_other_type_robustness_long": table2_other_type_robustness_long,
        "table2_prompt_robustness": table2_prompt_robustness,
        "table2_prompt_robustness_long": table2_prompt_robustness_long,
        "table2_source": table2_source,
    }


# =========================================================
# TABLE 3: LABEL PERSISTENCE (main pipeline version -- matches oracle header)
# =========================================================

TABLE3_MODELS = ["XL", "M35", "Flx", "Qwen"]


def build_table3_label_persistence(label_persistence_loo: pd.DataFrame, preferred_prompt_group: str) -> pd.DataFrame:
    d = label_persistence_loo[
        (label_persistence_loo["prompt_group"] == preferred_prompt_group)
        & label_persistence_loo["Model_key"].isin(TABLE3_MODELS)
    ].copy()

    key = ["Model_key", "occupation", "label"]
    g = d.groupby(key, as_index=False, sort=True).agg(
        label_n=("label_n", "sum"), total_n=("total_n", "sum"), chi_sq_loo=("chi_sq_loo", "sum"), df_loo=("df_loo", "sum")
    )
    g["pct"] = 100 * safe_div_na(g["label_n"].to_numpy(), g["total_n"].to_numpy())


    p_value_loo = np.ones(len(g))
    ok = (g["df_loo"] > 0) & np.isfinite(g["chi_sq_loo"])
    p_value_loo[ok.to_numpy()] = sstats.chi2.sf(g.loc[ok, "chi_sq_loo"].to_numpy(), g.loc[ok, "df_loo"].to_numpy())
    g["p_value_loo"] = p_value_loo

    g["Model_key"] = pd.Categorical(g["Model_key"], categories=TABLE3_MODELS, ordered=True)
    g["pct_round"] = np.round(g["pct"].to_numpy())
    g["chi_round"] = np.round(g["chi_sq_loo"].to_numpy(), 2)
    g["p_print"] = fmt_p(g["p_value_loo"].to_numpy())

    table3_base_fixed = g

    rank = table3_base_fixed.groupby(["occupation", "label"], as_index=False, sort=True).agg(
        n_models=("Model_key", "nunique"),
        mean_pct=("pct", "mean"),
        min_pct=("pct", "min"),
        mean_p=("p_value_loo", "mean"),
        max_chi=("chi_sq_loo", "max"),
    )
    rank = rank[rank["n_models"] == len(TABLE3_MODELS)]
    rank = rank.sort_values(
        ["mean_pct", "min_pct", "mean_p", "occupation", "label"],
        ascending=[False, False, True, True, True],
        kind="stable",
    ).head(TABLE3_N_ROWS)

    base_sel = table3_base_fixed[["occupation", "label", "Model_key", "pct_round", "chi_round", "p_print"]]
    base_sel = base_sel[
        base_sel.set_index(["occupation", "label"]).index.isin(rank.set_index(["occupation", "label"]).index)
    ]

    wide = base_sel.pivot_table(
        index=["occupation", "label"], columns="Model_key", values=["pct_round", "chi_round", "p_print"], aggfunc="first", observed=True
    )
    wide = wide.reset_index()

    out = pd.DataFrame({"occupation": wide["occupation"], "label": wide["label"]})
    for mk in TABLE3_MODELS:
        for vc in ["pct_round", "chi_round", "p_print"]:
            colname = f"{mk}_{vc}"
            if (vc, mk) in wide.columns:
                out[colname] = wide[(vc, mk)].to_numpy()
            else:
                out[colname] = np.nan

    out = out.merge(rank, on=["occupation", "label"], how="left")
    out = out.sort_values(
        ["mean_pct", "min_pct", "mean_p", "occupation", "label"],
        ascending=[False, False, True, True, True],
        kind="stable",
    ).reset_index(drop=True)

    required_cols = [
        "XL_pct_round", "M35_pct_round", "Flx_pct_round", "Qwen_pct_round",
        "XL_chi_round", "M35_chi_round", "Flx_chi_round", "Qwen_chi_round",
        "XL_p_print", "M35_p_print", "Flx_p_print", "Qwen_p_print",
    ]
    for c in required_cols:
        if c not in out.columns:
            out[c] = np.nan

    out = out[["occupation", "label"] + required_cols]
    out = out.rename(
        columns={
            "occupation": "Occupation", "label": "Label",
            "XL_pct_round": "XL_%", "M35_pct_round": "3.5_%", "Flx_pct_round": "Flx_%", "Qwen_pct_round": "Qwen_%",
            "XL_chi_round": "XL_chi2", "M35_chi_round": "3.5_chi2", "Flx_chi_round": "Flx_chi2", "Qwen_chi_round": "Qwen_chi2",
            "XL_p_print": "XL_p", "M35_p_print": "3.5_p", "Flx_p_print": "Flx_p", "Qwen_p_print": "Qwen_p",
        }
    )
    return out


# =========================================================
# FIGURE 2 / FIGURE 3 DATA
# =========================================================


def build_fig2_data(overall_prop_all: pd.DataFrame, fig_prompt_group: str) -> pd.DataFrame:
    d = overall_prop_all[
        (overall_prop_all["prompt_group"] == fig_prompt_group)
        & (~overall_prop_all["cohort"].isin(["totals", "safety"]))
        & overall_prop_all["label"].notna()
        & (overall_prop_all["label"] != "unknown")
    ].copy()

    d["Model_plot"] = d["Model"].map(PLOT_MODEL_MAP).fillna(d["Model"])
    d["Model_plot"] = pd.Categorical(d["Model_plot"], categories=["SD3.5", "SDXL", "Flux", "Qwen", "ALL"], ordered=True)
    d["Context_short"] = context_shortener(d["Context"])
    d["Context_short"] = pd.Categorical(d["Context_short"], categories=["CA-R", "CA-U", "CF"], ordered=True)
    d["dimension_family"] = dimension_to_family(d["cohort"])
    d["dimension_family"] = pd.Categorical(
        d["dimension_family"], categories=["People", "Scene", "Camera", "Objects"], ordered=True
    )

    dd = d[
        ["Model_plot", "Context_short", "dimension_family", "dimension", "cohort", "label", "prop_context"]
    ].drop_duplicates()

    inten = dd.groupby(
        ["Model_plot", "Context_short", "dimension_family", "dimension", "cohort"], as_index=False, sort=True, observed=True
    ).agg(intensity=("prop_context", lambda s: bias_severity_score(s.to_numpy())))

    out = inten.groupby(["Model_plot", "Context_short", "dimension_family"], as_index=False, sort=True, observed=True).agg(
        mean_intensity=("intensity", "mean")
    )
    out = out[out["dimension_family"].notna() & out["Context_short"].notna()]
    return out


def build_fig3_dim_data(overall_prop_all: pd.DataFrame, fig_prompt_group: str) -> pd.DataFrame:
    d = overall_prop_all[
        (overall_prop_all["prompt_group"] == fig_prompt_group)
        & (~overall_prop_all["cohort"].isin(["totals", "safety"]))
        & overall_prop_all["label"].notna()
        & (overall_prop_all["label"] != "unknown")
    ].copy()

    d["Model_plot"] = d["Model"].map(PLOT_MODEL_MAP).fillna(d["Model"])
    d["Model_plot"] = pd.Categorical(d["Model_plot"], categories=["SD3.5", "SDXL", "Flux", "Qwen", "ALL"], ordered=True)

    step1 = d.groupby(["Model_plot", "dimension", "cohort", "label"], as_index=False, sort=True, observed=True).agg(
        count=("count", "sum")
    )
    step2 = step1.groupby(["Model_plot", "dimension", "cohort"], as_index=False, sort=True, observed=True).agg(
        avg_intensity=("count", lambda s: bias_severity_score(s.to_numpy()))
    )
    step3 = step2.groupby(["Model_plot", "dimension"], as_index=False, sort=True, observed=True).agg(
        avg_intensity=("avg_intensity", "mean")
    )
    return step3


# =========================================================
# MAIN
# =========================================================


def main():
    overall_prop_all = build_overall_prop_all()
    print(f"overall_prop_all: {len(overall_prop_all)} rows", file=sys.stderr)
    overall_prop_all.to_csv(os.path.join(OUT_DIR, "overall_prop_all_loo.csv"), index=False)

    bias_occ_loo = build_bias_occ_loo(overall_prop_all)
    bias_occ_loo.to_csv(os.path.join(OUT_DIR, "bias_occ_loo.csv"), index=False)
    print(f"bias_occ_loo: {len(bias_occ_loo)} rows", file=sys.stderr)

    occ_table_all = build_occ_table_all(overall_prop_all)
    occ_table_all.to_csv(os.path.join(OUT_DIR, "occ_table_all.csv"), index=False)
    print(f"occ_table_all: {len(occ_table_all)} rows", file=sys.stderr)

    occ_table_all_wide = build_occ_table_all_wide(occ_table_all)
    occ_table_all_wide.to_csv(os.path.join(OUT_DIR, "occ_table_all_wide.csv"), index=False)
    print(f"occ_table_all_wide: {len(occ_table_all_wide)} rows", file=sys.stderr)

    preferred_prompt_group = get_preferred_prompt_group(overall_prop_all)
    print(f"preferred_prompt_group: {preferred_prompt_group}", file=sys.stderr)

    label_persistence_loo = build_label_persistence_loo(overall_prop_all)

    t2 = build_table2(overall_prop_all, preferred_prompt_group)
    t2["table2_no_type_isolated"].to_csv(os.path.join(OUT_DIR, "table2_no_type_isolated_baseline.csv"), index=False)
    t2["table2_no_type_isolated_long"].to_csv(os.path.join(OUT_DIR, "table2_no_type_isolated_baseline_long.csv"), index=False)
    t2["table2_other_type_robustness"].to_csv(os.path.join(OUT_DIR, "table2_other_type_robustness.csv"), index=False)
    t2["table2_other_type_robustness_long"].to_csv(os.path.join(OUT_DIR, "table2_other_type_robustness_long.csv"), index=False)
    t2["table2_prompt_robustness"].to_csv(os.path.join(OUT_DIR, "table2_prompt_robustness.csv"), index=False)
    t2["table2_prompt_robustness_long"].to_csv(os.path.join(OUT_DIR, "table2_prompt_robustness_long.csv"), index=False)

    table3 = build_table3_label_persistence(label_persistence_loo, preferred_prompt_group)
    table3.to_csv(os.path.join(OUT_DIR, "table3_label_persistence.csv"), index=False)
    print(f"table3_label_persistence: {len(table3)} rows", file=sys.stderr)

    fig2_data = build_fig2_data(overall_prop_all, preferred_prompt_group)
    fig2_data.to_csv(os.path.join(OUT_DIR, "fig2_data.csv"), index=False)

    fig3_dim_data = build_fig3_dim_data(overall_prop_all, preferred_prompt_group)
    fig3_dim_data.to_csv(os.path.join(OUT_DIR, "fig3_dim_data.csv"), index=False)

    print("DONE", file=sys.stderr)


if __name__ == "__main__":
    main()
