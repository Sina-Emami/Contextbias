"""Shared label-agreement helpers for the human-study analysis scripts.

Every reviewer/pipeline CSV is an aggregated (cohort, dimension, label, count, bin) table:
one row per distinct label that was ever selected for a given (cohort, dimension) across the
images in scope, with `count` giving how many images received that label. The "majority
label" for a (cohort, dimension) pair is simply the label with the highest count.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

_MODEL_TOKENS = {"sdxl", "3.5", "flux", "qwen"}


def role_model_from_filename(path: str) -> tuple[str, str, str]:
    """'.../web_developer_SDXL.csv' -> ('web_developer_SDXL', 'web developer', 'SDXL').

    The model is taken as the last underscore-separated token if it matches a known model
    name; otherwise the whole stem is treated as the role with an empty model, which keeps
    the caller's file lookups (which key off the stem, not the parsed role/model) working
    regardless.
    """
    stem = re.sub(r"\.csv$", "", path.split("/")[-1].split("\\")[-1])
    if "_" in stem:
        role_part, model_part = stem.rsplit("_", 1)
        if model_part.lower() in _MODEL_TOKENS:
            return stem, role_part.replace("_", " "), model_part
    return stem, stem.replace("_", " "), ""


def _majority_labels(df: pd.DataFrame, label_col: str = "label") -> pd.DataFrame:
    """One row per (cohort, dimension): the label with the highest count."""
    idx = df.groupby(["cohort", "dimension"])["count"].idxmax()
    out = df.loc[idx, ["cohort", "dimension", label_col if label_col in df.columns else "label"]].copy()
    if label_col != "label" and label_col not in out.columns:
        out = out.rename(columns={"label": label_col})
    return out.reset_index(drop=True)


def merge_majority_labels(
    df_a: pd.DataFrame, col_a: str, df_b: pd.DataFrame, col_b: str
) -> pd.DataFrame:
    """Inner-join two reviewers' majority labels on (cohort, dimension)."""
    maj_a = _majority_labels(df_a).rename(columns={"label": col_a})
    maj_b = _majority_labels(df_b).rename(columns={"label": col_b})
    return maj_a.merge(maj_b, on=["cohort", "dimension"], how="inner")


def merge_all_labels(reviewer_dfs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Inner-join every reviewer's majority labels on (cohort, dimension), one column per
    reviewer (keyed by the dict keys). Used for multi-rater statistics (Fleiss' kappa),
    where every item needs a label from every rater.
    """
    if not reviewer_dfs:
        return pd.DataFrame(columns=["cohort", "dimension"])

    merged = None
    for name, df in reviewer_dfs.items():
        maj = _majority_labels(df).rename(columns={"label": name})
        merged = maj if merged is None else merged.merge(maj, on=["cohort", "dimension"], how="inner")
    return merged


def cohens_kappa(labels_a, labels_b) -> float:
    labels_a = pd.Series(labels_a).reset_index(drop=True)
    labels_b = pd.Series(labels_b).reset_index(drop=True)
    n = len(labels_a)
    if n == 0:
        return float("nan")

    categories = sorted(set(labels_a) | set(labels_b))
    idx = {c: i for i, c in enumerate(categories)}
    k = len(categories)

    confusion = np.zeros((k, k), dtype=float)
    for a, b in zip(labels_a, labels_b):
        confusion[idx[a], idx[b]] += 1

    po = np.trace(confusion) / n
    row_marg = confusion.sum(axis=1) / n
    col_marg = confusion.sum(axis=0) / n
    pe = float(np.sum(row_marg * col_marg))

    if pe >= 1.0:
        return 1.0 if po >= 1.0 else 0.0
    return (po - pe) / (1 - pe)


def score_labels(labels_a, labels_b) -> dict:
    """Symmetric agreement stats treating `a` as ground truth for accuracy/macro-recall
    (only meaningful when the caller actually has a ground-truth side, e.g. human vs.
    pipeline; harmless but uninterpreted for a human-human pair)."""
    labels_a = pd.Series(labels_a).reset_index(drop=True)
    labels_b = pd.Series(labels_b).reset_index(drop=True)
    n = len(labels_a)
    if n == 0:
        return {"raw_agreement": float("nan"), "kappa": float("nan"), "accuracy": float("nan"), "macro_recall": float("nan")}

    matches = (labels_a == labels_b)
    raw_agreement = float(matches.mean())

    recalls = []
    for cls in sorted(labels_a.unique()):
        mask = labels_a == cls
        if mask.sum() == 0:
            continue
        recalls.append(float(matches[mask].mean()))
    macro_recall = float(np.mean(recalls)) if recalls else float("nan")

    return {
        "raw_agreement": raw_agreement,
        "kappa": cohens_kappa(labels_a, labels_b),
        "accuracy": raw_agreement,
        "macro_recall": macro_recall,
    }


def fleiss_kappa(rating_matrix: np.ndarray) -> float:
    """Standard Fleiss' kappa for N items x R raters, given as an N x K matrix of category
    counts per item (rating_matrix[i, k] = number of raters who assigned item i to category
    k). Use `category_counts` to build this from a wide label table.
    """
    n_items, n_categories = rating_matrix.shape
    n_raters = rating_matrix.sum(axis=1)
    if not np.allclose(n_raters, n_raters[0]):
        raise ValueError("Fleiss' kappa requires the same number of raters per item.")
    n_raters = n_raters[0]

    p_j = rating_matrix.sum(axis=0) / (n_items * n_raters)
    p_e = float(np.sum(p_j**2))

    p_i = (np.sum(rating_matrix**2, axis=1) - n_raters) / (n_raters * (n_raters - 1))
    p_bar = float(np.mean(p_i))

    if p_e >= 1.0:
        return 1.0 if p_bar >= 1.0 else 0.0
    return (p_bar - p_e) / (1 - p_e)


def category_counts(wide_labels: pd.DataFrame, rater_cols: list[str]) -> tuple[np.ndarray, list]:
    """wide_labels has one row per item and one column per rater (as produced by
    merge_all_labels). Returns (rating_matrix, categories) for fleiss_kappa."""
    categories = sorted(set(wide_labels[rater_cols].to_numpy().ravel()))
    idx = {c: i for i, c in enumerate(categories)}

    matrix = np.zeros((len(wide_labels), len(categories)), dtype=float)
    for row_i, (_, row) in enumerate(wide_labels[rater_cols].iterrows()):
        for value in row:
            matrix[row_i, idx[value]] += 1
    return matrix, categories
