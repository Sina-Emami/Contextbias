"""
Flatten and visualize summary_report.json distributions.

Features
- Load JSON summary report(s)
- Build tidy DataFrames for groups, attributes, and 3x3 spatial data
- Collapse small categories (<5%) into "Other (k)" with audit mapping
- Optional Wilson 95% CI for small samples (n < 30)
- Matplotlib-only visualizations (colorblind-safe palettes)

Usage
- Edit HARDCODED_INPUTS below to point at the summary_report.json file(s) or directories to process.
- Run `python -m decription_pipeline.data_processing.context_metrics` to build the CSVs and figures.
"""

import hashlib
import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ----------------------------
# Hardcoded configuration
# ----------------------------

# Edit this list to include summary_report.json files or directories containing them.
HARDCODED_INPUTS: List[Path | str] = [
    Path(
        "dataset/Context-aware_Related_CA-R/doctor/a_photo_of_a_doctor_in_a_hospital_room/descriptions/summary/summary_report.json"
    )
]

# Destination for generated CSVs/plots/manifests.
OUTPUT_DIR = Path("analysis_output")

# Processing flags.
COLLAPSE_THRESHOLD = 0.05
DROP_UNKNOWN = True


# ----------------------------
# Loading and flattening utils
# ----------------------------


def load_summary_report(path: Path | str) -> Dict[str, Any]:
    p = Path(path)
    data = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or "cohorts" not in data:
        raise ValueError(f"Invalid summary_report JSON at {p}")
    return data


def _safe_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def flatten_groups(report: Mapping[str, Any]) -> pd.DataFrame:
    """Return tidy groups dataframe: cohort|key|sub_key|label|total_count|normalized_share."""
    rows: List[Dict[str, Any]] = []
    for cohort in report.get("cohorts", []):
        cohort_name = cohort.get("cohort")
        for g in cohort.get("groups", []) or []:
            rows.append(
                {
                    "cohort": cohort_name,
                    "key": g.get("key"),
                    "sub_key": g.get("sub_key"),
                    "label": g.get("canonical_label"),
                    "total_count": int(g.get("total_count", 0) or 0),
                    "normalized_share": _safe_float(g.get("normalized_share")),
                }
            )
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(
            columns=["cohort", "key", "sub_key", "label", "total_count", "normalized_share"]
        )
    return df


def flatten_attributes(report: Mapping[str, Any]) -> pd.DataFrame:
    """Return tidy attributes dataframe: cohort|attribute|value|count|normalized."""
    rows: List[Dict[str, Any]] = []
    for cohort in report.get("cohorts", []):
        cohort_name = cohort.get("cohort")
        attrs = cohort.get("attributes") or {}
        for attr_name, payload in attrs.items():
            if not isinstance(payload, Mapping):
                # Some summaries store free-form strings (e.g., notes) here; skip those.
                continue
            vc = (payload or {}).get("value_counts") or {}
            norm = (payload or {}).get("normalized") or {}
            keys = set(vc.keys()) | set(norm.keys())
            for val in keys:
                rows.append(
                    {
                        "cohort": cohort_name,
                        "attribute": attr_name,
                        "value": val,
                        "count": int(vc.get(val, 0) or 0),
                        "normalized": _safe_float(norm.get(val, np.nan)),
                    }
                )
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(
            columns=["cohort", "attribute", "value", "count", "normalized"]
        )
    return df


def _split_plane_side(token: str) -> Optional[Tuple[str, str]]:
    if not isinstance(token, str) or "." not in token:
        return None
    parts = token.split(".")
    if len(parts) != 2:
        return None
    plane, side = parts
    if plane not in {"foreground", "midground", "background"}:
        return None
    if side not in {"left", "center", "right"}:
        return None
    return plane, side


def extract_spatial_layout(report: Mapping[str, Any]) -> pd.DataFrame:
    """Extract 3x3 spatial counts from either groups or attributes.

    Returns columns: plane|side|count|normalized_share
    """
    counts: Dict[Tuple[str, str], int] = {}
    norms: Dict[Tuple[str, str], float] = {}

    # 1) Inspect cohort groups for key == object_counts_by_position
    for cohort in report.get("cohorts", []):
        for g in cohort.get("groups", []) or []:
            if g.get("key") == "object_counts_by_position":
                split = _split_plane_side(g.get("sub_key"))
                if split:
                    counts[split] = counts.get(split, 0) + int(g.get("total_count", 0) or 0)
                    ns = g.get("normalized_share")
                    if ns is not None:
                        norms[split] = float(ns)

    # 2) Inspect attributes named like object_density/object_counts_by_position
    for cohort in report.get("cohorts", []):
        attrs = cohort.get("attributes") or {}
        for attr_name, payload in attrs.items():
            if not isinstance(payload, Mapping):
                continue
            if attr_name not in {"object_density", "object_counts_by_position", "object_counts_by_position_total"}:
                # Still allow spatial-looking keys under any attribute
                pass
            vc = (payload or {}).get("value_counts") or {}
            nm = (payload or {}).get("normalized") or {}
            for key, v in vc.items():
                split = _split_plane_side(key)
                if not split:
                    continue
                counts[split] = counts.get(split, 0) + int(v or 0)
            for key, v in nm.items():
                split = _split_plane_side(key)
                if not split:
                    continue
                norms[split] = float(v)

    # Convert to DataFrame
    rows: List[Dict[str, Any]] = []
    if counts:
        total = sum(counts.values())
        for (plane, side), c in counts.items():
            rows.append(
                {
                    "plane": plane,
                    "side": side,
                    "count": int(c),
                    "normalized_share": (c / total) if total else np.nan,
                    "normalized_share_from_attr": norms.get((plane, side), np.nan),
                }
            )
    elif norms:
        # Fallback if only normalized shares are provided
        for (plane, side), v in norms.items():
            rows.append(
                {
                    "plane": plane,
                    "side": side,
                    "count": np.nan,
                    "normalized_share": float(v),
                    "normalized_share_from_attr": float(v),
                }
            )
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=["plane", "side", "count", "normalized_share", "normalized_share_from_attr"])
    return df


# ----------------------------
# Aggregation helpers shared with dataset rollup
# ----------------------------


def make_dimension_name(cohort: str | None, key: str | None) -> str:
    cohort_token = str(cohort or "unknown")
    key_token = str(key or "unknown")
    prefix = f"{cohort_token}_"
    if key_token.startswith(prefix):
        key_token = key_token[len(prefix) :]
    return f"{cohort_token}.{key_token}"


def compute_dimension_counts(
    report: Mapping[str, Any], drop_unknown: bool = False
) -> Tuple[Dict[str, Dict[str, int]], Dict[Tuple[str, str], int], int]:
    """Aggregate group/attribute counts and spatial layout counts from a report."""

    aggregated: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    spatial_counts: Dict[Tuple[str, str], int] = defaultdict(int)

    metadata = report.get("metadata") or {}
    num_images = int(metadata.get("num_images", 0) or 0)

    group_df = flatten_groups(report)
    if not group_df.empty:
        data = group_df.copy()
        data["cohort"] = data["cohort"].fillna("unknown").astype(str)
        data["key"] = data["key"].fillna("unknown").astype(str)
        data["label"] = (
            data["label"].fillna(data["sub_key"]).fillna("unknown").astype(str)
        )
        if drop_unknown:
            data = data[data["sub_key"].astype(str).str.lower() != "unknown"]
        grouped = data.groupby(["cohort", "key", "label"], dropna=False)["total_count"].sum()
        for (cohort, key, label), count in grouped.items():
            dimension = make_dimension_name(cohort, key)
            aggregated[dimension][str(label)] += int(count)

    attr_df = flatten_attributes(report)
    if not attr_df.empty:
        data = attr_df.copy()
        data["cohort"] = data["cohort"].fillna("unknown").astype(str)
        data["attribute"] = data["attribute"].fillna("unknown").astype(str)
        data["value"] = data["value"].fillna("unknown").astype(str)
        if drop_unknown:
            data = data[data["value"].astype(str).str.lower() != "unknown"]
        grouped = data.groupby(["cohort", "attribute", "value"], dropna=False)["count"].sum()
        for (cohort, attribute, value), count in grouped.items():
            dimension = make_dimension_name(cohort, attribute)
            aggregated[dimension][str(value)] += int(count)

    spatial_df = extract_spatial_layout(report)
    if not spatial_df.empty:
        spatial_data = spatial_df.copy()
        if "count" in spatial_data.columns:
            spatial_data = spatial_data[pd.notna(spatial_data["count"])]
            grouped = spatial_data.groupby(["plane", "side"], dropna=False)["count"].sum()
            for (plane, side), count in grouped.items():
                spatial_counts[(str(plane), str(side))] += int(count)

    aggregated_out: Dict[str, Dict[str, int]] = {
        dimension: {label: int(value) for label, value in labels.items()}
        for dimension, labels in aggregated.items()
    }
    spatial_out: Dict[Tuple[str, str], int] = {
        (plane, side): int(value) for (plane, side), value in spatial_counts.items()
    }

    return aggregated_out, spatial_out, num_images


# ----------------------------
# Computation helpers
# ----------------------------


def wilson_ci(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n <= 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + (z * z) / n
    center = p + (z * z) / (2 * n)
    margin = z * math.sqrt((p * (1 - p)) / n + (z * z) / (4 * n * n))
    lo = max(0.0, (center - margin) / denom)
    hi = min(1.0, (center + margin) / denom)
    return lo, hi


@dataclass
class CollapseResult:
    df: pd.DataFrame
    audit_mapping: Dict[str, List[str]]


def collapse_small_categories(
    df: pd.DataFrame,
    label_col: str,
    count_col: str,
    share_col: str,
    threshold: float = 0.05,
) -> CollapseResult:
    """Merge categories with share < threshold into an Other bucket.

    Returns a new df plus audit mapping: {"Other (k)": [labels...]}
    """
    if df.empty:
        return CollapseResult(df.copy(), {})

    df = df.copy()
    small_mask = df[share_col].fillna(0) < threshold
    large = df.loc[~small_mask]
    small = df.loc[small_mask]

    if small.empty:
        return CollapseResult(df.sort_values(count_col, ascending=False).reset_index(drop=True), {})

    other_count = int(small[count_col].fillna(0).sum())
    other_share = float(small[share_col].fillna(0).sum())
    other_label = f"Other ({other_count})"

    other_row = {
        label_col: other_label,
        count_col: other_count,
        share_col: other_share,
    }
    out = pd.concat([large, pd.DataFrame([other_row])], ignore_index=True)
    out = out.sort_values(count_col, ascending=False).reset_index(drop=True)
    mapping = {other_label: sorted(map(str, small[label_col].tolist()))}
    return CollapseResult(out, mapping)


# ----------------------------
# Plotting helpers (matplotlib only)
# ----------------------------


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _slugify(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    slug = re.sub(r"-+", "-", text).strip("-")
    return slug or "value"


def _slugify_limited(text: str, max_length: int) -> str:
    slug = _slugify(text)
    if len(slug) <= max_length:
        return slug
    digest = hashlib.md5(text.encode("utf-8")).hexdigest()[:8]
    trimmed = slug[: max_length - 9].rstrip("-")
    if not trimmed:
        trimmed = slug[: max_length - 9]
    return f"{trimmed}-{digest}"


def _pick_palette(n: int, name: str = "tab10") -> List[str]:
    cmap = plt.get_cmap(name)
    # Matplotlib categorical colormaps wrap; sample evenly
    return [cmap(i % cmap.N) for i in range(n)]


def plot_bar_sorted(
    df: pd.DataFrame,
    label_col: str,
    count_col: str,
    share_col: str,
    title: str,
    outpath: Path,
    add_percent_labels: bool = True,
    ci_half_width: Optional[Sequence[float]] = None,
) -> None:
    if df.empty:
        return
    order = df.sort_values(count_col, ascending=True)
    labels = order[label_col].astype(str).tolist()
    counts = order[count_col].astype(float).tolist()
    shares = order[share_col].astype(float).tolist()

    fig, ax = plt.subplots(figsize=(8, max(2.5, 0.35 * len(labels))))
    colors = _pick_palette(len(labels), name="Set2")
    y = np.arange(len(labels))
    ax.barh(y, counts, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_yticks(y, labels)
    ax.set_xlabel("Count")
    ax.set_title(title)

    if add_percent_labels:
        for i, (c, s) in enumerate(zip(counts, shares)):
            pct = f"{s*100:.1f}%"
            ax.text(c + max(counts) * 0.01, i, f"{int(c)} ({pct})", va="center", fontsize=8)

    # Optional CI as error bars overlaid as thin whiskers near bar end (for small n)
    if ci_half_width is not None:
        # Draw as x-error bars centered at c with width = 2*half_width*n (on count axis)
        n_total = sum(counts)
        xerr = np.array(ci_half_width) * 2.0 * n_total
        ax.errorbar(counts, y, xerr=xerr, fmt="none", ecolor="gray", elinewidth=1)

    ax.grid(axis="x", linestyle=":", alpha=0.4)
    fig.tight_layout()
    _ensure_dir(outpath.parent)
    fig.savefig(outpath, dpi=200)
    plt.close(fig)


def plot_stacked_100(
    parts: Sequence[Tuple[str, float, int]],
    title: str,
    outpath: Path,
    bar_label: Optional[str] = None,
) -> None:
    """Draw a single 100% stacked bar from (label, share, count) segments."""
    if not parts:
        return
    labels, shares, counts = zip(*parts)
    shares = np.array(shares, dtype=float)
    # Sort by share desc
    order = np.argsort(-shares)
    labels = [labels[i] for i in order]
    shares = shares[order]
    counts = [counts[i] for i in order]

    fig, ax = plt.subplots(figsize=(8, 1.0 + 0.2 * len(labels)))
    colors = _pick_palette(len(labels), name="tab10")
    left = 0.0
    for i, (lab, sh, ct) in enumerate(zip(labels, shares, counts)):
        ax.barh([0], [sh], left=left, color=colors[i], edgecolor="black", linewidth=0.5, label=lab)
        # In-bar label if space allows
        if sh >= 0.06:
            ax.text(left + sh / 2, 0, f"{lab}\n{ct} ({sh*100:.0f}%)", ha="center", va="center", fontsize=8)
        left += sh
    ax.set_xlim(0, 1)
    ax.set_yticks([0], [bar_label or "Distribution"])
    ax.set_xlabel("Share (100%)")
    ax.set_title(title)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncols=min(4, len(labels)), fontsize=8)
    ax.grid(axis="x", linestyle=":", alpha=0.4)
    fig.tight_layout()
    _ensure_dir(outpath.parent)
    fig.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_dot_with_ci(
    df: pd.DataFrame,
    label_col: str,
    share_col: str,
    n_total: int,
    title: str,
    outpath: Path,
    cis: Optional[Sequence[Tuple[float, float]]] = None,
) -> None:
    if df.empty:
        return
    order = df.sort_values(share_col, ascending=True)
    labels = order[label_col].astype(str).tolist()
    shares = order[share_col].astype(float).tolist()
    y = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(8, max(2.5, 0.35 * len(labels))))
    colors = _pick_palette(len(labels), name="Set2")

    ax.scatter(shares, y, color=colors, s=40, edgecolor="black", linewidth=0.5)
    if cis is not None:
        lows = [max(0.0, s - ci[0]) for s, ci in zip(shares, cis)]
        highs = [max(0.0, ci[1] - s) for s, ci in zip(shares, cis)]
        ax.errorbar(shares, y, xerr=[lows, highs], fmt="none", ecolor="gray", elinewidth=1)

    for i, (s, lab) in enumerate(zip(shares, labels)):
        ax.text(s + 0.01, i, f"{s*100:.1f}%", va="center", fontsize=8)

    ax.set_yticks(y, labels)
    ax.set_xlabel(f"Share (n={n_total})")
    ax.set_xlim(0, 1)
    ax.set_title(title)
    ax.grid(axis="x", linestyle=":", alpha=0.4)
    fig.tight_layout()
    _ensure_dir(outpath.parent)
    fig.savefig(outpath, dpi=200)
    plt.close(fig)


def plot_spatial_heatmap(
    df_spatial: pd.DataFrame,
    use_normalized: bool,
    title: str,
    outpath: Path,
) -> None:
    if df_spatial.empty:
        return
    planes = ["foreground", "midground", "background"]
    sides = ["left", "center", "right"]
    mat = np.zeros((len(planes), len(sides)))
    for i, pl in enumerate(planes):
        for j, sd in enumerate(sides):
            row = df_spatial[(df_spatial["plane"] == pl) & (df_spatial["side"] == sd)]
            if row.empty:
                val = 0.0
            else:
                if use_normalized:
                    v = row["normalized_share_from_attr"].values[0]
                    val = float(v) if pd.notna(v) else float(row["normalized_share"].values[0])
                else:
                    v = row["count"].values[0]
                    val = float(v) if pd.notna(v) else 0.0
            mat[i, j] = val

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(mat, cmap="YlOrRd", aspect="equal", vmin=0, vmax=(1 if use_normalized else mat.max() or 1))
    ax.set_xticks(range(len(sides)), sides)
    ax.set_yticks(range(len(planes)), planes)
    ax.set_xlabel("Side")
    ax.set_ylabel("Plane")
    ax.set_title(title)
    # Add cell labels
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            txt = f"{mat[i,j]*100:.0f}%" if use_normalized else f"{int(mat[i,j])}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=9, color="black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    _ensure_dir(outpath.parent)
    fig.savefig(outpath, dpi=200)
    plt.close(fig)


def plot_plane_and_side_bars(
    df_spatial: pd.DataFrame,
    outdir: Path,
    base_title: str,
) -> None:
    if df_spatial.empty:
        return
    # Plane counts
    plane_df = (
        df_spatial.groupby("plane", as_index=False)["count"].sum().sort_values("count", ascending=False)
    )
    plot_bar_sorted(
        plane_df,
        label_col="plane",
        count_col="count",
        share_col="count",  # not used for labels here
        title=f"{base_title} – Objects by plane",
        outpath=outdir / "spatial_plane_counts.png",
        add_percent_labels=False,
    )
    # Side counts
    side_df = (
        df_spatial.groupby("side", as_index=False)["count"].sum().sort_values("count", ascending=False)
    )
    plot_bar_sorted(
        side_df,
        label_col="side",
        count_col="count",
        share_col="count",
        title=f"{base_title} – Objects by side",
        outpath=outdir / "spatial_side_counts.png",
        add_percent_labels=False,
    )


# ----------------------------
# Orchestration: build DFs and plots per summary file
# ----------------------------


def generate_all_plots_for_summary(
    summary_path: Path,
    output_dir: Path,
    collapse_threshold: float = 0.05,
    drop_unknown: bool = True,
) -> Dict[str, Any]:
    """Create DataFrames and charts for the given summary file.

    Returns a manifest with paths and audit mappings.
    """
    report = load_summary_report(summary_path)
    meta = report.get("metadata", {})
    num_images = int(meta.get("num_images", 0) or 0)

    # Output folder per summary
    try:
        rel_tag = summary_path.relative_to(Path.cwd())
    except ValueError:
        rel_tag = summary_path
    stem = _slugify_limited(str(rel_tag), max_length=80)
    outdir = output_dir / stem
    _ensure_dir(outdir)

    # Flatten
    df_groups = flatten_groups(report)
    df_attrs = flatten_attributes(report)
    df_spatial = extract_spatial_layout(report)

    # Save flattened data for reuse
    df_groups.to_csv(outdir / "groups.csv", index=False)
    df_attrs.to_csv(outdir / "attributes.csv", index=False)
    df_spatial.to_csv(outdir / "spatial.csv", index=False)

    manifest: Dict[str, Any] = {
        "summary_path": str(summary_path),
        "outdir": str(outdir),
        "num_images": num_images,
        "artifacts": [],
        "audits": {},
    }

    # 4a. Sorted bar plots for categorical distributions (per cohort/key)
    for (cohort, key), g in df_groups.groupby(["cohort", "key"], dropna=False):
        data = g.copy()
        if drop_unknown:
            data = data[data["sub_key"].astype(str).str.lower() != "unknown"]
        # Sum counts per sub_key (should already be unique)
        agg = (
            data.groupby(["cohort", "key", "sub_key", "label"], as_index=False)[
                ["total_count", "normalized_share"]
            ]
            .sum()
            .rename(columns={"total_count": "count", "normalized_share": "share"})
        )
        if agg.empty:
            continue
        # Collapse small categories
        collapsed = collapse_small_categories(
            agg.rename(columns={"sub_key": "category", "share": "normalized"}),
            label_col="category",
            count_col="count",
            share_col="normalized",
            threshold=collapse_threshold,
        )
        out = collapsed.df
        manifest["audits"][f"{cohort}|{key}"] = collapsed.audit_mapping

        title = f"{cohort} – {key} (n={int(out['count'].sum())})"
        cohort_slug = _slugify_limited(str(cohort), max_length=30)
        key_slug = _slugify_limited(str(key), max_length=40)
        bar_path = outdir / f"bar_{cohort_slug}_{key_slug}.png"
        plot_bar_sorted(
            out,
            label_col="category",
            count_col="count",
            share_col="normalized",
            title=title,
            outpath=bar_path,
        )
        manifest["artifacts"].append(str(bar_path))

        # 4b. 100% stacked bar if this key partitions a whole (~sums to 1)
        share_sum = float(out["normalized"].sum()) if not out.empty else 0.0
        if 0.98 <= share_sum <= 1.02 and len(out) > 1:
            parts = list(zip(out["category"].tolist(), out["normalized"].tolist(), out["count"].tolist()))
            stack_path = outdir / f"stack100_{cohort_slug}_{key_slug}.png"
            plot_stacked_100(
                parts,
                title=f"{cohort} – {key} (100% stack)",
                outpath=stack_path,
                bar_label=str(key),
            )
            manifest["artifacts"].append(str(stack_path))

        # 5. Optional Wilson 95% CI when sample size < 30 (dot plot)
        n_total = int(agg["count"].sum())
        if n_total and n_total < 30:
            cis = [wilson_ci(int(c), n_total) for c in agg["count"].tolist()]
            dot_ci_path = outdir / f"dot_{cohort_slug}_{key_slug}_ci.png"
            plot_dot_with_ci(
                agg.rename(columns={"sub_key": "category", "normalized_share": "share"}),
                label_col="category",
                share_col="share",
                n_total=n_total,
                title=f"{cohort} – {key} (proportions with 95% Wilson CI)",
                outpath=dot_ci_path,
                cis=cis,
            )
            manifest["artifacts"].append(str(dot_ci_path))

    # 4c. Dot plots for attributes per (cohort, attribute)
    for (cohort, attr), g in df_attrs.groupby(["cohort", "attribute"], dropna=False):
        data = g.copy()
        if drop_unknown:
            data = data[data["value"].astype(str).str.lower() != "unknown"]
        # Aggregate and normalize if needed
        agg = (
            data.groupby(["value"], as_index=False)[["count", "normalized"]]
            .sum()
            .sort_values("normalized", ascending=False)
        )
        if agg.empty:
            continue
        n_total = int(agg["count"].sum())
        cis = None
        if n_total and n_total < 30:
            cis = [wilson_ci(int(c), n_total) for c in agg["count"].tolist()]
        cohort_slug = _slugify_limited(str(cohort), max_length=30)
        attr_slug = _slugify_limited(str(attr), max_length=40)
        dot_attr_path = outdir / f"dot_attr_{cohort_slug}_{attr_slug}.png"
        plot_dot_with_ci(
            agg.rename(columns={"value": "category", "normalized": "share"}),
            label_col="category",
            share_col="share",
            n_total=n_total,
            title=f"{cohort} – attribute: {attr}",
            outpath=dot_attr_path,
            cis=cis,
        )
        manifest["artifacts"].append(str(dot_attr_path))

    # 4d. 3×3 heatmap for spatial object layout
    if not df_spatial.empty:
        title_counts = "Spatial object layout (counts)"
        plot_spatial_heatmap(
            df_spatial,
            use_normalized=False,
            title=title_counts,
            outpath=outdir / "heatmap_spatial_counts.png",
        )
        manifest["artifacts"].append(str(outdir / "heatmap_spatial_counts.png"))

        title_share = "Spatial object layout (normalized)"
        plot_spatial_heatmap(
            df_spatial,
            use_normalized=True,
            title=title_share,
            outpath=outdir / "heatmap_spatial_normalized.png",
        )
        manifest["artifacts"].append(str(outdir / "heatmap_spatial_normalized.png"))

        # 4e. Separate bars for plane and side
        plot_plane_and_side_bars(
            df_spatial,
            outdir=outdir,
            base_title="Spatial object counts",
        )
        manifest["artifacts"].extend(
            [
                str(outdir / "spatial_plane_counts.png"),
                str(outdir / "spatial_side_counts.png"),
            ]
        )

    # 4f. Compact row of 100% bars for camera & lighting settings
    # Gather keys of interest
    keys_interest = set()
    for k in [
        "camera_angle",
        "camera_depth_of_field",
        "lighting_color_temperature",
        "lighting_contrast_level",
        "lighting_saturation_level",
    ]:
        if ((df_groups["key"] == k).any()) and (df_groups[df_groups["key"] == k]["normalized_share"].sum() > 0):
            keys_interest.add(k)

    if keys_interest:
        # Prepare stacked bars side-by-side in one axes
        parts_per_key: List[Tuple[str, List[Tuple[str, float, int]]]] = []
        for key in sorted(keys_interest):
            sub = df_groups[df_groups["key"] == key]
            sub = sub[sub["sub_key"].astype(str).str.lower() != "unknown"] if drop_unknown else sub
            agg = (
                sub.groupby("sub_key", as_index=False)[["total_count", "normalized_share"]]
                .sum()
                .rename(columns={"total_count": "count", "normalized_share": "share"})
            )
            # Normalize shares
            ssum = float(agg["share"].sum())
            if ssum > 0:
                agg["share"] = agg["share"] / ssum
            parts = list(zip(agg["sub_key"].tolist(), agg["share"].tolist(), agg["count"].tolist()))
            parts_per_key.append((key, parts))

        # Plot
        fig, ax = plt.subplots(figsize=(max(8, 2.5 * len(parts_per_key)), 3.2))
        x_positions = np.arange(len(parts_per_key))
        for xi, (k, parts) in zip(x_positions, parts_per_key):
            left = 0.0
            colors = _pick_palette(len(parts), name="tab10")
            for i, (lab, sh, ct) in enumerate(sorted(parts, key=lambda t: -t[1])):
                ax.bar([xi], [sh], bottom=[left], color=colors[i], edgecolor="black", linewidth=0.5)
                if sh >= 0.10:
                    ax.text(xi, left + sh / 2, f"{lab}\n{ct} ({sh*100:.0f}%)", ha="center", va="center", fontsize=8)
                left += sh
        ax.set_xticks(x_positions, [k for k, _ in parts_per_key])
        ax.set_ylim(0, 1)
        ax.set_ylabel("Share (100%)")
        ax.set_title("Camera & lighting settings (100% stacks)")
        ax.grid(axis="y", linestyle=":", alpha=0.4)
        fig.tight_layout()
        outpath = outdir / "stack100_camera_lighting.png"
        _ensure_dir(outpath.parent)
        fig.savefig(outpath, dpi=200)
        plt.close(fig)
        manifest["artifacts"].append(str(outpath))

    # Save manifest and audits
    (outdir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if manifest.get("audits"):
        (outdir / "audits.json").write_text(json.dumps(manifest["audits"], indent=2), encoding="utf-8")

    return manifest


# ----------------------------
# CLI
# ----------------------------


def _iter_summary_files(input_path: Path) -> Iterable[Path]:
    if input_path.is_file():
        yield input_path
        return
    for p in input_path.rglob("summary_report.json"):
        if p.is_file():
            yield p


def main() -> int:
    if not HARDCODED_INPUTS:
        raise RuntimeError("HARDCODED_INPUTS is empty. Add summary_report paths before running.")

    output_dir = OUTPUT_DIR.resolve()
    _ensure_dir(output_dir)

    manifests = []
    for raw_path in HARDCODED_INPUTS:
        input_path = Path(raw_path).resolve()
        if not input_path.exists():
            raise FileNotFoundError(f"Configured input does not exist: {input_path}")
        for summary_file in _iter_summary_files(input_path):
            print(f"[Process] {summary_file}")
            manifest = generate_all_plots_for_summary(
                summary_path=summary_file,
                output_dir=output_dir,
                collapse_threshold=COLLAPSE_THRESHOLD,
                drop_unknown=DROP_UNKNOWN,
            )
            manifests.append(manifest)

    if len(manifests) > 1:
        (output_dir / "index.json").write_text(json.dumps(manifests, indent=2), encoding="utf-8")
    print(f"[Done] Wrote outputs to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
