"""Dataset-wide aggregation and visualization for summary_report.json files.

This script walks the dataset hierarchy, loads each prompt's summary_report.json,
aggregates counts by cohort/sub-key, writes the aggregation JSONs per prompt, and
emits count bar charts plus optional spatial heatmaps under each prompt.

Usage
-----
Run from the repository root:

    python -m decription_pipeline.analysis.dataset_rollup

Configuration constants near the top control dataset root and colour palettes.
"""

import hashlib
import json
from collections import defaultdict, OrderedDict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .context_metrics import compute_dimension_counts, load_summary_report

# ----------------------------
# Configuration
# ----------------------------

DATASET_ROOT = Path("dataset")
SUMMARY_RELATIVE_PATH = Path("descriptions/summary/summary_report.json")
AGGREGATION_DIR_NAME = "aggregation_counting"
VISUALIZATION_DIR_NAME = "visualization_analysis"

BAR_CMAP_NAME = "tab10"
HEATMAP_CMAP_NAME = "cividis"


# ----------------------------
# Helpers
# ----------------------------


def _slugify(text: str) -> str:
    stripped = "".join(ch.lower() if ch.isalnum() else "-" for ch in str(text))
    collapsed = "-".join(filter(None, stripped.split("-")))
    return collapsed or "value"


def _slugify_limited(text: str, max_length: int) -> str:
    slug = _slugify(text)
    if len(slug) <= max_length:
        return slug
    digest = hashlib.md5(str(text).encode("utf-8")).hexdigest()[:8]
    suffix = digest
    trimmed = slug[: max_length - len(suffix) - 1].rstrip("-")
    if not trimmed:
        trimmed = slug[: max_length - len(suffix) - 1]
    return f"{trimmed}-{suffix}"


def iter_prompt_dirs(root: Path) -> Iterable[Tuple[Path, Path, Path, Path]]:
    """Yield (context_dir, role_dir, prompt_dir, summary_path) tuples."""
    for context_dir in sorted(root.iterdir()):
        if not context_dir.is_dir():
            continue
        for role_dir in sorted(context_dir.iterdir()):
            if not role_dir.is_dir():
                continue
            for prompt_dir in sorted(role_dir.iterdir()):
                if not prompt_dir.is_dir():
                    continue
                if prompt_dir.name == AGGREGATION_DIR_NAME:
                    continue
                summary_path = prompt_dir / SUMMARY_RELATIVE_PATH
                if summary_path.exists():
                    yield context_dir, role_dir, prompt_dir, summary_path


def counts_to_ordered_dict(counts: Mapping[str, int]) -> OrderedDict[str, int]:
    return OrderedDict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def palette_colors(n: int, cmap_name: str = BAR_CMAP_NAME) -> List[str]:
    cmap = plt.get_cmap(cmap_name)
    return [cmap(i % cmap.N) for i in range(n)]


def plot_bar_chart(df: pd.DataFrame, title: str, out_path: Path) -> None:
    if df.empty:
        return
    df_sorted = df.sort_values("count", ascending=False)
    labels = df_sorted["label"].astype(str).tolist()
    counts = df_sorted["count"].astype(float).tolist()

    fig_width = max(6.0, min(16.0, 0.7 * len(labels)))
    fig, ax = plt.subplots(figsize=(fig_width, 4.5))
    colors = palette_colors(len(labels))
    bars = ax.bar(range(len(labels)), counts, color=colors, edgecolor="black", linewidth=0.5)

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.grid(axis="y", linestyle=":", alpha=0.4)

    max_height = max(counts) if counts else 0
    offset = max(0.5, max_height * 0.04)
    for bar, count in zip(bars, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + offset,
            f"{int(count)}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    fig.tight_layout()
    ensure_dir(out_path.parent)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_spatial_heatmap(spatial_counts: Mapping[Tuple[str, str], int], title: str, out_path: Path) -> None:
    planes = ["foreground", "midground", "background"]
    sides = ["left", "center", "right"]
    data = np.zeros((len(planes), len(sides)), dtype=float)
    has_data = False
    for i, plane in enumerate(planes):
        for j, side in enumerate(sides):
            value = spatial_counts.get((plane, side), 0)
            if value:
                has_data = True
            data[i, j] = value
    if not has_data:
        return

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(data, cmap=HEATMAP_CMAP_NAME)
    ax.set_xticks(range(len(sides)))
    ax.set_xticklabels(sides)
    ax.set_yticks(range(len(planes)))
    ax.set_yticklabels(planes)
    ax.set_xlabel("Side")
    ax.set_ylabel("Plane")
    ax.set_title(title)

    for i in range(len(planes)):
        for j in range(len(sides)):
            ax.text(j, i, f"{int(data[i, j])}", ha="center", va="center", color="white", fontsize=9)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Count")
    fig.tight_layout()
    ensure_dir(out_path.parent)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def write_aggregation_json(
    output_path: Path,
    data: Mapping[str, Mapping[str, int]],
    summary_rel: Path,
    num_images: int,
) -> None:
    ensure_dir(output_path.parent)
    payload = OrderedDict(
        [
            (dimension, counts_to_ordered_dict(values))
            for dimension, values in sorted(data.items())
        ]
    )
    output = {
        "source_summary": str(summary_rel).replace("\\", "/"),
        "num_images": int(num_images),
        "aggregated_counts": {k: dict(v) for k, v in payload.items()}
    }
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")


def process_prompt(summary_path: Path, role_dir: Path, prompt_dir: Path):
    report = load_summary_report(summary_path)
    aggregated, spatial_counts, num_images = compute_dimension_counts(report)

    # Save aggregation JSON
    aggregation_dir = role_dir / AGGREGATION_DIR_NAME
    ensure_dir(aggregation_dir)
    prompt_slug = _slugify_limited(prompt_dir.name, max_length=120)
    aggregation_path = aggregation_dir / f"{prompt_slug}_counts.json"
    summary_rel = summary_path.relative_to(DATASET_ROOT)
    write_aggregation_json(aggregation_path, aggregated, summary_rel, num_images)

    # Visualizations
    viz_dir = prompt_dir / VISUALIZATION_DIR_NAME
    ensure_dir(viz_dir)

    for dimension, counts in aggregated.items():
        if not counts:
            continue
        ordered = counts_to_ordered_dict(counts)
        df = pd.DataFrame({"label": list(ordered.keys()), "count": list(ordered.values())})
        cohort, subkey = dimension.split(".", 1)
        hash_suffix = hashlib.md5(dimension.encode("utf-8")).hexdigest()[:8]
        file_name = f"{_slugify_limited(cohort, 6)}_{hash_suffix}.png"
        title = f"{cohort} – {subkey}"
        plot_bar_chart(df, title, viz_dir / file_name)

    if spatial_counts:
        heatmap_title = "Object counts by position (3×3)"
        heatmap_name = "spatial_heatmap.png"
        plot_spatial_heatmap(spatial_counts, heatmap_title, viz_dir / heatmap_name)

    return aggregated, spatial_counts, num_images


def main() -> int:
    if not DATASET_ROOT.exists():
        raise FileNotFoundError(f"Dataset root not found: {DATASET_ROOT}")

    prompt_items = list(iter_prompt_dirs(DATASET_ROOT))
    if not prompt_items:
        print("[Info] No summary_report.json files found.")
        return 0

    role_rollups: Dict[Path, Dict[str, object]] = {}

    for context_dir, role_dir, prompt_dir, summary_path in prompt_items:
        print(f"[Process] {summary_path}")
        aggregated, spatial_counts, num_images = process_prompt(summary_path, role_dir, prompt_dir)

        entry = role_rollups.setdefault(
            role_dir,
            {
                "context": context_dir.name,
                "role": role_dir.name,
                "aggregated_counts": defaultdict(lambda: defaultdict(int)),
                "spatial_counts": defaultdict(int),
                "total_prompts": 0,
                "total_images": 0,
                "prompt_details": [],
            },
        )

        entry["total_prompts"] = int(entry["total_prompts"]) + 1
        entry["total_images"] = int(entry["total_images"]) + int(num_images)
        entry["prompt_details"].append(
            {
                "prompt": prompt_dir.name,
                "summary_report": str(summary_path.relative_to(DATASET_ROOT)).replace("\\", "/"),
                "num_images": int(num_images),
            }
        )

        aggregated_counts = entry["aggregated_counts"]
        for dimension, labels in aggregated.items():
            dest = aggregated_counts.setdefault(dimension, defaultdict(int))
            for label, count in labels.items():
                dest[label] += int(count)

        spatial_totals = entry["spatial_counts"]
        for (plane, side), count in spatial_counts.items():
            spatial_totals[(plane, side)] += int(count)

    # Write per-role rollups
    for role_dir, data in role_rollups.items():
        aggregation_dir = role_dir / AGGREGATION_DIR_NAME
        ensure_dir(aggregation_dir)

        ordered_counts = OrderedDict(
            [
                (dimension, counts_to_ordered_dict(labels))
                for dimension, labels in sorted(data["aggregated_counts"].items())
            ]
        )
        aggregated_counts_out = {dim: dict(order) for dim, order in ordered_counts.items()}
        spatial_out = {
            f"{plane}.{side}": int(count)
            for (plane, side), count in sorted(data["spatial_counts"].items())
            if int(count)
        }

        role_output = {
            "context": data["context"],
            "role": data["role"],
            "total_prompts": int(data["total_prompts"]),
            "total_images": int(data["total_images"]),
            "aggregated_counts": aggregated_counts_out,
            "prompts": data["prompt_details"],
        }
        if spatial_out:
            role_output["spatial_counts"] = spatial_out

        role_output_path = aggregation_dir / "role_counts.json"
        role_output_path.write_text(json.dumps(role_output, indent=2), encoding="utf-8")

        if data["spatial_counts"]:
            role_viz_dir = role_dir / VISUALIZATION_DIR_NAME
            ensure_dir(role_viz_dir)
            plot_spatial_heatmap(
                data["spatial_counts"],
                f"{data['role']} – object counts by position (all prompts)",
                role_viz_dir / "role_spatial_heatmap.png",
            )

    print("[Done] Aggregations and visualizations generated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
