"""
Prompt-level aggregation entry point.

This script walks every ``summary_report.json`` in the dataset, normalises the
dimension/label tokens, and writes per-prompt as well as per-role aggregates.
The normalisation logic is shared with :mod:`role_rollup`.

Key outputs
-----------
- For each prompt: ``dataset/<context>/<role>/aggregation_counting/<prompt>_counts.json``
- For each role: ``dataset/<context>/<role>/aggregation_counting/role_counts.csv`` and
  ``role_counts.json`` (accumulated across prompts in that specific context)
"""

import json
from collections import defaultdict, OrderedDict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Tuple

import pandas as pd

from .context_metrics import compute_dimension_counts, load_summary_report
from .role_rollup import (
    normalize_cohort,
    normalize_dimension_key,
    normalize_label,
)

DATASET_ROOT = Path("dataset")
SUMMARY_RELATIVE_PATH = Path("descriptions/summary/summary_report.json")
AGGREGATION_DIR_NAME = "aggregation_counting"

WINDOWS_SAFE_PATH_LIMIT = 240


def slugify(text: str) -> str:
    result = "".join(ch.lower() if ch.isalnum() else "-" for ch in str(text))
    parts = [part for part in result.split("-") if part]
    return "-".join(parts) or "value"


def slugify_limited(text: str, max_length: int) -> str:
    slug = slugify(text)
    if len(slug) <= max_length:
        return slug
    digest = abs(hash(slug)) % (10**8)
    suffix = f"{digest:08d}"
    trimmed = slug[: max_length - len(suffix) - 1].rstrip("-")
    if not trimmed:
        trimmed = slug[: max_length - len(suffix) - 1]
    return f"{trimmed}-{suffix}"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def ensure_safe_path(path: Path, hint: str) -> Path:
    if path.exists():
        return path
    absolute = path if path.is_absolute() else Path.cwd() / path
    if len(str(absolute)) <= WINDOWS_SAFE_PATH_LIMIT:
        return path
    suffix = path.suffix
    digest = abs(hash(str(path))) % (10**8)
    safe_name = f"{slugify_limited(hint, 40)}-{digest:08d}{suffix}"
    return path.with_name(safe_name)


def iter_prompt_dirs(root: Path) -> Iterable[Tuple[Path, Path, Path, Path]]:
    for context_dir in sorted(root.iterdir()):
        if not context_dir.is_dir():
            continue
        for role_dir in sorted(context_dir.iterdir()):
            if not role_dir.is_dir():
                continue
            for prompt_dir in sorted(role_dir.iterdir()):
                if not prompt_dir.is_dir():
                    continue
                summary_path = prompt_dir / SUMMARY_RELATIVE_PATH
                if summary_path.exists():
                    yield context_dir, role_dir, prompt_dir, summary_path


def normalize_dimension_name(raw_dimension: str) -> str:
    """
    Convert a raw dimension token (e.g. ``people.hair.color``) into the standard
    ``<cohort>.<dimension>`` form using :mod:`role_rollup` helpers.
    """
    if not raw_dimension:
        return "unknown.unknown"
    text = str(raw_dimension)
    if "." in text:
        cohort_raw, key_raw = text.split(".", 1)
    else:
        cohort_raw, key_raw = "unknown", text
    cohort = normalize_cohort(cohort_raw)
    key = normalize_dimension_key(key_raw)
    return f"{cohort}.{key}"


def normalize_counts(
    aggregated: Mapping[str, Mapping[str, int]]
) -> Dict[str, Dict[str, int]]:
    """
    Normalise the output of :func:`compute_dimension_counts`.
    """
    normalised: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for raw_dimension, labels in (aggregated or {}).items():
        dimension = normalize_dimension_name(raw_dimension)
        if "." in dimension:
            cohort, key = dimension.split(".", 1)
        else:
            cohort, key = "unknown", dimension
        for label, value in (labels or {}).items():
            normalised_label = normalize_label(cohort, key, label)
            normalised[dimension][normalised_label] += int(value)
    return {dim: dict(counts) for dim, counts in normalised.items()}


def process_prompt(summary_path: Path) -> Tuple[Dict[str, Dict[str, int]], Dict[Tuple[str, str], int], int]:
    report = load_summary_report(summary_path)
    aggregated, spatial_counts, num_images = compute_dimension_counts(report)
    normalised = normalize_counts(aggregated)
    spatial_totals: Dict[Tuple[str, str], int] = defaultdict(int)
    for (plane, side), count in spatial_counts.items():
        spatial_totals[(str(plane), str(side))] += int(count)
    return normalised, spatial_totals, int(num_images)


def write_prompt_json(
    output_path: Path,
    counts: Mapping[str, Mapping[str, int]],
    source_summary: Path,
    num_images: int,
) -> None:
    ordered = OrderedDict(
        (dimension, OrderedDict(sorted(labels.items(), key=lambda kv: (-kv[1], kv[0]))))
        for dimension, labels in sorted(counts.items())
    )
    payload = {
        "source_summary": str(source_summary).replace("\\", "/"),
        "num_images": int(num_images),
        "aggregated_counts": {dim: dict(values) for dim, values in ordered.items()},
    }
    ensure_dir(output_path.parent)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def run() -> int:
    if not DATASET_ROOT.exists():
        raise FileNotFoundError(f"Dataset root not found: {DATASET_ROOT}")

    prompt_items = list(iter_prompt_dirs(DATASET_ROOT))
    if not prompt_items:
        print("[Info] No summary_report.json files found.")
        return 0

    per_role_rollups: Dict[Path, Dict[str, object]] = {}

    for context_dir, role_dir, prompt_dir, summary_path in prompt_items:
        print(f"[Process] {summary_path}")
        counts, spatial_counts, num_images = process_prompt(summary_path)

        aggregation_dir = prompt_dir / AGGREGATION_DIR_NAME
        ensure_dir(aggregation_dir)
        prompt_slug = slugify_limited(prompt_dir.name, max_length=120)
        prompt_counts_path = ensure_safe_path(
            aggregation_dir / f"{prompt_slug}_counts.json",
            hint=f"{prompt_dir.name}_counts",
        )
        write_prompt_json(
            prompt_counts_path,
            counts,
            summary_path.relative_to(DATASET_ROOT),
            num_images,
        )

        entry = per_role_rollups.setdefault(
            role_dir,
            {
                "context": context_dir.name,
                "role": role_dir.name,
                "counts": defaultdict(lambda: defaultdict(int)),
                "spatial": defaultdict(int),
                "total_prompts": 0,
                "total_images": 0,
                "prompt_details": [],
            },
        )

        entry["total_prompts"] += 1
        entry["total_images"] += num_images
        entry["prompt_details"].append(
            {
                "prompt": prompt_dir.name,
                "summary_report": str(summary_path.relative_to(DATASET_ROOT)).replace("\\", "/"),
                "num_images": num_images,
            }
        )

        for dimension, labels in counts.items():
            dest = entry["counts"][dimension]
            for label, value in labels.items():
                dest[label] += int(value)

        for (plane, side), value in spatial_counts.items():
            entry["spatial"][(plane, side)] += int(value)

    for role_dir, data in per_role_rollups.items():
        aggregation_dir = role_dir / AGGREGATION_DIR_NAME
        ensure_dir(aggregation_dir)

        ordered_counts = OrderedDict(
            (dimension, OrderedDict(sorted(labels.items(), key=lambda kv: (-kv[1], kv[0]))))
            for dimension, labels in sorted(data["counts"].items())
        )
        spatial_out = {
            f"{plane}.{side}": count
            for (plane, side), count in sorted(data["spatial"].items())
            if count
        }

        role_json = {
            "context": data["context"],
            "role": data["role"],
            "total_prompts": data["total_prompts"],
            "total_images": data["total_images"],
            "aggregated_counts": {dim: dict(labels) for dim, labels in ordered_counts.items()},
            "prompts": data["prompt_details"],
        }
        if spatial_out:
            role_json["spatial_counts"] = spatial_out

        json_path = ensure_safe_path(
            aggregation_dir / "role_counts.json",
            hint=f"{data['role']}_counts",
        )
        json_path.write_text(json.dumps(role_json, indent=2), encoding="utf-8")

        rows: List[Dict[str, object]] = []
        for dimension, labels in ordered_counts.items():
            if "." in dimension:
                cohort, key = dimension.split(".", 1)
            else:
                cohort, key = dimension, ""
            for label, count in labels.items():
                rows.append(
                    {
                        "cohort": cohort,
                        "dimension": key,
                        "label": label,
                        "count": int(count),
                    }
                )
        frame = pd.DataFrame(rows, columns=["cohort", "dimension", "label", "count"])
        csv_path = ensure_safe_path(
            aggregation_dir / "role_counts.csv",
            hint=f"{data['role']}_counts_csv",
        )
        frame.to_csv(csv_path, index=False)

    print("[Done] Prompt and role aggregates generated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
