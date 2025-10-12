"""
Aggregate cleaned frequency outputs per role.

This script traverses the dataset directory, locates every
``clean_frequency/frequencies.json`` file produced by ``frequency_cleaner.py``,
groups them by role (the directory level immediately above the prompt), and
merges counts across all prompts for that role. Aggregated results are written
to ``dataset/role_counting`` as both JSON and CSV files.

Usage
=====

    python -m decription_pipeline.data_processing.role_frequency_aggregator

"""

import argparse
import json
import logging
import re
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from .frequency_schema import build_csv_rows, create_counts_structure, normalise_cohort_structure
from .io_utils import write_atomic_csv, write_atomic_text
from .semantic_utils import clean_token_counts

logger = logging.getLogger(__name__)

DATASET_ROOT_DEFAULT = Path("dataset")
OUTPUT_DIR_NAME = "role_counting"


# ---------------------------------------------------------------------------
# Filesystem helpers
# ---------------------------------------------------------------------------

def _load_frequency_json(path: Path) -> Optional[Dict[str, object]]:
    try:
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Unable to parse %s: %s", path, exc)
        return None
    if not isinstance(data, dict):
        logger.warning("Skipping %s because it does not contain a JSON object", path)
        return None
    return data


# ---------------------------------------------------------------------------
# Aggregation logic
# ---------------------------------------------------------------------------

def _sanitise_role_name(role: str) -> str:
    role = role.strip().lower()
    role = re.sub(r"[^a-z0-9]+", "_", role)
    role = role.strip("_")
    return role or "role"


def _iter_clean_frequency_files(dataset_root: Path) -> Iterable[Path]:
    pattern = "clean_frequency"
    for freq_dir in dataset_root.rglob(pattern):
        freq_json = freq_dir / "frequencies.json"
        if freq_json.is_file():
            yield freq_json


def _group_by_role(dataset_root: Path) -> Dict[str, List[Path]]:
    grouped: Dict[str, List[Path]] = defaultdict(list)
    for freq_json in _iter_clean_frequency_files(dataset_root):
        try:
            role_dir = freq_json.parent.parent.parent
        except ValueError:
            logger.debug("Skipping %s because role directory could not be determined", freq_json)
            continue
        role_name = role_dir.name
        grouped[role_name].append(freq_json)
    return grouped


def _collect_metadata(role: str, freq_paths: Iterable[Path]) -> Dict[str, object]:
    contexts = set()
    prompt_count = 0
    for freq_path in freq_paths:
        prompt_count += 1
        if len(freq_path.parents) >= 4:
            contexts.add(freq_path.parents[3].name)
    return {
        "role": role,
        "prompt_count": prompt_count,
        "contexts": sorted(contexts),
    }

def _aggregate_role(role: str, freq_paths: List[Path]) -> Dict[str, object]:
    aggregate = create_counts_structure()

    for freq_path in freq_paths:
        data = _load_frequency_json(freq_path)
        if not data:
            continue
        cohorts = normalise_cohort_structure(data.get("cohorts", data))
        for cohort, dimensions in cohorts.items():
            if cohort not in aggregate:
                aggregate[cohort] = OrderedDict()
            target_dimensions = aggregate[cohort]
            for dimension, labels in dimensions.items():
                if dimension not in target_dimensions:
                    target_dimensions[dimension] = {}
                target_labels = target_dimensions[dimension]
                if not isinstance(labels, dict):
                    continue
                for label, value in labels.items():
                    try:
                        numeric = int(round(float(value)))  # type: ignore[arg-type]
                    except (TypeError, ValueError):
                        logger.debug("Ignoring non-numeric value for %s/%s/%s: %r", cohort, dimension, label, value)
                        continue
                    label_str = str(label)
                    target_labels[label_str] = target_labels.get(label_str, 0) + numeric

    cleaned_cohorts = create_counts_structure()
    for cohort, dimensions in aggregate.items():
        if cohort not in cleaned_cohorts:
            cleaned_cohorts[cohort] = OrderedDict()
        target_dimensions = cleaned_cohorts[cohort]
        for dimension, labels in dimensions.items():
            if cohort == "totals":
                target_dimensions[dimension] = {label: int(count) for label, count in labels.items()}
            else:
                target_dimensions[dimension] = clean_token_counts(labels)

    metadata = _collect_metadata(role, freq_paths)
    return {"_meta": metadata, "cohorts": cleaned_cohorts}


def _determine_output_paths(output_dir: Path, role: str, existing: Dict[str, str]) -> Tuple[Path, Path]:
    base = _sanitise_role_name(role)
    if base in existing:
        suffix = 2
        while f"{base}_{suffix}" in existing.values():
            suffix += 1
        base = f"{base}_{suffix}"
    existing[role] = base
    json_path = output_dir / f"{base}.json"
    csv_path = output_dir / f"{base}.csv"
    return json_path, csv_path


def aggregate_roles(dataset_root: Path, output_dir: Optional[Path] = None) -> Dict[str, Dict[str, object]]:
    dataset_root = dataset_root.resolve()
    if output_dir is None:
        output_dir = dataset_root / OUTPUT_DIR_NAME
    else:
        output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    role_map = _group_by_role(dataset_root)
    if not role_map:
        logger.warning("No clean frequency files found under %s", dataset_root)
        return {}

    aggregated_results: Dict[str, Dict[str, object]] = {}
    filename_map: Dict[str, str] = {}
    for role, freq_paths in sorted(role_map.items()):
        aggregated = _aggregate_role(role, freq_paths)
        aggregated_results[role] = aggregated
        json_path, csv_path = _determine_output_paths(output_dir, role, filename_map)
        write_atomic_text(json_path, json.dumps(aggregated, indent=2))
        write_atomic_csv(csv_path, build_csv_rows(aggregated["cohorts"]))
        logger.info("Wrote role aggregates for %s to %s", role, json_path)

    summary_path = output_dir / "role_summary.json"
    roles_summary: List[Dict[str, object]] = []
    for role in sorted(aggregated_results):
        aggregated = aggregated_results[role]
        meta = aggregated.get("_meta", {})
        if isinstance(meta, dict):
            prompt_count = int(meta.get("prompt_count", 0) or 0)
        else:
            prompt_count = 0
        roles_summary.append(
            {"role": role, "file_stem": filename_map[role], "prompt_count": prompt_count}
        )
    summary_payload = {"dataset_root": str(dataset_root), "roles": roles_summary}
    write_atomic_text(summary_path, json.dumps(summary_payload, indent=2, sort_keys=True))
    return aggregated_results


def main(argv: Optional[Iterable[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate clean frequency counts per role."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DATASET_ROOT_DEFAULT,
        help="Path to the dataset root (default: ./dataset)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output directory (default: dataset/role_counting)",
    )
    parser.add_argument(
        "--log-level",
        choices=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"],
        default="INFO",
        help="Logging verbosity (default: INFO)",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level))

    aggregate_roles(args.dataset, args.output)


if __name__ == "__main__":
    main()
