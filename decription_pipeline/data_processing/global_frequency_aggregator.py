"""
Aggregate role-level frequency outputs into a single dataset-wide summary.

This utility reads every JSON file in ``dataset/role_counting`` (excluding the
summary file), merges counts across roles, normalises tokens using the semantic
cleaner, and writes combined JSON/CSV artefacts. Use it after running
``role_frequency_aggregator.py``.

Usage
=====

    python -m decription_pipeline.data_processing.global_frequency_aggregator

"""

from __future__ import annotations

import argparse
import json
import logging
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

from .frequency_schema import build_csv_rows, create_counts_structure, normalise_cohort_structure
from .io_utils import write_atomic_csv, write_atomic_text
from .semantic_utils import clean_token_counts

logger = logging.getLogger(__name__)

DATASET_ROOT_DEFAULT = Path("dataset")
ROLE_COUNTING_DIRNAME = "role_counting"
OUTPUT_BASENAME = "all_roles"


def _iter_role_jsons(role_dir: Path, skip_names: Set[str]) -> Iterable[Path]:
    for path in sorted(role_dir.glob("*.json")):
        stem = path.stem
        if stem in skip_names or path.name in skip_names:
            continue
        yield path


def _load_role_payload(path: Path) -> Optional[Dict[str, object]]:
    try:
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Skipping %s: %s", path, exc)
        return None
    if not isinstance(data, dict):
        logger.warning("Skipping %s because it does not contain a JSON object", path)
        return None
    return data


def _aggregate_roles(role_dir: Path, skip_names: Set[str]) -> Tuple[Dict[str, Dict[str, Dict[str, int]]], Dict[str, object]]:
    aggregate = create_counts_structure()
    contexts: Set[str] = set()
    roles: List[str] = []
    prompt_total = 0

    for role_json in _iter_role_jsons(role_dir, skip_names):
        payload = _load_role_payload(role_json)
        if not payload:
            continue
        meta_block = payload.get("_meta")
        role_name = None
        if isinstance(meta_block, dict):
            role_name = meta_block.get("role")
            prompt_total += int(meta_block.get("prompt_count", 0) or 0)
            ctx_values = meta_block.get("contexts", [])
            if isinstance(ctx_values, list):
                for ctx in ctx_values:
                    if isinstance(ctx, str):
                        contexts.add(ctx)
        if not isinstance(role_name, str):
            role_name = role_json.stem
        roles.append(role_name)

        cohorts = normalise_cohort_structure(payload.get("cohorts", payload))
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
                        continue
                    label_str = str(label)
                    target_labels[label_str] = target_labels.get(label_str, 0) + numeric

    cleaned = create_counts_structure()
    for cohort, dimensions in aggregate.items():
        if cohort not in cleaned:
            cleaned[cohort] = OrderedDict()
        target_dimensions = cleaned[cohort]
        for dimension, labels in dimensions.items():
            if cohort == "totals":
                target_dimensions[dimension] = {label: int(count) for label, count in labels.items()}
            else:
                target_dimensions[dimension] = clean_token_counts(labels)

    meta = {
        "role_count": len(roles),
        "roles": sorted(set(roles)),
        "contexts": sorted(contexts),
        "prompt_count": prompt_total,
    }
    return cleaned, meta


def aggregate_all_roles(dataset_root: Path, role_dir: Optional[Path] = None, output_basename: str = OUTPUT_BASENAME) -> Dict[str, object]:
    dataset_root = dataset_root.resolve()
    if role_dir is None:
        role_dir = dataset_root / ROLE_COUNTING_DIRNAME
    else:
        role_dir = role_dir.resolve()
    if not role_dir.exists():
        raise FileNotFoundError(f"Role counting directory not found: {role_dir}")

    skip_names: Set[str] = {"role_summary", "role_summary.json", output_basename, f"{output_basename}.json"}
    cohorts, meta = _aggregate_roles(role_dir, skip_names)
    has_values = any(labels for _, dimensions in cohorts.items() for labels in dimensions.values())
    if not has_values:
        logger.warning("No role data found in %s", role_dir)
        return {}

    payload: Dict[str, object] = {"_meta": meta, "cohorts": cohorts}

    json_path = role_dir / f"{output_basename}.json"
    csv_path = role_dir / f"{output_basename}.csv"
    write_atomic_text(json_path, json.dumps(payload, indent=2))
    write_atomic_csv(csv_path, build_csv_rows(cohorts))
    logger.info("Wrote global aggregates to %s and %s", json_path, csv_path)
    return payload


def main(argv: Optional[Iterable[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate and clean frequency counts across all roles."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DATASET_ROOT_DEFAULT,
        help="Path to the dataset root (default: ./dataset)",
    )
    parser.add_argument(
        "--role-dir",
        type=Path,
        default=None,
        help="Override the role counting directory (default: <dataset>/role_counting)",
    )
    parser.add_argument(
        "--output-name",
        default=OUTPUT_BASENAME,
        help="Base filename for output artefacts (default: all_roles)",
    )
    parser.add_argument(
        "--log-level",
        choices=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"],
        default="INFO",
        help="Logging verbosity (default: INFO)",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level))

    aggregate_all_roles(args.dataset, args.role_dir, args.output_name)


if __name__ == "__main__":
    main()
