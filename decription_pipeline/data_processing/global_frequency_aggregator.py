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
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

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


def _merge_category_counts(
    aggregate: Dict[str, Dict[str, int]],
    category: str,
    values: Dict[str, object],
) -> None:
    counter = aggregate.setdefault(category, {})
    for token, value in values.items():
        if not isinstance(token, str):
            token = str(token)
        try:
            numeric = int(round(float(value)))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            continue
        counter[token] = counter.get(token, 0) + numeric


def _merge_totals(totals: Dict[str, int], payload: Dict[str, object]) -> None:
    for key, value in payload.items():
        if not isinstance(key, str):
            key = str(key)
        try:
            numeric = int(round(float(value)))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            continue
        totals[key] = totals.get(key, 0) + numeric


def _aggregate_roles(role_dir: Path, skip_names: Set[str]) -> Tuple[Dict[str, Dict[str, int]], Dict[str, int], Dict[str, object]]:
    aggregate: Dict[str, Dict[str, int]] = {}
    totals: Dict[str, int] = {}
    contexts: Set[str] = set()
    roles: List[str] = []
    prompt_total = 0

    for role_json in _iter_role_jsons(role_dir, skip_names):
        payload = _load_role_payload(role_json)
        if not payload:
            continue
        role_name = payload.get("_meta", {}).get("role") if isinstance(payload.get("_meta"), dict) else None
        if not isinstance(role_name, str):
            role_name = role_json.stem
        roles.append(role_name)
        if isinstance(payload.get("_meta"), dict):
            meta = payload["_meta"]  # type: ignore[assignment]
            if isinstance(meta, dict):
                prompt_total += int(meta.get("prompt_count", 0) or 0)
                ctx_values = meta.get("contexts", [])
                if isinstance(ctx_values, list):
                    for ctx in ctx_values:
                        if isinstance(ctx, str):
                            contexts.add(ctx)
        for category, values in payload.items():
            if category in {"_meta", "totals"}:
                continue
            if not isinstance(values, dict):
                continue
            _merge_category_counts(aggregate, category, values)
        totals_payload = payload.get("totals")
        if isinstance(totals_payload, dict):
            _merge_totals(totals, totals_payload)

    meta = {
        "role_count": len(roles),
        "roles": sorted(set(roles)),
        "contexts": sorted(contexts),
        "prompt_count": prompt_total,
    }
    return aggregate, totals, meta


def _clean_aggregate(aggregate: Dict[str, Dict[str, int]]) -> Dict[str, Dict[str, int]]:
    cleaned: Dict[str, Dict[str, int]] = {}
    for category, counts in aggregate.items():
        cleaned[category] = clean_token_counts(counts)
    return cleaned


def _build_csv_rows(aggregated: Dict[str, object]) -> List[List[str]]:
    rows: List[List[str]] = [["category", "token", "count"]]
    for category, values in aggregated.items():
        if category in {"_meta", "totals"}:
            continue
        if not isinstance(values, dict):
            continue
        for token, count in values.items():
            rows.append([category, token, str(count)])
    totals = aggregated.get("totals")
    if isinstance(totals, dict):
        for key, value in totals.items():
            rows.append(["totals", key, str(value)])
    return rows


def aggregate_all_roles(dataset_root: Path, role_dir: Optional[Path] = None, output_basename: str = OUTPUT_BASENAME) -> Dict[str, object]:
    dataset_root = dataset_root.resolve()
    if role_dir is None:
        role_dir = dataset_root / ROLE_COUNTING_DIRNAME
    else:
        role_dir = role_dir.resolve()
    if not role_dir.exists():
        raise FileNotFoundError(f"Role counting directory not found: {role_dir}")

    skip_names: Set[str] = {"role_summary", "role_summary.json", output_basename, f"{output_basename}.json"}
    aggregate, totals, meta = _aggregate_roles(role_dir, skip_names)
    if not aggregate and not totals:
        logger.warning("No role data found in %s", role_dir)
        return {}

    cleaned = _clean_aggregate(aggregate)
    if totals:
        totals = dict(sorted(totals.items()))

    payload: Dict[str, object] = {"_meta": meta, "totals": totals}
    for category in sorted(cleaned):
        payload[category] = cleaned[category]

    json_path = role_dir / f"{output_basename}.json"
    csv_path = role_dir / f"{output_basename}.csv"
    write_atomic_text(json_path, json.dumps(payload, indent=2, sort_keys=True))
    write_atomic_csv(csv_path, _build_csv_rows(payload))
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
