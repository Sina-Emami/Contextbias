"""Aggregate gender presentation counts for SDXL and SD3.5 role CSVs."""

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

RowDict = Dict[str, object]
GroupKey = Tuple[str, str, str, str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize gender presentation counts from role_counting CSVs for "
            "both SDXL and SD3.5 models."
        )
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Repository root that contains the 92_job_description_* directories.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("gender_presentation_summary.csv"),
        help="Path to the CSV file that will store the aggregated results.",
    )
    return parser.parse_args()


def load_gender_counts(csv_path: Path) -> Dict[Tuple[str, str], List[Tuple[str, int]]]:
    grouped: Dict[Tuple[str, str], List[Tuple[str, int]]] = defaultdict(list)
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if (row.get("dimension") or "").strip().lower() != "gender_presentation":
                continue
            cohort = (row.get("cohort") or "").strip()
            label = (row.get("label") or "").strip()
            count_str = (row.get("count") or "").strip()
            if not cohort or not label or not count_str:
                continue
            try:
                count = int(float(count_str))
            except ValueError:
                raise ValueError(f"Count '{count_str}' in {csv_path} is not numeric.") from None
            grouped[(cohort, "gender_presentation")].append((label, count))
    return grouped


def build_summary_rows(model_dirs: Dict[str, Path]) -> List[RowDict]:
    rows: List[RowDict] = []
    for model, directory in model_dirs.items():
        if not directory.is_dir():
            raise FileNotFoundError(f"Directory not found: {directory}")
        for csv_path in sorted(directory.glob("*.csv")):
            if csv_path.name.lower() == "all_roles.csv":
                continue
            role = csv_path.stem
            grouped_counts = load_gender_counts(csv_path)
            for (cohort, dimension), entries in grouped_counts.items():
                total = sum(count for _, count in entries)
                if total == 0:
                    continue
                for label, count in entries:
                    rows.append(
                        {
                            "model": model,
                            "role": role,
                            "cohort": cohort,
                            "dimension": dimension,
                            "label": label,
                            "count": count,
                            "percentage": round(count / total, 6),
                        }
                    )
    rows.sort(key=lambda r: (r["model"], r["role"], r["cohort"], r["label"]))
    return rows


def write_output(rows: Iterable[RowDict], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["model", "role", "cohort", "dimension", "label", "count", "percentage"]
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    base_dir = args.base_dir.resolve()
    model_dirs = {
        "SDXL": base_dir / "92_job_description_SDXL" / "role_counting",
        "SD3.5": base_dir / "92_job_description_3.5" / "role_counting",
    }
    rows = build_summary_rows(model_dirs)
    write_output(rows, args.output.resolve())
    print(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # pragma: no cover - simple CLI safeguard
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
