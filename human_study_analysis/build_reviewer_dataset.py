"""Build one reviewer's aggregated dataset from a raw survey export.

Replaces the old three-step clean.py -> convertor.py -> aggregator.py
pipeline with a single command. Each row in a raw export (raw_surveys/*.csv)
is one respondent, and each respondent answers N_SECTIONS blocks of
questions (one block per image batch / model condition). This picks one
respondent and one set of sections (the batches answered under a single
model condition) and writes the aggregated cohort/dimension/label/count/bin
CSV consumed by compute_pipeline_agreement.py and compute_reviewer_agreement.py.

Usage
=====
    python human_study_analysis/build_reviewer_dataset.py \\
        human_study_analysis/raw_surveys/dancer.csv \\
        --respondent 0 --sections 4,5,6 \\
        --output human_study_analysis/clean_section/aggregated/human_dancer_3.5.csv
"""

import argparse
from pathlib import Path

from common.survey_pipeline import build_reviewer_dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csv_path", type=Path, help="Raw survey export (e.g. raw_surveys/dancer.csv)")
    parser.add_argument("--respondent", type=int, required=True, help="0-based respondent row index")
    parser.add_argument("--sections", required=True, help="Comma-separated 1-based section numbers, e.g. 4,5,6")
    parser.add_argument("--output", type=Path, required=True, help="Where to write the aggregated CSV")
    args = parser.parse_args()

    section_numbers = [int(s) for s in args.sections.split(",")]
    result = build_reviewer_dataset(args.csv_path, args.respondent, section_numbers)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(f"Wrote {args.output} ({len(result)} rows)")


if __name__ == "__main__":
    main()
