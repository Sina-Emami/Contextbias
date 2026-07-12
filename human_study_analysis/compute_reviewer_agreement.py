"""Compare human reviewers against each other (inter-rater reliability).

Every subdirectory under clean_section/reviewers/ holds one independent
reviewer's aggregated cohort/dimension/label/count/bin files, all sharing the
same <role>_<model>.csv naming. Rather than assuming a fixed number of
reviewers, this discovers whichever reviewer directories exist and scores
every pair of them against each other, for every role/model file they share.
Writes clean_section/inter_annotator_results.csv with raw agreement and
Cohen's kappa per (reviewer pair, role, model), plus an overall pooled row.
"""

import glob
import itertools
import os

import pandas as pd

from common.agreement import merge_majority_labels, role_model_from_filename, score_labels

BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "clean_section")
REVIEWERS_ROOT = os.path.join(BASE_DIR, "reviewers")

OUT_CSV = os.path.join(BASE_DIR, "inter_annotator_results.csv")


def reviewer_dirs():
    if not os.path.isdir(REVIEWERS_ROOT):
        return []
    return sorted(
        os.path.join(REVIEWERS_ROOT, name)
        for name in os.listdir(REVIEWERS_ROOT)
        if os.path.isdir(os.path.join(REVIEWERS_ROOT, name))
    )


def _rel(path):
    return os.path.relpath(path, BASE_DIR).replace(os.sep, "/")


def compute_all_pairs():
    rows = []
    all_a, all_b = [], []

    dirs = reviewer_dirs()
    if len(dirs) < 2:
        print(f"Need at least two reviewer directories under {REVIEWERS_ROOT} to compare.")
        return rows

    for dir_a, dir_b in itertools.combinations(dirs, 2):
        for path_a in sorted(glob.glob(os.path.join(dir_a, "*.csv"))):
            name, role, model = role_model_from_filename(path_a)
            path_b = os.path.join(dir_b, f"{name}.csv")

            if not os.path.exists(path_b):
                print(f"[WARN] No matching file for {_rel(path_a)} in {_rel(dir_b)}")
                continue

            reviewer_a = pd.read_csv(path_a)
            reviewer_b = pd.read_csv(path_b)
            merged = merge_majority_labels(reviewer_a, "label_a", reviewer_b, "label_b")
            if merged.empty:
                continue

            metrics = score_labels(merged["label_a"], merged["label_b"])
            print(
                f"{role} ({model}): raw_agreement = {metrics['raw_agreement']:.3f}, "
                f"kappa = {metrics['kappa']:.3f} over {len(merged)} (cohort,dimension) pairs"
            )

            rows.append(
                {
                    "role": role,
                    "model": model,
                    "annotator1_file": _rel(path_a),
                    "annotator2_file": _rel(path_b),
                    "raw_agreement": metrics["raw_agreement"],
                    "kappa": metrics["kappa"],
                }
            )
            all_a.extend(merged["label_a"])
            all_b.extend(merged["label_b"])

    if all_a:
        overall = score_labels(all_a, all_b)
    else:
        overall = {"raw_agreement": float("nan"), "kappa": float("nan")}

    print(f"\nOverall: raw_agreement = {overall['raw_agreement']:.3f}, kappa = {overall['kappa']:.3f}")
    rows.append(
        {
            "role": "ALL",
            "model": "ALL",
            "annotator1_file": "",
            "annotator2_file": "",
            "raw_agreement": overall["raw_agreement"],
            "kappa": overall["kappa"],
        }
    )
    return rows


def main():
    rows = compute_all_pairs()
    if not rows:
        return

    pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
    print(f"\nSaved inter-annotator results to {OUT_CSV}")


if __name__ == "__main__":
    main()
