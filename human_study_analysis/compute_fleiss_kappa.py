"""Fleiss' kappa across all reviewer directories, with a bootstrap confidence interval.

Cohen's kappa (compute_reviewer_agreement.py) only handles rater pairs; with three
annotators (as in this study) the paper reports Fleiss' kappa as the primary reliability
statistic instead, since it's the one defined for more than two raters, with a bootstrap
95% CI (paper Sec. 4.3: kappa = 0.89, 95% CI [0.87, 0.91], 1,000 replicates).

The bootstrap resamples items (the (cohort, dimension) rating units, matched across every
role/model file) with replacement -- the rater panel itself is fixed, so items are the
correct resampling unit here, unlike the role-level cluster bootstrap used for BI shifts
in bias_quantification/cluster_bootstrap.py.

Writes clean_section/fleiss_kappa_results.csv (kappa per role/model, plus overall) and
prints the overall 95% CI to stdout.
"""

import glob
import os

import numpy as np
import pandas as pd

from common.agreement import category_counts, fleiss_kappa, merge_all_labels, role_model_from_filename

BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "clean_section")
REVIEWERS_ROOT = os.path.join(BASE_DIR, "reviewers")
OUT_CSV = os.path.join(BASE_DIR, "fleiss_kappa_results.csv")

N_REPLICATES = int(os.environ.get("HUMAN_STUDY_BOOTSTRAP_REPLICATES", "1000"))
RANDOM_SEED = int(os.environ.get("HUMAN_STUDY_BOOTSTRAP_SEED", "0"))
CI_LOW, CI_HIGH = 2.5, 97.5


def reviewer_dirs() -> list[str]:
    if not os.path.isdir(REVIEWERS_ROOT):
        return []
    return sorted(
        os.path.join(REVIEWERS_ROOT, name)
        for name in os.listdir(REVIEWERS_ROOT)
        if os.path.isdir(os.path.join(REVIEWERS_ROOT, name))
    )


def bootstrap_ci(rating_matrix: np.ndarray, rng: np.random.Generator) -> tuple[float, float, float]:
    n_items = rating_matrix.shape[0]
    point = fleiss_kappa(rating_matrix)

    draws = rng.integers(0, n_items, size=(N_REPLICATES, n_items))
    values = np.empty(N_REPLICATES)
    for rep in range(N_REPLICATES):
        try:
            values[rep] = fleiss_kappa(rating_matrix[draws[rep]])
        except ValueError:
            values[rep] = np.nan
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return point, float("nan"), float("nan")
    return point, float(np.percentile(values, CI_LOW)), float(np.percentile(values, CI_HIGH))


def main() -> None:
    rng = np.random.default_rng(RANDOM_SEED)
    dirs = reviewer_dirs()
    if len(dirs) < 3:
        print(f"Need at least three reviewer directories under {REVIEWERS_ROOT}; found {len(dirs)}.")
        return

    reviewer_names = [os.path.basename(d) for d in dirs]
    file_stems = sorted(
        os.path.splitext(os.path.basename(p))[0] for p in glob.glob(os.path.join(dirs[0], "*.csv"))
    )

    rows = []
    all_matrices = []

    for stem in file_stems:
        _, role, model = role_model_from_filename(stem + ".csv")
        reviewer_dfs = {}
        for d, name in zip(dirs, reviewer_names):
            path = os.path.join(d, f"{stem}.csv")
            if not os.path.exists(path):
                break
            reviewer_dfs[name] = pd.read_csv(path)
        if len(reviewer_dfs) != len(dirs):
            print(f"[WARN] {stem}: not all reviewers have this file, skipping.")
            continue

        wide = merge_all_labels(reviewer_dfs)
        if wide.empty:
            continue

        matrix, _ = category_counts(wide, reviewer_names)
        point, ci_low, ci_high = bootstrap_ci(matrix, rng)
        print(f"{role} ({model}): kappa = {point:.3f} [{ci_low:.3f}, {ci_high:.3f}] over {len(wide)} items")

        rows.append(
            {
                "role": role,
                "model": model,
                "n_items": len(wide),
                "kappa": point,
                "ci_low": ci_low,
                "ci_high": ci_high,
            }
        )
        all_matrices.append(matrix)

    if not all_matrices:
        print("No matched reviewer files found; nothing to score.")
        return

    overall_matrix = np.concatenate(all_matrices, axis=0)
    overall_point, overall_lo, overall_hi = bootstrap_ci(overall_matrix, rng)
    print(f"\nOverall: kappa = {overall_point:.3f}, 95% CI = [{overall_lo:.3f}, {overall_hi:.3f}] over {len(overall_matrix)} items")

    rows.append(
        {
            "role": "ALL",
            "model": "ALL",
            "n_items": len(overall_matrix),
            "kappa": overall_point,
            "ci_low": overall_lo,
            "ci_high": overall_hi,
        }
    )

    pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
    print(f"Saved Fleiss' kappa results to {OUT_CSV}")


if __name__ == "__main__":
    main()
