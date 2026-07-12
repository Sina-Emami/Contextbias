"""Compare a human reviewer's ratings against the automated pipeline's predictions.

For every role/model file belonging to the designated reference reviewer
(see REFERENCE_REVIEWER below), finds the matching clean_section/pipeline/
file, takes the majority label per (cohort, dimension), and scores agreement
between the two. Writes:

  - clean_section/human_pipeline_agreement.csv  (accuracy, macro recall, kappa)
  - clean_section/Cohen's kappa_results.csv     (kappa-only view of the same rows)
"""

import glob
import os

import pandas as pd

from common.agreement import merge_majority_labels, role_model_from_filename, score_labels

BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "clean_section")
REVIEWERS_ROOT = os.path.join(BASE_DIR, "reviewers")
PIPE_DIR = os.path.join(BASE_DIR, "pipeline")

# Macro recall treats one side of the comparison as ground truth, so scoring
# against the pipeline needs one fixed reviewer to play that role rather than
# whichever reviewer happens to be discovered first.
REFERENCE_REVIEWER = "reviewer_2"
HUMAN_DIR = os.path.join(REVIEWERS_ROOT, REFERENCE_REVIEWER)

AGREEMENT_OUT = os.path.join(BASE_DIR, "human_pipeline_agreement.csv")
KAPPA_OUT = os.path.join(BASE_DIR, "Cohen's kappa_results.csv")


def compute_all_pairs():
    rows = []
    all_true, all_pred = [], []

    human_files = sorted(glob.glob(os.path.join(HUMAN_DIR, "*.csv")))
    if not human_files:
        print(f"No reference reviewer files found under: {HUMAN_DIR}")
        return rows

    for human_path in human_files:
        name, role, model = role_model_from_filename(human_path)
        pipe_path = os.path.join(PIPE_DIR, f"{name}.csv")

        if not os.path.exists(pipe_path):
            print(f"[WARN] Pipeline file not found for {os.path.basename(human_path)}: {pipe_path}")
            continue

        human = pd.read_csv(human_path)
        pipe = pd.read_csv(pipe_path)
        merged = merge_majority_labels(human, "human_label", pipe, "pipeline_label")
        if merged.empty:
            continue

        metrics = score_labels(merged["human_label"], merged["pipeline_label"])
        print(
            f"{role} ({model}): acc = {metrics['accuracy']:.3f}, "
            f"macro_rec = {metrics['macro_recall']:.3f}, kappa = {metrics['kappa']:.3f} "
            f"over {len(merged)} (cohort,dimension) pairs"
        )

        rows.append(
            {
                "role": role,
                "model": model,
                "human_file": os.path.relpath(human_path, BASE_DIR).replace(os.sep, "/"),
                "pipeline_file": os.path.basename(pipe_path),
                "accuracy": metrics["accuracy"],
                "macro_recall": metrics["macro_recall"],
                "kappa": metrics["kappa"],
            }
        )
        all_true.extend(merged["human_label"])
        all_pred.extend(merged["pipeline_label"])

    if all_true:
        overall = score_labels(all_true, all_pred)
    else:
        overall = {"accuracy": float("nan"), "macro_recall": float("nan"), "kappa": float("nan")}

    print(
        f"\nOverall: acc = {overall['accuracy']:.3f}, "
        f"macro_rec = {overall['macro_recall']:.3f}, kappa = {overall['kappa']:.3f}"
    )
    rows.append(
        {
            "role": "ALL",
            "model": "ALL",
            "human_file": "",
            "pipeline_file": "",
            "accuracy": overall["accuracy"],
            "macro_recall": overall["macro_recall"],
            "kappa": overall["kappa"],
        }
    )
    return rows


def main():
    rows = compute_all_pairs()
    if not rows:
        return

    agreement_df = pd.DataFrame(rows)
    agreement_df.to_csv(AGREEMENT_OUT, index=False)
    print(f"\nSaved human-pipeline agreement results to {AGREEMENT_OUT}")

    kappa_df = agreement_df[["role", "model", "human_file", "pipeline_file", "kappa"]]
    kappa_df.to_csv(KAPPA_OUT, index=False)
    print(f"Saved Cohen's kappa results to {KAPPA_OUT}")


if __name__ == "__main__":
    main()
