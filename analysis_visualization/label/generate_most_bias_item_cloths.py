from pathlib import Path

import pandas as pd


def collect_bias_rows(role_dir: Path) -> pd.DataFrame:
    """Aggregate clothing/items counts above threshold for each occupation."""
    target_dimensions = {"clothing_garment", "items"}
    records = []

    for csv_path in sorted(role_dir.glob("*.csv")):
        if csv_path.name.lower() == "all_roles.csv":
            continue

        role_df = pd.read_csv(csv_path)
        role_df = role_df.loc[
            role_df["dimension"].isin(target_dimensions) & (role_df["count"] > 20)
        ].copy()
        if role_df.empty:
            continue

        role_df["occupation"] = csv_path.stem
        records.append(role_df)

    if not records:
        return pd.DataFrame(
            columns=["dimension", "cohort", "label", "occupation", "sum_context_obs"]
        )

    combined = pd.concat(records, ignore_index=True)
    return (
        combined.groupby(["dimension", "cohort", "label", "occupation"], as_index=False)[
            "count"
        ]
        .sum()
        .rename(columns={"count": "sum_context_obs"})
    )


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    base_dirs = {
        project_root / "92_job_description_SDXL/role_counting": project_root
        / "analysis_visualization/label/most_bias_item_cloths_SDXL.csv",
        project_root / "92_job_description_3.5/role_counting": project_root
        / "analysis_visualization/label/most_bias_item_cloths_SD3.5.csv",
    }

    for role_dir, output_path in base_dirs.items():
        bias_df = collect_bias_rows(role_dir)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        bias_df.to_csv(output_path, index=False)
        print(f"Saved {len(bias_df)} rows to {output_path}")


if __name__ == "__main__":
    main()
