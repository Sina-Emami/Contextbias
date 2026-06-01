"""Load, process, and compute chi-square bias statistics from clean_frequency CSVs."""

from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - tqdm is optional
    tqdm = None


FILE_DIR = Path(__file__).resolve().parent
SIGNIFICANT_DIR = FILE_DIR / "significant_analysis_35"
SIGNIFICANT_DIR.mkdir(parents=True, exist_ok=True)


def progress_iter(iterable, description, unit=None, total=None, print_every=50):
    """
    Wrap an iterable with a tqdm progress bar when available.
    Falls back to a plain iterable with a status print otherwise.
    """
    if tqdm is not None:
        return tqdm(iterable, desc=description, unit=unit, total=total, leave=False)

    label = unit or "items"
    print(f"{description}...")
    def generator():
        count = 0
        for count, item in enumerate(iterable, start=1):
            if print_every and (count == 1 or count % print_every == 0):
                print(f"  processed {count} {label}")
            yield item
        if count:
            print(f"{description} complete ({count} {label}).")
        else:
            print(f"{description} complete (no {label}).")

    return generator()


def complete(df, group_cols, complete_cols):
    """
    Replicates tidyr's complete() by ensuring all combinations exist.
    """
    idx = pd.MultiIndex.from_product(
        [df[col].unique() for col in complete_cols],
        names=complete_cols
    )
    return (
        df.set_index(complete_cols)
        .groupby(group_cols)
        .apply(lambda d: d.reindex(idx))
        .drop(group_cols, axis=1)
        .reset_index()
    )


def load_and_merge_data(dataset_roots=None):
    # --- 1. Data Loading and Merging ---
    if dataset_roots is None:
        default_root = FILE_DIR.parent / "92_job_description_3.5"
        dataset_roots = [default_root]
    else:
        resolved_roots = []
        for root in dataset_roots:
            root_path = Path(root)
            if not root_path.is_absolute():
                root_path = (FILE_DIR.parent / root_path).resolve()
            resolved_roots.append(root_path)
        dataset_roots = resolved_roots

    csv_paths = {
        path.resolve()
        for root in dataset_roots
        if Path(root).exists()
        for path in Path(root).rglob("clean_frequency/*.csv")
        if path.is_file()
    }
    if not csv_paths:
        roots_list = ", ".join(str(Path(root)) for root in dataset_roots)
        raise ValueError(
            "No valid data files were found. Checked roots: "
            f"{roots_list}. Please verify the dataset location or pass dataset_roots."
        )

    csv_paths = sorted(csv_paths)

    merged_list = []
    skipped = 0
    for file_path in progress_iter(csv_paths, "Loading clean_frequency CSVs", "file", total=len(csv_paths)):
        file_posix = file_path.as_posix()
        parts = file_posix.split('/')

        try:
            clean_idx = parts.index('clean_frequency')
        except ValueError:
            print(f"Skipping file {file_posix}: unable to locate 'clean_frequency' segment.")
            continue

        try:
            high_level_context = parts[clean_idx - 3]
            occupation = parts[clean_idx - 2]
            specific_context = parts[clean_idx - 1]
        except IndexError:
            print(f"Skipping file {file_posix}: unexpected directory layout.")
            continue

        try:
            df = pd.read_csv(file_path)
        except (pd.errors.EmptyDataError, FileNotFoundError) as exc:
            skipped += 1
            print(f"Skipping file {file_posix} due to error: {exc}")
            continue

        df['occupation'] = occupation
        df['Context'] = high_level_context
        df['specific_context'] = specific_context
        merged_list.append(df)

    if skipped and merged_list:
        print(f"Finished loading files with {skipped} skipped due to errors.")

    if not merged_list:
        raise ValueError(
            "No valid data files could be read after scanning available paths. "
            "Please verify dataset contents or provide dataset_roots explicitly."
        )

    merged = pd.concat(merged_list, ignore_index=True)

    print("Data loaded successfully!")
    print(f"CSV files processed: {len(merged_list)}")
    print(f"Shape: {merged.shape}")
    print(f"Unique high-level contexts: {sorted(merged['Context'].unique())}")
    print(f"Unique occupations: {len(merged['occupation'].unique())}")
    print(f"Unique specific contexts: {len(merged['specific_context'].unique())}")
    print(f"Sample of contexts: {list(merged['Context'].unique())}")
    print(f"Sample of specific contexts: {list(merged['specific_context'].unique())[:5]}")

    return merged


def process_data(merged):
    # --- 2. Data Processing ---
    print("  Casting counts to numeric")
    merged['count'] = pd.to_numeric(merged['count'], errors='coerce').fillna(0)

    print("  Calculating label bins per (dimension, cohort)")
    overall_prop = (
        merged
        .groupby(['dimension', 'cohort'])
        .apply(lambda x: x.assign(bins=x['label'].nunique()))
        .reset_index(drop=True)
        .query('label != "unknown"')
        .copy()
    )

    def complete_r_style(df):
        contexts = df['Context'].unique()
        occupations = df['occupation'].unique()
        labels = df['label'].unique()

        all_combinations = list(product(contexts, occupations, labels))
        complete_df = pd.DataFrame(all_combinations, columns=['Context', 'occupation', 'label'])

        for col in ['dimension', 'cohort']:
            complete_df[col] = df[col].iloc[0]

        result = complete_df.merge(
            df,
            on=['dimension', 'cohort', 'Context', 'occupation', 'label'],
            how='left'
        )

        mask = result['specific_context'].isna()
        if mask.any():
            context_occupation_mapping = (
                df.groupby(['Context', 'occupation'])['specific_context']
                .first()
                .to_dict()
            )
            for idx in result[mask].index:
                key = (result.loc[idx, 'Context'], result.loc[idx, 'occupation'])
                if key in context_occupation_mapping:
                    result.loc[idx, 'specific_context'] = context_occupation_mapping[key]
        return result

    print("  Completing context/occupation combinations per (dimension, cohort)")
    overall_prop = (
        overall_prop
        .groupby(['dimension', 'cohort'])
        .apply(complete_r_style)
        .reset_index(drop=True)
    )

    overall_prop['bins'] = overall_prop.groupby(['dimension', 'cohort'])['label'].transform('nunique')
    overall_prop['uniform.prop'] = 1 / overall_prop['bins']

    print("  Filling missing counts and computing totals")
    overall_prop['count'] = overall_prop['count'].fillna(0)

    overall_prop['n.with.coh'] = overall_prop.groupby(['dimension', 'cohort'])['count'].transform('sum')

    print("  Deriving proportion metrics")
    overall_prop['prop_overall'] = (
        overall_prop.groupby(['dimension', 'cohort', 'label'])['count'].transform('sum') /
        overall_prop['n.with.coh']
    )

    overall_prop['n.with.coh.Cont'] = overall_prop.groupby(['Context', 'dimension', 'cohort'])['count'].transform('sum')
    overall_prop['prop_context'] = (
        overall_prop.groupby(['Context', 'dimension', 'cohort', 'label'])['count'].transform('sum') /
        overall_prop['n.with.coh.Cont']
    )

    overall_prop['n.with.coh.Cont.occ'] = overall_prop.groupby(
        ['Context', 'dimension', 'cohort', 'occupation']
    )['count'].transform('sum')
    overall_prop['prop_occ'] = (
        overall_prop.groupby(['Context', 'dimension', 'cohort', 'label', 'occupation'])['count'].transform('sum') /
        overall_prop['n.with.coh.Cont.occ']
    )

    overall_prop = overall_prop.drop(columns=['n.with.coh.Cont.occ', 'n.with.coh.Cont', 'n.with.coh'])

    print("  Computing expected/observed values and chi-square components")
    overall_prop['exp'] = overall_prop['prop_overall'] * overall_prop['count']
    overall_prop['ch.v'] = np.where(
        overall_prop['exp'] == 0,
        0,
        ((overall_prop['exp'] - overall_prop['count']) ** 2) / overall_prop['exp']
    )

    overall_prop['df'] = overall_prop.groupby(['dimension', 'cohort'])['label'].transform('nunique') - 1

    overall_prop['exp.whole'] = (
        overall_prop.groupby(['dimension', 'cohort'])['count'].transform('sum') *
        overall_prop['uniform.prop']
    )
    overall_prop['obs.whole'] = overall_prop.groupby(['dimension', 'cohort', 'label'])['count'].transform('sum')

    overall_prop['exp.whole.context'] = (
        overall_prop.groupby(['dimension', 'cohort', 'Context'])['count'].transform('sum') *
        overall_prop['prop_overall']
    )
    overall_prop['obs.whole.context'] = overall_prop.groupby(
        ['dimension', 'cohort', 'label', 'Context']
    )['count'].transform('sum')

    overall_prop['exp.whole.context.occ'] = (
        overall_prop.groupby(['dimension', 'cohort', 'Context', 'occupation'])['count'].transform('sum') *
        overall_prop['prop_overall']
    )
    overall_prop['obs.whole.context.occ'] = overall_prop.groupby(
        ['dimension', 'cohort', 'label', 'Context', 'occupation']
    )['count'].transform('sum')

    print("Data processing completed!")
    print(f"Shape: {overall_prop.shape}")
    print(f"Unique contexts: {sorted(overall_prop['Context'].unique())}")
    print(f"Non-zero exp.whole count: {(overall_prop['exp.whole'] > 0).sum()}")
    print(f"Non-zero obs.whole count: {(overall_prop['obs.whole'] > 0).sum()}")
    print(f"Non-zero exp.whole.context.occ count: {(overall_prop['exp.whole.context.occ'] > 0).sum()}")
    print(f"Non-zero obs.whole.context.occ count: {(overall_prop['obs.whole.context.occ'] > 0).sum()}")

    return overall_prop


def calculate_bias(overall_prop):
    Bias = (
        overall_prop[['cohort', 'dimension', 'label', 'exp.whole', 'obs.whole']]
        .drop_duplicates()
        .copy()
    )

    Bias['df'] = Bias.groupby(['dimension', 'cohort'])['label'].transform('nunique') - 1

    Bias['ch.v'] = np.where(
        Bias['exp.whole'] == 0,
        0,
        ((Bias['exp.whole'] - Bias['obs.whole']) ** 2) / Bias['exp.whole']
    )

    Bias_grouped = (
        Bias.groupby(['dimension', 'cohort', 'df'])
        .agg({'ch.v': 'sum'})
        .rename(columns={'ch.v': 'chi_value'})
        .reset_index()
    )

    Bias_grouped['p_value'] = 1 - chi2.cdf(Bias_grouped['chi_value'], Bias_grouped['df'])
    Bias_grouped['p_value'] = np.clip(Bias_grouped['p_value'], 0, 1)

    Bias = Bias_grouped.copy()

    cd_levels = (
        Bias.sort_values(['cohort', 'dimension'])
        .apply(lambda row: f"{row['cohort']}_{row['dimension']}", axis=1)
        .unique()
    )

    Bias['cohort_dimension'] = pd.Categorical(
        Bias.apply(lambda row: f"{row['cohort']}_{row['dimension']}", axis=1),
        categories=cd_levels,
        ordered=True
    )

    print("Bias calculation completed!")
    print(f"Bias shape: {Bias.shape}")
    print(f"P-value range: {Bias['p_value'].min():.6f} to {Bias['p_value'].max():.6f}")
    print(f"Chi-square value range: {Bias['chi_value'].min():.6f} to {Bias['chi_value'].max():.6f}")
    print("\nSample of Bias data:")
    print(Bias[['cohort', 'dimension', 'chi_value', 'df', 'p_value']].head())

    return Bias, cd_levels


def calculate_bias1(overall_prop, cd_levels):
    Bias1 = (
        overall_prop[['cohort', 'dimension', 'label', 'Context', 'exp.whole.context', 'obs.whole.context']]
        .drop_duplicates()
        .copy()
    )

    Bias1['df'] = Bias1.groupby(['dimension', 'cohort', 'Context'])['label'].transform('nunique') - 1

    Bias1['ch.v'] = np.where(
        Bias1['exp.whole.context'] == 0,
        0,
        ((Bias1['exp.whole.context'] - Bias1['obs.whole.context']) ** 2) /
        Bias1['exp.whole.context']
    )

    Bias1_grouped = (
        Bias1.groupby(['dimension', 'cohort', 'Context', 'df'])
        .agg({'ch.v': 'sum'})
        .rename(columns={'ch.v': 'chi_value'})
        .reset_index()
    )

    Bias1_grouped['p_value'] = 1 - chi2.cdf(Bias1_grouped['chi_value'], Bias1_grouped['df'])
    Bias1_grouped['p_value'] = np.clip(Bias1_grouped['p_value'], 0, 1)

    Bias1 = Bias1_grouped.copy()

    Bias1['cohort_dimension'] = pd.Categorical(
        Bias1.apply(lambda row: f"{row['cohort']}_{row['dimension']}", axis=1),
        categories=cd_levels,
        ordered=True
    )

    cohort_order = sorted([c for c in Bias1['cohort'].unique() if c != 'totals'])

    print("Bias1 calculation completed!")
    print(f"Bias1 shape: {Bias1.shape}")
    print(f"P-value range: {Bias1['p_value'].min():.6f} to {Bias1['p_value'].max():.6f}")
    print(f"Contexts: {sorted(Bias1['Context'].unique())}")

    return Bias1, cohort_order


def calculate_bias2(overall_prop):
    Bias2 = (
        overall_prop[
            ['cohort', 'dimension', 'label', 'Context', 'occupation', 'exp.whole.context.occ', 'obs.whole.context.occ']
        ]
        .drop_duplicates()
        .copy()
    )

    Bias2['df'] = Bias2.groupby(['dimension', 'cohort', 'Context', 'occupation'])['label'].transform('nunique') - 1

    Bias2['ch.v'] = np.where(
        Bias2['exp.whole.context.occ'] == 0,
        0,
        ((Bias2['exp.whole.context.occ'] - Bias2['obs.whole.context.occ']) ** 2) /
        Bias2['exp.whole.context.occ']
    )

    Bias2_grouped = (
        Bias2.groupby(['dimension', 'cohort', 'Context', 'occupation', 'df'])
        .agg({'ch.v': 'sum'})
        .rename(columns={'ch.v': 'chi_value'})
        .reset_index()
    )

    Bias2_grouped['p_value'] = 1 - chi2.cdf(Bias2_grouped['chi_value'], Bias2_grouped['df'])
    Bias2_grouped['p_value'] = np.clip(Bias2_grouped['p_value'], 0, 1)

    Bias2 = Bias2_grouped.copy()

    print("Bias2 calculation completed!")
    print(f"Bias2 shape: {Bias2.shape}")
    print(f"P-value range: {Bias2['p_value'].min():.6f} to {Bias2['p_value'].max():.6f}")
    print(f"Chi-square value range: {Bias2['chi_value'].min():.6f} to {Bias2['chi_value'].max():.6f}")
    print(f"Unique contexts: {sorted(Bias2['Context'].unique())}")
    print(f"Unique occupations: {Bias2['occupation'].nunique()}")
    print(f"Unique cohorts: {sorted(Bias2['cohort'].unique())}")
    print("\nSample of Bias2 data:")
    print(Bias2[['Context', 'cohort', 'dimension', 'occupation', 'chi_value', 'df', 'p_value']].head(10))

    return Bias2


def save_analysis_results(Bias, Bias1, Bias2, output_dir=SIGNIFICANT_DIR):
    output_path = Path(output_dir)
    if not output_path.is_absolute():
        output_path = (FILE_DIR / output_path).resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    bias_path = output_path / "bias_uniformity.csv"
    bias1_path = output_path / "bias_context_summary.csv"
    bias2_path = output_path / "bias2_chi_square_results.csv"

    Bias.to_csv(bias_path, index=False)
    print(f"✅ Saved: {bias_path}")

    Bias1.to_csv(bias1_path, index=False)
    print(f"✅ Saved: {bias1_path}")

    Bias2[['Context', 'cohort', 'dimension', 'occupation', 'chi_value', 'df', 'p_value']].to_csv(
        bias2_path,
        index=False
    )
    print(f"✅ Saved: {bias2_path}")

    return {
        "bias": bias_path,
        "bias1": bias1_path,
        "bias2": bias2_path,
    }


def main(output_dir=SIGNIFICANT_DIR, dataset_roots=None):
    print("Step 1/4: Loading and merging data")
    merged = load_and_merge_data(dataset_roots=dataset_roots)

    print("Step 2/4: Processing aggregated proportions")
    overall_prop = process_data(merged)

    print("Step 3/4: Calculating bias metrics")
    Bias, cd_levels = calculate_bias(overall_prop)
    Bias1, cohort_order = calculate_bias1(overall_prop, cd_levels)
    Bias2 = calculate_bias2(overall_prop)

    print("Step 4/4: Saving analysis outputs")
    paths = save_analysis_results(Bias, Bias1, Bias2, output_dir=output_dir)

    return Bias, Bias1, Bias2, paths


if __name__ == "__main__":
    main()
