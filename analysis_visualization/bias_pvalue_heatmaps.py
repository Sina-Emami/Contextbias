"""Generate p-value heatmaps for bias analysis across cohorts, contexts, and occupations."""

import re
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec

# ---------- configuration ----------
FILE_DIR = Path(__file__).resolve().parent
DATA_PATH = FILE_DIR / "significant_analysis" / "bias2_chi_square_results.csv"  # default to pipeline output

PLOTS_DIR = FILE_DIR / "plots"
PLOT1_DIR = PLOTS_DIR / "plot1_uniformity"
PLOT2_DIR = PLOTS_DIR / "plot2_context_summary"
CONTEXT_HEATMAP_DIR = PLOTS_DIR / "context_heatmaps"
COHORT_HEATMAP_DIR = PLOTS_DIR / "cohort_compare_heatmaps"

for path in [PLOTS_DIR, PLOT1_DIR, PLOT2_DIR, CONTEXT_HEATMAP_DIR, COHORT_HEATMAP_DIR]:
    path.mkdir(parents=True, exist_ok=True)

DEFAULT_COHORT_ORDER = ["camera", "objects", "people", "safety", "scene_appearance"]

# Font configuration (Times New Roman, smaller fonts)
mpl.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 9,
        "figure.titlesize": 14,
    }
)

# Seaborn colorblind palette for categorical use (available if needed downstream)
cb_palette = sns.color_palette("colorblind", n_colors=len(DEFAULT_COHORT_ORDER))
cohort_colors = dict(zip(DEFAULT_COHORT_ORDER, cb_palette))

# Diverging blue↔orange palette
cmap_p = LinearSegmentedColormap.from_list(
    "blue_orange_div",
    [
    "#275E8C",  # dark blue
    "#A3C5DF",  # your original blue
    "#F7F7F7",  # neutral
    "#FDC27C",  # light orange
    "#D9791F",  # darker orange
],
)
cmap_p.set_bad("white")
ALPHA = 0.05  # keep your threshold


def sanitize_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(value))


def resolve_data_path(path: Path | str) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = FILE_DIR / candidate
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"Data source not found at: {candidate}")


def load_bias_tables(data_path: Path):
    path = resolve_data_path(data_path)
    tables: dict[str, pd.DataFrame] = {}

    if path.suffix.lower() in {".xlsx", ".xls", ".xlsm"}:
        xls = pd.ExcelFile(path)
        required = {"Bias": "bias", "Bias1": "bias1", "Bias2": "bias2"}
        missing = [sheet for sheet in required if sheet not in xls.sheet_names]
        if missing:
            raise ValueError(f"Missing sheet(s) in Excel file: {missing}")
        for sheet, key in required.items():
            tables[key] = xls.parse(sheet)
    else:
        # Fall back to CSV files living next to the provided path
        tables["bias2"] = pd.read_csv(path)
        bias_csv = path.with_name("bias_uniformity.csv")
        bias1_csv = path.with_name("bias_context_summary.csv")
        if bias_csv.exists():
            tables["bias"] = pd.read_csv(bias_csv)
        if bias1_csv.exists():
            tables["bias1"] = pd.read_csv(bias1_csv)

    return tables


def prepare_bias_table(bias_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Index]:
    rename_map = {
        "Cohort": "cohort",
        "Dimension": "dimension",
        "Chi-square": "chi_value",
        "Chi_value": "chi_value",
        "p-value": "p_value",
        "pvalue": "p_value",
    }
    bias = bias_df.rename(columns=rename_map).copy()
    required = {"cohort", "dimension", "chi_value", "df", "p_value"}
    missing = required - set(bias.columns)
    if missing:
        raise ValueError(f"Bias sheet missing columns: {missing}")

    bias["p_value"] = pd.to_numeric(bias["p_value"], errors="coerce").clip(0, 1)
    bias["cohort"] = bias["cohort"].astype(str)
    bias["dimension"] = bias["dimension"].astype(str)

    cd_levels = (
        bias.sort_values(["cohort", "dimension"])
        .apply(lambda row: f"{row['cohort']}_{row['dimension']}", axis=1)
        .unique()
    )
    bias["cohort_dimension"] = pd.Categorical(
        bias.apply(lambda row: f"{row['cohort']}_{row['dimension']}", axis=1),
        categories=cd_levels,
        ordered=True,
    )

    return bias, cd_levels


def prepare_bias1_table(
    bias1_df: pd.DataFrame, cd_levels: pd.Index | None
) -> tuple[pd.DataFrame, list[str]]:
    rename_map = {
        "Cohort": "cohort",
        "Dimension": "dimension",
        "Context": "Context",
        "Chi-square": "chi_value",
        "Chi_value": "chi_value",
        "p-value": "p_value",
        "pvalue": "p_value",
    }
    bias1 = bias1_df.rename(columns=rename_map).copy()
    required = {"cohort", "dimension", "Context", "chi_value", "df", "p_value"}
    missing = required - set(bias1.columns)
    if missing:
        raise ValueError(f"Bias1 sheet missing columns: {missing}")

    bias1["p_value"] = pd.to_numeric(bias1["p_value"], errors="coerce").clip(0, 1)
    bias1["cohort"] = bias1["cohort"].astype(str)
    bias1["dimension"] = bias1["dimension"].astype(str)
    bias1["Context"] = bias1["Context"].astype(str)

    if cd_levels is None:
        cd_levels = (
            bias1.sort_values(["cohort", "dimension"])
            .apply(lambda row: f"{row['cohort']}_{row['dimension']}", axis=1)
            .unique()
        )

    bias1["cohort_dimension"] = pd.Categorical(
        bias1.apply(lambda row: f"{row['cohort']}_{row['dimension']}", axis=1),
        categories=cd_levels,
        ordered=True,
    )

    cohort_values = [c for c in bias1["cohort"].unique() if c != "totals"]
    ordered_cohorts = [
        c for c in DEFAULT_COHORT_ORDER if c in cohort_values
    ] + [c for c in cohort_values if c not in DEFAULT_COHORT_ORDER]

    bias1["cohort"] = pd.Categorical(bias1["cohort"], categories=ordered_cohorts, ordered=True)

    return bias1, ordered_cohorts


def prepare_bias2_table(
    bias2_df: pd.DataFrame, cohort_sequence: list[str] | None
) -> tuple[pd.DataFrame, list[str]]:
    rename_map = {
        "Cohort": "cohort",
        "Occupation": "occupation",
        "Dimension": "dimension",
        "Context": "Context",
        "p-value": "p_value",
        "pvalue": "p_value",
    }
    bias2 = bias2_df.rename(columns=rename_map).copy()
    required = {"Context", "cohort", "occupation", "dimension", "p_value"}
    missing = required - set(bias2.columns)
    if missing:
        raise ValueError(f"Bias2 sheet missing columns: {missing}")

    bias2["p_value"] = pd.to_numeric(bias2["p_value"], errors="coerce").clip(0, 1)
    bias2["cohort"] = bias2["cohort"].astype(str)
    bias2["Context"] = bias2["Context"].astype(str)
    bias2["occupation"] = bias2["occupation"].astype(str)
    bias2["dimension"] = bias2["dimension"].astype(str)

    cohorts = [c for c in bias2["cohort"].unique() if c != "totals"]
    if cohort_sequence is None:
        cohort_sequence = [
            c for c in DEFAULT_COHORT_ORDER if c in cohorts
        ] + [c for c in cohorts if c not in DEFAULT_COHORT_ORDER]

    bias2["cohort"] = pd.Categorical(bias2["cohort"], categories=cohort_sequence, ordered=True)

    return bias2, cohort_sequence


def _prepare_heatmap_pivot(df: pd.DataFrame, cohort_order: list[str] | None):
    work = df.copy()
    if "cohort_dimension" not in work.columns:
        work["cohort_dimension"] = work.apply(
            lambda row: f"{row['cohort']}_{row['dimension']}", axis=1
        )
    if hasattr(work["cohort_dimension"], "cat"):
        work["_order"] = work["cohort_dimension"].cat.codes
    else:
        work["_order"] = pd.factorize(work["cohort_dimension"])[0]
    work = work.sort_values("_order")
    work["dimension_label"] = work["cohort_dimension"].astype(str).str.split("_", n=1).str.get(1)
    order = work["dimension_label"].drop_duplicates().tolist()
    pivot = work.pivot(index="dimension_label", columns="cohort", values="p_value")
    pivot = pivot.reindex(order)
    if cohort_order:
        cols = [c for c in cohort_order if c in pivot.columns]
    else:
        cols = []
    remaining = [c for c in pivot.columns if c not in cols]
    pivot = pivot[cols + remaining]
    return pivot


def create_plot1(bias: pd.DataFrame):
    data = bias[bias["cohort"] != "totals"].copy()
    if data.empty:
        print("Plot 1 skipped: no data.")
        return None

    cohort_order = [
        c for c in DEFAULT_COHORT_ORDER if c in data["cohort"].unique()
    ] + [c for c in data["cohort"].unique() if c not in DEFAULT_COHORT_ORDER]

    pivot = _prepare_heatmap_pivot(data, cohort_order)
    if pivot.empty:
        print("Plot 1 skipped: unable to build pivot table.")
        return None

    fig, ax = plt.subplots(figsize=(10, 8))
    im = draw_heatmap(
        ax,
        pivot,
        title="Uniformity Test across all Context",
        show_y=True,
        x_tick_fontsize=10,
        y_tick_fontsize=9,
        annot_fontsize=4,
        y_label="Dimension (grouped by Cohort)",
        x_tick_rotation=45,
    )
    ax.set_xlabel("Cohort")
    fig.tight_layout()
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("p-value", fontsize=10)

    png_path = PLOT1_DIR / "plot1_uniformity_test.png"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved Plot 1: {png_path}")
    return png_path



def create_plot2(bias1: pd.DataFrame, cohort_order: list[str]):
    data = bias1[bias1["cohort"] != "totals"].copy()
    if data.empty:
        print("Plot 2 skipped: no data.")
        return None

    context_order = [
        ctx
        for ctx in [
            "Context-aware_Related_CA-R",
            "Context-aware_Unrelated_CA-U",
            "Context-free_CF",
        ]
        if ctx in data["Context"].unique()
    ]
    if not context_order:
        context_order = sorted(data["Context"].unique())

    fig_width = max(18, 6 * len(context_order))
    fig, axes = plt.subplots(1, len(context_order), figsize=(fig_width, 8), sharey=True)
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])

    last_im = None
    for idx, (ctx, ax) in enumerate(zip(context_order, axes)):
        subset = data[data["Context"] == ctx].copy()
        if subset.empty:
            ax.axis("off")
            ax.set_title(ctx, fontweight="bold", fontsize=12)
            continue

        pivot = _prepare_heatmap_pivot(subset, cohort_order)
        if pivot.empty:
            ax.axis("off")
            ax.set_title(ctx, fontweight="bold", fontsize=12)
            continue

        last_im = draw_heatmap(
            ax,
            pivot,
            title=ctx,
            show_y=(idx == 0),
            x_tick_fontsize=8,
            y_tick_fontsize=7,
            annot_fontsize=3,
            y_label="Dimension (grouped by Cohort)" if idx == 0 else "",
            x_tick_rotation=45,
        )
        ax.set_xlabel("Cohort")

    fig.suptitle(
        "Significance of p-values by Dimension and Cohort within each Context",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )
    fig.tight_layout(rect=[0, 0, 0.92, 0.95], w_pad=3.0)

    if last_im is not None:
        cbar = fig.colorbar(last_im, ax=axes.ravel().tolist(), fraction=0.025, pad=0.02)
        cbar.set_label("p-value", fontsize=10)

    png_path = PLOT2_DIR / "plot2_bias_by_context.png"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved Plot 2: {png_path}")
    return png_path



def draw_heatmap(
    ax,
    pivot_df,
    title=None,
    show_y=True,
    cohort_order=None,
    x_tick_fontsize=8,
    y_tick_fontsize=2,
    annot_fontsize=2.5,
    y_label="Occupation",
    x_tick_rotation=90,
):
    data = np.ma.masked_invalid(pivot_df.values)
    im = ax.imshow(
        data,
        aspect="auto",
        interpolation="none",
        cmap=cmap_p,
        vmin=0,
        vmax=1,
        origin="upper",
    )
    ax.set_xticks(np.arange(pivot_df.shape[1]))
    ax.set_xticklabels(
        pivot_df.columns, rotation=x_tick_rotation, ha="center", va="top", fontsize=x_tick_fontsize
    )
    if show_y:
        ax.set_yticks(np.arange(pivot_df.shape[0]))
        ax.set_yticklabels(pivot_df.index, fontsize=y_tick_fontsize)
        if y_label:
            ax.set_ylabel(y_label)
    else:
        ax.set_yticks([])
        ax.set_ylabel("")
    ax.set_title(title or "", fontweight="bold", fontsize=12)
    ax.set_xticks(np.arange(-0.5, pivot_df.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, pivot_df.shape[0], 1), minor=True)
    ax.grid(which="minor", color="lightgrey", linewidth=0.3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im


def plot_context_svg(df, ctx_name, cohort_order):
    ctx_df = df[df["Context"] == ctx_name]
    pivots, widths = [], []
    last_im = None

    for cohort in cohort_order:
        sub = ctx_df[ctx_df["cohort"] == cohort]
        if sub.empty:
            pivots.append(pd.DataFrame(index=[], columns=[]))
            widths.append(1)
        else:
            pivot = sub.pivot(index="occupation", columns="dimension", values="p_value").sort_index(axis=0)
            pivots.append(pivot)
            widths.append(max(1, pivot.shape[1]))

    fig = plt.figure(figsize=(16, 7))
    gs = GridSpec(1, len(cohort_order), width_ratios=widths, wspace=0.05, figure=fig)

    for idx, (cohort, pivot) in enumerate(zip(cohort_order, pivots)):
        ax = fig.add_subplot(gs[0, idx])
        if pivot.size == 0:
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(cohort, fontweight="bold", fontsize=12)
            ax.set_frame_on(True)
        else:
            last_im = draw_heatmap(ax, pivot, title=cohort, show_y=(idx == 0), cohort_order=cohort_order)

    fig.subplots_adjust(bottom=0.28, right=0.88, wspace=0.05)

    if last_im is not None:
        cbar = fig.colorbar(
            last_im, ax=fig.axes, location="right", fraction=0.06, pad=0.03, aspect=30
        )
        cbar.set_label("p-value", fontsize=10)

    fig.suptitle(
        f"Significance of p-values by Dimension and Cohort within each Occupation — {ctx_name}",
        y=0.98,
        fontsize=14,
        fontweight="bold",
    )
    fig.text(0.5, 0.02, "Dimension", ha="center", fontsize=10)

    context_dir = CONTEXT_HEATMAP_DIR / sanitize_filename(ctx_name)
    context_dir.mkdir(parents=True, exist_ok=True)
    output_path = context_dir / f"pvalue_heatmap_{sanitize_filename(ctx_name)}.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")



def plot_cohort_comparison_svg(df, cohort, contexts, cohort_order):
    pivots = []
    for ctx in contexts:
        sub = df[(df["cohort"] == cohort) & (df["Context"] == ctx)]
        if sub.empty:
            pivots.append((ctx, pd.DataFrame(index=[], columns=[])))
        else:
            pivot = sub.pivot(index="occupation", columns="dimension", values="p_value").sort_index(axis=0)
            pivots.append((ctx, pivot))

    all_dims = sorted({dim for _, pivot in pivots for dim in pivot.columns})
    aligned = [(ctx, pivot.reindex(columns=all_dims)) for ctx, pivot in pivots]
    n_contexts = len(contexts)
    fig_width = max(12, 4 * n_contexts)

    fig = plt.figure(figsize=(fig_width, 7))
    gs = GridSpec(1, n_contexts, figure=fig, wspace=0.05)
    last_im = None

    for idx, (ctx, pivot) in enumerate(aligned):
        ax = fig.add_subplot(gs[0, idx])
        if pivot.size == 0:
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(ctx, fontweight="bold", fontsize=12)
            ax.set_frame_on(True)
        else:
            last_im = draw_heatmap(ax, pivot, title=ctx, show_y=(idx == 0), cohort_order=cohort_order)

    fig.subplots_adjust(bottom=0.28, right=0.88, wspace=0.05)

    if last_im is not None:
        cbar = fig.colorbar(
            last_im, ax=fig.axes, location="right", fraction=0.06, pad=0.03, aspect=30
        )
        cbar.set_label("p-value", fontsize=10)

    fig.suptitle(
        f"{cohort}: p-values by Dimension & Occupation — comparison across contexts",
        y=0.98,
        fontsize=14,
        fontweight="bold",
    )
    fig.text(0.5, 0.02, "Dimension", ha="center", fontsize=10)

    cohort_dir = COHORT_HEATMAP_DIR / sanitize_filename(cohort)
    cohort_dir.mkdir(parents=True, exist_ok=True)
    output_path = cohort_dir / f"{sanitize_filename(cohort)}_compare_contexts.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")



def generate_plots(data_path: Path = DATA_PATH):
    tables = load_bias_tables(data_path)

    bias = tables.get("bias")
    bias1 = tables.get("bias1")
    bias2 = tables.get("bias2")

    cd_levels = None
    cohort_order = DEFAULT_COHORT_ORDER

    if bias is not None:
        bias, cd_levels = prepare_bias_table(bias)
        create_plot1(bias)
    else:
        print("Bias data not found; skipping Plot 1.")

    if bias1 is not None:
        bias1, inferred_order = prepare_bias1_table(bias1, cd_levels)
        cohort_order = inferred_order or cohort_order
        create_plot2(bias1, cohort_order)
    else:
        print("Bias1 data not found; skipping Plot 2.")

    if bias2 is not None:
        bias2, cohort_order = prepare_bias2_table(bias2, cohort_order)
        contexts = list(bias2["Context"].dropna().unique())
        for context in contexts:
            plot_context_svg(bias2, context, cohort_order)
        for cohort in cohort_order:
            if pd.isna(cohort):
                continue
            if (bias2["cohort"] == cohort).any():
                plot_cohort_comparison_svg(bias2, cohort, contexts, cohort_order)
    else:
        print("Bias2 data not found; skipping heatmaps.")


if __name__ == "__main__":
    generate_plots(DATA_PATH)
