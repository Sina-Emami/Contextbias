import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# 1. Load the CSV
# ---------------------------------------------------------
# Put your CSV in the same folder as this script or change the path
df = pd.read_csv("analysis_visualization/label/gender_summary.csv")

# ---------------------------------------------------------
# 2. Helper: get top-10 female, male, and balanced roles
#    for one model (SDXL or SD3.5)
# ---------------------------------------------------------
def compute_top_roles_for_model(df, model, min_count=25, top_n=10):
    """
    For a given model:
      - top_n female-label roles with count_female > min_count
      - top_n male-label roles with count_male > min_count
      - top_n most balanced roles (smallest |F - M|) with
        total_count > min_count and both genders present
      - returns also a 'selected' dataframe with up to 30 roles
        used for plotting (union of those 3 groups)
    """
    dfm = df[df["model"] == model].copy()
    if dfm.empty:
        raise ValueError(f"No rows for model {model}")

    # Pivot so each (role, cohort, dimension) has both genders on one row
    pivot = dfm.pivot_table(
        index=["role", "cohort", "dimension"],
        columns="label",
        values=["percentage", "count"],
        aggfunc="sum",
    )

    # Flatten multi-index columns into simple names
    pivot.columns = [f"{metric}_{gender}" for metric, gender in pivot.columns]

    # Make sure the expected columns exist
    for col in ["percentage_female", "percentage_male",
                "count_female", "count_male"]:
        if col not in pivot.columns:
            pivot[col] = 0.0

    pivot = pivot.fillna(0.0)
    pivot["total_count"] = pivot["count_female"] + pivot["count_male"]

    # ---- Top 10 female-label roles (count_female > min_count) ----
    top_female = (
        pivot[pivot["count_female"] > min_count]
        .sort_values("percentage_female", ascending=False)
        .head(top_n)
    )

    # ---- Top 10 male-label roles (count_male > min_count) ----
    top_male = (
        pivot[pivot["count_male"] > min_count]
        .sort_values("percentage_male", ascending=False)
        .head(top_n)
    )

    # ---- Top 10 most balanced roles ----
    # require both genders present and enough total samples
    balanced = pivot[
        (pivot["count_female"] > 0)
        & (pivot["count_male"] > 0)
        & (pivot["total_count"] > min_count)
    ].copy()

    balanced["diff"] = (balanced["percentage_female"]
                        - balanced["percentage_male"]).abs()

    top_balanced = (
        balanced.sort_values("diff", ascending=True)
        .head(top_n)
    )

    # ---- Union of the indices for plotting (up to 30 roles) ----
    # order: strongest male, then balanced, then strongest female
    union_idx = list(
        dict.fromkeys(
            list(top_male.index)
            + list(top_balanced.index)
            + list(top_female.index)
        )
    )

    selected = pivot.loc[union_idx].copy()
    selected["female_pct"] = selected["percentage_female"] * 100
    selected["male_pct"] = selected["percentage_male"] * 100

    # Sort so male-dominated roles appear at the bottom,
    # female-dominated at the top, balanced in the middle
    selected = selected.sort_values("male_pct", ascending=False)

    return pivot, top_female, top_male, top_balanced, selected


# ---------------------------------------------------------
# 3. Compute & print the top-10 lists for each model
# ---------------------------------------------------------
models = df["model"].unique()
results = {}

for model in models:
    pivot, top_f, top_m, top_b, selected = compute_top_roles_for_model(
        df, model, min_count=25, top_n=10
    )
    results[model] = {
        "pivot": pivot,
        "top_female": top_f,
        "top_male": top_m,
        "top_balanced": top_b,
        "selected": selected,
    }

    print(f"\n================ {model} ================")

    # Top 10 female-label roles
    print("\nTop 10 roles with highest FEMALE percentage (count_female > 25):")
    tmp = top_f[["percentage_female", "count_female"]].copy()
    tmp["percentage_female"] = (tmp["percentage_female"] * 100).round(1)
    tmp = tmp.rename(columns={"percentage_female": "female_%", "count_female": "female_count"})
    print(tmp)

    # Top 10 male-label roles
    print("\nTop 10 roles with highest MALE percentage (count_male > 25):")
    tmp = top_m[["percentage_male", "count_male"]].copy()
    tmp["percentage_male"] = (tmp["percentage_male"] * 100).round(1)
    tmp = tmp.rename(columns={"percentage_male": "male_%", "count_male": "male_count"})
    print(tmp)

    # Top 10 most balanced roles
    print("\nTop 10 MOST BALANCED roles (smallest |F% - M%|, total_count > 25):")
    tmp = top_b[["percentage_female", "percentage_male", "diff", "total_count"]].copy()
    tmp["percentage_female"] = (tmp["percentage_female"] * 100).round(1)
    tmp["percentage_male"] = (tmp["percentage_male"] * 100).round(1)
    tmp["diff"] = (tmp["diff"] * 100).round(1)
    tmp = tmp.rename(columns={
        "percentage_female": "female_%",
        "percentage_male": "male_%",
        "diff": "|F% - M%|"
    })
    print(tmp)


# ---------------------------------------------------------
# 4. Plotting: diverging bar chart like your first image
#    (two charts, one for each model, 30 roles each)
#    Colors similar to the grouped bar chart (blue & orange)
# ---------------------------------------------------------
# def plot_gender_chart_for_model(model, selected,
#                                 female_color="#A3C5DF",   # paler blue
#                                 male_color="#FAA43A"):    # same orange
#     """
#     selected: dataframe from compute_top_roles_for_model
#               with columns female_pct and male_pct
#     Produces a horizontal diverging bar chart:
#       - female bars to the left (negative)
#       - male bars to the right (positive)
#       - up to 30 roles, combining top female, top male, balanced
#     """
#     selected = selected.copy()

#     # y-axis labels (role only)
#     labels = []
#     for (role, cohort, dimension), row in selected.iterrows():
#         labels.append(role)

#     y = np.arange(len(selected))
#     female = -selected["female_pct"].values
#     male = selected["male_pct"].values
#     max_pct = max(selected["female_pct"].max(), selected["male_pct"].max())

#     fig, ax = plt.subplots(figsize=(12, 0.4 * len(selected) + 2))

#     ax.barh(y, female, color=female_color, label="Female")
#     ax.barh(y, male,   color=male_color,   label="Male")

#     ax.axvline(0, color="black", linewidth=1)

#     ax.set_yticks(y)
#     ax.set_yticklabels(labels)
#     ax.set_xlabel("Percentage of depictions")
#     ax.set_title(f"Gender distribution for top roles – {model}")

#     ax.set_xlim(-max_pct - 5, max_pct + 5)
#     ax.xaxis.grid(True, linestyle="--", alpha=0.4)
#     ax.legend(loc="lower right")
#     plt.tight_layout()
#     return fig, ax



def plot_gender_chart_for_model(model, selected,
                                cmap_name="viridis"):
    selected = selected.copy()

    # Create labels
    labels = []
    for (role, cohort, dimension), row in selected.iterrows():
        labels.append(role)

    y = np.arange(len(selected))
    female = -selected["female_pct"].values
    male = selected["male_pct"].values
    max_pct = max(selected["female_pct"].max(), selected["male_pct"].max())

    # === use the exact viridis palette ===
    cmap = plt.cm.get_cmap(cmap_name)
    female_color = cmap(0.92) # yellow-green side
    male_color = cmap(0.2) # purple-blue side
    # swap these if you want the opposite mapping

    fig, ax = plt.subplots(figsize=(12, 0.4 * len(selected) + 2))

    ax.barh(y, female, color=female_color, label="Female")
    ax.barh(y, male,   color=male_color,   label="Male")

    ax.axvline(0, color="black", linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Percentage of depictions")
    ax.set_title(f"Gender distribution for top roles – {model}")

    ax.set_xlim(-max_pct - 5, max_pct + 5)
    ax.xaxis.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="lower right")
    plt.tight_layout()
    return fig, ax



# ---------------------------------------------------------
# 5. Draw the two charts (one per model) with 30 roles each
# ---------------------------------------------------------
# ---------------------------------------------------------
# 5. Draw the two charts (one per model) with 30 roles each
#    and SAVE them instead of showing
# ---------------------------------------------------------
for model in models:
    selected = results[model]["selected"]
    fig, ax = plot_gender_chart_for_model(model, selected)

    # save each chart as a PNG (you can change the name / format)
    out_name = f"{model}_gender_roles.png"
    fig.savefig(out_name, dpi=300, bbox_inches="tight")
    plt.close(fig)  # free memory / avoid extra windows
