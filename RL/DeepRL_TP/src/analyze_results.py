from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RESULTS_DIR = PROJECT_ROOT / "results"

ANALYSIS_DIR = PROJECT_ROOT / "analysis"

ANALYSIS_DIR.mkdir(exist_ok=True)


# ============================================================
# 1. Load all evaluation CSV files
# ============================================================

csv_files = sorted(RESULTS_DIR.glob("*.csv"))

if len(csv_files) == 0:
    raise RuntimeError(
        f"No CSV files found in {RESULTS_DIR}"
    )

print(f"CSV files found: {len(csv_files)}")


dataframes = []

for csv_file in csv_files:

    df = pd.read_csv(csv_file)

    df["source_file"] = csv_file.name

    dataframes.append(df)


all_results = pd.concat(
    dataframes,
    ignore_index=True,
)


print(f"Total evaluation episodes: {len(all_results)}")


# ============================================================
# 2. Create condition label
# ============================================================

all_results["condition"] = np.where(
    all_results["layer_norm"],
    all_results["algorithm"] + " + LN",
    all_results["algorithm"],
)


# ============================================================
# 3. Save all 400 evaluation episodes in one CSV
# ============================================================

all_results_path = (
    ANALYSIS_DIR / "all_episode_results.csv"
)

all_results.to_csv(
    all_results_path,
    index=False,
)

print(
    f"All episode results saved to: "
    f"{all_results_path}"
)


# ============================================================
# 4. Model-level summary
#
# IMPORTANT:
#
# One row = one independently trained model.
#
# We have:
#
# 4 conditions × 5 training seeds = 20 models
#
# Return:
#   calculated over ALL evaluation episodes
#
# Bias:
#   calculated only over naturally terminated episodes
#   because truncated episodes have NaN mean_bias.
# ============================================================

model_summary = (
    all_results
    .groupby(
        [
            "algorithm",
            "layer_norm",
            "condition",
            "training_seed",
        ],
        as_index=False,
    )
    .agg(
        mean_return=(
            "return",
            "mean",
        ),

        std_return=(
            "return",
            "std",
        ),

        mean_bias=(
            "mean_bias",
            "mean",
        ),

        std_bias=(
            "mean_bias",
            "std",
        ),

        evaluation_episodes=(
            "episode",
            "count",
        ),

        valid_bias_episodes=(
            "mean_bias",
            "count",
        ),

        truncated_episodes=(
            "truncated",
            "sum",
        ),

        mean_episode_length=(
            "steps",
            "mean",
        ),
    )
)


model_summary = model_summary.sort_values(
    [
        "algorithm",
        "layer_norm",
        "training_seed",
    ]
)


model_summary_path = (
    ANALYSIS_DIR / "model_summary.csv"
)

model_summary.to_csv(
    model_summary_path,
    index=False,
)


print(
    f"Model summary saved to: "
    f"{model_summary_path}"
)


# ============================================================
# 5. Condition-level summary
#
# Statistics here are computed across TRAINING SEEDS.
#
# This is important:
#
# n = 5 models per condition,
# NOT n = 100 evaluation episodes per condition.
# ============================================================

condition_summary = (
    model_summary
    .groupby(
        [
            "algorithm",
            "layer_norm",
            "condition",
        ],
        as_index=False,
    )
    .agg(
        models=(
            "training_seed",
            "count",
        ),

        mean_return=(
            "mean_return",
            "mean",
        ),

        std_return_across_models=(
            "mean_return",
            "std",
        ),

        mean_bias=(
            "mean_bias",
            "mean",
        ),

        std_bias_across_models=(
            "mean_bias",
            "std",
        ),

        mean_truncated_episodes=(
            "truncated_episodes",
            "mean",
        ),

        mean_valid_bias_episodes=(
            "valid_bias_episodes",
            "mean",
        ),
    )
)


condition_order = [
    "DQN",
    "DQN + LN",
    "DDQN",
    "DDQN + LN",
]


condition_summary["condition"] = pd.Categorical(
    condition_summary["condition"],
    categories=condition_order,
    ordered=True,
)


condition_summary = condition_summary.sort_values(
    "condition"
)


condition_summary_path = (
    ANALYSIS_DIR / "condition_summary.csv"
)

condition_summary.to_csv(
    condition_summary_path,
    index=False,
)


print(
    f"Condition summary saved to: "
    f"{condition_summary_path}"
)


# ============================================================
# 6. Print model-level results
# ============================================================

print()
print("=" * 70)
print("MODEL-LEVEL RESULTS")
print("=" * 70)

print(
    model_summary[
        [
            "condition",
            "training_seed",
            "mean_return",
            "mean_bias",
            "valid_bias_episodes",
            "truncated_episodes",
        ]
    ].to_string(index=False)
)


# ============================================================
# 7. Print condition-level results
# ============================================================

print()
print("=" * 70)
print("CONDITION-LEVEL RESULTS")
print("=" * 70)

print(
    condition_summary[
        [
            "condition",
            "models",
            "mean_return",
            "std_return_across_models",
            "mean_bias",
            "std_bias_across_models",
            "mean_truncated_episodes",
        ]
    ].to_string(index=False)
)


# ============================================================
# 8. Figure: Mean return by condition
# ============================================================

plot_data = []

for condition in condition_order:

    values = model_summary.loc[
        model_summary["condition"] == condition,
        "mean_return",
    ].values

    plot_data.append(values)


plt.figure(figsize=(9, 6))

plt.boxplot(
    plot_data,
    tick_labels=condition_order,
)

plt.ylabel(
    "Mean evaluation return"
)

plt.xlabel(
    "Condition"
)

plt.title(
    "Evaluation performance across training seeds"
)

plt.grid(
    axis="y",
    alpha=0.3,
)

plt.tight_layout()

return_plot_path = (
    ANALYSIS_DIR / "return_by_condition.png"
)

plt.savefig(
    return_plot_path,
    dpi=300,
)

plt.close()


# ============================================================
# 9. Figure: Mean bias by condition
# ============================================================

plot_data = []

for condition in condition_order:

    values = model_summary.loc[
        model_summary["condition"] == condition,
        "mean_bias",
    ].dropna().values

    plot_data.append(values)


plt.figure(figsize=(9, 6))

plt.boxplot(
    plot_data,
    tick_labels=condition_order,
)

plt.axhline(
    y=0,
    linestyle="--",
)

plt.ylabel(
    "Mean overestimation bias"
)

plt.xlabel(
    "Condition"
)

plt.title(
    "Overestimation bias across training seeds"
)

plt.grid(
    axis="y",
    alpha=0.3,
)

plt.tight_layout()

bias_plot_path = (
    ANALYSIS_DIR / "bias_by_condition.png"
)

plt.savefig(
    bias_plot_path,
    dpi=300,
)

plt.close()


# ============================================================
# 10. Figure: Bias vs performance
#
# IMPORTANT:
#
# Each point represents ONE trained model.
# Therefore there are at most 20 points.
# ============================================================

plt.figure(figsize=(9, 6))


markers = {
    "DQN": "o",
    "DQN + LN": "s",
    "DDQN": "^",
    "DDQN + LN": "D",
}


for condition in condition_order:

    subset = model_summary[
        model_summary["condition"] == condition
    ]

    plt.scatter(
        subset["mean_bias"],
        subset["mean_return"],
        label=condition,
        marker=markers[condition],
        s=70,
    )


plt.xlabel(
    "Mean overestimation bias"
)

plt.ylabel(
    "Mean evaluation return"
)

plt.title(
    "Relationship between overestimation bias and performance"
)

plt.axvline(
    x=0,
    linestyle="--",
)

plt.grid(
    alpha=0.3,
)

plt.legend()

plt.tight_layout()


scatter_path = (
    ANALYSIS_DIR / "bias_vs_performance.png"
)

plt.savefig(
    scatter_path,
    dpi=300,
)

plt.close()


# ============================================================
# 11. Correlation between bias and performance
#
# One observation = one trained model.
# ============================================================

valid_models = model_summary.dropna(
    subset=[
        "mean_bias",
        "mean_return",
    ]
)


if len(valid_models) >= 2:

    pearson_correlation = (
        valid_models[
            [
                "mean_bias",
                "mean_return",
            ]
        ]
        .corr(method="pearson")
        .iloc[0, 1]
    )

    spearman_correlation = (
        valid_models[
            [
                "mean_bias",
                "mean_return",
            ]
        ]
        .corr(method="spearman")
        .iloc[0, 1]
    )

else:

    pearson_correlation = np.nan
    spearman_correlation = np.nan


print()
print("=" * 70)
print("BIAS vs PERFORMANCE")
print("=" * 70)

print(
    f"Models used: "
    f"{len(valid_models)}"
)

print(
    f"Pearson correlation:  "
    f"{pearson_correlation:.4f}"
)

print(
    f"Spearman correlation: "
    f"{spearman_correlation:.4f}"
)


# ============================================================
# 12. Save correlation results
# ============================================================

correlation_results = pd.DataFrame(
    {
        "metric": [
            "pearson",
            "spearman",
        ],

        "correlation": [
            pearson_correlation,
            spearman_correlation,
        ],

        "n_models": [
            len(valid_models),
            len(valid_models),
        ],
    }
)


correlation_path = (
    ANALYSIS_DIR
    / "bias_performance_correlations.csv"
)

correlation_results.to_csv(
    correlation_path,
    index=False,
)


# ============================================================
# 13. Final message
# ============================================================

print()
print("=" * 70)
print("ANALYSIS COMPLETED")
print("=" * 70)

print(
    f"Analysis directory: "
    f"{ANALYSIS_DIR}"
)

print()
print("Generated files:")

print(
    " - all_episode_results.csv"
)

print(
    " - model_summary.csv"
)

print(
    " - condition_summary.csv"
)

print(
    " - bias_performance_correlations.csv"
)

print(
    " - return_by_condition.png"
)

print(
    " - bias_by_condition.png"
)

print(
    " - bias_vs_performance.png"
)

print("=" * 70)