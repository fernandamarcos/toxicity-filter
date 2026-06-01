import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# =========================
# LOAD DATA
# =========================
nr_df = pd.read_csv("results/NR_metrics.csv")
sr_df = pd.read_csv("results/SR_metrics.csv")

os.makedirs("results/plots", exist_ok=True)

# =========================
# AGGREGATE (mean + std across seeds)
# =========================
def summarize(df):
    return df.groupby("task").agg(["mean", "std"])

nr = summarize(nr_df)
sr = summarize(sr_df)

# flatten columns
nr.columns = ["_".join(col) for col in nr.columns]
sr.columns = ["_".join(col) for col in sr.columns]

# =========================
# PLOT FUNCTION
# =========================
def plot_group(df, metrics, title, out_file):

    tasks = df.index.tolist()
    x = np.arange(len(tasks))

    plt.figure(figsize=(12, 5))

    width = 0.25

    for i, m in enumerate(metrics):

        mean = df[f"{m}_mean"].values
        std = df[f"{m}_std"].values

        plt.bar(
            x + i * width,
            mean,
            width=width,
            yerr=std,
            capsize=4,
            label=m,
            alpha=0.9
        )

    plt.xticks(x + width, tasks, rotation=45, ha="right")
    plt.ylim(0, 1)
    plt.title(title)
    plt.ylabel("Score")
    plt.legend()
    plt.tight_layout()

    plt.savefig(out_file, dpi=300)
    plt.close()

# =========================
# NR PLOTS
# =========================
plot_group(
    nr,
    metrics=["acc", "bacc", "auc"],
    title="NR Tasks: Accuracy / Balanced Accuracy / AUC",
    out_file="results/plots/nr_main.png"
)

plot_group(
    nr,
    metrics=["fpr", "fnr", "tpr", "tnr"],
    title="NR Tasks: Error Rates",
    out_file="results/plots/nr_rates.png"
)

# =========================
# SR PLOTS
# =========================
plot_group(
    sr,
    metrics=["acc", "bacc", "auc"],
    title="SR Tasks: Accuracy / Balanced Accuracy / AUC",
    out_file="results/plots/sr_main.png"
)

plot_group(
    sr,
    metrics=["fpr", "fnr", "tpr", "tnr"],
    title="SR Tasks: Error Rates",
    out_file="results/plots/sr_rates.png"
)

print("DONE — plots saved in results/plots/")