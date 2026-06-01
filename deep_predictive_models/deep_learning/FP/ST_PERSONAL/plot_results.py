import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

results_dir = "results"
plot_dir = "plots_summary/tasks_individual"
os.makedirs(plot_dir, exist_ok=True)

# =========================
# LOAD DATA
# =========================
files = glob.glob(f"{results_dir}/*_baseline.csv")
df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

metrics = ["acc", "f1", "auc", "bacc", "brier", "fpr", "fnr"]

# =========================
# AGGREGATE
# =========================
summary = df.groupby("task")[metrics].agg(["mean", "std"])
summary.columns = ["_".join(c) for c in summary.columns]
summary = summary.reset_index()

# =========================
# PLOT EACH TASK SEPARATELY
# =========================
x = np.arange(len(metrics))

for i, row in summary.iterrows():

    task = row["task"]

    means = [row[f"{m}_mean"] for m in metrics]
    stds  = [row[f"{m}_std"] for m in metrics]

    plt.figure(figsize=(6, 4))

    plt.bar(x, means, yerr=stds, capsize=4)

    plt.xticks(x, metrics, rotation=45)
    plt.ylim(0, 1)

    plt.title(f"Tox21 - {task}")
    plt.ylabel("Score")

    plt.tight_layout()

    save_path = f"{plot_dir}/{task}_metrics.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved: {save_path}")