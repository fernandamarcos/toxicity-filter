import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from sklearn.model_selection import train_test_split

morgan_bits   = 2048
morgan_radius = 2
GLOBAL_SEED   = 42

tox21_tasks = [
    'NR-AR', 'NR-Aromatase', 'NR-PPAR-gamma', 'SR-HSE',
    'NR-AR-LBD', 'NR-ER', 'SR-ARE', 'SR-MMP',
    'NR-AhR', 'NR-ER-LBD', 'SR-ATAD5', 'SR-p53'
]

data = pd.read_csv("data/datasets/tox21/raw_data/tox21.csv")
_, temp_df = train_test_split(data, test_size=0.2, random_state=GLOBAL_SEED)
_, test_df = train_test_split(temp_df, test_size=0.5, random_state=GLOBAL_SEED)
test_df = test_df.reset_index(drop=True)

def compute_fp(df):
    fps = []
    for smi in df["smiles"]:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            fps.append(np.zeros(morgan_bits, dtype=np.float32))
            continue
        fp  = AllChem.GetMorganFingerprintAsBitVect(mol, morgan_radius, nBits=morgan_bits)
        arr = np.zeros((morgan_bits,), dtype=np.float32)
        DataStructs.ConvertToNumpyArray(fp, arr)
        fps.append(arr)
    return np.array(fps, dtype=np.float32)

X_test_binary = compute_fp(test_df)
Y_test        = test_df[tox21_tasks].fillna(-1).values.astype(np.float32)

clusters = np.load(
    "analysis/cluster_permutation/clusters.npy",
    allow_pickle=True
).item()

imp_df  = pd.read_csv("analysis/cluster_permutation/cluster_permutation_importance.csv")
avg_imp = (
    imp_df
    .groupby(["task", "cluster"])["importance"]
    .mean()
    .reset_index()
    .sort_values(["task", "importance"], ascending=[True, False])
)

rows = []

for _, imp_row in avg_imp.iterrows():

    task = imp_row["task"]
    cid  = int(imp_row["cluster"])

    bit_indices = clusters[cid]
    activation  = X_test_binary[:, bit_indices].mean(axis=1)
    active_mask = activation > activation.mean()   # ← cambio clave

    n_active = active_mask.sum()
    if n_active == 0:
        continue

    t_idx  = tox21_tasks.index(task)
    labels = Y_test[active_mask, t_idx]

    valid_mask   = labels >= 0
    n_valid      = valid_mask.sum()
    if n_valid == 0:
        continue

    valid_labels = labels[valid_mask]
    n_toxic      = (valid_labels == 1).sum()
    n_nontoxic   = (valid_labels == 0).sum()

    rows.append({
        "cluster":      cid,
        "task":         task,
        "importance":   round(float(imp_row["importance"]), 6),
        "n_active":     int(n_active),
        "n_valid":      int(n_valid),
        "n_toxic":      int(n_toxic),
        "n_nontoxic":   int(n_nontoxic),
        "pct_toxic":    round(n_toxic    / n_valid * 100, 2),
        "pct_nontoxic": round(n_nontoxic / n_valid * 100, 2),
    })

dist_df = pd.DataFrame(rows)
dist_df["rank"] = (
    dist_df
    .groupby("task")["importance"]
    .rank(ascending=False, method="first")
    .astype(int)
)
dist_df = dist_df.sort_values(["task", "rank"])

dist_df.to_csv(
    "analysis/cluster_shap/all_clusters_toxicity_distribution.csv",
    index=False
)

print(dist_df.head(30).to_string(index=False))

import os
import matplotlib.pyplot as plt

output_dir = "analysis/cluster_permutation/toxicity_plots"
os.makedirs(output_dir, exist_ok=True)

for task in tox21_tasks:

    task_df = (
        dist_df[dist_df["task"] == task]
        .sort_values("rank")
        .reset_index(drop=True)
    )

    if len(task_df) == 0:
        continue

    plt.figure(figsize=(12, 5))

    plt.bar(
        range(len(task_df)),
        task_df["pct_toxic"]
    )

    plt.xticks(
        range(len(task_df)),
        task_df["cluster"].astype(str),
        rotation=90
    )

    plt.xlabel("Cluster (ordenado por importancia)")
    plt.ylabel("% moléculas tóxicas")
    plt.title(f"{task} - Toxicidad por cluster")

    plt.tight_layout()

    plt.savefig(
        os.path.join(output_dir, f"{task}_pct_toxic_by_cluster.png"),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

print(f"Plots guardados en: {output_dir}")

_, temp_df = train_test_split(data, test_size=0.2, random_state=GLOBAL_SEED)
_, test_df = train_test_split(temp_df, test_size=0.5, random_state=GLOBAL_SEED)


_, temp_df = train_test_split(
    data,
    test_size=0.2,
    random_state=GLOBAL_SEED
)

valid_df, test_df = train_test_split(
    temp_df,
    test_size=0.5,
    random_state=GLOBAL_SEED
)

validation_stats = []

for task in tox21_tasks:

    labels = valid_df[task].fillna(-1).values

    valid_mask = labels >= 0
    valid_labels = labels[valid_mask]

    n_valid = len(valid_labels)
    n_positive = (valid_labels == 1).sum()
    n_negative = (valid_labels == 0).sum()

    validation_stats.append({
        "task": task,
        "n_valid": int(n_valid),
        "n_positive": int(n_positive),
        "n_negative": int(n_negative),
        "pct_positive": round(100 * n_positive / n_valid, 2)
            if n_valid > 0 else np.nan
    })

validation_stats_df = pd.DataFrame(validation_stats)

print(validation_stats_df.to_string(index=False))