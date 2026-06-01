#!/usr/bin/env python
# coding: utf-8

"""
NeuralSens analysis at cluster level for Tox21 single-task DNN models.

For each cluster of Morgan bits, importance is the mean absolute
gradient contribution summed across all bits in the cluster,
then normalised by sqrt(cluster_size) for fair comparison across
clusters of different sizes.

Output mirrors the cluster_permutation_importance.csv schema so
both explanation methods can be compared directly.
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

from sklearn.model_selection import train_test_split
from scipy.stats import spearmanr
from itertools import combinations
from tqdm import tqdm

# =========================================================
# CONFIG
# =========================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

morgan_bits   = 2048
morgan_radius = 2
GLOBAL_SEED   = 42
top_k         = 50

tox21_tasks = [
    'NR-AR', 'NR-Aromatase', 'NR-PPAR-gamma', 'SR-HSE',
    'NR-AR-LBD', 'NR-ER', 'SR-ARE', 'SR-MMP',
    'NR-AhR', 'NR-ER-LBD', 'SR-ATAD5', 'SR-p53'
]

seeds = ["seed_122", "seed_123", "seed_124", "seed_125", "seed_126"]

# =========================================================
# DATA
# =========================================================
data = pd.read_csv("data/datasets/tox21/raw_data/tox21.csv")

train_df, temp_df = train_test_split(data, test_size=0.2, random_state=GLOBAL_SEED)
valid_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=GLOBAL_SEED)

train_df = train_df.reset_index(drop=True)
valid_df  = valid_df.reset_index(drop=True)
test_df   = test_df.reset_index(drop=True)

# =========================================================
# FINGERPRINTS
# =========================================================
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

# =========================================================
# MODEL
# =========================================================
class SingleTaskDNN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc1     = nn.Linear(input_dim, 1024)
        self.ln1     = nn.BatchNorm1d(1024)
        self.fc2     = nn.Linear(1024, 512)
        self.ln2     = nn.BatchNorm1d(512)
        self.fc3     = nn.Linear(512, 256)
        self.dropout = nn.Dropout(0.2)
        self.out     = nn.Linear(256, 1)
        self.leaky   = nn.LeakyReLU(0.05)

    def forward(self, x):
        x = self.leaky(self.ln1(self.fc1(x)))
        x = self.dropout(x)
        x = self.leaky(self.ln2(self.fc2(x)))
        x = self.dropout(x)
        x = self.leaky(self.fc3(x))
        return self.out(x)

def load_model(path):
    model = SingleTaskDNN(morgan_bits).to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model

# =========================================================
# BIT-LEVEL GRADIENTS  (one backward pass over all samples)
# =========================================================
def compute_bit_gradients(model, X):
    """
    Returns mean |grad * input| per bit — shape (morgan_bits,).
    Single backward pass: efficient and numerically equivalent to
    per-sample accumulation.
    """
    model.eval()
    X_req = X.clone().detach().requires_grad_(True)
    probs = torch.sigmoid(model(X_req))   # (N, 1)
    probs.sum().backward()
    grads = X_req.grad.detach()           # (N, D)
    # gradient × input captures signed contribution; abs + mean = sensitivity
    bit_importance = (grads * X_req.detach()).abs().mean(dim=0).cpu().numpy()
    return bit_importance                 # (morgan_bits,)

# =========================================================
# CLUSTER AGGREGATION
# =========================================================
def aggregate_to_clusters(bit_importance, clusters):
    """
    Sum bit-level sensitivities within each cluster, then normalise
    by sqrt(cluster_size) — same normalisation as permutation importance
    so both methods are directly comparable.

    Returns np.array shape (n_clusters,) indexed by cluster id.
    """
    n_clusters   = len(clusters)
    cluster_imp  = np.zeros(n_clusters, dtype=np.float32)

    for cid, bit_indices in clusters.items():
        cluster_imp[cid] = (
            bit_importance[bit_indices].sum() / np.sqrt(len(bit_indices))
        )

    return cluster_imp

# =========================================================
# STABILITY
# =========================================================
def compute_stability(seed_results, top_k=50):
    seed_keys = list(seed_results.keys())
    rows = []
    for task in seed_results[seed_keys[0]]:
        jaccards, spearmans = [], []
        for s1, s2 in combinations(seed_keys, 2):
            if task not in seed_results[s1] or task not in seed_results[s2]:
                continue
            imp1 = seed_results[s1][task]
            imp2 = seed_results[s2][task]
            top1 = set(np.argsort(imp1)[::-1][:top_k])
            top2 = set(np.argsort(imp2)[::-1][:top_k])
            jaccards.append(len(top1 & top2) / len(top1 | top2))
            spear, _ = spearmanr(imp1, imp2)
            spearmans.append(spear)
        if jaccards:
            rows.append({
                "task":              task,
                "jaccard_topk_mean": np.mean(jaccards),
                "jaccard_topk_std":  np.std(jaccards),
                "spearman_mean":     np.mean(spearmans),
                "spearman_std":      np.std(spearmans),
                "n_seed_pairs":      len(jaccards),
            })
    return pd.DataFrame(rows)

# =========================================================
# MAIN
# =========================================================
os.makedirs("analysis/neuralsens", exist_ok=True)

# ── Load clusters ─────────────────────────────────────────
print("Loading clusters...")
clusters = np.load(
    "analysis/cluster_permutation/clusters.npy", allow_pickle=True
).item()
n_clusters = len(clusters)
print(f"  {n_clusters} clusters loaded.")

# ── Test fingerprints ─────────────────────────────────────
print("Computing test fingerprints...")
X_np = compute_fp(test_df) - 0.5
X    = torch.tensor(X_np, dtype=torch.float32, device=device)

# ── Seed / task loop ──────────────────────────────────────
out_path    = "analysis/neuralsens/neuralsens_cluster_importance.csv"
first_write = True
seed_results: dict = {}

for seed in tqdm(seeds, desc="Seeds"):
    seed_results[seed] = {}

    for task in tqdm(tox21_tasks, desc=seed, leave=False):
        model_path = f"models_baseline/{seed}/{task}.pt"
        if not os.path.exists(model_path):
            continue

        model = load_model(model_path)

        # 1. bit-level sensitivities
        bit_imp = compute_bit_gradients(model, X)

        # 2. aggregate to cluster level
        cluster_imp = aggregate_to_clusters(bit_imp, clusters)

        seed_results[seed][task] = cluster_imp

        sorted_clusters = np.argsort(cluster_imp)[::-1]

        rows = [
            {
                "seed":       seed,
                "task":       task,
                "cluster":    int(cid),
                "rank":       rank,
                "importance": float(cluster_imp[cid]),
                "n_features": len(clusters[cid]),
            }
            for rank, cid in enumerate(sorted_clusters)
        ]

        df_chunk = pd.DataFrame(rows)
        if first_write:
            df_chunk.to_csv(out_path, index=False)
            first_write = False
        else:
            df_chunk.to_csv(out_path, mode="a", header=False, index=False)

# ── Seed-averaged importance per task ─────────────────────
print("Computing seed-averaged importance...")
mean_rows = []
for task in tox21_tasks:
    task_imps = [seed_results[s][task] for s in seeds
                 if task in seed_results.get(s, {})]
    if not task_imps:
        continue
    mean_imp      = np.stack(task_imps).mean(axis=0)
    sorted_clusters = np.argsort(mean_imp)[::-1]
    for rank, cid in enumerate(sorted_clusters):
        mean_rows.append({
            "task":       task,
            "cluster":    int(cid),
            "rank":       rank,
            "importance": float(mean_imp[cid]),
            "n_features": len(clusters[cid]),
        })

pd.DataFrame(mean_rows).to_csv(
    "analysis/neuralsens/neuralsens_cluster_importance_mean.csv", index=False
)

# ── Stability ─────────────────────────────────────────────
print("Computing stability metrics...")
stability_df = compute_stability(seed_results, top_k=top_k)
stability_df.to_csv(
    "analysis/neuralsens/neuralsens_cluster_stability.csv", index=False
)
print(stability_df.to_string(index=False))

print("DONE")