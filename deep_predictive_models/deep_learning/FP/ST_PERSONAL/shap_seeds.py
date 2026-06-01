#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import shap

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from scipy.stats import spearmanr
from itertools import combinations
from collections import Counter

# =========================
# CONFIG  (mirror training)
# =========================
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

morgan_bits   = 2048
morgan_radius = 2

n_clusters   = 300
top_k        = 50
n_background = 100   # samples used as GradientExplainer background
n_explain    = 500   # samples to compute SHAP values on (None = full split)

tox21_tasks = [
    'NR-AR', 'NR-Aromatase', 'NR-PPAR-gamma', 'SR-HSE',
    'NR-AR-LBD', 'NR-ER', 'SR-ARE', 'SR-MMP',
    'NR-AhR', 'NR-ER-LBD', 'SR-ATAD5', 'SR-p53'
]

seeds = ["seed_122", "seed_123", "seed_124", "seed_125", "seed_126"]

GLOBAL_SEED = 42

# =========================
# DATA
# =========================
data = pd.read_csv("data/datasets/tox21/raw_data/tox21.csv")

train_df, temp_df = train_test_split(data, test_size=0.2, random_state=GLOBAL_SEED)
valid_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=GLOBAL_SEED)

# =========================
# FINGERPRINTS
# =========================
def compute_fp(df: pd.DataFrame) -> np.ndarray:
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

# =========================
# MODEL
# =========================
class SingleTaskDNN(nn.Module):
    def __init__(self, input_dim: int):
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


def load_model(path: str) -> SingleTaskDNN:
    model = SingleTaskDNN(morgan_bits).to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model

# =========================
# CLUSTER SHAP
# =========================
def cluster_shap_importance(
    model: nn.Module,
    X_background: torch.Tensor,   # (n_bg, D)
    X_explain: torch.Tensor,       # (n_exp, D)
    clusters: dict,
) -> np.ndarray:
    """
    Returns array of shape (n_clusters,) — mean |SHAP| per cluster,
    summed over features within each cluster then averaged over samples.

    GradientExplainer computes E[grad * (x - x')] by sampling background
    points independently for each explain sample, so no pairing is needed
    and arbitrary n_background / n_explain combinations are fine.
    """
    explainer = shap.GradientExplainer(model, X_background)

    # returns list of 1 ndarray for single-output model → shape (n_exp, D)
    shap_vals = explainer.shap_values(X_explain)
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[0]                  # (n_exp, D)

    abs_shap = np.abs(shap_vals)                  # (n_exp, D)

    importances = np.zeros(len(clusters), dtype=np.float32)
    for cid, feat_idx in clusters.items():
        cluster_abs      = abs_shap[:, feat_idx]  # (n_exp, k)
        importances[cid] = cluster_abs.sum(axis=1).mean()

    return importances

# =========================
# STABILITY
# =========================
def compute_stability(seed_results: dict, top_k: int = 10) -> pd.DataFrame:
    seed_keys = list(seed_results.keys())
    rows = []
    for task in seed_results[seed_keys[0]]:
        jaccards, spearmans = [], []
        for s1, s2 in combinations(seed_keys, 2):
            if task not in seed_results[s1] or task not in seed_results[s2]:
                continue
            imp1, imp2 = seed_results[s1][task], seed_results[s2][task]
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

# =========================
# MAIN
# =========================
os.makedirs("analysis/cluster_shap", exist_ok=True)

# ── Fixed clusters ───────────────────────────────────────────────────────────
cluster_path = "analysis/cluster_permutation/clusters.npy"
if os.path.exists(cluster_path):
    print("Loading existing clusters...")
    clusters = np.load(cluster_path, allow_pickle=True).item()
else:
    raise FileNotFoundError(
        f"Cluster file not found at {cluster_path}. "
        "Run the permutation-importance script first to build clusters."
    )

# ── Fixed fingerprints and sampling (outside seed loop) ──────────────────────
rng = np.random.default_rng(GLOBAL_SEED)

X_split_np = compute_fp(test_df) - 0.5   # centred, matches training
n_total    = len(X_split_np)

bg_idx = rng.choice(n_total, size=min(n_background, n_total), replace=False)
X_bg   = torch.tensor(X_split_np[bg_idx], dtype=torch.float32, device=device)

if n_explain is None or n_explain >= n_total:
    X_exp = torch.tensor(X_split_np, dtype=torch.float32, device=device)
else:
    exp_idx = rng.choice(n_total, size=n_explain, replace=False)
    X_exp   = torch.tensor(X_split_np[exp_idx], dtype=torch.float32, device=device)

# ── Seed loop ─────────────────────────────────────────────────────────────────
out_path    = "analysis/cluster_shap/cluster_shap_importance.csv"
first_write = True
seed_results: dict = {}

for seed in tqdm(seeds, desc="Seeds"):
    seed_results[seed] = {}

    for task in tqdm(tox21_tasks, desc=f"{seed}", leave=False):
        model_path = f"models_baseline/{seed}/{task}.pt"
        if not os.path.exists(model_path):
            continue

        model       = load_model(model_path)
        importances = cluster_shap_importance(model, X_bg, X_exp, clusters)

        seed_results[seed][task] = importances
        sorted_clusters          = np.argsort(importances)[::-1]

        rows = [
            {
                "seed":       seed,
                "task":       task,
                "cluster":    int(cid), 
                "rank":       rank,
                "importance": float(importances[cid]),
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

# ── Stability ─────────────────────────────────────────────────────────────────
print("Computing stability metrics...")
stability_df = compute_stability(seed_results, top_k=top_k)
stability_df.to_csv("analysis/cluster_shap/stability_metrics.csv", index=False)

print("DONE")