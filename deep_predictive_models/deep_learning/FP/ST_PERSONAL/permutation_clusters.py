#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
from sklearn.cluster import AgglomerativeClustering
from sklearn.model_selection import train_test_split
from collections import Counter
from itertools import combinations
from tqdm import tqdm
from scipy.stats import spearmanr

# =========================
# CONFIG
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True

morgan_bits   = 2048
morgan_radius = 2

GLOBAL_SEED = 42
n_clusters  = 300
n_repeats   = 5
top_k       = 50

tox21_tasks = [
    'NR-AR', 'NR-Aromatase', 'NR-PPAR-gamma', 'SR-HSE',
    'NR-AR-LBD', 'NR-ER', 'SR-ARE', 'SR-MMP',
    'NR-AhR', 'NR-ER-LBD', 'SR-ATAD5', 'SR-p53'
]

seeds = ["seed_122", "seed_123", "seed_124", "seed_125", "seed_126"]

# =========================
# FUNCTIONAL GROUP LIBRARY
# Named SMARTS patterns used for human-readable cluster labeling.
# Matching is done directly on molecules (not via ECFP bit SMARTS),
# avoiding Morgan bit hash-collision noise.
# =========================
FUNCTIONAL_GROUPS = {
    "Primary amine":         "[NH2][CX4]",
    "Aromatic amine":        "[NH2]c",
    "Secondary amine":       "[NH1]([CX4])[CX4]",
    "Tertiary amine":        "[NX3]([CX4])([CX4])[CX4]",
    "Amide":                 "[NX3][CX3](=[OX1])",
    "Sulfonamide":           "[NX3][SX4](=[OX1])(=[OX1])",
    "Nitro group":           "[$([NX3](=O)=O),$([NX3+](=O)[O-])][!#8]",
    "Hydroxyl (aliphatic)":  "[OX2H][CX4]",
    "Phenol":                "[OX2H]c",
    "Carboxylic acid":       "[CX3](=O)[OX2H1]",
    "Ester":                 "[CX3](=O)[OX2][CX4]",
    "Ether":                 "[OD2]([#6])[#6]",
    "Thiol":                 "[SX2H]",
    "Thioether":             "[SX2]([#6])[#6]",
    "Sulfoxide":             "[SX3](=[OX1])([#6])[#6]",
    "Sulfone":               "[SX4](=[OX1])(=[OX1])([#6])[#6]",
    "Halogen (F)":           "[F]",
    "Halogen (Cl)":          "[Cl]",
    "Halogen (Br)":          "[Br]",
    "Halogen (I)":           "[I]",
    "Cyano":                 "[CX2]#[NX1]",
    "Aldehyde":              "[CX3H1](=O)[#6]",
    "Ketone":                "[#6][CX3](=O)[#6]",
    "Benzene ring":          "c1ccccc1",
    "Pyridine":              "c1ccncc1",
    "Pyrimidine":            "c1ncncn1",
    "Piperidine":            "C1CCNCC1",
    "Morpholine":            "C1COCCN1",
    "Piperazine":            "C1CNCCN1",
    "Furan":                 "c1ccoc1",
    "Thiophene":             "c1ccsc1",
    "Indole":                "c1ccc2[nH]ccc2c1",
}

_COMPILED_FG = {
    name: Chem.MolFromSmarts(smarts)
    for name, smarts in FUNCTIONAL_GROUPS.items()
    if Chem.MolFromSmarts(smarts) is not None
}


def get_functional_groups(mol):
    """Return list of FG names present in mol."""
    return [name for name, patt in _COMPILED_FG.items()
            if mol.HasSubstructMatch(patt)]


# =========================
# DATA
# =========================
data = pd.read_csv("data/datasets/tox21/raw_data/tox21.csv")

train_df, temp_df = train_test_split(data, test_size=0.2, random_state=GLOBAL_SEED)
valid_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=GLOBAL_SEED)

# Reset indices so iloc positions align with fingerprint matrix rows
train_df = train_df.reset_index(drop=True)
valid_df = valid_df.reset_index(drop=True)
test_df  = test_df.reset_index(drop=True)

# =========================
# FINGERPRINTS
# =========================
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

# =========================
# CLUSTER EXPLANATIONS
# Molecule-direct approach: for each cluster, find training molecules
# that meaningfully activate it (≥threshold fraction of cluster bits set),
# then count named functional groups across those molecules.
#
# This avoids the Morgan bit hash-collision problem of the SMARTS-pooling
# approach (where different substructures can share the same bit index).
# =========================
def explain_clusters(clusters, df, X_fp_binary, smiles_col="smiles",
                     threshold=0.3, top_n=5):
    """
    Parameters
    ----------
    clusters     : dict {cluster_id: array of bit indices}
    df           : training DataFrame, reset_index(drop=True)
    X_fp_binary  : (n_mols, n_bits) binary fingerprint matrix, dtype int/float
                   rows must align with df
    threshold    : min fraction of cluster bits active to include a molecule
    top_n        : number of top FG labels to return per cluster

    Returns
    -------
    dict {cluster_id: [fg_name, ...]}
    """
    assert len(df) == X_fp_binary.shape[0], (
        f"df rows ({len(df)}) != fingerprint rows ({X_fp_binary.shape[0]}). "
        "Ensure the same split is used for both."
    )

    cluster_explanations = {}

    for cid, bit_indices in tqdm(clusters.items(), desc="Explaining clusters"):
        activation = X_fp_binary[:, bit_indices].mean(axis=1)
        active_idx = np.where(activation >= threshold)[0]

        fg_counter = Counter()
        for i in active_idx:
            mol = Chem.MolFromSmiles(df[smiles_col].iloc[i])
            if mol is None:
                continue
            for fg_name in get_functional_groups(mol):
                fg_counter[fg_name] += 1

        top = fg_counter.most_common(top_n)
        cluster_explanations[cid] = [name for name, _ in top]

    return cluster_explanations

# =========================
# MODEL
# =========================
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


def predict(model, X):
    with torch.no_grad():
        return torch.sigmoid(model(X))

# =========================
# CLUSTERING
# =========================
def build_clusters(X_np, n_clusters=300):
    corr = np.corrcoef(X_np.T)
    dist = 1 - np.abs(corr)
    clustering = AgglomerativeClustering(
        n_clusters=n_clusters,
        metric="precomputed",
        linkage="average"
    )
    labels = clustering.fit_predict(dist)
    return {i: np.where(labels == i)[0] for i in range(n_clusters)}

# =========================
# PERMUTATION IMPORTANCE
# =========================
def cluster_permutation_importance(model, X, clusters_t, n_repeats=5):
    """
    Output-based cluster permutation importance averaged over n_repeats.
    Importance = mean |pred_base - pred_permuted|, normalised by sqrt(cluster_size).

    GPU efficiency:
    - clusters_t: pre-converted dict of {cid: LongTensor} — no per-call allocation
    - in-place column restore instead of full X.clone() per repeat
    - float accumulation instead of stacking GPU tensors
    """
    base_pred = predict(model, X)
    n_samples = X.shape[0]
    importances = []

    for cid, feat_idx_t in clusters_t.items():
        orig       = X[:, feat_idx_t].clone()
        impact_sum = 0.0

        for _ in range(n_repeats):
            perm             = torch.randperm(n_samples, device=device)
            X[:, feat_idx_t] = X[perm][:, feat_idx_t]
            new_pred         = predict(model, X)
            impact_sum      += torch.mean(torch.abs(base_pred - new_pred)).item()
            X[:, feat_idx_t] = orig

        importance = (impact_sum / n_repeats) / np.sqrt(len(feat_idx_t))
        importances.append(importance)

    return np.array(importances)

# =========================
# STABILITY
# =========================
def compute_stability(seed_results, top_k=10):
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

# =========================
# MAIN
# =========================
os.makedirs("analysis/cluster_permutation", exist_ok=True)

# ── Clusters (train only, no leakage) ────────────────────────────────────────
print("Preparing train fingerprints for clustering...")
X_train_np = compute_fp(train_df) - 0.5

cluster_path = "analysis/cluster_permutation/clusters.npy"
if os.path.exists(cluster_path):
    print("Loading existing clusters...")
    clusters = np.load(cluster_path, allow_pickle=True).item()
else:
    print("Building clusters...")
    clusters = build_clusters(X_train_np, n_clusters=n_clusters)
    np.save(cluster_path, clusters, allow_pickle=True)

# ── Cluster explanations (molecule-direct, train only) ───────────────────────
# X_fp_binary: restore binary 0/1 from the centred matrix.
# Only valid-SMILES rows contribute; zero-FP rows (failed parses) have
# activation=0 for every cluster and are naturally excluded at threshold>0.
print("Explaining clusters...")
X_train_binary     = (X_train_np + 0.5).round().astype(np.float32)
cluster_explanations = explain_clusters(
    clusters, train_df, X_train_binary, threshold=0.3, top_n=5
)

# ── Pre-convert cluster indices to GPU tensors (once, reused every seed/task) ─
clusters_t = {
    cid: torch.tensor(idx, dtype=torch.long, device=device)
    for cid, idx in clusters.items()
}

# ── Fixed evaluation tensor (outside seed loop) ───────────────────────────────
print("Computing test fingerprints...")
X_split_np = compute_fp(test_df) - 0.5   # swap to valid_df for dev runs
X_split    = torch.tensor(X_split_np, dtype=torch.float32, device=device)

# ── Seed loop ─────────────────────────────────────────────────────────────────
out_path    = "analysis/cluster_permutation/cluster_permutation_importance.csv"
first_write = True
seed_results: dict = {}

for seed in tqdm(seeds, desc="Seeds"):
    seed_results[seed] = {}

    for task in tqdm(tox21_tasks, desc=f"{seed}", leave=False):
        model_path = f"models_baseline/{seed}/{task}.pt"
        if not os.path.exists(model_path):
            continue

        model = load_model(model_path)

        # pass a contiguous copy so in-place ops don't affect X_split
        X_work      = X_split.clone()
        importances = cluster_permutation_importance(model, X_work, clusters_t, n_repeats)

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
                # human-readable named FGs, e.g. "Aromatic amine; Sulfonamide"
                "motifs":     "; ".join(cluster_explanations.get(cid, [])),
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
stability_df.to_csv("analysis/cluster_permutation/stability_metrics.csv", index=False)

print("DONE")