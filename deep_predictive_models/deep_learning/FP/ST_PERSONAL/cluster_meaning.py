#!/usr/bin/env python
# coding: utf-8
"""
explain_consensus_clusters.py
───────────────────────────────────────────────────────────────────────────────
Finds clusters that appear in the top-K across ALL seeds for each task,
then explains only those consensus clusters using enrichment-based FG labeling.

Works with any explanation method CSV that has columns:
    seed, task, cluster, rank

Usage
-----
    python explain_consensus_clusters.py \
        --importance  analysis/cluster_permutation/cluster_permutation_importance.csv \
        --clusters    analysis/cluster_permutation/clusters.npy \
        --data        data/datasets/tox21/raw_data/tox21.csv \
        --top_k       50 \
        --top_n       5 \
        --output      analysis/consensus_cluster_meanings.csv

Output columns
--------------
    task            : Tox21 task name
    cluster         : cluster id
    n_bits          : number of Morgan bits in this cluster
    mean_rank       : mean rank across seeds (lower = more important)
    rank_std        : std of rank across seeds
    n_seeds         : number of seeds where this cluster appears in top_k
    top_fg          : top FGs ranked by enrichment
    top_fg_enrichment : enrichment scores
"""

import argparse
import os

import numpy as np
import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
MORGAN_BITS   = 2048
MORGAN_RADIUS = 2
GLOBAL_SEED   = 42

# ─────────────────────────────────────────────────────────────────────────────
# FUNCTIONAL GROUP LIBRARY
# ─────────────────────────────────────────────────────────────────────────────
FUNCTIONAL_GROUPS = {
    "Primary amine":        "[NH2][CX4]",
    "Aromatic amine":       "[NH2]c",
    "Secondary amine":      "[NH1]([CX4])[CX4]",
    "Tertiary amine":       "[NX3]([CX4])([CX4])[CX4]",
    "Amide":                "[NX3][CX3](=[OX1])",
    "Sulfonamide":          "[NX3][SX4](=[OX1])(=[OX1])",
    "Nitro group":          "[$([NX3](=O)=O),$([NX3+](=O)[O-])][!#8]",
    "Hydroxyl (aliphatic)": "[OX2H][CX4]",
    "Phenol":               "[OX2H]c",
    "Carboxylic acid":      "[CX3](=O)[OX2H1]",
    "Ester":                "[CX3](=O)[OX2][CX4]",
    "Ether":                "[OD2]([#6])[#6]",
    "Thiol":                "[SX2H]",
    "Thioether":            "[SX2]([#6])[#6]",
    "Sulfoxide":            "[SX3](=[OX1])([#6])[#6]",
    "Sulfone":              "[SX4](=[OX1])(=[OX1])([#6])[#6]",
    "Halogen (F)":          "[F]",
    "Halogen (Cl)":         "[Cl]",
    "Halogen (Br)":         "[Br]",
    "Halogen (I)":          "[I]",
    "Cyano":                "[CX2]#[NX1]",
    "Aldehyde":             "[CX3H1](=O)[#6]",
    "Ketone":               "[#6][CX3](=O)[#6]",
    "Benzene ring":         "c1ccccc1",
    "Pyridine":             "c1ccncc1",
    "Pyrimidine":           "c1ncncn1",
    "Piperidine":           "C1CCNCC1",
    "Morpholine":           "C1COCCN1",
    "Piperazine":           "C1CNCCN1",
    "Furan":                "c1ccoc1",
    "Thiophene":            "c1ccsc1",
    "Indole":               "c1ccc2[nH]ccc2c1",
}

_COMPILED_FG = {
    name: Chem.MolFromSmarts(smarts)
    for name, smarts in FUNCTIONAL_GROUPS.items()
    if Chem.MolFromSmarts(smarts) is not None
}


def get_functional_groups(mol):
    return [name for name, patt in _COMPILED_FG.items()
            if mol.HasSubstructMatch(patt)]


# ─────────────────────────────────────────────────────────────────────────────
# FINGERPRINTS
# ─────────────────────────────────────────────────────────────────────────────
def compute_fp(df, smiles_col="smiles"):
    fps = []
    for smi in df[smiles_col]:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            fps.append(np.zeros(MORGAN_BITS, dtype=np.float32))
            continue
        fp  = AllChem.GetMorganFingerprintAsBitVect(
            mol, MORGAN_RADIUS, nBits=MORGAN_BITS
        )
        arr = np.zeros((MORGAN_BITS,), dtype=np.float32)
        DataStructs.ConvertToNumpyArray(fp, arr)
        fps.append(arr)
    return np.array(fps, dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# BACKGROUND FG FREQUENCY
# ─────────────────────────────────────────────────────────────────────────────
def background_fg_freq(df, smiles_col="smiles"):
    counter = Counter()
    total   = 0
    for smi in tqdm(df[smiles_col], desc="Background FG frequencies"):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        for fg in get_functional_groups(mol):
            counter[fg] += 1
        total += 1
    return {fg: count / total for fg, count in counter.items()}, total


# ─────────────────────────────────────────────────────────────────────────────
# CONSENSUS CLUSTERS
# ─────────────────────────────────────────────────────────────────────────────
def find_consensus_clusters(importance_df, top_k=50):
    """
    For each task, find clusters that appear in the top_k for every seed.

    Returns a dict:
        {task: DataFrame with columns [cluster, mean_rank, rank_std, n_seeds]}
    """
    seeds   = importance_df["seed"].unique()
    n_seeds = len(seeds)
    consensus = {}

    for task, task_df in importance_df.groupby("task"):
        # for each seed, get the set of top_k cluster ids
        seed_top_sets = []
        rank_records  = []

        for seed, seed_df in task_df.groupby("seed"):
            top = seed_df.nsmallest(top_k, "rank")
            seed_top_sets.append(set(top["cluster"].tolist()))
            for _, row in top.iterrows():
                rank_records.append({
                    "cluster": row["cluster"],
                    "rank":    row["rank"],
                })

        # intersection across all seeds
        common = set.intersection(*seed_top_sets)

        if not common:
            print(f"  [{task}] No clusters in top {top_k} across all seeds.")
            consensus[task] = pd.DataFrame(
                columns=["cluster", "mean_rank", "rank_std", "n_seeds"]
            )
            continue

        # compute mean/std rank over seeds for the consensus clusters
        rank_df = pd.DataFrame(rank_records)
        stats   = (
            rank_df[rank_df["cluster"].isin(common)]
            .groupby("cluster")["rank"]
            .agg(mean_rank="mean", rank_std="std", n_seeds="count")
            .reset_index()
        )
        stats = stats.sort_values("mean_rank").reset_index(drop=True)
        consensus[task] = stats
        print(f"  [{task}] {len(common)} consensus clusters in top {top_k} "
              f"across all {n_seeds} seeds.")

    return consensus


# ─────────────────────────────────────────────────────────────────────────────
# EXPLAIN CONSENSUS CLUSTERS
# ─────────────────────────────────────────────────────────────────────────────
def explain_consensus(consensus, clusters, df, X_fp_binary, bg_freq,
                      smiles_col="smiles", top_n=5):
    """
    For each task's consensus clusters, compute enrichment-based FG labels.
    Only clusters that appear in the consensus are processed — no redundant work.
    """
    # collect all unique cluster ids across all tasks to avoid
    # recomputing the same cluster's FG profile more than once
    all_consensus_ids = set()
    for stats_df in consensus.values():
        all_consensus_ids.update(stats_df["cluster"].tolist())

    print(f"Explaining {len(all_consensus_ids)} unique consensus clusters...")

    cluster_fg: dict = {}
    for cid in tqdm(sorted(all_consensus_ids), desc="FG enrichment"):
        bit_indices = clusters[cid]
        activation  = X_fp_binary[:, bit_indices].mean(axis=1)
        active_idx  = np.where(activation > activation.mean())[0]
        n_active    = len(active_idx)

        fg_counter = Counter()
        for i in active_idx:
            mol = Chem.MolFromSmiles(df[smiles_col].iloc[i])
            if mol is None:
                continue
            for fg in get_functional_groups(mol):
                fg_counter[fg] += 1

        enrichment = {
            fg: (count / n_active) / bg_freq[fg]
            for fg, count in fg_counter.items()
            if bg_freq.get(fg, 0) > 0
        }
        top = sorted(enrichment, key=enrichment.get, reverse=True)[:top_n]
        cluster_fg[cid] = {
            "n_bits":            len(bit_indices),
            "n_active_mols":     n_active,
            "top_fg":            "; ".join(top),
            "top_fg_enrichment": "; ".join(f"{enrichment[fg]:.2f}" for fg in top),
        }

    # assemble final DataFrame: one row per (task, cluster)
    rows = []
    for task, stats_df in consensus.items():
        for _, row in stats_df.iterrows():
            cid  = int(row["cluster"])
            info = cluster_fg.get(cid, {})
            rows.append({
                "task":              task,
                "cluster":           cid,
                "n_bits":            info.get("n_bits", 0),
                "mean_rank":         row["mean_rank"],
                "rank_std":          row["rank_std"],
                "n_seeds":           int(row["n_seeds"]),
                "n_active_mols":     info.get("n_active_mols", 0),
                "top_fg":            info.get("top_fg", ""),
                "top_fg_enrichment": info.get("top_fg_enrichment", ""),
            })

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main(args):
    # ── Load importance CSV ──────────────────────────────────────────────────
    print(f"Loading importance scores from {args.importance} ...")
    imp_df = pd.read_csv(args.importance)
    required = {"seed", "task", "cluster", "rank"}
    assert required.issubset(imp_df.columns), (
        f"importance CSV must have columns: {required}. "
        f"Found: {set(imp_df.columns)}"
    )

    # ── Load clusters ────────────────────────────────────────────────────────
    print(f"Loading clusters from {args.clusters} ...")
    clusters = np.load(args.clusters, allow_pickle=True).item()
    print(f"  {len(clusters)} clusters loaded.")

    # ── Load training data ───────────────────────────────────────────────────
    print(f"Loading data from {args.data} ...")
    data     = pd.read_csv(args.data)
    train_df, _ = train_test_split(data, test_size=0.2, random_state=GLOBAL_SEED)
    train_df = train_df.reset_index(drop=True)
    print(f"  {len(train_df)} training molecules.")

    # ── Fingerprints ─────────────────────────────────────────────────────────
    print("Computing fingerprints ...")
    X_fp = compute_fp(train_df, smiles_col=args.smiles_col)
    assert X_fp.shape == (len(train_df), MORGAN_BITS)

    # ── Background frequencies ───────────────────────────────────────────────
    bg_freq, n_valid = background_fg_freq(train_df, smiles_col=args.smiles_col)
    print(f"  Background computed over {n_valid} valid molecules.")

    # ── Find consensus clusters ──────────────────────────────────────────────
    print(f"\nFinding clusters in top {args.top_k} across ALL seeds...")
    consensus = find_consensus_clusters(imp_df, top_k=args.top_k)

    # ── Explain ──────────────────────────────────────────────────────────────
    result_df = explain_consensus(
        consensus, clusters, train_df, X_fp, bg_freq,
        smiles_col=args.smiles_col,
        top_n=args.top_n,
    )

    # ── Save ─────────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    result_df.to_csv(args.output, index=False)
    print(f"\nSaved → {args.output}")

    # print summary
    summary = (
        result_df.groupby("task")["cluster"]
        .count()
        .rename("n_consensus_clusters")
        .reset_index()
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Explain clusters that are stable across all seeds (top-K intersection)."
    )
    parser.add_argument(
        "--importance",  required=True,
        help="CSV with columns: seed, task, cluster, rank. "
             "E.g. cluster_permutation_importance.csv or neuralsens_cluster_importance.csv"
    )
    parser.add_argument(
        "--clusters",    default="analysis/cluster_permutation/clusters.npy",
    )
    parser.add_argument(
        "--data",        default="data/datasets/tox21/raw_data/tox21.csv",
    )
    parser.add_argument(
        "--smiles_col",  default="smiles",
    )
    parser.add_argument(
        "--top_k",       type=int, default=50,
        help="Rank cutoff — cluster must be in top_k for every seed (default 50)."
    )
    parser.add_argument(
        "--top_n",       type=int, default=5,
        help="Number of FG labels per cluster (default 5)."
    )
    parser.add_argument(
        "--output",      default="analysis/consensus_cluster_meanings.csv",
    )
    args = parser.parse_args()
    main(args)  