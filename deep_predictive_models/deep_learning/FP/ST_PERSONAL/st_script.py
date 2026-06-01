#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
import pandas as pd
import random
from tqdm import tqdm

import torch
import torch.nn as nn

from rdkit import DataStructs
from rdkit.Chem import AllChem

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.warning')


from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix
)

import matplotlib.pyplot as plt

# =========================
# DEVICE
# =========================
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True

# =========================
# SETTINGS
# =========================
morgan_bits = 2048
morgan_radius = 2

epochs = 50
batch_size = 256
lr = 1e-3

tox21_tasks = [
    'NR-AR','NR-Aromatase','NR-PPAR-gamma','SRssss-HSE',
    'NR-AR-LBD','NR-ER','SR-ARE','SR-MMP',
    'NR-AhR','NR-ER-LBD','SR-ATAD5','SR-p53'
]

NR_TASKS = tox21_tasks[:5]
SR_TASKS = tox21_tasks[5:]

data = pd.read_csv('data/datasets/tox21/raw_data/tox21.csv')

# =========================
# FINGERPRINTS
# =========================
def compute_fp(df):
    mols = [AllChem.MolFromSmiles(x) for x in df['smiles']]
    fps = []

    for mol in mols:
        if mol is None:
            fps.append(np.zeros(morgan_bits, dtype=np.float32))
            continue

        fp = AllChem.GetMorganFingerprintAsBitVect(
            mol, morgan_radius, nBits=morgan_bits
        )

        arr = np.zeros((morgan_bits,), dtype=np.float32)
        DataStructs.ConvertToNumpyArray(fp, arr)
        fps.append(arr)

    return np.array(fps, dtype=np.float32)

# =========================
# MODEL
# =========================
class SingleTaskDNN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        self.fc1 = nn.Linear(input_dim, 1024)
        self.ln1 = nn.BatchNorm1d(1024)

        self.fc2 = nn.Linear(1024, 512)
        self.ln2 = nn.BatchNorm1d(512)

        self.fc3 = nn.Linear(512, 256)

        self.dropout = nn.Dropout(0.2)
        self.out = nn.Linear(256, 1)
        self.act = nn.LeakyReLU(0.05)

    def forward(self, x):
        x = self.act(self.ln1(self.fc1(x)))
        x = self.dropout(x)

        x = self.act(self.ln2(self.fc2(x)))
        x = self.dropout(x)

        x = self.act(self.fc3(x))
        return self.out(x).squeeze(1)

# =========================
# METRICS
# =========================
def compute_metrics(y_true, y_prob):

    mask = y_true >= 0
    y_true = y_true[mask]
    y_prob = y_prob[mask]

    if len(np.unique(y_true)) < 2:
        return None

    y_pred = (y_prob >= 0.2).astype(int)

    acc = accuracy_score(y_true, y_pred)
    bacc = balanced_accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    fpr = fp / (fp + tn + 1e-6)
    fnr = fn / (fn + tp + 1e-6)
    tpr = tp / (tp + fn + 1e-6)
    tnr = tn / (tn + fp + 1e-6)

    return {
        "acc": acc,
        "bacc": bacc,
        "auc": auc,
        "fpr": fpr,
        "fnr": fnr,
        "tpr": tpr,
        "tnr": tnr
    }

# =========================
# TRAIN
# =========================
def train_model(Xtr, ytr, Xval, yval, task_idx):

    model = SingleTaskDNN(Xtr.shape[1]).to(device)

    # ---- dynamic class weight ----
    y_task = ytr[:, task_idx]
    pos = np.sum(y_task == 1)
    neg = np.sum(y_task == 0)
    pos_weight = neg / (pos + 1e-6)

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight], device=device)
    )

    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    Xtr = torch.tensor(Xtr, dtype=torch.float32, device=device)
    Xval = torch.tensor(Xval, dtype=torch.float32, device=device)

    ytr = torch.tensor(ytr[:, task_idx], dtype=torch.float32, device=device)
    yval_np = yval[:, task_idx]

    best_auc = 0
    best_state = None

    for _ in range(epochs):

        model.train()
        perm = torch.randperm(Xtr.size(0), device=device)

        for i in range(0, len(perm), batch_size):
            idx = perm[i:i+batch_size]

            x = Xtr[idx]
            y = ytr[idx]

            mask = y >= 0
            if mask.sum() == 0:
                continue

            logits = model(x)
            loss = criterion(logits[mask], y[mask])

            opt.zero_grad()
            loss.backward()
            opt.step()

        # ---- validation ----
        model.eval()
        with torch.no_grad():
            probs = torch.sigmoid(model(Xval)).cpu().numpy()

        mask = yval_np >= 0
        auc_val = roc_auc_score(yval_np[mask], probs[mask])

        if auc_val > best_auc:
            best_auc = auc_val
            best_state = model.state_dict()

    model.load_state_dict(best_state)
    return model

# =========================
# MAIN
# =========================
seeds = [122, 123, 124, 125, 126, 127, 128, 129, 130]

os.makedirs("models_baseline", exist_ok=True)

NR_results = []
SR_results = []

for seed in tqdm(seeds, desc="Seeds"):

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    seed_dir = f"models_baseline/seed_{seed}"
    os.makedirs(seed_dir, exist_ok=True)

    train, temp = train_test_split(data, test_size=0.2, random_state=seed)
    valid, test = train_test_split(temp, test_size=0.5, random_state=seed)

    X_train = compute_fp(train) - 0.5
    X_valid = compute_fp(valid) - 0.5
    X_test  = compute_fp(test)  - 0.5

    Y_train = train[tox21_tasks].fillna(-1).values.astype(np.float32)
    Y_valid = valid[tox21_tasks].fillna(-1).values.astype(np.float32)
    Y_test  = test[tox21_tasks].fillna(-1).values.astype(np.float32)

    NR_seed, SR_seed = [], []

    for t_idx, task in enumerate(tqdm(tox21_tasks, leave=False)):

        model = train_model(
            X_train, Y_train,
            X_valid, Y_valid,
            t_idx
        )

        # ✅ SAVE MODEL
        model_path = f"{seed_dir}/{task}.pt"
        torch.save(model.state_dict(), model_path)

        # ✅ TEST EVAL
        model.eval()
        with torch.no_grad():
            probs = torch.sigmoid(
                model(torch.tensor(X_test, dtype=torch.float32, device=device))
            ).cpu().numpy()

        metrics = compute_metrics(Y_test[:, t_idx], probs)
        if metrics is None:
            continue

        row = {"seed": seed, "task": task, **metrics}

        if task in NR_TASKS:
            NR_seed.append(row)
        else:
            SR_seed.append(row)

    NR_results.extend(NR_seed)
    SR_results.extend(SR_seed)

# =========================
# SAVE RESULTS
# =========================
NR_df = pd.DataFrame(NR_results)
SR_df = pd.DataFrame(SR_results)

os.makedirs("results", exist_ok=True)

NR_df.to_csv("results/NR_metrics.csv", index=False)
SR_df.to_csv("results/SR_metrics.csv", index=False)

print("DONE")