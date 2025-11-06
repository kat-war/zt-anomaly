#!/usr/bin/env python3
"""
Train & evaluate IsolationForest on engineered features.
- Trains on NORMAL rows only (is_anomaly==0)
- Evaluates on a holdout split
- Saves artifacts and a scored CSV
"""

from __future__ import annotations
import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split

FEATS_PATH = "data/processed/zt_features.csv"
ART_DIR = "artifacts/isoforest"
SCORED_PATH = "data/processed/isoforest_scored.csv"

def load_data(path=FEATS_PATH):
    df = pd.read_csv(path, low_memory=False)
    # Ensure bools are numeric for sklearn
    for c in df.columns:
        if df[c].dtype == bool:
            df[c] = df[c].astype(int)
    # Separate target
    y = df["is_anomaly"].astype(int).values
    X = df.drop(columns=["is_anomaly"])
    return X, y, df

def train_and_eval():
    os.makedirs(ART_DIR, exist_ok=True)

    X, y, df_full = load_data()
    # Split holdout (keep label only for metrics; model is unsupervised)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    # Train on NORMAL-only rows (unsupervised one-class flavor)
    normal_mask = y_train == 0
    X_train_norm = X_train[normal_mask]

    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X_train_norm)
    Xte = scaler.transform(X_test)

    iso = IsolationForest(
        n_estimators=400,
        contamination=0.02,   # expected anomaly rate (tune)
        random_state=42,
        n_jobs=-1,
        verbose=0
    )
    iso.fit(Xtr)

    # Scores: higher = more anomalous
    scores = -iso.decision_function(Xte)

    # Metrics (since synthetic labels exist)
    try:
        auroc = roc_auc_score(y_test, scores)
        ap = average_precision_score(y_test, scores)
    except Exception:
        auroc, ap = float("nan"), float("nan")

    # Choose threshold via percentile (align to contamination)
    thresh = np.percentile(scores, 100 * (1 - 0.02))  # top 2% flagged
    preds = (scores >= thresh).astype(int)

    report = {
        "samples_train": int(Xtr.shape[0]),
        "samples_test": int(Xte.shape[0]),
        "features": list(X.columns),
        "auroc": float(auroc),
        "average_precision": float(ap),
        "threshold": float(thresh),
        "contamination": 0.02,
    }

    # Save artifacts
    joblib.dump(scaler, f"{ART_DIR}/scaler.joblib")
    joblib.dump(iso, f"{ART_DIR}/model.joblib")
    with open(f"{ART_DIR}/report.json", "w") as f:
        json.dump(report, f, indent=2)

    # Build a scored dataframe for the WHOLE dataset (not just test split)
    X_all_scaled = scaler.transform(X)  # uses all rows
    all_scores = -iso.decision_function(X_all_scaled)
    all_preds = (all_scores >= thresh).astype(int)

    scored = df_full.copy()
    scored["anomaly_score"] = all_scores
    scored["pred_is_anomaly"] = all_preds
    scored.to_csv(SCORED_PATH, index=False)

    print("[+] IsolationForest training complete")
    print(f"    AUROC: {auroc:.4f} | AP: {ap:.4f}")
    print(f"    Threshold: {thresh:.6f} (flags ~2%)")
    print(f"[+] Artifacts saved in: {ART_DIR}")
    print(f"[+] Scored CSV: {SCORED_PATH}")
    print(scored.head())

if __name__ == "__main__":
    train_and_eval()
