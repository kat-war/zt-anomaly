#!/usr/bin/env python3
"""
Feature engineering for Zero Trust anomaly detection.

Input : data/raw/zt_logs.csv  (or --in path)
Output: data/processed/zt_features.csv and .parquet (or --out-* paths)

Safe against mixed dtypes/NaNs; adds time, velocity, change/novelty flags, and one-hots.
"""

from __future__ import annotations
import argparse
import os
import sys
import numpy as np
import pandas as pd


# -----------------------
# Defaults (overridable)
# -----------------------
DEF_IN = "data/raw/zt_logs.csv"
DEF_OUT_CSV = "data/processed/zt_features.csv"
DEF_OUT_PQ = "data/processed/zt_features.parquet"


# -----------------------
# Utilities
# -----------------------
def haversine_km(lat1, lon1, lat2, lon2) -> np.ndarray:
    """Vectorized haversine distance in kilometers."""
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(
        np.radians, [lat1.astype(float), lon1.astype(float), lat2.astype(float), lon2.astype(float)]
    )
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def ensure_cols(df: pd.DataFrame, cols_with_defaults: dict) -> pd.DataFrame:
    """Create any missing columns with sensible defaults."""
    for col, default in cols_with_defaults.items():
        if col not in df.columns:
            df[col] = default
    return df


def as_str_series(s: pd.Series) -> pd.Series:
    """Return a safe, lowercase string series (handles non-string & NaN)."""
    return s.astype("string").fillna("").str.lower()


# -----------------------
# Core pipeline
# -----------------------
def build_features(
    df: pd.DataFrame,
    one_hot_cols=("role", "action", "country", "resource"),
) -> pd.DataFrame:
    # 1) Basic normalization / required columns
    required_defaults = {
        "timestamp": pd.NaT,
        "user": "",
        "device_id": "",
        "device_trust": "",
        "result": "",
        "mfa": "",
        "role": "",
        "action": "",
        "country": "",
        "resource": "",
        "lat": 0.0,
        "lon": 0.0,
        "is_anomaly": 0,
    }
    df = ensure_cols(df.copy(), required_defaults)

    # Parse timestamps and sort
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=False)
    df = df.sort_values(["user", "timestamp"], kind="mergesort").reset_index(drop=True)

    # 2) Time features
    df["hour"] = df["timestamp"].dt.hour.fillna(0).astype(int)
    df["dayofweek"] = df["timestamp"].dt.dayofweek.fillna(0).astype(int)  # 0=Mon
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)
    df["is_off_hours"] = df["hour"].isin([0, 1, 2, 3, 4, 5, 22, 23]).astype(int)

    # 3) Per-user previous event context
    grp = df.groupby("user", sort=False, dropna=False)
    df["prev_lat"] = grp["lat"].shift(1)
    df["prev_lon"] = grp["lon"].shift(1)
    df["prev_timestamp"] = grp["timestamp"].shift(1)
    df["prev_country"] = grp["country"].shift(1)
    df["prev_resource"] = grp["resource"].shift(1)
    df["prev_device"] = grp["device_id"].shift(1)

    # 4) Travel distance and velocity
    #    If previous lat/lon missing, use current to yield 0 km
    lat1 = df["prev_lat"].fillna(df["lat"])
    lon1 = df["prev_lon"].fillna(df["lon"])
    lat2 = df["lat"].fillna(0.0)
    lon2 = df["lon"].fillna(0.0)
    df["km_travelled"] = haversine_km(lat1, lon1, lat2, lon2)

    dt_hours = (df["timestamp"] - df["prev_timestamp"]).dt.total_seconds() / 3600.0
    # Avoid div-by-zero and negative time deltas (clock skew)
    dt_hours = dt_hours.where((dt_hours > 0) & np.isfinite(dt_hours), np.nan)
    df["km_per_hour"] = (df["km_travelled"] / dt_hours).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    # Cap extreme velocities to reduce model distortion (optional)
    df["km_per_hour"] = df["km_per_hour"].clip(upper=20000)  # faster than a jet? suspicious but cap anyway

    # 5) Change flags
    df["country_changed"] = (df["country"] != df["prev_country"]).fillna(False).astype(int)
    df["resource_changed"] = (df["resource"] != df["prev_resource"]).fillna(False).astype(int)
    df["device_changed"] = (df["device_id"] != df["prev_device"]).fillna(False).astype(int)

    # 6) Device novelty for user
    df["user_device"] = df["user"].astype("string") + "::" + df["device_id"].astype("string")
    df["is_new_device_for_user"] = (~df.duplicated(subset=["user_device"])).astype(int)

    # 7) Trust / MFA / result flags (safe string handling)
    df["is_untrusted_device"] = (as_str_series(df["device_trust"]) == "untrusted").astype(int)
    df["mfa_false"] = (as_str_series(df["mfa"]) == "false").astype(int)
    df["result_failure"] = (as_str_series(df["result"]) == "failure").astype(int)

    # 8) One-hot encode selected low-cardinality columns
    #    (If some are missing, no problem: ensure_cols created them above.)
    df_oh = pd.get_dummies(df, columns=[c for c in one_hot_cols if c in df.columns], drop_first=True)

    # 9) Final column selection
    to_drop_text = [
        "timestamp",
        "user",
        "device_id",
        "device_trust",
        "result",
        "mfa",
        "prev_lat",
        "prev_lon",
        "prev_timestamp",
        "prev_country",
        "prev_resource",
        "prev_device",
        "user_device",
    ]
    keep_cols = [c for c in df_oh.columns if c not in to_drop_text]
    out = df_oh[keep_cols].copy()

    # Ensure target column present and last
    if "is_anomaly" not in out.columns:
        out["is_anomaly"] = 0
    else:
        # move to end
        y = out.pop("is_anomaly")
        out["is_anomaly"] = y.astype(int)

    # Basic sanity: replace any remaining NaNs with 0
    out = out.replace([np.inf, -np.inf], np.nan).fillna(0)

    return out


def run(in_path: str, out_csv: str, out_parquet: str) -> None:
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    # Read as "object" to avoid pandas trying to coerce and breaking .str ops
    df = pd.read_csv(in_path, dtype="object", low_memory=False)
    # lat/lon should be numeric; coerce if present
    for col in ("lat", "lon"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    feats = build_features(df)

    feats.to_csv(out_csv, index=False)
    try:
        feats.to_parquet(out_parquet, index=False)
    except Exception as e:
        # Parquet optional (missing pyarrow/fastparquet etc.)
        print(f"[!] Parquet save failed ({e}). CSV saved OK.", file=sys.stderr)

    print(f"[+] Features saved")
    print(f"    CSV : {out_csv}")
    print(f"    PQ  : {out_parquet}")
    print(f"[+] Shape: {feats.shape[0]} rows x {feats.shape[1]} cols")
    # Show a quick peek
    with pd.option_context("display.max_columns", 20):
        print(feats.head())


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Build features for ZT anomaly detection.")
    p.add_argument("--in", dest="in_path", default=DEF_IN, help="Input CSV of raw logs")
    p.add_argument("--out-csv", dest="out_csv", default=DEF_OUT_CSV, help="Output CSV path")
    p.add_argument("--out-parquet", dest="out_parquet", default=DEF_OUT_PQ, help="Output Parquet path")
    return p.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    run(args.in_path, args.out_csv, args.out_parquet)
