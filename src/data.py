"""
src/data.py
===========
Data Collection & Cleaning utilities for the Web Attacks ML Pipeline.

Dataset: shengqin/web-attacks-long  (Hugging Face)
Classes: 1=XSS, 2=SQLi, 3=Normal

Usage
-----
    from src.data import load_raw_data, clean_data
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

# ── Optional: load via Hugging Face datasets library ──────────────────────────
try:
    from datasets import load_dataset as hf_load_dataset
    _HF_AVAILABLE = True
except ImportError:
    _HF_AVAILABLE = False

# ── Project paths ──────────────────────────────────────────────────────────────
from src.paths import DATA_DIR, RAW_CSV

DATASET_URL = (
    "https://huggingface.co/datasets/shengqin/web-attacks-long"
    "/resolve/main/train.csv"
)
DATASET_ID = "shengqin/web-attacks-long"

LABEL_MAP = {1: "XSS", 2: "SQLi", 3: "Normal"}


# ══════════════════════════════════════════════════════════════════════════════
# 1. DATA COLLECTION
# ══════════════════════════════════════════════════════════════════════════════

def load_raw_data(
    prefer_local: bool = True,
    save_local:   bool = True,
) -> pd.DataFrame:
    """
    Load the web-attacks dataset.

    Strategy
    --------
    1. If ``prefer_local`` and the local CSV already exists → read from disk.
    2. Else try the Hugging Face ``datasets`` library (streaming-friendly).
    3. Fall back to a direct ``pd.read_csv`` over the HF CDN URL.

    Parameters
    ----------
    prefer_local : bool
        When True, return the cached CSV without hitting the network.
    save_local : bool
        When True, save the downloaded data as ``data/raw_train.csv``.

    Returns
    -------
    pd.DataFrame
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if prefer_local and RAW_CSV.exists():
        print(f"[data] Loading from local cache: {RAW_CSV}")
        return pd.read_csv(RAW_CSV)

    # ── Try HF datasets library first (handles auth / gated repos) ────────────
    if _HF_AVAILABLE:
        try:
            print(f"[data] Fetching '{DATASET_ID}' via Hugging Face datasets …")
            hf_ds = hf_load_dataset(DATASET_ID, split="train")
            df = hf_ds.to_pandas()
        except Exception as exc:
            print(f"[data] HF library failed ({exc}). Falling back to direct CSV …")
            df = pd.read_csv(DATASET_URL)
    else:
        print(f"[data] Fetching CSV from: {DATASET_URL}")
        df = pd.read_csv(DATASET_URL)

    print(f"[data] Loaded {df.shape[0]:,} rows × {df.shape[1]} columns")

    if save_local:
        df.to_csv(RAW_CSV, index=False)
        print(f"[data] Saved raw copy → {RAW_CSV}")

    return df


# ══════════════════════════════════════════════════════════════════════════════
# 2. DATA CLEANING
# ══════════════════════════════════════════════════════════════════════════════

def clean_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    """
    Clean the raw DataFrame.

    Steps
    -----
    * Remove duplicate rows.
    * Handle missing / null values (numeric → median, categorical → mode).
    * Drop constant / near-constant columns (unique ratio < 0.001).
    * Strip leading / trailing whitespace from string columns.

    Returns
    -------
    (cleaned_df, report_dict)
    """
    report: dict = {}
    df = df.copy()

    # ── Duplicates ─────────────────────────────────────────────────────────────
    before = len(df)
    df = df.drop_duplicates()
    report["duplicates_removed"] = before - len(df)

    # ── Missing values ─────────────────────────────────────────────────────────
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    report["missing_before"] = missing.to_dict()

    for col in missing.index:
        if df[col].dtype in ("float64", "int64"):
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna(df[col].mode().iloc[0])

    report["missing_after"] = int(df.isnull().sum().sum())

    # ── Strip whitespace ───────────────────────────────────────────────────────
    str_cols = df.select_dtypes(include="object").columns
    for col in str_cols:
        df[col] = df[col].str.strip()

    # ── Constant / near-constant columns ──────────────────────────────────────
    unique_ratios = df.nunique() / len(df)
    constant_cols = unique_ratios[unique_ratios < 0.001].index.tolist()
    # Always keep the target columns even if they look constant
    protect = {"Label", "text_label", "label"}
    constant_cols = [c for c in constant_cols if c not in protect]
    if constant_cols:
        df = df.drop(columns=constant_cols)
    report["constant_cols_dropped"] = constant_cols

    return df, report


# ══════════════════════════════════════════════════════════════════════════════
# 3. FEATURE ENGINEERING helpers (payload-level)
# ══════════════════════════════════════════════════════════════════════════════

def _shannon_entropy(text: str) -> float:
    """Character-level Shannon entropy of a string."""
    if not text:
        return 0.0
    freq = np.array([text.count(c) for c in set(text)]) / len(text)
    return -np.sum(freq * np.log2(freq + 1e-12))


def add_payload_features(df: pd.DataFrame, payload_col: str = "Payload") -> pd.DataFrame:
    """
    Derive hand-crafted features from the raw payload string.

    New columns
    -----------
    payload_len      : character count
    payload_entropy  : Shannon entropy
    special_char_cnt : count of '<', '>', '"', "'", '(', ')'
    digit_ratio      : fraction of digit characters
    upper_ratio      : fraction of upper-case characters
    kw_sql           : 1 if payload contains common SQL keywords
    kw_xss           : 1 if payload contains common XSS tokens
    """
    df = df.copy()
    col = df[payload_col].astype(str)

    df["payload_len"]      = col.str.len()
    df["payload_entropy"]  = col.apply(_shannon_entropy)
    df["special_char_cnt"] = col.apply(lambda x: sum(x.count(c) for c in '<>"\'()'))
    df["digit_ratio"]      = col.apply(lambda x: sum(c.isdigit() for c in x) / max(len(x), 1))
    df["upper_ratio"]      = col.apply(lambda x: sum(c.isupper() for c in x) / max(len(x), 1))

    sql_kw = ["select", "union", "insert", "drop", "delete", "update", "from", "where", "--", "/*"]
    xss_kw = ["<script", "alert(", "onerror", "onload", "javascript:", "eval(", "<svg", "<img"]

    low = col.str.lower()
    df["kw_sql"] = low.apply(lambda x: int(any(k in x for k in sql_kw)))
    df["kw_xss"] = low.apply(lambda x: int(any(k in x for k in xss_kw)))

    return df


if __name__ == "__main__":
    raw = load_raw_data(prefer_local=False)
    clean, report = clean_data(raw)
    print("\nCleaning report:", report)
    enriched = add_payload_features(clean)
    print(enriched.head())
    print(f"\nFinal shape: {enriched.shape}")
