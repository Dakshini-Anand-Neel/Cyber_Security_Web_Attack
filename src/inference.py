"""
src/inference.py
================
Unified, vectorized inference layer for web attack detection.

Used by the Streamlit app, Flask API, and CLI predict module.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.paths import MODEL_DIR

# ── Constants ──────────────────────────────────────────────────────────────────

SQL_KW = (
    "select", "union", "insert", "drop", "delete", "update",
    "from", "where", "--", "/*",
)
XSS_KW = (
    "<script", "alert(", "onerror", "onload", "javascript:",
    "eval(", "<svg", "<img",
)

FEATURE_COLS = [
    "payload_len", "payload_entropy", "special_char_cnt",
    "digit_ratio", "upper_ratio", "kw_sql", "kw_xss",
]

LABEL_MAP = {
    "1": "XSS", "2": "SQLi", "3": "Normal",
    1: "XSS", 2: "SQLi", 3: "Normal",
}

CLASS_COLORS = {"XSS": "#7c3aed", "SQLi": "#06b6d4", "Normal": "#10b981"}
CLASS_ORDER = ["XSS", "SQLi", "Normal"]

SEVERITY = {
    "XSS": ("CRITICAL", "#ef4444", 95),
    "SQLi": ("HIGH", "#f59e0b", 85),
    "Normal": ("SAFE", "#10b981", 10),
}

ATTACK_LIBRARY = {
    "XSS": [
        ("Reflected XSS", "<script>alert('XSS')</script>"),
        ("DOM XSS", "<img src=x onerror=alert(document.cookie)>"),
        ("SVG Injection", "<svg/onload=alert(1)>"),
        ("Event Handler", "<body onload=alert('hacked')>"),
        ("JS Protocol", "<a href=\"javascript:alert(1)\">click</a>"),
    ],
    "SQLi": [
        ("Classic OR 1=1", "1' OR '1'='1'--"),
        ("UNION Attack", "1 UNION SELECT null,table_name FROM information_schema.tables--"),
        ("DROP TABLE", "'; DROP TABLE users;--"),
        ("Blind SQLi", "1' AND SLEEP(5)--"),
        ("Comment Bypass", "admin'--"),
    ],
    "Normal": [
        ("Search Query", "What is the weather in Chennai today?"),
        ("Product Page", "Select a paint color for the living room"),
        ("User Input", "Hello, my name is Alex and I need help"),
        ("Form Data", "email=john@example.com&name=John+Doe"),
    ],
}

_SPECIAL_RE = re.compile(r'[<>\"\'()]')
_DIGIT_RE = re.compile(r"[0-9]")
_UPPER_RE = re.compile(r"[A-Z]")
_SQL_PATTERN = re.compile("|".join(re.escape(k) for k in SQL_KW), re.IGNORECASE)
_XSS_PATTERN = re.compile("|".join(re.escape(k) for k in XSS_KW), re.IGNORECASE)


def shannon_entropy(text: str) -> float:
    if not text:
        return 0.0
    freq = np.array([text.count(c) for c in set(text)], dtype=float) / len(text)
    return float(-np.sum(freq * np.log2(freq + 1e-12)))


def extract_features(payloads: list[str]) -> pd.DataFrame:
    """Vectorized feature extraction from raw payload strings."""
    col = pd.Series(payloads, dtype=str)
    lengths = col.str.len().clip(lower=1)
    low = col.str.lower()

    return pd.DataFrame({
        "payload_len": col.str.len(),
        "payload_entropy": col.map(shannon_entropy),
        "special_char_cnt": col.str.count(_SPECIAL_RE),
        "digit_ratio": col.str.count(_DIGIT_RE) / lengths,
        "upper_ratio": col.str.count(_UPPER_RE) / lengths,
        "kw_sql": low.str.contains(_SQL_PATTERN, regex=True).astype(int),
        "kw_xss": low.str.contains(_XSS_PATTERN, regex=True).astype(int),
    })


def load_artifacts(model_dir: Path | None = None) -> dict[str, Any] | None:
    """Load model artifacts from disk. Returns None if files are missing."""
    base = model_dir or MODEL_DIR
    try:
        with open(base / "metadata.json", encoding="utf-8") as fh:
            metadata = json.load(fh)
        return {
            "model": joblib.load(base / "final_model.joblib"),
            "scaler": joblib.load(base / "scaler.joblib"),
            "target_encoder": joblib.load(base / "target_encoder.joblib"),
            "metadata": metadata,
        }
    except (FileNotFoundError, OSError):
        return None


def _resolve_label(raw_label: Any) -> str:
    return LABEL_MAP.get(raw_label, str(raw_label))


def predict(
    payloads: list[str],
    artifacts: dict[str, Any],
) -> dict[str, Any]:
    """
    Run inference on a list of payloads.

    Returns
    -------
    dict with keys: labels, confidence, probabilities, features
    """
    feat_df = extract_features(payloads)
    X_scaled = artifacts["scaler"].transform(feat_df[FEATURE_COLS])
    raw_preds = artifacts["model"].predict(X_scaled)
    labels = [_resolve_label(l) for l in artifacts["target_encoder"].inverse_transform(raw_preds)]

    confidence: list[float] | None = None
    probabilities: list[dict[str, float]] | None = None

    if hasattr(artifacts["model"], "predict_proba"):
        proba = artifacts["model"].predict_proba(X_scaled)
        classes = artifacts["target_encoder"].classes_
        class_names = [_resolve_label(c) for c in classes]

        confidence = (proba.max(axis=1) * 100).round(1).tolist()
        probabilities = [
            {name: round(float(p) * 100, 1) for name, p in zip(class_names, row)}
            for row in proba
        ]

    return {
        "labels": labels,
        "confidence": confidence,
        "probabilities": probabilities,
        "features": feat_df,
    }


@lru_cache(maxsize=1)
def load_model_comparison() -> pd.DataFrame:
    """Load model comparison CSV from reports."""
    path = MODEL_DIR.parent / "reports" / "model_comparison.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if df.columns[0] == "Unnamed: 0" or df.columns[0] == "":
        df = df.rename(columns={df.columns[0]: "Model"})
    return df


@lru_cache(maxsize=1)
def load_class_distribution() -> pd.DataFrame:
    """Load class distribution from cached dataset."""
    from src.paths import RAW_CSV
    if not RAW_CSV.exists():
        return pd.DataFrame({"Class": CLASS_ORDER, "Count": [5932, 7567, 2902]})
    df = pd.read_csv(RAW_CSV, usecols=["Label"])
    counts = df["Label"].value_counts()
    return pd.DataFrame({
        "Class": [LABEL_MAP[i] for i in sorted(counts.index)],
        "Count": [int(counts[i]) for i in sorted(counts.index)],
    })
