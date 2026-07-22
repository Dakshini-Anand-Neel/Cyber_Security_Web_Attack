"""
src/features.py
===============
Preprocessing & Feature Selection utilities.

Covers:
  • Label Encoding / One-Hot Encoding
  • Five scaling strategies (MinMax, Standard, Robust, MaxAbs, Unit-Vector)
  • Filter method  (mutual information, correlation)
  • Wrapper method (RFE with a fast estimator)
  • Embedded method (tree-based feature importances)
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import (
    RFE,
    SelectFromModel,
    SelectKBest,
    mutual_info_classif,
)
from sklearn.preprocessing import (
    LabelEncoder,
    MaxAbsScaler,
    MinMaxScaler,
    Normalizer,
    OneHotEncoder,
    RobustScaler,
    StandardScaler,
)

# ══════════════════════════════════════════════════════════════════════════════
# Encoding
# ══════════════════════════════════════════════════════════════════════════════

def encode_labels(
    df: pd.DataFrame,
    target_col: str,
) -> Tuple[pd.DataFrame, LabelEncoder]:
    """Label-encode the target column. Returns modified df + fitted encoder."""
    df = df.copy()
    le = LabelEncoder()
    df[target_col] = le.fit_transform(df[target_col].astype(str))
    return df, le


def encode_features(
    df: pd.DataFrame,
    cat_cols: Optional[List[str]] = None,
    strategy: str = "label",
) -> Tuple[pd.DataFrame, Dict]:
    """
    Encode categorical feature columns.

    Parameters
    ----------
    strategy : "label" | "onehot"
    """
    df = df.copy()
    if cat_cols is None:
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    encoders: Dict = {}

    if strategy == "label":
        for col in cat_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le

    elif strategy == "onehot":
        ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        encoded = ohe.fit_transform(df[cat_cols].astype(str))
        enc_cols = ohe.get_feature_names_out(cat_cols)
        df = df.drop(columns=cat_cols)
        df = pd.concat([df, pd.DataFrame(encoded, columns=enc_cols, index=df.index)], axis=1)
        encoders["__ohe__"] = ohe

    return df, encoders


# ══════════════════════════════════════════════════════════════════════════════
# Scaling
# ══════════════════════════════════════════════════════════════════════════════

SCALERS = {
    "minmax":      MinMaxScaler(),
    "standard":    StandardScaler(),
    "robust":      RobustScaler(),
    "maxabs":      MaxAbsScaler(),
    "unit_vector": Normalizer(norm="l2"),
}


def scale_features(
    X: pd.DataFrame,
    method: str = "standard",
) -> Tuple[pd.DataFrame, object]:
    """
    Scale numeric columns of X.

    Parameters
    ----------
    method : one of "minmax" | "standard" | "robust" | "maxabs" | "unit_vector"

    Returns
    -------
    (scaled_df, fitted_scaler)
    """
    if method not in SCALERS:
        raise ValueError(f"Unknown scaling method '{method}'. Choose from {list(SCALERS)}")

    num_cols = X.select_dtypes(include=["float64", "int64", "float32"]).columns.tolist()
    scaler = SCALERS[method]
    X = X.copy()
    X[num_cols] = scaler.fit_transform(X[num_cols])
    return X, scaler


# ══════════════════════════════════════════════════════════════════════════════
# Feature Selection
# ══════════════════════════════════════════════════════════════════════════════

def filter_selection(
    X: pd.DataFrame,
    y: np.ndarray,
    k: int = 10,
) -> List[str]:
    """
    Filter method: select top-k features by mutual information.

    Returns list of selected column names.
    """
    selector = SelectKBest(score_func=mutual_info_classif, k=min(k, X.shape[1]))
    selector.fit(X, y)
    mask = selector.get_support()
    selected = X.columns[mask].tolist()
    return selected


def wrapper_selection(
    X: pd.DataFrame,
    y: np.ndarray,
    n_features: int = 10,
) -> List[str]:
    """
    Wrapper method: Recursive Feature Elimination (RFE)
    using a shallow RandomForest as the estimator.
    """
    estimator = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
    rfe = RFE(estimator, n_features_to_select=min(n_features, X.shape[1]), step=1)
    rfe.fit(X, y)
    selected = X.columns[rfe.support_].tolist()
    return selected


def embedded_selection(
    X: pd.DataFrame,
    y: np.ndarray,
    threshold: str = "mean",
) -> Tuple[List[str], pd.Series]:
    """
    Embedded method: SelectFromModel using a RandomForest.

    Returns (selected_columns, feature_importances_series).
    """
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
    selector = SelectFromModel(rf, threshold=threshold, prefit=True)
    mask = selector.get_support()
    selected = X.columns[mask].tolist()
    return selected, importances
