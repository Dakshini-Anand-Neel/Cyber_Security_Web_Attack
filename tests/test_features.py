"""
tests/test_features.py
======================
Unit tests for src/features.py — encoding, scaling, and feature selection.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import pytest

from src.features import (
    encode_labels,
    encode_features,
    scale_features,
    filter_selection,
    wrapper_selection,
    embedded_selection,
    SCALERS,
)


@pytest.fixture
def numeric_df():
    np.random.seed(42)
    return pd.DataFrame({
        "payload_len":      np.random.randint(5, 200, 100).astype(float),
        "payload_entropy":  np.random.uniform(1, 4, 100),
        "special_char_cnt": np.random.randint(0, 20, 100).astype(float),
        "digit_ratio":      np.random.uniform(0, 0.5, 100),
        "upper_ratio":      np.random.uniform(0, 0.3, 100),
        "kw_sql":           np.random.randint(0, 2, 100).astype(float),
        "kw_xss":           np.random.randint(0, 2, 100).astype(float),
    })


@pytest.fixture
def cat_df():
    return pd.DataFrame({
        "Payload":    ["<script>", "SELECT *", "Hello"],
        "text_label": ["XSS",     "SQLi",     "normal"],
        "Label":      [1, 2, 3],
    })


@pytest.fixture
def y_array():
    np.random.seed(0)
    return np.random.choice([0, 1, 2], size=100)


# ── Encoding tests ─────────────────────────────────────────────────────────────

class TestEncodeLabels:
    def test_returns_int_encoded(self, cat_df):
        df, le = encode_labels(cat_df, "Label")
        assert df["Label"].dtype in (int, np.int64)

    def test_encoder_classes(self, cat_df):
        _, le = encode_labels(cat_df, "Label")
        assert len(le.classes_) == 3


class TestEncodeFeatures:
    def test_label_encoding(self, cat_df):
        df, encoders = encode_features(cat_df, cat_cols=["Payload","text_label"], strategy="label")
        assert df["Payload"].dtype in (int, np.int64, np.int32)
        assert "Payload" in encoders

    def test_onehot_encoding(self, cat_df):
        df, encoders = encode_features(cat_df, cat_cols=["text_label"], strategy="onehot")
        # original col removed, new OHE cols added
        assert "text_label" not in df.columns
        assert "__ohe__" in encoders

    def test_unknown_strategy_raises(self, cat_df):
        with pytest.raises(Exception):
            encode_features(cat_df, cat_cols=["text_label"], strategy="bad")


# ── Scaling tests ──────────────────────────────────────────────────────────────

class TestScaleFeatures:
    @pytest.mark.parametrize("method", ["minmax","standard","robust","maxabs","unit_vector"])
    def test_all_methods_run(self, numeric_df, method):
        scaled, scaler = scale_features(numeric_df, method=method)
        assert scaled.shape == numeric_df.shape

    def test_minmax_range(self, numeric_df):
        scaled, _ = scale_features(numeric_df, method="minmax")
        assert scaled.min().min() >= -1e-9
        assert scaled.max().max() <= 1 + 1e-9

    def test_standard_mean_zero(self, numeric_df):
        scaled, _ = scale_features(numeric_df, method="standard")
        means = scaled.mean()
        assert (means.abs() < 0.01).all()

    def test_unknown_method_raises(self, numeric_df):
        with pytest.raises(ValueError, match="Unknown scaling method"):
            scale_features(numeric_df, method="nonexistent")


# ── Feature selection tests ────────────────────────────────────────────────────

class TestFeatureSelection:
    def test_filter_returns_list(self, numeric_df, y_array):
        selected = filter_selection(numeric_df, y_array, k=4)
        assert isinstance(selected, list)
        assert len(selected) == 4
        assert all(c in numeric_df.columns for c in selected)

    def test_wrapper_returns_list(self, numeric_df, y_array):
        selected = wrapper_selection(numeric_df, y_array, n_features=3)
        assert isinstance(selected, list)
        assert len(selected) == 3

    def test_embedded_returns_importances(self, numeric_df, y_array):
        selected, importances = embedded_selection(numeric_df, y_array)
        assert isinstance(selected, list)
        assert len(importances) == numeric_df.shape[1]
        assert abs(importances.sum() - 1.0) < 1e-6  # importances sum to 1
