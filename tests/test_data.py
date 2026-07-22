"""
tests/test_data.py
==================
Unit tests for src/data.py — loading, cleaning, and feature engineering.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import pytest

from src.data import clean_data, add_payload_features


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_df():
    """Minimal dataset mimicking the web-attacks schema."""
    return pd.DataFrame({
        "Payload": [
            "<script>alert('XSS')</script>",
            "1' OR '1'='1'--",
            "Select a paint color for the room.",
            "1' OR '1'='1'--",           # duplicate
            None,                          # missing value
            "  hello world  ",             # whitespace
        ],
        "Label":      [1, 2, 3, 2, 3, 3],
        "text_label": ["XSS", "SQLi", "normal", "SQLi", "normal", "normal"],
        "ID":         [1, 2, 3, 2, 5, 6],
    })


# ── clean_data tests ──────────────────────────────────────────────────────────

class TestCleanData:
    def test_removes_duplicates(self, sample_df):
        clean, report = clean_data(sample_df)
        assert report["duplicates_removed"] == 1
        assert len(clean) == len(sample_df) - 1

    def test_fills_missing_strings(self, sample_df):
        clean, report = clean_data(sample_df)
        assert clean["Payload"].isnull().sum() == 0

    def test_strips_whitespace(self, sample_df):
        clean, _ = clean_data(sample_df)
        for val in clean["Payload"].dropna():
            assert val == val.strip()

    def test_returns_dict_report(self, sample_df):
        _, report = clean_data(sample_df)
        assert isinstance(report, dict)
        assert "duplicates_removed" in report
        assert "missing_before" in report

    def test_no_constant_cols_dropped_on_clean_df(self):
        df = pd.DataFrame({
            "Payload": ["a","b","c"],
            "Label":   [1, 2, 3],
            "text_label": ["XSS","SQLi","normal"],
            "ID": [10,11,12],
        })
        _, report = clean_data(df)
        assert report["constant_cols_dropped"] == []


# ── add_payload_features tests ────────────────────────────────────────────────

class TestPayloadFeatures:
    @pytest.fixture
    def simple_df(self):
        return pd.DataFrame({
            "Payload": [
                "<script>alert(1)</script>",     # XSS
                "SELECT * FROM users WHERE 1=1",  # SQLi
                "Hello world",                    # Normal
            ]
        })

    def test_adds_payload_len(self, simple_df):
        out = add_payload_features(simple_df)
        assert "payload_len" in out.columns
        assert out["payload_len"].iloc[0] == len("<script>alert(1)</script>")

    def test_kw_xss_detected(self, simple_df):
        out = add_payload_features(simple_df)
        assert out["kw_xss"].iloc[0] == 1
        assert out["kw_xss"].iloc[2] == 0

    def test_kw_sql_detected(self, simple_df):
        out = add_payload_features(simple_df)
        assert out["kw_sql"].iloc[1] == 1
        assert out["kw_sql"].iloc[2] == 0

    def test_entropy_positive(self, simple_df):
        out = add_payload_features(simple_df)
        assert (out["payload_entropy"] >= 0).all()

    def test_digit_ratio_range(self, simple_df):
        out = add_payload_features(simple_df)
        assert (out["digit_ratio"] >= 0).all()
        assert (out["digit_ratio"] <= 1).all()

    def test_upper_ratio_range(self, simple_df):
        out = add_payload_features(simple_df)
        assert (out["upper_ratio"] >= 0).all()
        assert (out["upper_ratio"] <= 1).all()

    def test_output_shape(self, simple_df):
        out = add_payload_features(simple_df)
        # Original columns + 7 new features
        assert out.shape[1] == simple_df.shape[1] + 7
