"""
tests/test_train.py
===================
Unit tests for src/train.py — model training and hyperparameter tuning.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest

from src.train import (
    train_and_evaluate,
    tune_model,
    cross_validate_model,
    DEFAULT_MODELS,
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier


@pytest.fixture
def synthetic_data():
    """Small balanced 3-class dataset (enough to train quickly)."""
    np.random.seed(99)
    n = 120
    # Class 0: low values, Class 1: mid, Class 2: high
    X = np.vstack([
        np.random.randn(n, 7) - 2,
        np.random.randn(n, 7),
        np.random.randn(n, 7) + 2,
    ])
    y = np.array([0]*n + [1]*n + [2]*n)
    from sklearn.model_selection import train_test_split
    return train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)


class TestTrainAndEvaluate:
    def test_returns_results_and_models(self, synthetic_data):
        X_tr, X_te, y_tr, y_te = synthetic_data
        fast_models = {
            "LR": LogisticRegression(max_iter=200, random_state=42),
            "DT": DecisionTreeClassifier(max_depth=3, random_state=42),
        }
        results, trained = train_and_evaluate(X_tr, X_te, y_tr, y_te, models=fast_models)
        assert set(results.keys()) == {"LR", "DT"}
        assert set(trained.keys()) == {"LR", "DT"}

    def test_metric_keys(self, synthetic_data):
        X_tr, X_te, y_tr, y_te = synthetic_data
        fast_models = {"LR": LogisticRegression(max_iter=200, random_state=42)}
        results, _ = train_and_evaluate(X_tr, X_te, y_tr, y_te, models=fast_models)
        assert set(results["LR"].keys()) == {"accuracy","precision","recall","f1"}

    def test_scores_in_range(self, synthetic_data):
        X_tr, X_te, y_tr, y_te = synthetic_data
        fast_models = {"LR": LogisticRegression(max_iter=200, random_state=42)}
        results, _ = train_and_evaluate(X_tr, X_te, y_tr, y_te, models=fast_models)
        for v in results["LR"].values():
            assert 0.0 <= v <= 1.0


class TestTuneModel:
    def test_grid_search_returns_fitted_model(self, synthetic_data):
        X_tr, _, y_tr, _ = synthetic_data
        base = LogisticRegression(max_iter=200)
        tuned = tune_model("LogisticRegression", base, X_tr, y_tr, strategy="grid", cv=3)
        assert hasattr(tuned, "predict")

    def test_unknown_name_returns_original(self, synthetic_data):
        X_tr, _, y_tr, _ = synthetic_data
        base = LogisticRegression(max_iter=200)
        tuned = tune_model("NoSuchModel", base, X_tr, y_tr)
        assert tuned is base


class TestCrossValidate:
    def test_returns_mean_std(self, synthetic_data):
        X_tr, X_te, y_tr, y_te = synthetic_data
        X_all = np.vstack([X_tr, X_te])
        y_all = np.hstack([y_tr, y_te])
        model = LogisticRegression(max_iter=200, random_state=42)
        result = cross_validate_model(model, X_all, y_all, cv=3)
        assert "mean" in result
        assert "std" in result
        assert 0 <= result["mean"] <= 1
        assert result["std"] >= 0
