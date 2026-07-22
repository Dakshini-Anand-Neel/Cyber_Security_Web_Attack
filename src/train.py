"""
src/train.py
============
Model Training & Hyperparameter Tuning for the Web Attacks ML Pipeline.

Supervised classifiers: LogisticRegression, DecisionTree, RandomForest,
                         KNN, SVM, NaiveBayes, NeuralNetwork (MLP)
Tuning: GridSearchCV + RandomizedSearchCV with 5-fold cross-validation
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
)

# ══════════════════════════════════════════════════════════════════════════════
# Model registry
# ══════════════════════════════════════════════════════════════════════════════

DEFAULT_MODELS: Dict[str, Any] = {
    "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
    "DecisionTree":       DecisionTreeClassifier(random_state=42),
    "RandomForest":       RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
    "KNN":                KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
    "SVM":                SVC(kernel="rbf", probability=True, random_state=42),
    "GaussianNB":         GaussianNB(),
    "MLP":                MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=300, random_state=42),
}

PARAM_GRIDS: Dict[str, Dict] = {
    "LogisticRegression": {"C": [0.01, 0.1, 1, 10, 100]},
    "DecisionTree":       {"max_depth": [5, 10, 20, None], "min_samples_split": [2, 5, 10]},
    "RandomForest":       {"n_estimators": [100, 200, 300], "max_depth": [10, 20, None]},
    "KNN":                {"n_neighbors": [3, 5, 7, 11]},
    "SVM":                {"C": [0.1, 1, 10], "kernel": ["rbf", "linear"]},
    "GaussianNB":         {"var_smoothing": [1e-9, 1e-8, 1e-7]},
    "MLP":                {"hidden_layer_sizes": [(64,), (128, 64), (256, 128, 64)]},
}


# ══════════════════════════════════════════════════════════════════════════════
# Training helpers
# ══════════════════════════════════════════════════════════════════════════════

def _avg_strategy(y: np.ndarray) -> str:
    return "binary" if len(np.unique(y)) == 2 else "weighted"


def train_and_evaluate(
    X_train: np.ndarray,
    X_test:  np.ndarray,
    y_train: np.ndarray,
    y_test:  np.ndarray,
    models:  Optional[Dict] = None,
) -> Tuple[Dict[str, Dict], Dict[str, Any]]:
    """
    Train every model in *models* and compute hold-out metrics.

    Returns
    -------
    results       : dict  { model_name → {accuracy, precision, recall, f1} }
    trained_models: dict  { model_name → fitted estimator }
    """
    if models is None:
        models = DEFAULT_MODELS

    avg = _avg_strategy(y_train)
    results: Dict[str, Dict] = {}
    trained: Dict[str, Any]  = {}

    for name, model in models.items():
        print(f"\n  ▶ Training {name} …")
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        acc  = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds, average=avg, zero_division=0)
        rec  = recall_score(y_test, preds, average=avg, zero_division=0)
        f1   = f1_score(y_test, preds, average=avg, zero_division=0)

        results[name] = {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}
        trained[name] = model
        print(f"    Acc={acc:.4f}  Prec={prec:.4f}  Rec={rec:.4f}  F1={f1:.4f}")

    return results, trained


# ══════════════════════════════════════════════════════════════════════════════
# Hyperparameter Tuning
# ══════════════════════════════════════════════════════════════════════════════

def tune_model(
    best_name:  str,
    best_model: Any,
    X_train:    np.ndarray,
    y_train:    np.ndarray,
    strategy:   str = "grid",
    cv:         int = 5,
    n_iter:     int = 20,
) -> Any:
    """
    Hyperparameter tuning via Grid Search or Randomized Search.

    Parameters
    ----------
    strategy : "grid" | "random"

    Returns
    -------
    Fitted best estimator.
    """
    grid = PARAM_GRIDS.get(best_name)
    if grid is None:
        print(f"  No param grid defined for {best_name} — returning original model.")
        return best_model

    base = type(best_model)()  # fresh instance

    if strategy == "random":
        search = RandomizedSearchCV(
            base, grid, n_iter=n_iter, cv=cv,
            scoring="f1_weighted", n_jobs=-1, random_state=42, verbose=0,
        )
    else:
        search = GridSearchCV(
            base, grid, cv=cv,
            scoring="f1_weighted", n_jobs=-1, verbose=0,
        )

    search.fit(X_train, y_train)
    print(f"  Best params : {search.best_params_}")
    print(f"  Best CV F1  : {search.best_score_:.4f}")
    return search.best_estimator_


# ══════════════════════════════════════════════════════════════════════════════
# Cross-Validation helper
# ══════════════════════════════════════════════════════════════════════════════

def cross_validate_model(
    model:   Any,
    X:       np.ndarray,
    y:       np.ndarray,
    cv:      int = 5,
    scoring: str = "f1_weighted",
) -> Dict[str, float]:
    """Return mean ± std of cross-validation scores."""
    scores = cross_val_score(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)
    return {"mean": scores.mean(), "std": scores.std(), "all": scores.tolist()}


# ══════════════════════════════════════════════════════════════════════════════
# Save / Load
# ══════════════════════════════════════════════════════════════════════════════

def save_model(model: Any, path: str) -> None:
    joblib.dump(model, path)
    print(f"  Model saved → {path}")


def load_model(path: str) -> Any:
    model = joblib.load(path)
    print(f"  Model loaded ← {path}")
    return model
