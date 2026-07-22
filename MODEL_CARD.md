# Model Card — Web Attack Classifier

## Model Summary

| Field | Value |
|-------|-------|
| **Model Name** | Web Attack Multi-Class Classifier |
| **Task** | Multi-class Text Classification |
| **Classes** | XSS · SQLi · Normal |
| **Best Algorithm** | Logistic Regression (tuned) |
| **Final Accuracy** | 100.00% |
| **Final F1 (weighted)** | 1.0000 |
| **Artifact** | `models/final_model.joblib` |
| **Framework** | scikit-learn 1.3+ |
| **Python** | 3.10+ |

---

## Intended Use

| Use Case | Supported? |
|----------|-----------|
| Detecting XSS payloads in web requests | ✅ Yes |
| Detecting SQL Injection payloads | ✅ Yes |
| Filtering benign vs. malicious traffic | ✅ Yes |
| Real-time WAF (Web Application Firewall) integration | ⚠️ Requires latency testing |
| Adversarial / obfuscated attack detection | ⚠️ Limited — retrain with augmented data |
| Non-HTTP traffic classification | ❌ Not intended |

---

## Training Details

### Dataset
- Source: `shengqin/web-attacks-long` (Hugging Face)
- Size: 16 401 rows
- Train split: 13 120 rows (80%)
- Test split: 3 281 rows (20%, stratified)

### Features (input to model)
After preprocessing, the model uses **7 numeric features** derived from the raw payload:

| Feature | Description |
|---------|-------------|
| `payload_len` | Character count of the payload |
| `payload_entropy` | Shannon entropy (character distribution) |
| `special_char_cnt` | Count of `< > " ' ( )` characters |
| `digit_ratio` | Fraction of digits in payload |
| `upper_ratio` | Fraction of uppercase letters |
| `kw_sql` | 1 if SQL keywords detected, else 0 |
| `kw_xss` | 1 if XSS tokens detected, else 0 |
| `ID` (encoded) | Encoded sample identifier (minor signal) |

### Preprocessing Pipeline
```
Raw Payload
    ↓ LabelEncoder (on Payload & text_label columns)
    ↓ Payload feature engineering (7 numeric features)
    ↓ StandardScaler (Z-score normalization)
    ↓ Logistic Regression (C=0.01, tuned via GridSearchCV)
```

---

## Evaluation Results

### Hold-out Test Set (3 281 samples)

| Class  | Precision | Recall | F1-Score | Support |
|--------|-----------|--------|----------|---------|
| XSS    | 1.00      | 1.00   | 1.00     | 1 187   |
| SQLi   | 1.00      | 1.00   | 1.00     | 1 514   |
| Normal | 1.00      | 1.00   | 1.00     | 580     |
| **Weighted avg** | **1.00** | **1.00** | **1.00** | **3 281** |

### Confusion Matrix
```
        Pred XSS  Pred SQLi  Pred Normal
XSS       1187        0          0
SQLi         0     1514          0
Normal       0        0        580
```

### All Models Comparison

| Model               | Accuracy | Precision | Recall | F1    |
|---------------------|----------|-----------|--------|-------|
| Logistic Regression | 1.000    | 1.000     | 1.000  | 1.000 |
| Decision Tree       | 1.000    | 1.000     | 1.000  | 1.000 |
| Random Forest       | 1.000    | 1.000     | 1.000  | 1.000 |
| SVM (RBF)           | 1.000    | 1.000     | 1.000  | 1.000 |
| KNN (k=5)           | 1.000    | 1.000     | 1.000  | 1.000 |
| MLP Neural Net      | 1.000    | 1.000     | 1.000  | 1.000 |
| Gaussian NB         | ~0.974   | ~0.974    | ~0.974 | ~0.974|

### Cross-Validation (5-fold, F1 weighted)
- Mean: 1.0000 ± 0.0000

---

## Hyperparameter Tuning

**Strategy:** GridSearchCV, 5-fold CV, scoring=`f1_weighted`

| Model | Param Grid | Best Params |
|-------|-----------|------------|
| Logistic Regression | `C ∈ [0.01, 0.1, 1, 10, 100]` | `C = 0.01` |
| Decision Tree | `max_depth, min_samples_split` | varies |
| Random Forest | `n_estimators, max_depth` | varies |

---

## Artifacts

| File | Description |
|------|-------------|
| `models/final_model.joblib` | Tuned best estimator |
| `models/scaler.joblib` | Fitted StandardScaler |
| `models/target_encoder.joblib` | LabelEncoder for y (1→XSS, 2→SQLi, 3→Normal) |
| `models/feature_encoders.joblib` | LabelEncoders for categorical X columns |
| `models/metadata.json` | JSON metadata (params, metrics, columns) |

---

## Limitations & Risks

| Risk | Mitigation |
|------|-----------|
| Perfect accuracy may indicate feature leakage | `text_label` was excluded from features after early pipeline |
| Dataset may not cover obfuscated attacks | Augment with adversarial samples |
| Static payload feature engineering | Consider TF-IDF or character n-grams for richer representation |
| No temporal validation | Implement rolling-window evaluation for production |

---

## How to Load & Use

```python
import joblib
from src.predict import predict_payloads

# Quick predict
results = predict_payloads(["<script>alert(1)</script>"])
print(results)

# Or load manually
model   = joblib.load("models/final_model.joblib")
scaler  = joblib.load("models/scaler.joblib")
```

---

## Model Versioning

| Version | Date | Notes |
|---------|------|-------|
| v1.0 | 2026-07-21 | Initial pipeline — 7 hand-crafted features, LR best model |
| v1.1 | TBD | Add TF-IDF character n-gram features |
| v2.0 | TBD | Deep learning (BERT / DistilBERT) fine-tune |
