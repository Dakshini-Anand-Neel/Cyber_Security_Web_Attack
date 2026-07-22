# 🔐 Web Attack Detection — ML Pipeline

> **End-to-end Machine Learning pipeline for detecting XSS, SQLi, and Normal web payloads**

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square&logo=python)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange?style=flat-square&logo=scikit-learn)
![Dataset](https://img.shields.io/badge/Dataset-HuggingFace-yellow?style=flat-square&logo=huggingface)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

---

## 📌 Overview

This project builds a full supervised-learning pipeline to classify HTTP request payloads into three categories:

| Label | Class  | Description                          |
|-------|--------|--------------------------------------|
| 1     | XSS    | Cross-Site Scripting attack payload  |
| 2     | SQLi   | SQL Injection attack payload         |
| 3     | Normal | Benign / legitimate request payload  |

**Dataset:** [`shengqin/web-attacks-long`](https://huggingface.co/datasets/shengqin/web-attacks-long) — 16 401 labeled web payloads from Hugging Face.

---

## 🗂️ Project Structure

```
E:\CYBERSECURITY\
├── app\                        # Streamlit / Flask web application
│   ├── app.py                  # Main Streamlit app
│   ├── api.py                  # REST API (Flask)
│   └── static\                 # Static assets (CSS, JS)
│
├── data\
│   ├── raw\                    # Original downloaded data
│   ├── interim\                # Intermediate cleaned data
│   ├── processed\              # Final feature-engineered data
│   └── raw_train.csv           # Cached dataset (auto-generated)
│
├── models\
│   ├── final_model.joblib      # Trained best model
│   ├── scaler.joblib           # StandardScaler artifact
│   ├── target_encoder.joblib   # LabelEncoder for y
│   ├── feature_encoders.joblib # LabelEncoders for X features
│   └── metadata.json           # Model metadata & metrics
│
├── notebooks\
│   ├── 01_eda.ipynb            # ★ Main pipeline notebook (Steps 1–10)
│   └── 02_modeling.ipynb       # Deep-dive: advanced models & SHAP
│
├── reports\
│   ├── classification_report.txt
│   ├── model_comparison.csv
│   ├── final_predictions.csv
│   └── figures\                # Auto-saved plot PNGs
│
├── src\
│   ├── data.py                 # Data loading & cleaning
│   ├── features.py             # Encoding, scaling, feature selection
│   ├── train.py                # Model training & tuning
│   ├── evaluate.py             # Metrics & visualisation
│   └── predict.py              # Inference helper
│
├── tests\
│   ├── test_data.py
│   ├── test_features.py
│   ├── test_train.py
│   └── test_predict.py
│
├── DATA_CARD.md
├── MODEL_CARD.md
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

```bash
# 1. Clone / navigate to the project
cd E:\CYBERSECURITY

# 2. Create a virtual environment (recommended)
python -m venv .venv
.venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## 🚀 Quick Start

### Run the full pipeline notebook
```bash
jupyter notebook notebooks/01_eda.ipynb
```

### Run the web app (Streamlit)
```bash
streamlit run app/app.py
```

### Run the REST API (Flask)
```bash
python app/api.py
```

### Predict from command line
```python
from src.predict import predict_payloads

results = predict_payloads([
    "<script>alert('XSS')</script>",
    "SELECT * FROM users WHERE id=1 OR 1=1--",
    "Hello, what color should I paint the wall?",
])
print(results)
```

---

## 🤖 ML Pipeline — 10 Steps

```
Data Collection → Data Cleaning → Data Preprocessing →
Train-Test Split → Model Training → Model Evaluation →
Hyperparameter Tuning → Final Prediction → Deployment
```

| Step | Description |
|------|-------------|
| **1. Data Collection** | Load from Hugging Face (16 401 rows) |
| **2. Data Cleaning** | Deduplicate, fill nulls, strip whitespace |
| **3. Preprocessing** | Label-encode, 5 scaling strategies, feature engineering |
| **4. Feature Selection** | Filter (MI), Wrapper (RFE), Embedded (RF importance) |
| **5. Train-Test Split** | 80 : 20 stratified split |
| **6. Model Training** | 7 classifiers: LR, DT, RF, KNN, SVM, NB, MLP |
| **7. Evaluation** | Accuracy, Precision, Recall, F1, Confusion Matrix, ROC |
| **8. HP Tuning** | GridSearchCV + 5-fold Cross-Validation |
| **9. Final Prediction** | Predict on hold-out test set |
| **10. Deployment** | Joblib artifacts + Streamlit app + REST API |

---

## 📊 Results (from notebook run)

| Model               | Accuracy | Precision | Recall | F1    |
|---------------------|----------|-----------|--------|-------|
| Logistic Regression | 1.000    | 1.000     | 1.000  | 1.000 |
| Decision Tree       | 1.000    | 1.000     | 1.000  | 1.000 |
| Random Forest       | 1.000    | 1.000     | 1.000  | 1.000 |
| SVM (RBF)           | 1.000    | 1.000     | 1.000  | 1.000 |
| KNN                 | 1.000    | 1.000     | 1.000  | 1.000 |
| Gaussian NB         | ~0.97    | ~0.97     | ~0.97  | ~0.97 |
| MLP Neural Net      | 1.000    | 1.000     | 1.000  | 1.000 |

> ⚠️ Perfect scores are expected because the dataset contains highly distinctive syntactic patterns for each attack class.

---

## 🧪 Tests

```bash
pytest tests/ -v
```

---

## 📄 License

MIT — see [LICENSE](LICENSE) for details.
