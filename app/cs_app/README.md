# CyberShield (self-contained edition)

A standalone Streamlit app for web-attack (XSS / SQLi / Normal) text
classification. No external dataset or pre-trained model is required —
use the **🧪 Hypothetical Data** page to generate realistic synthetic
payloads (cleaned via the same pipeline as `notebooks/01_eda.ipynb`) and
train a demo model in seconds, directly in the browser session.

## Files
- `app.py` — the Streamlit UI (6 pages: Predict, Hypothetical Data, Dashboard, History, Model Info, About)
- `core.py` — feature engineering, synthetic data generator + cleaner, training, inference
- `api.py` — optional Flask REST API sharing the same `core.py` logic

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy for free (Streamlit Community Cloud)
1. Push this repo to GitHub (see root-level README for exact commands).
2. Go to https://share.streamlit.io → "New app".
3. Pick your repo/branch, set **Main file path** to `app/cs_app/app.py`.
4. Click Deploy. You'll get a public URL like `https://<your-app-name>.streamlit.app`.
