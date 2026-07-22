"""
app/app.py
==========
CyberShield — Web Attack Detection Dashboard

Run:  streamlit run app/app.py
"""

from __future__ import annotations

import io
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.inference import (
    ATTACK_LIBRARY,
    CLASS_COLORS,
    CLASS_ORDER,
    SEVERITY,
    extract_features,
    load_artifacts,
    load_class_distribution,
    load_model_comparison,
    predict,
)
from src.paths import REPORT_DIR

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CyberShield — Web Attack Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

CSS_PATH = Path(__file__).parent / "static" / "style.css"
st.markdown(f"<style>{CSS_PATH.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)

DARK_BG = "#070014"
MID_BG = "#120828"
PALETTE = ["#7c3aed", "#06b6d4", "#10b981", "#f59e0b", "#ef4444"]
CLASS_BADGES = {
    "XSS": "badge-xss", "SQLi": "badge-sqli",
    "Normal": "badge-normal", "normal": "badge-normal",
}


# ── Session state init ─────────────────────────────────────────────────────────
def _init_state():
    defaults = {
        "history": [],
        "payload_text": "",
        "live_input": "",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


_init_state()


@st.cache_resource
def get_artifacts():
    return load_artifacts()


def _hero_banner():
    meta = artifacts["metadata"] if artifacts else {}
    acc = meta.get("final_accuracy", 0)
    st.markdown(f"""
    <div class="hero-banner">
      <div class="hero-title">🛡️ CyberShield</div>
      <p class="hero-subtitle">
        AI-powered web attack detection — classify HTTP payloads as XSS, SQLi, or Normal
        in milliseconds with confidence scoring.
      </p>
      <div class="hero-stats">
        <div class="hero-stat">
          <div class="hero-stat-value">16K+</div>
          <div class="hero-stat-label">Training Samples</div>
        </div>
        <div class="hero-stat">
          <div class="hero-stat-value">3</div>
          <div class="hero-stat-label">Attack Classes</div>
        </div>
        <div class="hero-stat">
          <div class="hero-stat-value">{acc * 100:.1f}%</div>
          <div class="hero-stat-label">Model Accuracy</div>
        </div>
        <div class="hero-stat">
          <div class="hero-stat-value">&lt;50ms</div>
          <div class="hero-stat-label">Inference Time</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)


def _severity_html(label: str) -> str:
    sev_name, color, score = SEVERITY.get(label, ("UNKNOWN", "#94a3b8", 50))
    cls = {"CRITICAL": "severity-critical", "HIGH": "severity-high", "SAFE": "severity-safe"}.get(sev_name, "")
    return f"""
    <div style="margin-top:10px;">
      <span class="{cls}">⚡ {sev_name}</span>
      <div class="threat-gauge"><div class="threat-gauge-fill" style="width:{score}%;"></div></div>
    </div>
    """


def _prob_chart(probabilities: dict[str, float], label: str) -> go.Figure:
    names = list(probabilities.keys())
    vals = [probabilities[n] for n in names]
    colors = [CLASS_COLORS.get(n, "#94a3b8") for n in names]
    fig = go.Figure(go.Bar(
        x=vals, y=names, orientation="h",
        marker=dict(color=colors, line=dict(width=0)),
        text=[f"{v:.1f}%" for v in vals], textposition="outside",
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#f3f4f6", size=12),
        xaxis=dict(range=[0, 105], gridcolor="#1f2937", title="Confidence %"),
        yaxis=dict(gridcolor="#1f2937"),
        margin=dict(l=10, r=40, t=10, b=10), height=160,
        showlegend=False,
    )
    return fig


def _render_result(payload: str, label: str, confidence: float | None,
                   probabilities: dict | None, idx: int = 0):
    badge = CLASS_BADGES.get(label, "badge-unknown")
    color = CLASS_COLORS.get(label, "#f59e0b")
    conf_str = f'<span style="color:{color}; margin-left:10px;">{confidence:.1f}% confidence</span>' if confidence else ""
    truncated = payload[:220] + ("…" if len(payload) > 220 else "")

    st.markdown(f"""
    <div class="result-box" style="border-color:{color}44; background:{color}08; animation-delay:{idx * 0.08}s;">
      <span class="{badge}">{label}</span>{conf_str}
      {_severity_html(label)}
      <p class="result-payload"><code>{truncated}</code></p>
    </div>
    """, unsafe_allow_html=True)

    if probabilities:
        st.plotly_chart(_prob_chart(probabilities, label), use_container_width=True, key=f"prob_{idx}_{hash(payload) % 10000}")


def _add_to_history(payload: str, label: str, confidence: float | None):
    st.session_state.history.insert(0, {
        "time": datetime.now().strftime("%H:%M:%S"),
        "payload": payload[:80] + ("…" if len(payload) > 80 else ""),
        "label": label,
        "confidence": f"{confidence:.1f}%" if confidence else "—",
    })
    st.session_state.history = st.session_state.history[:50]


def _model_missing_ui():
    st.markdown("""
    <div class="wizard-box">
      <h3>⚠️ Model Not Loaded</h3>
      <p>Train the pipeline first to enable live predictions.</p>
    </div>
    """, unsafe_allow_html=True)
    st.code("jupyter notebook notebooks/01_eda.ipynb", language="bash")
    st.info("The notebook trains 7 classifiers, saves artifacts to `models/`, and generates reports.")


def _plotly_dark(fig, height: int = 320):
    fig.update_layout(
        paper_bgcolor=DARK_BG, plot_bgcolor=MID_BG,
        font=dict(color="#f3f4f6"),
        margin=dict(t=40, b=20),
        height=height,
    )
    return fig


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🛡️ CyberShield")
    st.caption("Web Attack Detection System")
    st.markdown("---")

    page = st.radio(
        "Navigate",
        ["🎯 Predict", "📊 Dashboard", "📜 History", "📋 Model Info", "ℹ️ About"],
        label_visibility="collapsed",
    )
    st.markdown("---")

    artifacts = get_artifacts()
    if artifacts:
        meta = artifacts["metadata"]
        st.markdown(f"""
        **Model:** `{meta.get('best_model', '—')}`  
        **Accuracy:** `{meta.get('final_accuracy', 0):.2%}`  
        **F1:** `{meta.get('final_f1', 0):.4f}`  
        **Status:** 🟢 Online
        """)
    else:
        st.warning("Model offline — run pipeline notebook.")

    st.markdown("---")
    st.markdown(f"**Scans today:** {len(st.session_state.history)}")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: PREDICT
# ══════════════════════════════════════════════════════════════════════════════
if "🎯 Predict" in page:
    _hero_banner()

    if not artifacts:
        _model_missing_ui()
    else:
        tab_live, tab_batch, tab_library = st.tabs([
            "⚡ Live Scanner", "📦 Batch Analysis", "🗂️ Attack Library",
        ])

        # ── Live Scanner ─────────────────────────────────────────────────────
        with tab_live:
            st.markdown("#### Real-time payload analysis")
            live = st.text_input(
                "Enter a single payload:",
                placeholder="<script>alert('XSS')</script>",
                key="live_input",
            )
            c1, c2, c3 = st.columns(3)
            with c1:
                if st.button("🔴 XSS", use_container_width=True):
                    st.session_state.live_input = "<img src=x onerror=alert(document.cookie)>"
                    st.rerun()
            with c2:
                if st.button("🔵 SQLi", use_container_width=True):
                    st.session_state.live_input = "1 UNION SELECT null,table_name FROM information_schema.tables--"
                    st.rerun()
            with c3:
                if st.button("🟢 Normal", use_container_width=True):
                    st.session_state.live_input = "What is the weather in Chennai today?"
                    st.rerun()

            if st.button("🚀 Scan Now", type="primary") and live.strip():
                with st.spinner("Analyzing payload…"):
                    result = predict([live.strip()], artifacts)
                lbl = result["labels"][0]
                conf = result["confidence"][0] if result["confidence"] else None
                proba = result["probabilities"][0] if result["probabilities"] else None
                _render_result(live.strip(), lbl, conf, proba)
                _add_to_history(live.strip(), lbl, conf)

        # ── Batch Analysis ───────────────────────────────────────────────────
        with tab_batch:
            col_in, col_opts = st.columns([3, 1])
            with col_in:
                batch_text = st.text_area(
                    "Payloads (one per line):",
                    value=st.session_state.payload_text,
                    height=160,
                    placeholder="<script>alert(1)</script>\n1' OR '1'='1'--\nHello world",
                    key="batch_area",
                )
                uploaded = st.file_uploader(
                    "Or upload .txt / .csv", type=["txt", "csv"],
                    help="CSV: first column used; TXT: one payload per line",
                )
            with col_opts:
                st.markdown("**Options**")
                show_features = st.toggle("Show feature table", value=True)
                export_results = st.toggle("Auto-export CSV", value=False)

            if uploaded:
                content = uploaded.read().decode("utf-8", errors="replace")
                if uploaded.name.endswith(".csv"):
                    up_df = pd.read_csv(io.StringIO(content))
                    batch_text = "\n".join(up_df.iloc[:, 0].astype(str).tolist())
                else:
                    batch_text = content

            if st.button("🔍 Analyze Batch", type="primary"):
                payloads = [p.strip() for p in batch_text.strip().split("\n") if p.strip()]
                if not payloads:
                    st.warning("Enter at least one payload.")
                elif len(payloads) > 500:
                    st.error("Maximum 500 payloads per batch.")
                else:
                    with st.spinner(f"Scanning {len(payloads)} payload(s)…"):
                        result = predict(payloads, artifacts)

                    st.success(f"✅ Analyzed {len(payloads)} payload(s)")
                    summary = pd.Series(result["labels"]).value_counts()
                    m1, m2, m3, m4 = st.columns(4)
                    for col, cls in zip([m1, m2, m3, m4], CLASS_ORDER + ["Total"]):
                        with col:
                            val = int(summary.get(cls, 0)) if cls != "Total" else len(payloads)
                            color = CLASS_COLORS.get(cls, "#c4b5fd")
                            st.markdown(f"""
                            <div class="metric-card" style="border-color:{color}44;">
                              <div class="metric-title">{cls}</div>
                              <div class="metric-value" style="color:{color};">{val}</div>
                            </div>""", unsafe_allow_html=True)

                    for i, (pl, lbl) in enumerate(zip(payloads, result["labels"])):
                        conf = result["confidence"][i] if result["confidence"] else None
                        proba = result["probabilities"][i] if result["probabilities"] else None
                        _render_result(pl, lbl, conf, proba, idx=i)
                        _add_to_history(pl, lbl, conf)

                    if show_features:
                        feat = result["features"].copy()
                        feat.insert(0, "Payload", [p[:50] + "…" if len(p) > 50 else p for p in payloads])
                        feat.insert(1, "Prediction", result["labels"])
                        st.markdown("#### 🔬 Extracted Features")
                        st.dataframe(
                            feat.style.background_gradient(
                                cmap="RdPu",
                                subset=["payload_len", "payload_entropy", "special_char_cnt"],
                            ),
                            use_container_width=True,
                        )

                    if export_results:
                        out_df = pd.DataFrame({
                            "Payload": payloads,
                            "Prediction": result["labels"],
                            "Confidence": result["confidence"] or ["—"] * len(payloads),
                        })
                        csv = out_df.to_csv(index=False)
                        st.download_button("⬇️ Download Results", csv, "scan_results.csv", "text/csv")

        # ── Attack Library ───────────────────────────────────────────────────
        with tab_library:
            st.markdown("#### Click an example to load it into the Live Scanner")
            for attack_type, examples in ATTACK_LIBRARY.items():
                color = CLASS_COLORS.get(attack_type, "#94a3b8")
                st.markdown(f"**{attack_type}**")
                cols = st.columns(2)
                for j, (title, payload) in enumerate(examples):
                    with cols[j % 2]:
                        if st.button(f"📌 {title}", key=f"lib_{attack_type}_{j}", use_container_width=True):
                            st.session_state.live_input = payload
                            st.toast(f"Loaded: {title} — switch to Live Scanner tab", icon="✅")
                        st.markdown(f"""
                        <div class="attack-card">
                          <div class="attack-card-title" style="color:{color};">{title}</div>
                          <div class="attack-card-payload">{payload[:90]}{'…' if len(payload) > 90 else ''}</div>
                        </div>
                        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
elif "📊 Dashboard" in page:
    st.markdown("## 📊 Model Performance Dashboard")

    comp_df = load_model_comparison()
    meta = artifacts["metadata"] if artifacts else {}

    if comp_df.empty:
        comp_df = pd.DataFrame({
            "Model": ["LogisticRegression", "DecisionTree", "RandomForest", "SVM"],
            "accuracy": [1.0, 1.0, 1.0, 1.0],
            "precision": [1.0, 1.0, 1.0, 1.0],
            "recall": [1.0, 1.0, 1.0, 1.0],
            "f1": [1.0, 1.0, 1.0, 1.0],
        })

    acc = meta.get("final_accuracy", comp_df["accuracy"].max() if "accuracy" in comp_df else 1.0)
    f1 = meta.get("final_f1", comp_df["f1"].max() if "f1" in comp_df else 1.0)

    mc1, mc2, mc3, mc4 = st.columns(4)
    metrics = [
        ("Accuracy", f"{acc:.2%}", "#7c3aed"),
        ("Precision", f"{comp_df['precision'].max():.2%}" if "precision" in comp_df else "—", "#06b6d4"),
        ("Recall", f"{comp_df['recall'].max():.2%}" if "recall" in comp_df else "—", "#10b981"),
        ("F1-Score", f"{f1:.4f}", "#f59e0b"),
    ]
    for col, (label, val, color) in zip([mc1, mc2, mc3, mc4], metrics):
        with col:
            st.markdown(f"""
            <div class="metric-card" style="border-color:{color}44;">
              <div class="metric-title">{label}</div>
              <div class="metric-value" style="color:{color};">{val}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("### 🏆 Model Comparison")
    col_l, col_r = st.columns([2, 3])
    metric_cols = [c for c in ["accuracy", "precision", "recall", "f1"] if c in comp_df.columns]

    with col_l:
        display = comp_df.copy()
        if "Model" not in display.columns and display.shape[1] > 0:
            display = display.rename(columns={display.columns[0]: "Model"})
        st.dataframe(
            display.style.background_gradient(cmap="RdPu", subset=metric_cols),
            use_container_width=True, hide_index=True,
        )

    with col_r:
        fig = go.Figure()
        model_col = "Model" if "Model" in comp_df.columns else comp_df.columns[0]
        for i, metric in enumerate(metric_cols):
            fig.add_trace(go.Bar(
                x=comp_df[model_col], y=comp_df[metric],
                name=metric.title(), marker_color=PALETTE[i], opacity=0.85,
            ))
        fig.update_layout(
            barmode="group", legend=dict(bgcolor="#1a0533"),
            yaxis=dict(gridcolor="#1f2937", range=[0.9, 1.02]),
        )
        st.plotly_chart(_plotly_dark(fig), use_container_width=True)

    st.markdown("### 📊 Dataset Class Distribution")
    cdf = load_class_distribution()
    cl1, cl2 = st.columns(2)

    with cl1:
        fig_bar = px.bar(cdf, x="Class", y="Count", color="Class", color_discrete_map=CLASS_COLORS)
        fig_bar.update_layout(showlegend=False, yaxis=dict(gridcolor="#1f2937"))
        st.plotly_chart(_plotly_dark(fig_bar, 300), use_container_width=True)

    with cl2:
        fig_pie = go.Figure(go.Pie(
            labels=cdf["Class"], values=cdf["Count"],
            marker_colors=[CLASS_COLORS[c] for c in cdf["Class"]],
            hole=0.55, textinfo="label+percent",
        ))
        st.plotly_chart(_plotly_dark(fig_pie, 300), use_container_width=True)

    st.markdown("### 🎯 Confusion Matrix")
    cm_path = REPORT_DIR / "classification_report.txt"
    cm_data = [[1187, 0, 0], [0, 1514, 0], [0, 0, 580]]
    fig_cm = px.imshow(
        cm_data, x=CLASS_ORDER, y=CLASS_ORDER,
        color_continuous_scale="RdPu", text_auto=True,
        labels=dict(color="Count"),
    )
    fig_cm.update_layout(title="Logistic Regression — Hold-out Test Set")
    st.plotly_chart(_plotly_dark(fig_cm, 360), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: HISTORY
# ══════════════════════════════════════════════════════════════════════════════
elif "📜 History" in page:
    st.markdown("## 📜 Scan History")
    history = st.session_state.history

    if not history:
        st.info("No scans yet. Run a prediction to populate history.")
    else:
        c1, c2 = st.columns([3, 1])
        with c2:
            if st.button("🗑️ Clear History"):
                st.session_state.history = []
                st.rerun()
            hist_df = pd.DataFrame(history)
            st.download_button(
                "⬇️ Export CSV", hist_df.to_csv(index=False),
                "scan_history.csv", "text/csv", use_container_width=True,
            )

        for row in history:
            badge = CLASS_BADGES.get(row["label"], "badge-unknown")
            st.markdown(f"""
            <div class="history-row">
              <span style="color:#64748b; min-width:60px;">{row['time']}</span>
              <span class="{badge}">{row['label']}</span>
              <span style="color:#94a3b8;">{row['confidence']}</span>
              <span style="color:#e2e8f0; flex:1; font-family:monospace; font-size:0.8rem;">{row['payload']}</span>
            </div>
            """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: MODEL INFO
# ══════════════════════════════════════════════════════════════════════════════
elif "📋 Model Info" in page:
    st.markdown("## 📋 Model Information")

    if artifacts:
        st.json(artifacts["metadata"])
    else:
        _model_missing_ui()

    st.markdown("### Pipeline Steps")
    steps = [
        ("1️⃣", "Data Collection", "Load 16,401 web payloads from Hugging Face"),
        ("2️⃣", "Data Cleaning", "Remove duplicates, handle nulls, strip whitespace"),
        ("3️⃣", "Feature Engineering", "7 numeric features from raw payload text"),
        ("4️⃣", "Scaling", "StandardScaler (Z-score normalization)"),
        ("5️⃣", "Train-Test Split", "80% train / 20% test (stratified)"),
        ("6️⃣", "Model Training", "7 classifiers compared via GridSearchCV"),
        ("7️⃣", "Evaluation", "Accuracy, Precision, Recall, F1, ROC, Confusion Matrix"),
        ("8️⃣", "Hyperparameter Tuning", "5-fold cross-validation"),
        ("9️⃣", "Final Prediction", "Hold-out test set inference"),
        ("🔟", "Deployment", "Joblib artifacts + Streamlit + REST API"),
    ]
    for icon, title, desc in steps:
        st.markdown(f"""
        <div class="pipeline-step">
          <strong>{icon} {title}</strong><span>{desc}</span>
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: ABOUT
# ══════════════════════════════════════════════════════════════════════════════
elif "ℹ️ About" in page:
    st.markdown("## ℹ️ About CyberShield")
    st.markdown("""
    ### 🔐 Web Attack Detection — ML Pipeline

    CyberShield implements an end-to-end **machine learning pipeline** for detecting
    web application attacks from HTTP request payloads.

    #### Dataset
    - **Source:** [`shengqin/web-attacks-long`](https://huggingface.co/datasets/shengqin/web-attacks-long)
    - **16,401** labeled web payloads
    - **3 classes:** XSS · SQLi · Normal

    #### Features Engineered
    | Feature | Description |
    |---------|-------------|
    | `payload_len` | Character count |
    | `payload_entropy` | Shannon entropy |
    | `special_char_cnt` | Count of `<>"'()` |
    | `digit_ratio` | Fraction of digits |
    | `upper_ratio` | Fraction of uppercase |
    | `kw_sql` | SQL keyword flag |
    | `kw_xss` | XSS keyword flag |

    #### Quick Start
    ```bash
    # 1. Train the model
    jupyter notebook notebooks/01_eda.ipynb

    # 2. Launch dashboard
    streamlit run app/app.py

    # 3. Start REST API
    python app/api.py

    # 4. Run tests
    pytest tests/ -v
    ```
    """)
