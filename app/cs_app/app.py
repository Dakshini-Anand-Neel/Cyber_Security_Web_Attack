"""
app.py
======
CyberShield — Web Attack Detection Dashboard (self-contained edition)

Run:
    pip install streamlit pandas numpy plotly scikit-learn joblib
    streamlit run app.py
"""

from __future__ import annotations

import io
from datetime import datetime

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from core import (
    CLASS_ORDER, CLASS_COLORS, CLASS_BADGES, SEVERITY, ATTACK_LIBRARY,
    generate_synthetic_dataset, train_model, predict, try_load_real_model,
)


st.set_page_config(
    page_title="CyberShield — Web Attack Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.hero-banner { padding: 1.5rem; border-radius: 14px;
  background: linear-gradient(135deg, #120828, #1e0a3c); margin-bottom: 1rem; }
.hero-title { font-size: 2rem; font-weight: 800; }
.hero-subtitle { color: #c4b5fd; max-width: 700px; }
.result-box { border: 1px solid; border-radius: 12px; padding: 1rem; margin: 0.6rem 0; }
.result-payload { font-family: monospace; color: #e2e8f0; word-break: break-all; }
.badge-xss, .badge-sqli, .badge-normal, .badge-unknown {
  padding: 3px 10px; border-radius: 999px; font-weight: 700; font-size: 0.8rem; }
.badge-xss { background:#ef444422; color:#ef4444; }
.badge-sqli { background:#7c3aed22; color:#7c3aed; }
.badge-normal { background:#10b98122; color:#10b981; }
.badge-unknown { background:#94a3b822; color:#94a3b8; }
.severity-critical { color:#ef4444; font-weight:700; }
.severity-high { color:#f59e0b; font-weight:700; }
.severity-safe { color:#10b981; font-weight:700; }
.threat-gauge { background:#1f2937; border-radius:6px; height:8px; margin-top:6px; overflow:hidden; }
.threat-gauge-fill { background:linear-gradient(90deg,#10b981,#f59e0b,#ef4444); height:100%; }
.metric-card { border:1px solid; border-radius:12px; padding:0.9rem; text-align:center; }
.metric-title { color:#94a3b8; font-size:0.85rem; }
.metric-value { font-size:1.5rem; font-weight:800; }
.attack-card { border:1px solid #ffffff22; border-radius:10px; padding:0.6rem; margin-bottom:0.6rem; }
.attack-card-title { font-weight:700; }
.attack-card-payload { font-family:monospace; color:#94a3b8; font-size:0.8rem; }
.history-row { display:flex; gap:10px; align-items:center; padding:6px 0; border-bottom:1px solid #ffffff11; }
.pipeline-step { border-left:3px solid #7c3aed; padding:6px 12px; margin-bottom:6px; }
.pipeline-step span { color:#94a3b8; margin-left:8px; }
.wizard-box { border:1px dashed #7c3aed77; border-radius:12px; padding:1rem; text-align:center; }
.explain-box { background:#0891b214; border-left:3px solid #06b6d4; border-radius:6px;
  padding:0.7rem 1rem; margin:0.6rem 0; color:#cffafe; }
</style>
""", unsafe_allow_html=True)


def _init_state():
    defaults = {
        "history": [],
        "live_input": "",
        "payload_text": "",
        "trained_bundle": None,
        "mode": "Beginner",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


_init_state()


@st.cache_resource
def get_real_artifacts():
    return try_load_real_model()


def explain(text: str):
    if st.session_state.mode == "Beginner":
        st.markdown(f'<div class="explain-box">💡 {text}</div>', unsafe_allow_html=True)


def expert_detail(md_text: str):
    with st.expander("🔬 Technical details"):
        st.markdown(md_text)


def _load_example(payload: str, toast_msg: str | None = None):
    st.session_state.live_input = payload
    if toast_msg:
        st.session_state["_pending_toast"] = toast_msg


real_artifacts = get_real_artifacts()
active_model = None
active_metadata = {}
model_source = None

if real_artifacts:
    active_model = real_artifacts["model"]
    active_metadata = real_artifacts["metadata"]
    model_source = f"Real model loaded from {real_artifacts['source']}"
elif st.session_state.trained_bundle:
    active_model = st.session_state.trained_bundle["model"]
    active_metadata = st.session_state.trained_bundle["metadata"]
    model_source = "Synthetic (hypothetical-data) model trained this session"


def _severity_html(label: str) -> str:
    sev_name, color, score = SEVERITY.get(label, ("UNKNOWN", "#94a3b8", 50))
    cls = {"CRITICAL": "severity-critical", "HIGH": "severity-high", "SAFE": "severity-safe"}.get(sev_name, "")
    return f"""
    <div style="margin-top:10px;">
      <span class="{cls}">⚡ {sev_name}</span>
      <div class="threat-gauge"><div class="threat-gauge-fill" style="width:{score}%;"></div></div>
    </div>
    """


def _prob_chart(probabilities: dict, label: str) -> go.Figure:
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


def _render_result(payload: str, label: str, confidence, probabilities, idx: int = 0):
    badge = CLASS_BADGES.get(label, "badge-unknown")
    color = CLASS_COLORS.get(label, "#f59e0b")
    conf_str = f'<span style="color:{color}; margin-left:10px;">{confidence:.1f}% confidence</span>' if confidence else ""
    truncated = payload[:220] + ("…" if len(payload) > 220 else "")

    st.markdown(f"""
    <div class="result-box" style="border-color:{color}44; background:{color}08;">
      <span class="{badge}">{label}</span>{conf_str}
      {_severity_html(label)}
      <p class="result-payload"><code>{truncated}</code></p>
    </div>
    """, unsafe_allow_html=True)

    if probabilities:
        st.plotly_chart(_prob_chart(probabilities, label), use_container_width=True,
                         key=f"prob_{idx}_{abs(hash(payload)) % 10000}")


def _add_to_history(payload: str, label: str, confidence):
    st.session_state.history.insert(0, {
        "time": datetime.now().strftime("%H:%M:%S"),
        "payload": payload[:80] + ("…" if len(payload) > 80 else ""),
        "label": label,
        "confidence": f"{confidence:.1f}%" if confidence else "—",
    })
    st.session_state.history = st.session_state.history[:50]


def _plotly_dark(fig, height: int = 320):
    fig.update_layout(
        paper_bgcolor="#070014", plot_bgcolor="#120828",
        font=dict(color="#f3f4f6"),
        margin=dict(t=40, b=20),
        height=height,
    )
    return fig


with st.sidebar:
    st.markdown("## 🛡️ CyberShield")
    st.caption("Web Attack Detection System")

    st.session_state.mode = st.radio(
        "Reading level",
        ["Beginner", "Expert"],
        horizontal=True,
        index=0 if st.session_state.mode == "Beginner" else 1,
        help="Beginner adds plain-language explanations. Expert keeps things compact.",
    )

    st.markdown("---")
    page = st.radio(
        "Navigate",
        ["🎯 Predict", "🧪 Hypothetical Data", "📊 Dashboard", "📜 History", "📋 Model Info", "ℹ️ About"],
        label_visibility="collapsed",
    )
    st.markdown("---")

    if active_model:
        st.success("🟢 Model online")
        st.markdown(f"""
        **Model:** `{active_metadata.get('best_model', '—')}`
        **Accuracy:** `{active_metadata.get('final_accuracy', 0):.2%}`
        **F1:** `{active_metadata.get('final_f1', 0):.4f}`
        """)
        if active_metadata.get("cv_f1_mean") is not None:
            st.caption(f"5-fold CV F1: {active_metadata['cv_f1_mean']:.2%} ± {active_metadata['cv_f1_std']:.2%}")
        st.caption(model_source)
    else:
        st.warning("🟡 No model loaded yet")
        st.caption("Go to '🧪 Hypothetical Data' to generate example data and train one in seconds.")

    st.markdown("---")
    st.markdown(f"**Scans this session:** {len(st.session_state.history)}")


if "🎯 Predict" in page:
    st.markdown("""
    <div class="hero-banner">
      <div class="hero-title">🛡️ CyberShield</div>
      <p class="hero-subtitle">
        Paste any piece of text — a search query, a form field, a URL parameter —
        and this tool guesses whether it looks like a normal message, a Cross-Site
        Scripting (XSS) attack, or a SQL Injection (SQLi) attack.
      </p>
    </div>
    """, unsafe_allow_html=True)

    explain(
        "<b>XSS</b> and <b>SQLi</b> are two common ways attackers try to break websites: "
        "XSS sneaks a bit of script into a page so it runs in someone else's browser; "
        "SQLi sneaks database commands into a text field to trick the database into "
        "leaking or changing data. This page shows you what those look like and lets "
        "you test your own examples."
    )

    if not active_model:
        st.markdown("""
        <div class="wizard-box">
          <h3>⚠️ No model loaded yet</h3>
          <p>You don't need a real dataset to try this out.</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("🧪 Generate hypothetical data & train a demo model now", type="primary"):
            with st.spinner("Inventing realistic example payloads and training a model…"):
                df_synth = generate_synthetic_dataset()
                bundle = train_model(df_synth)
            st.session_state.trained_bundle = bundle
            st.rerun()
    else:
        tab_live, tab_batch, tab_library = st.tabs([
            "⚡ Live Scanner", "📦 Batch Analysis", "🗂️ Attack Library",
        ])

        with tab_live:
            st.markdown("#### Real-time payload analysis")
            live = st.text_input(
                "Enter a single payload:",
                placeholder="<script>alert('XSS')</script>",
                key="live_input",
            )
            c1, c2, c3 = st.columns(3)
            with c1:
                st.button("🔴 Try an XSS example", use_container_width=True,
                          on_click=_load_example,
                          args=("<img src=x onerror=alert(document.cookie)>",))
            with c2:
                st.button("🔵 Try a SQLi example", use_container_width=True,
                          on_click=_load_example,
                          args=("1 UNION SELECT null,table_name FROM information_schema.tables--",))
            with c3:
                st.button("🟢 Try a normal example", use_container_width=True,
                          on_click=_load_example,
                          args=("What is the weather in Chennai today?",))

            if st.button("🚀 Scan Now", type="primary") and live.strip():
                with st.spinner("Analyzing payload…"):
                    result = predict([live.strip()], active_model)
                lbl = result["labels"][0]
                conf = result["confidence"][0] if result["confidence"] else None
                proba = result["probabilities"][0] if result["probabilities"] else None
                _render_result(live.strip(), lbl, conf, proba)
                _add_to_history(live.strip(), lbl, conf)
                explain(
                    "The confidence bars show how sure the model is about each "
                    "possible label. Higher isn't always right — always double-check "
                    "anything flagged as critical before acting on it."
                )

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
                        result = predict(payloads, active_model)

                    st.success(f"✅ Analyzed {len(payloads)} payload(s)")
                    summary = pd.Series(result["labels"]).value_counts()
                    cols = st.columns(len(CLASS_ORDER) + 1)
                    for col, cls in zip(cols, CLASS_ORDER + ["Total"]):
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
                        explain(
                            "These numbers are what the model actually looks at — things "
                            "like text length, how many odd symbols like `<` or `'` appear, "
                            "and whether SQL/script keywords show up."
                        )
                        st.dataframe(feat, use_container_width=True)

                    if export_results:
                        out_df = pd.DataFrame({
                            "Payload": payloads,
                            "Prediction": result["labels"],
                            "Confidence": result["confidence"] or ["—"] * len(payloads),
                        })
                        csv = out_df.to_csv(index=False)
                        st.download_button("⬇️ Download Results", csv, "scan_results.csv", "text/csv")

        with tab_library:
            st.markdown("#### Click an example to load it into the Live Scanner")
            for attack_type, examples in ATTACK_LIBRARY.items():
                color = CLASS_COLORS.get(attack_type, "#94a3b8")
                st.markdown(f"**{attack_type}**")
                cols = st.columns(2)
                for j, (title, payload) in enumerate(examples):
                    with cols[j % 2]:
                        st.button(
                            f"📌 {title}", key=f"lib_{attack_type}_{j}",
                            use_container_width=True,
                            on_click=_load_example,
                            args=(payload, f"Loaded: {title} — switch to Live Scanner tab"),
                        )
                        st.markdown(f"""
                        <div class="attack-card">
                          <div class="attack-card-title" style="color:{color};">{title}</div>
                          <div class="attack-card-payload">{payload[:90]}{'…' if len(payload) > 90 else ''}</div>
                        </div>
                        """, unsafe_allow_html=True)


elif "🧪 Hypothetical Data" in page:
    st.markdown("## 🧪 Hypothetical Data Generator")
    explain(
        "You don't need a real attack dataset to explore this app. Click the button "
        "below and the app will invent realistic-looking XSS, SQLi, and normal text "
        "examples, run them through the SAME cleaning steps as the project notebook "
        "(duplicate removal, missing-value handling, whitespace stripping, constant-"
        "column dropping), then train a small model on them — all in your browser, "
        "in a few seconds."
    )
    expert_detail(
        "Synthetic examples are drawn from 12–14 distinct templates per class "
        "(covering obfuscation like URL-encoding, mixed casing, blind/time-based SQLi, "
        "and event-handler variety for XSS), plus deliberately tricky Normal-class "
        "sentences that contain attack-adjacent words (`select`, `union`, `cookie`) to "
        "create realistic class overlap. The generated frame is passed through "
        "`core.clean_dataframe()`, which mirrors `notebooks/01_eda.ipynb` Step 2 "
        "(dedupe → missing-value fill → whitespace strip → constant-column drop) before "
        "any features are engineered. A configurable fraction of labels is randomly "
        "flipped (`label_noise`, default 3%) to simulate real annotation noise. The "
        "classifier is a depth-capped `RandomForestClassifier` evaluated with stratified "
        "cross-validation, not just a single train/test split."
    )

    c1, c2 = st.columns(2)
    with c1:
        n_per_class = st.slider("Examples per class", 50, 1000, 250, step=50)
    with c2:
        noise_pct = st.slider("Label noise (realism)", 0, 15, 3, step=1,
                               help="Percent of labels randomly flipped to simulate real-world annotation noise.")

    if st.button("🎲 Generate & Train", type="primary"):
        with st.spinner("Generating hypothetical data, cleaning it, and training…"):
            df_synth = generate_synthetic_dataset(n_per_class=n_per_class, label_noise=noise_pct / 100)
            bundle = train_model(df_synth)
        st.session_state.trained_bundle = bundle
        st.success(f"Trained on {len(df_synth)} synthetic examples (after cleaning).")
        st.rerun()

    if st.session_state.trained_bundle:
        bundle = st.session_state.trained_bundle
        st.markdown("### Preview of generated & cleaned data")
        preview_df = generate_synthetic_dataset(n_per_class=5)
        st.dataframe(preview_df, use_container_width=True)

        st.markdown("### Training result")
        m = bundle["metrics"]
        c1, c2, c3, c4 = st.columns(4)
        for col, (name, val) in zip([c1, c2, c3, c4], m.items()):
            with col:
                st.markdown(f"""
                <div class="metric-card" style="border-color:#7c3aed44;">
                  <div class="metric-title">{name.title()}</div>
                  <div class="metric-value" style="color:#7c3aed;">{val:.2%}</div>
                </div>""", unsafe_allow_html=True)
        st.caption(
            f"Cross-validation F1: {bundle['metadata']['cv_f1_mean']:.2%} "
            f"± {bundle['metadata']['cv_f1_std']:.2%} — a single train/test split can get lucky; "
            f"cross-validation gives a steadier, more trustworthy number."
        )


elif "📊 Dashboard" in page:
    st.markdown("## 📊 Model Performance Dashboard")

    if not active_model:
        st.info("No model loaded yet. Visit '🧪 Hypothetical Data' to train one, or place a "
                 "real trained model at `models/final_model.joblib` (relative to the project root).")
    else:
        bundle = st.session_state.trained_bundle
        acc = active_metadata.get("final_accuracy")
        f1 = active_metadata.get("final_f1")

        if acc is not None and f1 is not None:
            mc1, mc2 = st.columns(2)
            for col, (label, val, color) in zip(
                [mc1, mc2],
                [("Accuracy", acc, "#7c3aed"), ("F1-Score", f1, "#f59e0b")],
            ):
                with col:
                    st.markdown(f"""
                    <div class="metric-card" style="border-color:{color}44;">
                      <div class="metric-title">{label}</div>
                      <div class="metric-value" style="color:{color};">{val:.2%}</div>
                    </div>""", unsafe_allow_html=True)

        if bundle:
            st.markdown("### 📊 Class Distribution (synthetic training data)")
            dist = bundle["class_distribution"]
            fig_bar = px.bar(x=dist.index, y=dist.values, color=dist.index,
                              color_discrete_map=CLASS_COLORS,
                              labels={"x": "Class", "y": "Count"})
            fig_bar.update_layout(showlegend=False, yaxis=dict(gridcolor="#1f2937"))
            st.plotly_chart(_plotly_dark(fig_bar, 280), use_container_width=True)

            st.markdown("### 🎯 Confusion Matrix (computed from the hold-out test set)")
            explain(
                "Each row is what the true label actually was; each column is what the "
                "model guessed. Numbers on the diagonal are correct guesses — everything "
                "off the diagonal is a mistake. A few mistakes is normal and expected."
            )
            labels = bundle["labels"]
            fig_cm = px.imshow(
                bundle["confusion_matrix"], x=labels, y=labels,
                color_continuous_scale="RdPu", text_auto=True,
                labels=dict(color="Count", x="Predicted", y="Actual"),
            )
            st.plotly_chart(_plotly_dark(fig_cm, 340), use_container_width=True)

            col_a, col_b = st.columns(2)

            with col_a:
                st.markdown("### 📈 ROC Curves")
                explain("A curve hugging the top-left corner means the model separates "
                        "that class from the rest almost perfectly. AUC of 1.0 is perfect; 0.5 is a coin flip.")
                fig_roc = go.Figure()
                for cls, d in bundle["roc_data"].items():
                    fig_roc.add_trace(go.Scatter(
                        x=d["fpr"], y=d["tpr"], mode="lines",
                        name=f"{cls} (AUC={d['auc']:.3f})",
                        line=dict(color=CLASS_COLORS.get(cls, "#94a3b8")),
                    ))
                fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                                              line=dict(dash="dash", color="#475569"), name="Random"))
                fig_roc.update_layout(
                    xaxis=dict(title="False Positive Rate", gridcolor="#1f2937"),
                    yaxis=dict(title="True Positive Rate", gridcolor="#1f2937"),
                    legend=dict(bgcolor="#1a0533"),
                )
                st.plotly_chart(_plotly_dark(fig_roc, 340), use_container_width=True)

            with col_b:
                st.markdown("### 📉 Precision-Recall Curves")
                explain("Shows the trade-off between catching real attacks (recall) and "
                        "not crying wolf on normal traffic (precision).")
                fig_pr = go.Figure()
                for cls, d in bundle["pr_data"].items():
                    fig_pr.add_trace(go.Scatter(
                        x=d["recall"], y=d["precision"], mode="lines", name=cls,
                        line=dict(color=CLASS_COLORS.get(cls, "#94a3b8")),
                    ))
                fig_pr.update_layout(
                    xaxis=dict(title="Recall", gridcolor="#1f2937", range=[0, 1.02]),
                    yaxis=dict(title="Precision", gridcolor="#1f2937", range=[0, 1.02]),
                    legend=dict(bgcolor="#1a0533"),
                )
                st.plotly_chart(_plotly_dark(fig_pr, 340), use_container_width=True)

            col_c, col_d = st.columns(2)

            with col_c:
                st.markdown("### 🧩 Feature Importance")
                explain("Which signals the model actually relies on most to make a call.")
                fi = bundle["feature_importance"]
                fig_fi = px.bar(x=fi.values, y=fi.index, orientation="h",
                                 labels={"x": "Importance", "y": "Feature"},
                                 color=fi.values, color_continuous_scale="Purples")
                fig_fi.update_layout(showlegend=False, coloraxis_showscale=False,
                                      yaxis=dict(gridcolor="#1f2937", autorange="reversed"),
                                      xaxis=dict(gridcolor="#1f2937"))
                st.plotly_chart(_plotly_dark(fig_fi, 320), use_container_width=True)

            with col_d:
                st.markdown("### 🌀 2D Feature Map (PCA)")
                explain("Each dot is one payload, projected down to 2 dimensions. "
                        "Tight, separate clusters mean the classes are easy to tell apart; "
                        "overlap means some payloads genuinely look ambiguous.")
                pca_df = bundle["pca_df"]
                fig_pca = px.scatter(pca_df, x="PC1", y="PC2", color="label",
                                      color_discrete_map=CLASS_COLORS, opacity=0.7)
                fig_pca.update_layout(legend=dict(bgcolor="#1a0533"),
                                       xaxis=dict(gridcolor="#1f2937"), yaxis=dict(gridcolor="#1f2937"))
                st.plotly_chart(_plotly_dark(fig_pca, 320), use_container_width=True)

            st.markdown("### 🔁 Cross-Validation Stability")
            explain("Instead of trusting one lucky train/test split, the model is "
                    "retrained on several different slices of data. Consistent bars mean stable performance.")
            cv_scores = bundle["cv_scores"]
            fig_cv = px.bar(x=[f"Fold {i+1}" for i in range(len(cv_scores))], y=cv_scores,
                             labels={"x": "", "y": "F1 Score"}, color=cv_scores,
                             color_continuous_scale="Tealgrn", range_y=[0, 1.05])
            fig_cv.update_layout(showlegend=False, coloraxis_showscale=False,
                                  yaxis=dict(gridcolor="#1f2937"))
            st.plotly_chart(_plotly_dark(fig_cv, 280), use_container_width=True)
        else:
            st.info("Dashboard charts for a real, on-disk model will appear here once one is loaded. "
                     "Train a synthetic model on the Hypothetical Data page to see live charts now.")


elif "📜 History" in page:
    st.markdown("## 📜 Scan History")
    history = st.session_state.history

    if not history:
        st.info("No scans yet. Run a prediction on the Predict page to populate history.")
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

        hist_df_plot = pd.DataFrame(history)
        if len(hist_df_plot) > 1:
            counts = hist_df_plot["label"].value_counts()
            fig_hist = px.pie(values=counts.values, names=counts.index,
                               color=counts.index, color_discrete_map=CLASS_COLORS, hole=0.5)
            st.plotly_chart(_plotly_dark(fig_hist, 260), use_container_width=True)

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


elif "📋 Model Info" in page:
    st.markdown("## 📋 Model Information")

    if active_metadata:
        st.json(active_metadata)
    else:
        st.markdown("""
        <div class="wizard-box">
          <h3>⚠️ Model Not Loaded</h3>
          <p>Visit '🧪 Hypothetical Data' to generate example data and train one instantly.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### Pipeline Steps")
    steps = [
        ("1️⃣", "Data Collection", "Load real payloads, or generate realistic hypothetical ones on the fly"),
        ("2️⃣", "Data Cleaning", "Remove duplicates, handle nulls, strip whitespace, drop constant columns"),
        ("3️⃣", "Feature Engineering", "7 numeric features extracted from raw payload text"),
        ("4️⃣", "Train-Test Split", "80% train / 20% test (stratified)"),
        ("5️⃣", "Model Training", "Depth-capped RandomForestClassifier (150 trees, max_depth=6)"),
        ("6️⃣", "Cross-Validation", "Stratified CV for a stable performance estimate"),
        ("7️⃣", "Evaluation", "Accuracy, Precision, Recall, F1, ROC/AUC, PR curves, Confusion Matrix"),
        ("8️⃣", "Live Inference", "Type or paste any text on the Predict page"),
        ("9️⃣", "Deployment", "This Streamlit app, or the Flask REST API (api.py)"),
    ]
    for icon, title, desc in steps:
        st.markdown(f"""
        <div class="pipeline-step">
          <strong>{icon} {title}</strong><span>{desc}</span>
        </div>
        """, unsafe_allow_html=True)


elif "ℹ️ About" in page:
    st.markdown("## ℹ️ About CyberShield")
    st.markdown("""
    ### 🔐 Web Attack Detection — plain-language overview

    CyberShield looks at a short piece of text (like something typed into a
    search box or web form) and guesses whether it's:

    - **Normal** — an everyday message, harmless
    - **XSS (Cross-Site Scripting)** — text designed to run a hidden script in
      someone's browser, often to steal login info
    - **SQLi (SQL Injection)** — text designed to trick a database into running
      commands it shouldn't, often to steal or delete data

    #### How it decides
    It doesn't read the text like a person would. It counts things like: how long
    is it, how many odd symbols (`<`, `'`, `--`) does it contain, does it include
    words like `SELECT` or `script`. Those counts go into a model that has seen
    many labeled examples before and learned which patterns go with which label.

    #### Why isn't accuracy 100%?
    It shouldn't be. Real attack text is messy and sometimes looks a lot like
    normal text (a support message that happens to mention "cookie" or "select"
    isn't an attack). A model that claims perfect accuracy on realistic data is
    usually a sign something's wrong — either the data is too easy/templated, or
    the model has memorized rather than learned. This app's synthetic data is
    built to include exactly that kind of overlap, and the numbers you see
    reflect it honestly.
    """)

    expert_detail("""
**Features extracted per payload:**

| Feature | Description |
|---|---|
| `payload_len` | Character count |
| `payload_entropy` | Shannon entropy |
| `special_char_cnt` | Count of `<>"'()=;-%` |
| `digit_ratio` | Fraction of characters that are digits |
| `upper_ratio` | Fraction of characters that are uppercase |
| `kw_sql` | Binary flag for SQL keywords (`select`, `union`, `sleep(`, …) |
| `kw_xss` | Binary flag for XSS keywords (`<script`, `onerror`, `%3cscript`, …) |

**Model:** `RandomForestClassifier(n_estimators=150, max_depth=6, min_samples_leaf=3)`,
evaluated with an 80/20 stratified split plus stratified cross-validation.
Depth and leaf-size caps are deliberate regularization to prevent memorizing the
synthetic templates. When a real trained model exists at `models/final_model.joblib`
(project root), it's loaded automatically and takes priority over the synthetic model.
""")

    st.markdown("""
    #### Quick Start
    ```bash
    pip install streamlit pandas numpy plotly scikit-learn joblib
    streamlit run app.py
    ```
    No dataset or pre-trained model required — use the **Hypothetical Data** page
    to generate one on the spot.
    """)
