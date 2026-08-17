"""
Synthetic Identity Fraud Detection - Research Portal & Interactive Web Dashboard
Run with: python -m streamlit run app/dashboard.py
"""
import streamlit as st
import joblib
import numpy as np
import pandas as pd
import os
import json

st.set_page_config(
    page_title="Synthetic Identity Fraud Detection Research Portal",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE, "model", "fraud_model.joblib")
FEAT_PATH = os.path.join(BASE, "model", "feature_cols.joblib")
METRICS_PATH = os.path.join(BASE, "model", "metrics.json")
DATA_PATH = os.path.join(BASE, "data", "synthetic_kyc_behavioral.csv")

@st.cache_resource
def load_model_assets():
    model = joblib.load(MODEL_PATH)
    feature_cols = joblib.load(FEAT_PATH)
    metrics = {}
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH, "r") as f:
            metrics = json.load(f)
    return model, feature_cols, metrics

@st.cache_data
def load_dataset():
    if os.path.exists(DATA_PATH):
        return pd.read_csv(DATA_PATH)
    return None

model, feature_cols, metrics = load_model_assets()
df_sample = load_dataset()

st.title("🛡️ Multimodal Synthetic Identity Fraud Risk Engine")
st.caption("IEEE Research Implementation — KYC Verification, Identity Freshness, Behavioral Biometrics & Network Telemetry")

tab1, tab2, tab3, tab4 = st.tabs([
    "🎯 Live Risk Scorer",
    "📊 5-Model Benchmark Suite",
    "🔍 Signal Explainability",
    "📁 Batch Telemetry Inspector"
])

# TAB 1: Live Risk Scorer
with tab1:
    st.subheader("Real-Time Application Risk Inference")
    preset = st.selectbox(
        "Select Profile Preset",
        ["Custom Input", "Typical Legit Applicant", "Suspicious Synthetic Applicant", "Bot / Scripted Fraud Ring"]
    )

    defaults = {
        "Typical Legit Applicant": dict(
            name_address_mismatch_score=0.05, dob_pan_mismatch=0, document_reuse_count=0, ssn_pan_issuance_gap_years=15.0, commercial_address_flag=0,
            phone_age_days=1200, email_age_days=1000, credit_bureau_hit=1, bureau_file_depth_months=84, social_footprint_score=0.85,
            session_fill_time_sec=260, typing_speed_variance=0.30, backspace_count=16, paste_event_ratio=0.10, field_hesitation_ms=950,
            device_reuse_across_apps=0, application_velocity_24h=0, ip_geolocation_mismatch=0, identity_graph_degree_centrality=0.02, subnet_risk_score=0.05
        ),
        "Suspicious Synthetic Applicant": dict(
            name_address_mismatch_score=0.75, dob_pan_mismatch=1, document_reuse_count=3, ssn_pan_issuance_gap_years=1.2, commercial_address_flag=1,
            phone_age_days=8, email_age_days=5, credit_bureau_hit=0, bureau_file_depth_months=2, social_footprint_score=0.12,
            session_fill_time_sec=45, typing_speed_variance=0.03, backspace_count=1, paste_event_ratio=0.85, field_hesitation_ms=120,
            device_reuse_across_apps=4, application_velocity_24h=5, ip_geolocation_mismatch=1, identity_graph_degree_centrality=0.65, subnet_risk_score=0.78
        ),
        "Bot / Scripted Fraud Ring": dict(
            name_address_mismatch_score=0.88, dob_pan_mismatch=1, document_reuse_count=5, ssn_pan_issuance_gap_years=0.5, commercial_address_flag=1,
            phone_age_days=2, email_age_days=1, credit_bureau_hit=0, bureau_file_depth_months=0, social_footprint_score=0.05,
            session_fill_time_sec=15, typing_speed_variance=0.01, backspace_count=0, paste_event_ratio=0.98, field_hesitation_ms=20,
            device_reuse_across_apps=8, application_velocity_24h=9, ip_geolocation_mismatch=1, identity_graph_degree_centrality=0.92, subnet_risk_score=0.95
        )
    }
    vals = defaults.get(preset, {})

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown("##### 📄 KYC & Entity Matching")
        name_address_mismatch_score = st.slider("Name/Address Mismatch", 0.0, 1.0, vals.get("name_address_mismatch_score", 0.15))
        dob_pan_mismatch = st.selectbox("DOB-PAN Mismatch", [0, 1], index=vals.get("dob_pan_mismatch", 0))
        document_reuse_count = st.number_input("Document Reuse Count", 0, 10, vals.get("document_reuse_count", 0))
        ssn_pan_issuance_gap_years = st.number_input("ID Issuance Gap (years)", 0.0, 50.0, vals.get("ssn_pan_issuance_gap_years", 10.0))
        commercial_address_flag = st.selectbox("Commercial Mailbox", [0, 1], index=vals.get("commercial_address_flag", 0))

    with c2:
        st.markdown("##### ⏳ Identity Freshness")
        phone_age_days = st.number_input("Phone Age (days)", 0, 3000, vals.get("phone_age_days", 500))
        email_age_days = st.number_input("Email Age (days)", 0, 3000, vals.get("email_age_days", 450))
        credit_bureau_hit = st.selectbox("Bureau Record Exists", [0, 1], index=vals.get("credit_bureau_hit", 1))
        bureau_file_depth_months = st.number_input("Bureau History (months)", 0, 240, vals.get("bureau_file_depth_months", 48))
        social_footprint_score = st.slider("Social Footprint Score", 0.0, 1.0, vals.get("social_footprint_score", 0.70))

    with c3:
        st.markdown("##### 🖱️ Behavioral Biometrics")
        session_fill_time_sec = st.number_input("Fill Time (sec)", 5, 900, vals.get("session_fill_time_sec", 240))
        typing_speed_variance = st.slider("Typing Cadence Variance", 0.0, 1.0, vals.get("typing_speed_variance", 0.25))
        backspace_count = st.number_input("Backspace Count", 0, 50, vals.get("backspace_count", 12))
        paste_event_ratio = st.slider("Paste Ratio", 0.0, 1.0, vals.get("paste_event_ratio", 0.15))
        field_hesitation_ms = st.number_input("Field Hesitation (ms)", 0, 5000, vals.get("field_hesitation_ms", 850))

    with c4:
        st.markdown("##### 🌐 Network & Graph")
        device_reuse_across_apps = st.number_input("Identities on Device", 0, 10, vals.get("device_reuse_across_apps", 0))
        application_velocity_24h = st.number_input("Velocity (24h apps)", 0, 10, vals.get("application_velocity_24h", 0))
        ip_geolocation_mismatch = st.selectbox("IP Geolocation Mismatch", [0, 1], index=vals.get("ip_geolocation_mismatch", 0))
        identity_graph_degree_centrality = st.slider("Graph Centrality", 0.0, 1.0, vals.get("identity_graph_degree_centrality", 0.05))
        subnet_risk_score = st.slider("Subnet Risk Score", 0.0, 1.0, vals.get("subnet_risk_score", 0.10))

    st.divider()
    if st.button("⚡ Score Application with Random Forest Model", type="primary", use_container_width=True):
        row = pd.DataFrame([{
            "name_address_mismatch_score": name_address_mismatch_score,
            "dob_pan_mismatch": dob_pan_mismatch,
            "document_reuse_count": document_reuse_count,
            "ssn_pan_issuance_gap_years": ssn_pan_issuance_gap_years,
            "commercial_address_flag": commercial_address_flag,
            "phone_age_days": phone_age_days,
            "email_age_days": email_age_days,
            "credit_bureau_hit": credit_bureau_hit,
            "bureau_file_depth_months": bureau_file_depth_months,
            "social_footprint_score": social_footprint_score,
            "session_fill_time_sec": session_fill_time_sec,
            "typing_speed_variance": typing_speed_variance,
            "backspace_count": backspace_count,
            "paste_event_ratio": paste_event_ratio,
            "field_hesitation_ms": field_hesitation_ms,
            "device_reuse_across_apps": device_reuse_across_apps,
            "application_velocity_24h": application_velocity_24h,
            "ip_geolocation_mismatch": ip_geolocation_mismatch,
            "identity_graph_degree_centrality": identity_graph_degree_centrality,
            "subnet_risk_score": subnet_risk_score,
        }])[feature_cols]

        proba = model.predict_proba(row)[0, 1]
        pct = proba * 100.0

        r1, r2 = st.columns([1, 2])
        with r1:
            st.metric("Fraud Probability Score", f"{pct:.1f}%")
            if proba > 0.60:
                st.error("🚨 **HIGH RISK APPLICATION**\n\nAction: **REJECT / BLOCK ACCOUNT**")
            elif proba > 0.30:
                st.warning("⚠️ **MEDIUM RISK APPLICATION**\n\nAction: **STEP-UP VERIFICATION (Video KYC / OTP)**")
            else:
                st.success("✅ **LOW RISK APPLICATION**\n\nAction: **AUTO-APPROVE & DISBURSE**")

        with r2:
            st.markdown("##### 📈 Top Feature Importances (Model Global)")
            imp = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=True).tail(8)
            st.bar_chart(imp)

# TAB 2: Model Benchmark Suite
with tab2:
    st.subheader("5-Fold Cross-Validation Model Benchmarks")
    bench_data = metrics.get("benchmark_comparison", {})
    if bench_data:
        b_df = pd.DataFrame(bench_data).T
        st.dataframe(b_df, use_container_width=True)

    c_roc, c_pr = st.columns(2)
    roc_img = os.path.join(BASE, "report", "images", "roc_curves_comparison.png")
    pr_img = os.path.join(BASE, "report", "images", "pr_curves_comparison.png")
    radar_img = os.path.join(BASE, "report", "images", "radar_model_benchmark.png")

    with c_roc:
        if os.path.exists(roc_img):
            st.image(roc_img, caption="Multi-Model ROC Curves", use_container_width=True)
    with c_pr:
        if os.path.exists(pr_img):
            st.image(pr_img, caption="Multi-Model Precision-Recall Curves", use_container_width=True)

    if os.path.exists(radar_img):
        st.divider()
        st.image(radar_img, caption="Model Benchmark Metric Trade-off", use_container_width=True)

# TAB 3: Signal Explainability
with tab3:
    st.subheader("Feature Attribution across 20 Multimodal Signals")
    fi_img = os.path.join(BASE, "report", "images", "feature_importance.png")
    if os.path.exists(fi_img):
        st.image(fi_img, caption="Gini Feature Importance Ranking (All 20 Signals)", use_container_width=True)

# TAB 4: Batch Telemetry
with tab4:
    st.subheader("Batch Dataset Inspection (10,000 Applications)")
    if df_sample is not None:
        st.write(f"Dataset preview (`data/synthetic_kyc_behavioral.csv` - Total: {len(df_sample)} rows):")
        st.dataframe(df_sample.head(25), use_container_width=True)

        if st.button("⚡ Score First 100 Applications"):
            sub = df_sample.head(100).copy()
            sub["fraud_probability"] = model.predict_proba(sub[feature_cols])[:, 1]
            sub["risk_tier"] = pd.cut(sub["fraud_probability"], bins=[-0.01, 0.30, 0.60, 1.0], labels=["Low Risk", "Medium Risk", "High Risk"])
            st.dataframe(sub[["application_id", "is_fraud", "fraud_probability", "risk_tier"] + list(feature_cols[:4])], use_container_width=True)
            st.bar_chart(sub["risk_tier"].value_counts())
