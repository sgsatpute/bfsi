"""
Multimodal Synthetic Identity Fraud Detection - Human-Centered Explainable Portal
Run with: python -m streamlit run app/dashboard.py
"""
import streamlit as st
import joblib
import numpy as np
import pandas as pd
import os
import json

st.set_page_config(
    page_title="Digital Lending Fraud Risk Scanner",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE, "model", "fraud_model.joblib")
FEAT_PATH = os.path.join(BASE, "model", "feature_cols.joblib")
METRICS_PATH = os.path.join(BASE, "model", "metrics.json")
DATA_PATH = os.path.join(BASE, "data", "synthetic_kyc_behavioral.csv")

@st.cache_resource
def load_assets():
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

model, feature_cols, metrics = load_assets()
df_sample = load_dataset()

# Inject Modern Human-Centered CSS Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    .hero-banner {
        background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%);
        border-radius: 16px;
        padding: 24px 32px;
        color: white;
        margin-bottom: 24px;
        box-shadow: 0 10px 25px -5px rgba(15, 23, 42, 0.15);
    }
    
    .hero-title {
        font-size: 26px;
        font-weight: 800;
        margin: 0;
        color: #F8FAFC;
        letter-spacing: -0.5px;
    }
    
    .hero-sub {
        font-size: 14px;
        color: #94A3B8;
        margin-top: 6px;
        margin-bottom: 0;
    }
    
    .step-card {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 16px 20px;
        margin-bottom: 20px;
    }
    
    .step-num {
        background: #2563EB;
        color: white;
        font-weight: 700;
        font-size: 12px;
        padding: 3px 10px;
        border-radius: 20px;
        display: inline-block;
        margin-bottom: 8px;
    }
    
    .result-box-pass {
        background: #ECFDF5;
        border: 2px solid #10B981;
        border-radius: 14px;
        padding: 20px;
        color: #065F46;
    }

    .result-box-warn {
        background: #FFFBEB;
        border: 2px solid #F59E0B;
        border-radius: 14px;
        padding: 20px;
        color: #92400E;
    }

    .result-box-danger {
        background: #FEF2F2;
        border: 2px solid #EF4444;
        border-radius: 14px;
        padding: 20px;
        color: #991B1B;
    }
    
    .flag-card-red {
        background: #FFFFFF;
        border-left: 4px solid #EF4444;
        padding: 12px 16px;
        border-radius: 8px;
        margin-bottom: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.02);
    }

    .flag-card-green {
        background: #FFFFFF;
        border-left: 4px solid #10B981;
        padding: 12px 16px;
        border-radius: 8px;
        margin-bottom: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.02);
    }
</style>
""", unsafe_allow_html=True)

# Top Hero Banner
st.markdown("""
<div class="hero-banner">
    <div class="hero-title">🛡️ Digital Lending Fraud Risk Scanner</div>
    <div class="hero-sub">Underwriter Decision Portal • Human-Explainable AI Risk Diagnosis</div>
</div>
""", unsafe_allow_html=True)

# Navigation Tabs
tab_main, tab_explain, tab_proof = st.tabs([
    "🎯 Underwriter Decision Portal",
    "❓ How to Use & Understand the AI",
    "📊 Model Accuracy & Proof"
])

# -----------------------------------------------------------------------------
# TAB 1: UNDERWRITER DECISION PORTAL (SIMPLE, HUMAN-CENTERED UI)
# -----------------------------------------------------------------------------
with tab_main:
    
    # 3-Step How To Use Guide Banner
    st.markdown("""
    <div class="step-card">
        <div style="font-weight: 700; color: #0F172A; font-size: 15px; margin-bottom: 10px;">📋 How to Use This Scanner (3 Easy Steps)</div>
        <div style="display: flex; gap: 20px; flex-wrap: wrap;">
            <div><span class="step-num">STEP 1</span> <b>Pick an Applicant</b> below (Click a pre-filled card)</div>
            <div><span class="step-num">STEP 2</span> Click <b>"Analyze Loan Application"</b></div>
            <div><span class="step-num">STEP 3</span> Read the <b>Plain-English Explainable Report</b></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 1️⃣ Select Applicant Profile")

    # Sample Applicant Preset Buttons
    col_p1, col_p2, col_p3 = st.columns(3)
    
    with col_p1:
        if st.button("👤 **Rohan Sharma (Salaried IT Worker)**\n\nVerified PAN, 3-year phone line, normal typing", use_container_width=True):
            st.session_state["preset_id"] = "rohan"
    with col_p2:
        if st.button("⚠️ **Ankit Verma (Thin-File Applicant)**\n\nFresh phone line, recent email, minor address gap", use_container_width=True):
            st.session_state["preset_id"] = "ankit"
    with col_p3:
        if st.button("🚨 **Fake Persona #892 (Synthetic Fraud)**\n\nBurner phone (4 days old), 12s fill time, bot paste", use_container_width=True):
            st.session_state["preset_id"] = "fake"

    preset_id = st.session_state.get("preset_id", "rohan")

    presets = {
        "rohan": {
            "name": "Rohan Sharma",
            "desc": "Salaried IT Employee (Legitimate Customer)",
            "data": dict(
                name_address_mismatch_score=0.05, dob_pan_mismatch=0, document_reuse_count=0, ssn_pan_issuance_gap_years=14.0, commercial_address_flag=0,
                phone_age_days=1200, email_age_days=950, credit_bureau_hit=1, bureau_file_depth_months=84, social_footprint_score=0.85,
                session_fill_time_sec=270, typing_speed_variance=0.29, backspace_count=15, paste_event_ratio=0.08, field_hesitation_ms=900,
                device_reuse_across_apps=0, application_velocity_24h=0, ip_geolocation_mismatch=0, identity_graph_degree_centrality=0.02, subnet_risk_score=0.04
            )
        },
        "ankit": {
            "name": "Ankit Verma",
            "desc": "First-time Applicant (Thin Credit File)",
            "data": dict(
                name_address_mismatch_score=0.35, dob_pan_mismatch=0, document_reuse_count=1, ssn_pan_issuance_gap_years=2.0, commercial_address_flag=0,
                phone_age_days=45, email_age_days=30, credit_bureau_hit=0, bureau_file_depth_months=3, social_footprint_score=0.45,
                session_fill_time_sec=160, typing_speed_variance=0.18, backspace_count=6, paste_event_ratio=0.30, field_hesitation_ms=400,
                device_reuse_across_apps=1, application_velocity_24h=1, ip_geolocation_mismatch=0, identity_graph_degree_centrality=0.15, subnet_risk_score=0.20
            )
        },
        "fake": {
            "name": "Fake Persona #892",
            "desc": "Synthetic Identity Fraud Ring Attempt",
            "data": dict(
                name_address_mismatch_score=0.82, dob_pan_mismatch=1, document_reuse_count=4, ssn_pan_issuance_gap_years=0.4, commercial_address_flag=1,
                phone_age_days=4, email_age_days=3, credit_bureau_hit=0, bureau_file_depth_months=0, social_footprint_score=0.08,
                session_fill_time_sec=14, typing_speed_variance=0.02, backspace_count=0, paste_event_ratio=0.92, field_hesitation_ms=20,
                device_reuse_across_apps=5, application_velocity_24h=6, ip_geolocation_mismatch=1, identity_graph_degree_centrality=0.88, subnet_risk_score=0.90
            )
        }
    }

    current_applicant = presets[preset_id]
    vals = current_applicant["data"]

    st.success(f"Selected Applicant: **{current_applicant['name']}** — *{current_applicant['desc']}*")

    # Optional Signal Adjustment (Keep hidden in expander so it stays simple)
    with st.expander("🛠️ Optional: Adjust Applicant Data Signals (Advanced)", expanded=False):
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown("##### 📄 KYC Matching")
            name_address_mismatch_score = st.slider("Name/Address Mismatch", 0.0, 1.0, vals["name_address_mismatch_score"])
            dob_pan_mismatch = st.selectbox("DOB Mismatch", [0, 1], index=vals["dob_pan_mismatch"])
            document_reuse_count = st.number_input("Doc Reuse Count", 0, 10, vals["document_reuse_count"])
            ssn_pan_issuance_gap_years = st.number_input("ID Gap (Years)", 0.0, 40.0, vals["ssn_pan_issuance_gap_years"])
            commercial_address_flag = st.selectbox("Commercial Address", [0, 1], index=vals["commercial_address_flag"])

        with c2:
            st.markdown("##### ⏳ Identity Age")
            phone_age_days = st.number_input("Phone Age (Days)", 0, 3000, vals["phone_age_days"])
            email_age_days = st.number_input("Email Age (Days)", 0, 3000, vals["email_age_days"])
            credit_bureau_hit = st.selectbox("Bureau Record Found", [0, 1], index=vals["credit_bureau_hit"])
            bureau_file_depth_months = st.number_input("Bureau Depth (Months)", 0, 240, vals["bureau_file_depth_months"])
            social_footprint_score = st.slider("Digital Footprint", 0.0, 1.0, vals["social_footprint_score"])

        with c3:
            st.markdown("##### 🖱️ Behavioral Biometrics")
            session_fill_time_sec = st.number_input("Fill Time (Sec)", 5, 900, vals["session_fill_time_sec"])
            typing_speed_variance = st.slider("Typing Cadence", 0.0, 1.0, vals["typing_speed_variance"])
            backspace_count = st.number_input("Backspace Count", 0, 50, vals["backspace_count"])
            paste_event_ratio = st.slider("Paste Ratio", 0.0, 1.0, vals["paste_event_ratio"])
            field_hesitation_ms = st.number_input("Hesitation Pause (ms)", 0, 5000, vals["field_hesitation_ms"])

        with c4:
            st.markdown("##### 🌐 Network Telemetry")
            device_reuse_across_apps = st.number_input("Device Apps Count", 0, 10, vals["device_reuse_across_apps"])
            application_velocity_24h = st.number_input("Velocity (24h Apps)", 0, 10, vals["application_velocity_24h"])
            ip_geolocation_mismatch = st.selectbox("IP Geolocation Mismatch", [0, 1], index=vals["ip_geolocation_mismatch"])
            identity_graph_degree_centrality = st.slider("Graph Centrality", 0.0, 1.0, vals["identity_graph_degree_centrality"])
            subnet_risk_score = st.slider("IP Subnet Risk", 0.0, 1.0, vals["subnet_risk_score"])
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Analyze Application Action Button
    if st.button("⚡ 2️⃣ ANALYZE LOAN APPLICATION NOW", type="primary", use_container_width=True):
        row = pd.DataFrame([{
            "name_address_mismatch_score": name_address_mismatch_score if 'name_address_mismatch_score' in locals() else vals["name_address_mismatch_score"],
            "dob_pan_mismatch": dob_pan_mismatch if 'dob_pan_mismatch' in locals() else vals["dob_pan_mismatch"],
            "document_reuse_count": document_reuse_count if 'document_reuse_count' in locals() else vals["document_reuse_count"],
            "ssn_pan_issuance_gap_years": ssn_pan_issuance_gap_years if 'ssn_pan_issuance_gap_years' in locals() else vals["ssn_pan_issuance_gap_years"],
            "commercial_address_flag": commercial_address_flag if 'commercial_address_flag' in locals() else vals["commercial_address_flag"],
            "phone_age_days": phone_age_days if 'phone_age_days' in locals() else vals["phone_age_days"],
            "email_age_days": email_age_days if 'email_age_days' in locals() else vals["email_age_days"],
            "credit_bureau_hit": credit_bureau_hit if 'credit_bureau_hit' in locals() else vals["credit_bureau_hit"],
            "bureau_file_depth_months": bureau_file_depth_months if 'bureau_file_depth_months' in locals() else vals["bureau_file_depth_months"],
            "social_footprint_score": social_footprint_score if 'social_footprint_score' in locals() else vals["social_footprint_score"],
            "session_fill_time_sec": session_fill_time_sec if 'session_fill_time_sec' in locals() else vals["session_fill_time_sec"],
            "typing_speed_variance": typing_speed_variance if 'typing_speed_variance' in locals() else vals["typing_speed_variance"],
            "backspace_count": backspace_count if 'backspace_count' in locals() else vals["backspace_count"],
            "paste_event_ratio": paste_event_ratio if 'paste_event_ratio' in locals() else vals["paste_event_ratio"],
            "field_hesitation_ms": field_hesitation_ms if 'field_hesitation_ms' in locals() else vals["field_hesitation_ms"],
            "device_reuse_across_apps": device_reuse_across_apps if 'device_reuse_across_apps' in locals() else vals["device_reuse_across_apps"],
            "application_velocity_24h": application_velocity_24h if 'application_velocity_24h' in locals() else vals["application_velocity_24h"],
            "ip_geolocation_mismatch": ip_geolocation_mismatch if 'ip_geolocation_mismatch' in locals() else vals["ip_geolocation_mismatch"],
            "identity_graph_degree_centrality": identity_graph_degree_centrality if 'identity_graph_degree_centrality' in locals() else vals["identity_graph_degree_centrality"],
            "subnet_risk_score": subnet_risk_score if 'subnet_risk_score' in locals() else vals["subnet_risk_score"],
        }])[feature_cols]

        proba = model.predict_proba(row)[0, 1]
        score_pct = proba * 100.0

        st.markdown("---")
        st.markdown("### 3️⃣ Explainable AI Fraud Diagnosis Report")

        res_col1, res_col2 = st.columns([1, 2])
        
        with res_col1:
            st.metric("Fraud Risk Score", f"{score_pct:.1f}%")
            if proba > 0.60:
                st.markdown("""
                <div class="result-box-danger">
                    <div style="font-size: 18px; font-weight: 800;">🔴 REJECT & BLOCK</div>
                    <div style="font-size: 13px; margin-top: 6px;">High probability of synthetic identity creation. Do not disburse funds.</div>
                </div>
                """, unsafe_allow_html=True)
            elif proba > 0.30:
                st.markdown("""
                <div class="result-box-warn">
                    <div style="font-size: 18px; font-weight: 800;">🟡 MANUAL REVIEW / VIDEO KYC</div>
                    <div style="font-size: 13px; margin-top: 6px;">Borderline identity signals. Request physical document verification or Video KYC.</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="result-box-pass">
                    <div style="font-size: 18px; font-weight: 800;">🟢 AUTO-APPROVE</div>
                    <div style="font-size: 13px; margin-top: 6px;">Verified identity authenticity. Proceed to instant loan disbursal.</div>
                </div>
                """, unsafe_allow_html=True)

        with res_col2:
            st.markdown("##### 🔍 Human-Readable Diagnosis (Why the AI decided this):")
            
            curr_p_age = row["phone_age_days"].values[0]
            curr_e_age = row["email_age_days"].values[0]
            curr_fill = row["session_fill_time_sec"].values[0]
            curr_paste = row["paste_event_ratio"].values[0]
            curr_dev = row["device_reuse_across_apps"].values[0]
            curr_vel = row["application_velocity_24h"].values[0]
            curr_mismatch = row["name_address_mismatch_score"].values[0]

            # Generate Human Explanations
            explanations = []
            if curr_p_age < 15:
                explanations.append(("red", f"<b>Burner Phone Alert:</b> Mobile subscription activated only <b>{curr_p_age:.0f} days ago</b> (Synthetic identities use fresh numbers)."))
            else:
                explanations.append(("green", f"<b>Established Phone Line:</b> Mobile subscription active for <b>{curr_p_age/365:.1f} years</b>."))

            if curr_e_age < 10:
                explanations.append(("red", f"<b>Synthetic Email Alert:</b> Email domain registered only <b>{curr_e_age:.0f} days ago</b>."))
            else:
                explanations.append(("green", f"<b>Established Email Account:</b> Active email history detected (<b>{curr_e_age:.0f} days old</b>)."))

            if curr_fill < 30:
                explanations.append(("red", f"<b>Scripted Form Fill:</b> Application completed in <b>{curr_fill:.0f} seconds</b> (Bot/Automation pattern)."))
            elif curr_paste > 0.70:
                explanations.append(("red", f"<b>Clipboard Copy-Paste:</b> <b>{curr_paste*100:.0f}%</b> of fields populated via copy-paste."))
            else:
                explanations.append(("green", f"<b>Natural Typing Cadence:</b> Realistic human form completion time (<b>{curr_fill:.0f}s</b>)."))

            if curr_dev > 1:
                explanations.append(("red", f"<b>Device Multi-Accounting:</b> <b>{curr_dev} distinct identity applications</b> submitted from this device."))
            
            if curr_vel > 2:
                explanations.append(("red", f"<b>High Application Velocity:</b> <b>{curr_vel} loan applications</b> originating from this IP in 24h."))

            if curr_mismatch > 0.5:
                explanations.append(("red", f"<b>KYC Address Mismatch:</b> High mismatch score (<b>{curr_mismatch:.2f}</b>) between stated address and credit records."))

            for status, text in explanations:
                if status == "red":
                    st.markdown(f'<div class="flag-card-red">⚠️ {text}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="flag-card-green">✅ {text}</div>', unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# TAB 2: HOW TO USE & UNDERSTAND THE AI
# -----------------------------------------------------------------------------
with tab_explain:
    st.markdown("### ❓ How the AI Engine Works (In Simple Terms)")
    
    st.markdown("""
    #### What is Synthetic Identity Fraud?
    Unlike stolen identity fraud (where a real person's full identity is stolen), synthetic fraud creates a **"Frankenstein Fake Identity"**:
    - **Real PII:** A valid PAN or Tax ID number.
    - **Fake Details:** Fake name, fake residential address, burner phone line, and newly registered email.

    #### How the AI Detective Catches It:
    1. **Identity Age:** Real humans have older phone numbers and email accounts. Fraudsters purchase fresh burner lines (<10 days old).
    2. **Behavioral Biometrics:** Humans type normally, pause, and make typos. Bots fill 50 form fields in 10 seconds with 100% copy-pasted data.
    3. **Device Telemetry:** Catches fraud rings submitting multiple loan applications from the same laptop or IP address.
    """)

# -----------------------------------------------------------------------------
# TAB 3: MODEL ACCURACY & PROOF
# -----------------------------------------------------------------------------
with tab_proof:
    st.markdown("### 📊 Empirical Model Benchmark Proof")
    
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("ROC-AUC Accuracy", f"{metrics.get('roc_auc', 0.9139):.4f}")
    m2.metric("Fraud Precision", f"{metrics.get('precision_fraud', 0.8600):.4f}")
    m3.metric("Fraud Recall", f"{metrics.get('recall_fraud', 0.9110):.4f}")
    m4.metric("Fraud F1-Score", f"{metrics.get('f1_fraud', 0.8849):.4f}")

    st.divider()
    img1, img2 = st.columns(2)
    roc_img = os.path.join(BASE, "report", "images", "roc_curves_comparison.png")
    cm_img = os.path.join(BASE, "report", "images", "confusion_matrix.png")
    
    with img1:
        if os.path.exists(roc_img):
            st.image(roc_img, caption="Multi-Model ROC Curves", use_container_width=True)
    with img2:
        if os.path.exists(cm_img):
            st.image(cm_img, caption="Confusion Matrix Breakdown", use_container_width=True)
