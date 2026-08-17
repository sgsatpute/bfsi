"""
Multimodal Synthetic Identity Fraud Detection - Enterprise Decision Portal
Run with: python -m streamlit run app/dashboard.py
"""
import streamlit as st
import joblib
import numpy as np
import pandas as pd
import os
import json

st.set_page_config(
    page_title="Enterprise Loan Fraud Risk Portal",
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
        df = pd.read_csv(DATA_PATH)
        return df
    return None

model, feature_cols, metrics = load_assets()
df_sample = load_dataset()

# Custom High-Contrast Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    .hero-banner {
        background: linear-gradient(135deg, #0B0F19 0%, #1A2332 100%);
        border: 1px solid #2A364F;
        border-radius: 16px;
        padding: 24px 32px;
        color: #F8FAFC !important;
        margin-bottom: 24px;
        box-shadow: 0 12px 30px rgba(0,0,0,0.3);
    }
    
    .hero-title {
        font-size: 28px;
        font-weight: 800;
        margin: 0;
        color: #FFFFFF !important;
        letter-spacing: -0.5px;
    }
    
    .hero-sub {
        font-size: 14px;
        color: #94A3B8 !important;
        margin-top: 6px;
    }
    
    /* Result Badges */
    .result-box-pass {
        background-color: #064E3B !important;
        border: 2px solid #10B981 !important;
        border-radius: 14px;
        padding: 20px;
        color: #ECFDF5 !important;
        text-align: center;
    }

    .result-box-warn {
        background-color: #78350F !important;
        border: 2px solid #F59E0B !important;
        border-radius: 14px;
        padding: 20px;
        color: #FEF3C7 !important;
        text-align: center;
    }

    .result-box-danger {
        background-color: #7F1D1D !important;
        border: 2px solid #EF4444 !important;
        border-radius: 14px;
        padding: 20px;
        color: #FEE2E2 !important;
        text-align: center;
    }
    
    /* Diagnosis Cards */
    .flag-card-red {
        background-color: #2D1214 !important;
        border-left: 5px solid #EF4444 !important;
        border-top: 1px solid #451A1C !important;
        border-right: 1px solid #451A1C !important;
        border-bottom: 1px solid #451A1C !important;
        padding: 12px 16px !important;
        border-radius: 10px !important;
        margin-bottom: 8px !important;
        color: #FEE2E2 !important;
        font-size: 13.5px !important;
        font-weight: 500 !important;
    }

    .flag-card-green {
        background-color: #062C1E !important;
        border-left: 5px solid #10B981 !important;
        border-top: 1px solid #0B4530 !important;
        border-right: 1px solid #0B4530 !important;
        border-bottom: 1px solid #0B4530 !important;
        padding: 12px 16px !important;
        border-radius: 10px !important;
        margin-bottom: 8px !important;
        color: #D1FADF !important;
        font-size: 13.5px !important;
        font-weight: 500 !important;
    }

    .score-number {
        font-size: 52px !important;
        font-weight: 800 !important;
        line-height: 1.0 !important;
        margin-top: 6px !important;
        margin-bottom: 10px !important;
    }
</style>
""", unsafe_allow_html=True)

# Top Hero Header
st.markdown("""
<div class="hero-banner">
    <div class="hero-title">🛡️ Digital Lending Fraud Risk Portal</div>
    <div class="hero-sub">Enterprise Onboarding Risk Engine • Human-Explainable AI Portal</div>
</div>
""", unsafe_allow_html=True)

# Main Navigation Tabs
tab_form, tab_search, tab_preset, tab_batch, tab_proof = st.tabs([
    "📋 Fill Live Loan Form",
    "🔍 Search 10,000 Applications",
    "⚡ Preset Profiles (6 Cases)",
    "📂 Batch CSV Scorer",
    "📊 Model Benchmarks"
])

def render_diagnosis_report(row_df):
    proba = model.predict_proba(row_df[feature_cols])[0, 1]
    score_pct = proba * 100.0

    res_col1, res_col2 = st.columns([1, 2])
    
    with res_col1:
        score_color = "#EF4444" if proba > 0.60 else ("#F59E0B" if proba > 0.30 else "#10B981")
        st.markdown(f'<div style="font-size:14px; font-weight:700; color:#94A3B8;">Fraud Risk Score</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="score-number" style="color:{score_color};">{score_pct:.1f}%</div>', unsafe_allow_html=True)

        if proba > 0.60:
            st.markdown("""
            <div class="result-box-danger">
                <div style="font-size: 20px; font-weight: 800;">🔴 REJECT & BLOCK</div>
                <div style="font-size: 12px; margin-top: 4px;">High probability of synthetic identity creation. Do not disburse funds.</div>
            </div>
            """, unsafe_allow_html=True)
        elif proba > 0.30:
            st.markdown("""
            <div class="result-box-warn">
                <div style="font-size: 20px; font-weight: 800;">🟡 STEP-UP VIDEO KYC</div>
                <div style="font-size: 12px; margin-top: 4px;">Borderline identity signals. Request physical document verification or Video KYC.</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="result-box-pass">
                <div style="font-size: 20px; font-weight: 800;">🟢 AUTO-APPROVE</div>
                <div style="font-size: 12px; margin-top: 4px;">Verified identity authenticity. Proceed to instant loan disbursal.</div>
            </div>
            """, unsafe_allow_html=True)

    with res_col2:
        st.markdown("##### 🔍 Human-Readable Explainable AI Diagnosis")
        
        curr_p_age = row_df["phone_age_days"].values[0]
        curr_e_age = row_df["email_age_days"].values[0]
        curr_fill = row_df["session_fill_time_sec"].values[0]
        curr_paste = row_df["paste_event_ratio"].values[0]
        curr_dev = row_df["device_reuse_across_apps"].values[0]
        curr_vel = row_df["application_velocity_24h"].values[0]
        curr_mismatch = row_df["name_address_mismatch_score"].values[0]

        explanations = []
        if curr_p_age < 15:
            explanations.append(("red", f"⚠️ <b>Burner Phone Alert:</b> Mobile subscription activated only <b>{curr_p_age:.0f} days ago</b> (Synthetic identities use fresh numbers)."))
        else:
            explanations.append(("green", f"✅ <b>Established Phone Line:</b> Mobile subscription active for <b>{curr_p_age/365:.1f} years</b>."))

        if curr_e_age < 10:
            explanations.append(("red", f"⚠️ <b>Synthetic Email Alert:</b> Email domain registered only <b>{curr_e_age:.0f} days ago</b>."))
        else:
            explanations.append(("green", f"✅ <b>Established Email Account:</b> Active email history detected (<b>{curr_e_age:.0f} days old</b>)."))

        if curr_fill < 30:
            explanations.append(("red", f"⚠️ <b>Scripted Form Fill:</b> Application completed in <b>{curr_fill:.0f} seconds</b> (Bot/Automation pattern)."))
        elif curr_paste > 0.70:
            explanations.append(("red", f"⚠️ <b>Clipboard Copy-Paste:</b> <b>{curr_paste*100:.0f}%</b> of fields populated via copy-paste."))
        else:
            explanations.append(("green", f"✅ <b>Natural Typing Cadence:</b> Realistic human form completion time (<b>{curr_fill:.0f}s</b>)."))

        if curr_dev > 1:
            explanations.append(("red", f"⚠️ <b>Device Multi-Accounting:</b> <b>{curr_dev} distinct identity applications</b> submitted from this device."))
        
        if curr_vel > 2:
            explanations.append(("red", f"⚠️ <b>High Application Velocity:</b> <b>{curr_vel} loan applications</b> originating from this IP in 24h."))

        if curr_mismatch > 0.5:
            explanations.append(("red", f"⚠️ <b>KYC Address Mismatch:</b> High mismatch score (<b>{curr_mismatch:.2f}</b>) between stated address and credit records."))

        for status, text in explanations:
            if status == "red":
                st.markdown(f'<div class="flag-card-red">{text}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="flag-card-green">{text}</div>', unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# MODE 1: LIVE LOAN APPLICATION FORM (CUSTOM APPLICANT CREATOR)
# -----------------------------------------------------------------------------
with tab_form:
    st.markdown("### 📋 Fill & Score a Custom Loan Application")
    st.caption("Enter loan application details below to calculate real-time fraud risk.")

    with st.form("loan_application_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            applicant_name = st.text_input("Applicant Full Name", "Vikram Malhotra")
            loan_amount = st.number_input("Requested Loan Amount (₹)", 10000, 1000000, 150000, step=10000)
            phone_age = st.number_input("Mobile Phone Subscription Age (Days)", 0, 3650, 450)
            email_age = st.number_input("Email Account Age (Days)", 0, 3650, 600)
            
        with c2:
            pan_dob_match = st.selectbox("DOB Matches Government Database?", ["Yes (Match)", "No (Mismatch)"])
            address_mismatch = st.slider("Stated Address Mismatch Score", 0.0, 1.0, 0.12)
            fill_time = st.number_input("Form Completion Time (Seconds)", 5, 900, 220)
            paste_ratio = st.slider("Clipboard Paste Ratio", 0.0, 1.0, 0.15)

        with c3:
            device_apps = st.number_input("Other Applications from Same Device", 0, 20, 0)
            ip_velocity = st.number_input("Applications from Same IP in 24 Hours", 0, 20, 0)
            credit_bureau = st.selectbox("Active Credit Bureau File Exists?", ["Yes", "No"])
            commercial_addr = st.selectbox("Is Commercial Mailbox Address?", ["No", "Yes"])

        submit_btn = st.form_submit_button("⚡ SUBMIT & CALCULATE FRAUD RISK", type="primary", use_container_width=True)

    if submit_btn or "form_row" in st.session_state:
        # Map form fields to 20-signal dictionary
        row_dict = {
            "name_address_mismatch_score": address_mismatch,
            "dob_pan_mismatch": 1 if pan_dob_match.startswith("No") else 0,
            "document_reuse_count": device_apps,
            "ssn_pan_issuance_gap_years": 10.0 if credit_bureau == "Yes" else 1.0,
            "commercial_address_flag": 1 if commercial_addr == "Yes" else 0,
            "phone_age_days": phone_age,
            "email_age_days": email_age,
            "credit_bureau_hit": 1 if credit_bureau == "Yes" else 0,
            "bureau_file_depth_months": 48 if credit_bureau == "Yes" else 0,
            "social_footprint_score": 0.75 if phone_age > 100 else 0.10,
            "session_fill_time_sec": fill_time,
            "typing_speed_variance": 0.25 if fill_time > 60 else 0.02,
            "backspace_count": 8 if fill_time > 60 else 0,
            "paste_event_ratio": paste_ratio,
            "field_hesitation_ms": 600 if fill_time > 60 else 30,
            "device_reuse_across_apps": device_apps,
            "application_velocity_24h": ip_velocity,
            "ip_geolocation_mismatch": 1 if address_mismatch > 0.5 else 0,
            "identity_graph_degree_centrality": 0.05 if device_apps == 0 else 0.75,
            "subnet_risk_score": 0.05 if ip_velocity < 2 else 0.85
        }
        form_df = pd.DataFrame([row_dict])
        st.session_state["form_row"] = form_df
        st.markdown("---")
        st.markdown(f"### 🎯 Results for Applicant: **{applicant_name}** (Loan Amount: ₹{loan_amount:,})")
        render_diagnosis_report(form_df)

# -----------------------------------------------------------------------------
# MODE 2: SEARCH 10,000 DATASET APPLICATIONS
# -----------------------------------------------------------------------------
with tab_search:
    st.markdown("### 🔍 Search & Inspect Any of the 10,000 Dataset Applications")
    
    if df_sample is not None:
        c_s1, c_s2 = st.columns([1, 2])
        with c_s1:
            app_list = df_sample["application_id"].tolist()
            selected_app_id = st.selectbox("Select Application ID to Inspect:", app_list, index=41) # APP-00042
            
        row_match = df_sample[df_sample["application_id"] == selected_app_id]
        actual_label = row_match["is_fraud"].values[0]
        actual_str = "🔴 FRAUDSTER" if actual_label == 1 else "🟢 GENUINE"
        
        st.info(f"Inspecting Application **{selected_app_id}** (Ground Truth Dataset Label: **{actual_str}**)")
        render_diagnosis_report(row_match)
        
        with st.expander("📄 View Full 20 Signals Data Row"):
            st.dataframe(row_match, use_container_width=True)

# -----------------------------------------------------------------------------
# MODE 3: PRESET LIBRARY (6 DIVERSE CASE STUDIES)
# -----------------------------------------------------------------------------
with tab_preset:
    st.markdown("### ⚡ Preset Applicant Case Studies")
    
    presets_6 = {
        "rohan": ("👤 Rohan Sharma", "Salaried IT Professional", dict(
            name_address_mismatch_score=0.05, dob_pan_mismatch=0, document_reuse_count=0, ssn_pan_issuance_gap_years=14.0, commercial_address_flag=0,
            phone_age_days=1200, email_age_days=950, credit_bureau_hit=1, bureau_file_depth_months=84, social_footprint_score=0.85,
            session_fill_time_sec=270, typing_speed_variance=0.29, backspace_count=15, paste_event_ratio=0.08, field_hesitation_ms=900,
            device_reuse_across_apps=0, application_velocity_24h=0, ip_geolocation_mismatch=0, identity_graph_degree_centrality=0.02, subnet_risk_score=0.04
        )),
        "priya": ("🎓 Priya Patel", "Student (Thin Credit File, Genuine)", dict(
            name_address_mismatch_score=0.15, dob_pan_mismatch=0, document_reuse_count=0, ssn_pan_issuance_gap_years=1.5, commercial_address_flag=0,
            phone_age_days=450, email_age_days=600, credit_bureau_hit=0, bureau_file_depth_months=2, social_footprint_score=0.65,
            session_fill_time_sec=190, typing_speed_variance=0.22, backspace_count=9, paste_event_ratio=0.12, field_hesitation_ms=550,
            device_reuse_across_apps=0, application_velocity_24h=0, ip_geolocation_mismatch=0, identity_graph_degree_centrality=0.04, subnet_risk_score=0.08
        )),
        "fake": ("🚨 Synthetic Burner #892", "Burner Phone & Synthetic Email Ring", dict(
            name_address_mismatch_score=0.82, dob_pan_mismatch=1, document_reuse_count=4, ssn_pan_issuance_gap_years=0.4, commercial_address_flag=1,
            phone_age_days=4, email_age_days=3, credit_bureau_hit=0, bureau_file_depth_months=0, social_footprint_score=0.08,
            session_fill_time_sec=14, typing_speed_variance=0.02, backspace_count=0, paste_event_ratio=0.92, field_hesitation_ms=20,
            device_reuse_across_apps=5, application_velocity_24h=6, ip_geolocation_mismatch=1, identity_graph_degree_centrality=0.88, subnet_risk_score=0.90
        )),
        "bot": ("🤖 Scripted Bot Attack", "Automation Script Filling 50 Fields in 8s", dict(
            name_address_mismatch_score=0.95, dob_pan_mismatch=1, document_reuse_count=8, ssn_pan_issuance_gap_years=0.1, commercial_address_flag=1,
            phone_age_days=2, email_age_days=1, credit_bureau_hit=0, bureau_file_depth_months=0, social_footprint_score=0.02,
            session_fill_time_sec=8, typing_speed_variance=0.00, backspace_count=0, paste_event_ratio=1.00, field_hesitation_ms=5,
            device_reuse_across_apps=10, application_velocity_24h=15, ip_geolocation_mismatch=1, identity_graph_degree_centrality=0.95, subnet_risk_score=0.98
        )),
        "mailbox": ("🏢 Virtual Office Mailbox", "Stated Address Maps to Commercial Mailbox", dict(
            name_address_mismatch_score=0.75, dob_pan_mismatch=0, document_reuse_count=2, ssn_pan_issuance_gap_years=4.0, commercial_address_flag=1,
            phone_age_days=180, email_age_days=120, credit_bureau_hit=0, bureau_file_depth_months=6, social_footprint_score=0.25,
            session_fill_time_sec=110, typing_speed_variance=0.12, backspace_count=2, paste_event_ratio=0.55, field_hesitation_ms=250,
            device_reuse_across_apps=3, application_velocity_24h=3, ip_geolocation_mismatch=1, identity_graph_degree_centrality=0.45, subnet_risk_score=0.60
        )),
        "velocity": ("🌐 IP Velocity Ring", "7 Loan Applications from Same Wi-Fi Router", dict(
            name_address_mismatch_score=0.60, dob_pan_mismatch=0, document_reuse_count=3, ssn_pan_issuance_gap_years=3.0, commercial_address_flag=0,
            phone_age_days=25, email_age_days=18, credit_bureau_hit=0, bureau_file_depth_months=0, social_footprint_score=0.15,
            session_fill_time_sec=45, typing_speed_variance=0.08, backspace_count=1, paste_event_ratio=0.80, field_hesitation_ms=100,
            device_reuse_across_apps=4, application_velocity_24h=7, ip_geolocation_mismatch=1, identity_graph_degree_centrality=0.70, subnet_risk_score=0.82
        ))
    }

    cols = st.columns(3)
    idx = 0
    for key, (title, desc, pdata) in presets_6.items():
        with cols[idx % 3]:
            if st.button(f"{title}\n\n{desc}", key=f"btn_{key}", use_container_width=True):
                st.session_state["active_preset_6"] = key
        idx += 1

    active_p = st.session_state.get("active_preset_6", "rohan")
    p_title, p_desc, p_dict = presets_6[active_p]
    
    st.info(f"Selected Profile: **{p_title}** — *{p_desc}*")
    render_diagnosis_report(pd.DataFrame([p_dict]))

# -----------------------------------------------------------------------------
# MODE 4: BATCH CSV SCORER
# -----------------------------------------------------------------------------
with tab_batch:
    st.markdown("### 📂 Upload & Score Batch Loan Applications (CSV)")
    st.caption("Upload a CSV file containing application records to generate a batch risk report.")

    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])
    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
            if all(col in batch_df.columns for col in feature_cols):
                batch_df["fraud_probability"] = model.predict_proba(batch_df[feature_cols])[:, 1]
                batch_df["risk_tier"] = pd.cut(batch_df["fraud_probability"], bins=[-0.01, 0.30, 0.60, 1.0], labels=["Low Risk (Auto-Approve)", "Medium Risk (Video KYC)", "High Risk (Reject)"])
                
                st.success(f"Successfully scored {len(batch_df):,} application records!")
                
                m1, m2, m3 = st.columns(3)
                m1.metric("Total Scored Applications", f"{len(batch_df):,}")
                m2.metric("Flagged High Risk", f"{(batch_df['fraud_probability'] > 0.60).sum():,}")
                m3.metric("Auto-Approved", f"{(batch_df['fraud_probability'] <= 0.30).sum():,}")
                
                st.dataframe(batch_df[["application_id", "fraud_probability", "risk_tier"] + list(feature_cols[:4])], use_container_width=True)
                st.bar_chart(batch_df["risk_tier"].value_counts())
            else:
                st.error("Uploaded CSV is missing required 20 signal columns.")
        except Exception as e:
            st.error(f"Error processing CSV file: {e}")
    else:
        st.write("Alternatively, score the first 100 records from the dataset:")
        if st.button("⚡ Score Sample 100 Dataset Records"):
            sub = df_sample.head(100).copy()
            sub["fraud_probability"] = model.predict_proba(sub[feature_cols])[:, 1]
            sub["risk_tier"] = pd.cut(sub["fraud_probability"], bins=[-0.01, 0.30, 0.60, 1.0], labels=["Low Risk", "Medium Risk", "High Risk"])
            st.dataframe(sub[["application_id", "is_fraud", "fraud_probability", "risk_tier"] + list(feature_cols[:4])], use_container_width=True)

# -----------------------------------------------------------------------------
# MODE 5: MODEL ACCURACY BENCHMARKS
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
