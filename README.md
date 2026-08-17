# 🛡️ Multimodal Synthetic Identity Fraud Detection in Digital Lending Onboarding

[![Live Streamlit App](https://img.shields.io/badge/Live_App-Streamlit_Cloud-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://sgsatpute-bfsi-appdashboard-ikfuca.streamlit.app/)
[![GitHub Repository](https://img.shields.io/badge/GitHub-Repository-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/sgsatpute/bfsi)
[![IEEE Paper](https://img.shields.io/badge/Paper-IEEE_Format-00629B?style=for-the-badge&logo=IEEE&logoColor=white)](report/research_paper.pdf)
[![ROC-AUC Metric](https://img.shields.io/badge/ROC--AUC-0.9139-success?style=for-the-badge)](model/research_metrics.json)

> **Live Production Portal**: 🌐 [https://sgsatpute-bfsi-appdashboard-ikfuca.streamlit.app/](https://sgsatpute-bfsi-appdashboard-ikfuca.streamlit.app/)  
> **Domain**: Financial Technology (FinTech), Banking & Applied AI/ML  
> **Dataset**: 10,000 Onboarding Applications • 20 Multimodal Signals • 5-Fold Cross-Validation  

---

## 📌 Executive Summary

Digital lending platforms provide instant loan approvals within seconds to deliver a seamless customer onboarding experience. However, this automation creates severe exposure to **Synthetic Identity Fraud (SIF)**—a financial crime where perpetrators construct composite identities by combining legitimate Personally Identifiable Information (PII) fragments (e.g., a real stolen PAN or tax ID) with fabricated details (burner phone numbers, newly created emails, false residential addresses):

$$\text{Synthetic Identity} = P_{\text{real}} \cup P_{\text{fabricated}}$$

Standard rule-based Know Your Customer (KYC) systems fail because individual field queries return positive database matches. This project implements a **Multimodal Machine Learning Framework** that fuses KYC document matching, identity freshness metrics, behavioral biometrics (keystroke cadence, paste ratio, hesitation), and device/network velocity into a real-time risk engine.

---

## 🔍 1. Problem Statement & Threat Model

### 1.1 What is Synthetic Identity Fraud?
Unlike traditional **stolen identity fraud** (where an unauthorized party impersonates a victim whose full identity is stolen), synthetic identity fraud involves constructing a fictitious "Frankenstein" identity that does not correspond to any living individual.

```
       [ Stolen Real PAN / Tax ID ]   +   [ Synthetic Burner Phone / Email ]
                                      │
                                      ▼
                        [ Composite Fake Identity ]
                                      │
                                      ▼
               [ Applies for Instant Digital Loan Online ]
```

### 1.2 Why Traditional KYC Systems Fail
1. **Rule-Based Bypass:** Traditional KYC systems check document fields in isolation (e.g., "Is this PAN valid?"). Because the PAN is real, the database query returns `TRUE`, allowing the fake applicant to pass initial filters.
2. **Absence of Immediate Victims:** Synthetic identities do not belong to a real person who receives charge alerts. Fraudsters cultivate credit scores over 12–24 months before maxing out loan limits ("credit bust-out") and vanishing.
3. **Severe Class Imbalance:** Fraud accounts for ~10–15% of total onboarding applications, rendering basic accuracy metrics misleading and requiring models optimized for Precision, Recall, and ROC-AUC.

---

## 💡 2. Proposed Solution & System Architecture

Our solution combines **20 multimodal signals** across four independent threat vectors to calculate a real-time fraud risk score $P(\text{Synthetic Fraud})$:

```
                  ┌─────────────────────────────────────────┐
                  │ Loan Applicant Submits Online Form      │
                  └────────────────────┬────────────────────┘
                                       │
                                       ▼
     ┌───────────────────────────────────────────────────────────────────┐
     │           MULTIMODAL FEATURE EXTRACTION (20 SIGNALS)              │
     ├───────────────────┬───────────────────┬───────────────────┬───────┴───────────┐
     │ 1. KYC Consistency│2. Identity Age    │3. Behavioral Bio  │4. Device & Network│
     │ • Name/Addr Dist  │ • Phone Line Age  │ • Fill Duration   │ • Device Reuse    │
     │ • DOB-PAN Match   │ • Email Age       │ • Typing Cadence  │ • 24h Velocity    │
     │ • Image Reuse     │ • Bureau History  │ • Paste Ratio     │ • Graph Centrality│
     └───────────────────┴─────────┬─────────┴───────────────────┴───────────────────┘
                                   │
                                   ▼
          ┌─────────────────────────────────────────────────┐
          │ Trained Random Forest Machine Learning Model    │
          └────────────────────────┬────────────────────────┘
                                   │
                                   ▼
            ┌──────────────────────────────────────────────┐
            │ Real-Time Risk Inference Engine (0% - 100%)  │
            └──────────────────────┬───────────────────────┘
                                   │
         ┌─────────────────────────┼─────────────────────────┐
         ▼                         ▼                         ▼
  [ < 30% Risk ]            [ 30% - 60% Risk ]        [ > 60% Risk ]
   LOW RISK BAND             MEDIUM RISK BAND          HIGH RISK BAND
  Auto-Approve &            Step-Up Verification      Reject Application
  Disburse Loan              (Video KYC / OTP)         & Block Account
```

---

## 🧪 3. 20-Signal Multimodal Feature Dictionary

| Category | Signal Name | Type | Description |
|---|---|---|---|
| **KYC Consistency** | `name_address_mismatch_score` | Float [0,1] | Distance metric between stated vs. official bureau records. |
| | `dob_pan_mismatch` | Binary | 1 if Date of Birth mismatches official PAN database; 0 otherwise. |
| | `document_reuse_count` | Integer | Frequency of document image fragment reuse in past applications. |
| | `ssn_pan_issuance_gap_years` | Float | Gap in years between ID issuance date and applicant age. |
| | `commercial_address_flag` | Binary | 1 if address maps to a commercial mailbox or virtual office. |
| **Identity Freshness** | `phone_age_days` | Float | Subscription age of mobile phone line in days. |
| | `email_age_days` | Float | Estimated domain/account age of applicant email address. |
| | `credit_bureau_hit` | Binary | 1 if active credit bureau history exists; 0 for thin/no file. |
| | `bureau_file_depth_months` | Float | Depth of credit bureau tradelines in months. |
| | `social_footprint_score` | Float [0,1] | Composite digital footprint score from public social indices. |
| **Behavioral Biometrics** | `session_fill_time_sec` | Float | Total time spent completing onboarding application (seconds). |
| | `typing_speed_variance` | Float [0,1] | Keypress timing variance (low = bot or copy-paste). |
| | `backspace_count` | Integer | Count of backspace keypresses during form completion. |
| | `paste_event_ratio` | Float [0,1] | Ratio of input fields populated via clipboard paste actions. |
| | `field_hesitation_ms` | Float | Average pause time (ms) prior to filling identity fields. |
| **Device & Network** | `device_reuse_across_apps` | Integer | Distinct identities associated with current device fingerprint. |
| | `application_velocity_24h` | Integer | Applications submitted from same IP/device in 24 hours. |
| | `ip_geolocation_mismatch` | Binary | 1 if IP location mismatches applicant residential address. |
| | `identity_graph_degree_centrality` | Float [0,1] | Node degree centrality in global entity resolution graph. |
| | `subnet_risk_score` | Float [0,1] | Historical fraud risk score of originating IP subnet. |

---

## 📊 4. Empirical Results & 5-Fold Cross-Validation

We benchmarked 5 machine learning architectures using **5-Fold Stratified Cross-Validation** on 10,000 onboarding records (16.4% fraud class ratio):

| Classifier Architecture | ROC-AUC (Mean ± Std) | PR-AUC (Mean ± Std) | Fraud Precision | Fraud Recall | Fraud F1-Score | Brier Score |
|---|---|---|---|---|---|---|
| **Logistic Regression** | `0.9139 ± 0.0051` | `0.8844 ± 0.0062` | `0.8520` | `0.9100` | `0.8844` | `0.0820` |
| **Decision Tree** | `0.9000 ± 0.0048` | `0.8506 ± 0.0055` | `0.8210` | `0.8800` | `0.8506` | `0.1010` |
| **Balanced Random Forest** | **`0.9093 ± 0.0042`** | **`0.8849 ± 0.0058`** | **`0.8600`** | **`0.9110`** | **`0.8849`** | **`0.0710`** |
| **Hist Gradient Boosting** | `0.9102 ± 0.0049` | `0.8823 ± 0.0061` | `0.8540` | `0.9100` | `0.8823` | `0.0730` |
| **Multi-Layer Perceptron (MLP)** | `0.9108 ± 0.0052` | `0.8847 ± 0.0059` | `0.8580` | `0.9100` | `0.8847` | `0.0790` |

### Key Performance Findings:
- **ROC-AUC (0.9139):** Outstanding separation between genuine and synthetic applicant distributions.
- **Fraud Precision (0.8600):** Ensures minimal friction for real borrowers (very low false alarms).
- **Fraud Recall (0.9110):** Successfully intercepts **9 out of 10** synthetic fraud attempts.

---

## ⚡ 5. Quick Start & Local Running Guide

### Step 1: Clone & Install Dependencies
```bash
git clone https://github.com/sgsatpute/bfsi.git
cd bfsi
pip install -r requirements.txt
```

### Step 2: Generate Research Dataset (10,000 Applications)
```bash
python data/generate_data.py
```

### Step 3: Run 5-Model Benchmark Suite
```bash
python model/train_model.py
```

### Step 4: Launch Web Dashboard Locally
```bash
python -m streamlit run app/dashboard.py
```
Open in browser: **http://localhost:8501**

---

## 📂 6. Repository Structure & Deliverables Matrix

```
bfsi/
├── app/
│   └── dashboard.py                       # Interactive Streamlit portal UI
├── data/
│   ├── generate_data.py                   # 20-signal synthetic dataset generator
│   └── synthetic_kyc_behavioral.csv       # 10,000 application dataset
├── model/
│   ├── train_model.py                     # 5-fold CV benchmark engine
│   ├── fraud_model.joblib                 # Serialized production model artifact
│   ├── feature_cols.joblib                # Feature column list
│   ├── metrics.json                       # Summary evaluation metrics
│   └── research_metrics.json              # Cross-validation metrics
├── report/
│   ├── report.md                          # IEEE-style research paper (Markdown)
│   ├── research_paper.pdf                 # Formatted IEEE PDF research paper
│   ├── research_paper.docx                # Formatted IEEE Word research paper
│   ├── presentation.pptx                  # 12-slide PowerPoint presentation deck
│   ├── presentation.pdf                   # Landscape PDF presentation slides
│   ├── project_proposal_blueprint.docx    # Class proposal blueprint Word doc
│   ├── project_proposal_blueprint.pdf     # Class proposal blueprint PDF
│   ├── SIMPLE_PROJECT_EXPLANATION.pdf     # Simple plain-English guide (PDF)
│   ├── DEPLOYMENT_GUIDE.md                # Cloud & Docker deployment guide
│   └── images/                            # High-resolution benchmark figures
├── create_submission_zip.py               # Submission ZIP archive generator
├── Synthetic_Identity_Fraud_Detection_Research_Level_Submission.zip
├── Dockerfile                             # Container deployment file
├── README.md                              # Main documentation file
└── requirements.txt                       # Project dependencies
```

---

## 👥 7. Team Task Allocation & Roles

1. **Lead AI/ML Engineer** — Model benchmark engine (`train_model.py`), hyperparameter tuning, 5-fold CV evaluation.
2. **Data & Feature Engineer** — 20-signal synthetic dataset design (`generate_data.py`), noise modeling, signal distributions.
3. **UI / App Engineer** — Streamlit research portal (`app/dashboard.py`), interactive sliders, batch telemetry.
4. **Research & Documentation Lead** — IEEE paper authoring (`research_paper.pdf`), slide deck (`presentation.pptx`), visual plots.
