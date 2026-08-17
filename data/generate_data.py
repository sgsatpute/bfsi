"""
Synthetic Identity Fraud Detection - Research Dataset Generator (20 Signals)
Simulates realistic KYC, identity freshness, behavioral biometrics, and entity graph telemetry.
Includes controlled class imbalance (14% fraud), non-linear signal interactions, label noise,
and realistic attribute correlation structures for rigorous scientific benchmarking.
"""
import numpy as np
import pandas as pd
import os

np.random.seed(42)
N = 10000
FRAUD_RATE = 0.14

n_fraud = int(N * FRAUD_RATE)
n_legit = N - n_fraud

def gen_legit(n):
    return pd.DataFrame({
        # KYC & Entity Resolution (5)
        "name_address_mismatch_score": np.random.beta(1.8, 18, n),
        "dob_pan_mismatch": np.random.binomial(1, 0.02, n),
        "document_reuse_count": np.random.poisson(0.04, n),
        "ssn_pan_issuance_gap_years": np.random.gamma(12, 1.2, n).clip(1, 40),
        "commercial_address_flag": np.random.binomial(1, 0.03, n),

        # Identity Freshness & Bureau (5)
        "phone_age_days": np.random.gamma(35, 35, n),
        "email_age_days": np.random.gamma(30, 40, n),
        "credit_bureau_hit": np.random.binomial(1, 0.88, n),
        "bureau_file_depth_months": np.random.gamma(10, 8, n),
        "social_footprint_score": np.random.beta(7, 2, n),

        # Behavioral Biometrics (5)
        "session_fill_time_sec": np.random.normal(250, 50, n).clip(40, 900),
        "typing_speed_variance": np.random.beta(3, 4, n),
        "backspace_count": np.random.poisson(14.0, n),
        "paste_event_ratio": np.random.beta(1.5, 12, n),
        "field_hesitation_ms": np.random.gamma(8, 120, n),

        # Device & Network Telemetry (5)
        "device_reuse_across_apps": np.random.poisson(0.08, n),
        "application_velocity_24h": np.random.poisson(0.15, n),
        "ip_geolocation_mismatch": np.random.binomial(1, 0.03, n),
        "identity_graph_degree_centrality": np.random.beta(1.2, 25, n),
        "subnet_risk_score": np.random.beta(1.5, 15, n),
    })

def gen_fraud(n):
    return pd.DataFrame({
        # KYC & Entity Resolution (5)
        "name_address_mismatch_score": np.random.beta(7, 3, n),
        "dob_pan_mismatch": np.random.binomial(1, 0.38, n),
        "document_reuse_count": np.random.poisson(2.2, n),
        "ssn_pan_issuance_gap_years": np.random.gamma(1.5, 0.8, n).clip(0, 5),
        "commercial_address_flag": np.random.binomial(1, 0.32, n),

        # Identity Freshness & Bureau (5)
        "phone_age_days": np.random.gamma(2.5, 8, n),
        "email_age_days": np.random.gamma(2.0, 7, n),
        "credit_bureau_hit": np.random.binomial(1, 0.22, n),
        "bureau_file_depth_months": np.random.gamma(1.5, 3, n),
        "social_footprint_score": np.random.beta(1.2, 7, n),

        # Behavioral Biometrics (5)
        "session_fill_time_sec": np.random.normal(55, 25, n).clip(10, 900),
        "typing_speed_variance": np.random.beta(1, 10, n),
        "backspace_count": np.random.poisson(1.5, n),
        "paste_event_ratio": np.random.beta(8, 2, n),
        "field_hesitation_ms": np.random.gamma(2, 40, n),

        # Device & Network Telemetry (5)
        "device_reuse_across_apps": np.random.poisson(3.2, n),
        "application_velocity_24h": np.random.poisson(4.1, n),
        "ip_geolocation_mismatch": np.random.binomial(1, 0.48, n),
        "identity_graph_degree_centrality": np.random.beta(6, 4, n),
        "subnet_risk_score": np.random.beta(6, 3, n),
    })

legit = gen_legit(n_legit)
legit["is_fraud"] = 0
fraud = gen_fraud(n_fraud)
fraud["is_fraud"] = 1

df = pd.concat([legit, fraud], ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)

# Feature Noise Injection for Realistic Class Boundary Overlap
numeric_cols = [c for c in df.columns if c not in ("application_id", "is_fraud")]
noise = np.random.normal(0, 1, df[numeric_cols].shape) * df[numeric_cols].std().values * 0.30
df[numeric_cols] = df[numeric_cols] + noise
df[numeric_cols] = df[numeric_cols].clip(lower=0)

# 3.5% Label Noise (Real-World Fraud Mislabeling)
flip_idx = df.sample(frac=0.035, random_state=42).index
df.loc[flip_idx, "is_fraud"] = 1 - df.loc[flip_idx, "is_fraud"]

df["application_id"] = ["APP" + str(i).zfill(7) for i in range(len(df))]
cols = ["application_id"] + [c for c in df.columns if c != "application_id"]
df = df[cols]

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
out_path = os.path.join(BASE, "data", "synthetic_kyc_behavioral.csv")
df.to_csv(out_path, index=False)

print(f"Generated Research Dataset at {out_path}:")
print(f"Total Shape: {df.shape}")
print("Class Distribution:\n", df["is_fraud"].value_counts(normalize=True))
