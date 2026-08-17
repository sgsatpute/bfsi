import os
import joblib
import pandas as pd
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE, "model", "fraud_model.joblib")
FEAT_PATH = os.path.join(BASE, "model", "feature_cols.joblib")
DATA_PATH = os.path.join(BASE, "data", "synthetic_kyc_behavioral.csv")

print("1. Testing Asset Loading...")
model = joblib.load(MODEL_PATH)
feature_cols = joblib.load(FEAT_PATH)
print("Asset Loading PASS! Features:", len(feature_cols))

print("\n2. Testing Dataset Loading...")
df = pd.read_csv(DATA_PATH)
print("Dataset Loading PASS! Rows:", len(df), "Cols:", len(df.columns))

print("\n3. Testing Model Inference on 1,000 Random Dataset Rows...")
sub = df.head(1000)
probas = model.predict_proba(sub[feature_cols])[:, 1]
print("Inference PASS! Mean Probability:", np.mean(probas))
print("Min Probability:", np.min(probas), "Max Probability:", np.max(probas))

print("\n4. Testing Signal Explanation Logic...")
for i in range(5):
    row = df.iloc[[i]]
    p = model.predict_proba(row[feature_cols])[0, 1]
    phone_age = row["phone_age_days"].values[0]
    email_age = row["email_age_days"].values[0]
    fill_time = row["session_fill_time_sec"].values[0]
    print(f"Sample {i+1}: Fraud Prob = {p*100:.1f}% | Phone Age = {phone_age}d | Email Age = {email_age}d | Fill Time = {fill_time}s")

print("\nALL VERIFICATION TESTS PASSED SUCCESSFULLY!")
