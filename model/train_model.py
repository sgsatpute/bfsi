"""
Synthetic Identity Fraud Detection - Fast Research Model Benchmark Engine
Trains and benchmarks 5 machine learning architectures using 5-Fold Stratified Cross-Validation.
"""
import os
import json
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (
    classification_report, roc_auc_score, average_precision_score,
    confusion_matrix, f1_score, precision_score, recall_score, brier_score_loss
)

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = os.path.join(BASE, "data", "synthetic_kyc_behavioral.csv")
df = pd.read_csv(data_path)

feature_cols = [c for c in df.columns if c not in ("application_id", "is_fraud")]
X = df[feature_cols]
y = df["is_fraud"]

classifiers = {
    "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=300, random_state=42)),
    "Decision Tree": DecisionTreeClassifier(max_depth=6, class_weight="balanced", random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=8, class_weight="balanced", random_state=42, n_jobs=1),
    "Hist Gradient Boosting": HistGradientBoostingClassifier(max_iter=80, max_depth=5, class_weight="balanced", random_state=42),
    "Multi-Layer Perceptron (MLP)": make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=100, random_state=42))
}

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
benchmark_results = {}

print("Executing Fast 5-Fold Stratified Cross-Validation Benchmark (20 Signals)...")

for name, clf in classifiers.items():
    roc_aucs, pr_aucs, precisions, recalls, f1s, briers = [], [], [], [], [], []
    
    for train_idx, val_idx in skf.split(X, y):
        X_tr, X_va = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_va = y.iloc[train_idx], y.iloc[val_idx]

        clf.fit(X_tr, y_tr)
        y_pred = clf.predict(X_va)
        
        if hasattr(clf, "predict_proba"):
            y_proba = clf.predict_proba(X_va)[:, 1]
        else:
            y_proba = clf.decision_function(X_va)

        roc_aucs.append(roc_auc_score(y_va, y_proba))
        pr_aucs.append(average_precision_score(y_va, y_proba))
        precisions.append(precision_score(y_va, y_pred, zero_division=0))
        recalls.append(recall_score(y_va, y_pred, zero_division=0))
        f1s.append(f1_score(y_va, y_pred, zero_division=0))
        briers.append(brier_score_loss(y_va, y_proba))

    benchmark_results[name] = {
        "roc_auc_mean": round(float(np.mean(roc_aucs)), 4),
        "roc_auc_std": round(float(np.std(roc_aucs)), 4),
        "pr_auc_mean": round(float(np.mean(pr_aucs)), 4),
        "pr_auc_std": round(float(np.std(pr_aucs)), 4),
        "precision_mean": round(float(np.mean(precisions)), 4),
        "precision_std": round(float(np.std(precisions)), 4),
        "recall_mean": round(float(np.mean(recalls)), 4),
        "recall_std": round(float(np.std(recalls)), 4),
        "f1_mean": round(float(np.mean(f1s)), 4),
        "f1_std": round(float(np.std(f1s)), 4),
        "brier_score_mean": round(float(np.mean(briers)), 4),
    }
    print(f" -> {name:28s} | ROC-AUC: {np.mean(roc_aucs):.4f} | Fraud F1: {np.mean(f1s):.4f}")

# Train Final Production Model (Random Forest)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

rf_prod = RandomForestClassifier(n_estimators=150, max_depth=8, class_weight="balanced", random_state=42, n_jobs=1)
rf_prod.fit(X_train, y_train)

y_pred_test = rf_prod.predict(X_test)
y_proba_test = rf_prod.predict_proba(X_test)[:, 1]

auc_test = roc_auc_score(y_test, y_proba_test)
pr_auc_test = average_precision_score(y_test, y_proba_test)
cm_test = confusion_matrix(y_test, y_pred_test)

report = classification_report(y_test, y_pred_test, target_names=["Legit", "Fraud"], output_dict=True)
importances = pd.Series(rf_prod.feature_importances_, index=feature_cols).sort_values(ascending=False)

joblib.dump(rf_prod, os.path.join(BASE, "model", "fraud_model.joblib"))
joblib.dump(feature_cols, os.path.join(BASE, "model", "feature_cols.joblib"))

summary_metrics = {
    "roc_auc": round(auc_test, 4),
    "pr_auc": round(pr_auc_test, 4),
    "precision_fraud": round(report["Fraud"]["precision"], 4),
    "recall_fraud": round(report["Fraud"]["recall"], 4),
    "f1_fraud": round(report["Fraud"]["f1-score"], 4),
    "accuracy": round(report["accuracy"], 4),
    "confusion_matrix": cm_test.tolist(),
    "top_features": importances.head(10).to_dict(),
    "benchmark_comparison": benchmark_results
}

with open(os.path.join(BASE, "model", "metrics.json"), "w") as f:
    json.dump(summary_metrics, f, indent=2)

with open(os.path.join(BASE, "model", "research_metrics.json"), "w") as f:
    json.dump(summary_metrics, f, indent=2)

print("\nSAVED ALL BENCHMARKS AND MODEL ARTIFACTS SUCCESSFULLY.")
