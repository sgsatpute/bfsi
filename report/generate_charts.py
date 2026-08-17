import os
import json
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, auc, average_precision_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMG_DIR = os.path.join(BASE, "report", "images")
os.makedirs(IMG_DIR, exist_ok=True)

df = pd.read_csv(os.path.join(BASE, "data", "synthetic_kyc_behavioral.csv"))
model = joblib.load(os.path.join(BASE, "model", "fraud_model.joblib"))
feature_cols = joblib.load(os.path.join(BASE, "model", "feature_cols.joblib"))

X = df[feature_cols]
y = df["is_fraud"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 5 Trained models for ROC & PR curves plot
clfs = {
    "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=300, random_state=42)).fit(X_train, y_train),
    "Decision Tree": DecisionTreeClassifier(max_depth=6, class_weight="balanced", random_state=42).fit(X_train, y_train),
    "Random Forest": model,
    "Hist Gradient Boosting": HistGradientBoostingClassifier(max_iter=80, max_depth=5, class_weight="balanced", random_state=42).fit(X_train, y_train),
    "Multi-Layer Perceptron": make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=100, random_state=42)).fit(X_train, y_train)
}

colors_map = {
    "Logistic Regression": "#3B82F6",
    "Decision Tree": "#F59E0B",
    "Random Forest": "#10B981",
    "Hist Gradient Boosting": "#6366F1",
    "Multi-Layer Perceptron": "#EC4899"
}

fig_color = '#0F172A'

# 1. Multi-Model ROC Curves Comparison Plot
plt.figure(figsize=(8, 6), dpi=300)
for name, clf in clfs.items():
    proba = clf.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, proba)
    score = auc(fpr, tpr)
    plt.plot(fpr, tpr, color=colors_map[name], lw=2.2, label=f'{name} (AUC = {score:.4f})')

plt.plot([0, 1], [0, 1], color='#94A3B8', lw=1.5, linestyle='--', label='Random Baseline (AUC = 0.50)')
plt.xlim([-0.01, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=11, fontweight='bold', color=fig_color)
plt.ylabel('True Positive Rate (Recall)', fontsize=11, fontweight='bold', color=fig_color)
plt.title('Receiver Operating Characteristic (ROC) Benchmark', fontsize=13, fontweight='bold', color=fig_color, pad=12)
plt.legend(loc="lower right", frameon=True, facecolor='#F8FAFC', edgecolor='#CBD5E1', fontsize=9.5)
plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, "roc_curve.png"), dpi=300)
plt.savefig(os.path.join(IMG_DIR, "roc_curves_comparison.png"), dpi=300)
plt.close()

# 2. Precision-Recall Curves Plot
plt.figure(figsize=(8, 6), dpi=300)
for name, clf in clfs.items():
    proba = clf.predict_proba(X_test)[:, 1]
    prec, rec, _ = precision_recall_curve(y_test, proba)
    ap = average_precision_score(y_test, proba)
    plt.plot(rec, prec, color=colors_map[name], lw=2.2, label=f'{name} (AP = {ap:.4f})')

plt.xlabel('Recall (Fraud Catch Rate)', fontsize=11, fontweight='bold', color=fig_color)
plt.ylabel('Precision (Positive Predictive Value)', fontsize=11, fontweight='bold', color=fig_color)
plt.title('Precision-Recall (PR) Curves Benchmark', fontsize=13, fontweight='bold', color=fig_color, pad=12)
plt.legend(loc="lower left", frameon=True, facecolor='#F8FAFC', edgecolor='#CBD5E1', fontsize=9.5)
plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, "pr_curves_comparison.png"), dpi=300)
plt.close()

# 3. Production Model Confusion Matrix
y_pred_prod = model.predict(X_test)
cm = confusion_matrix(y_test, y_pred_prod)
fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
cax = ax.matshow(cm, cmap='Blues')
fig.colorbar(cax)

labels = ['Legitimate', 'Fraudulent']
ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_xticklabels(labels, fontsize=11, fontweight='bold')
ax.set_yticklabels(labels, fontsize=11, fontweight='bold')
ax.xaxis.set_ticks_position('bottom')

for i in range(2):
    for j in range(2):
        color = "white" if cm[i, j] > cm.max() / 2 else "black"
        ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center", color=color, fontsize=16, fontweight='bold')

plt.xlabel('Predicted Label', fontsize=11, fontweight='bold', color=fig_color)
plt.ylabel('True Label', fontsize=11, fontweight='bold', color=fig_color)
plt.title('Production Model Confusion Matrix (Test Evaluation)', fontsize=12, fontweight='bold', color=fig_color, pad=12)
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, "confusion_matrix.png"), dpi=300)
plt.close()

# 4. Feature Importance Ranking (All 20 Signals)
importances = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=True)

plt.figure(figsize=(10, 8), dpi=300)
colors_bar = ['#10B981' if i >= len(importances)-5 else '#64748B' for i in range(len(importances))]
bars = plt.barh(importances.index, importances.values, color=colors_bar, height=0.65)
plt.xlabel('Gini Feature Importance', fontsize=11, fontweight='bold', color=fig_color)
plt.title('Feature Attribution Analysis across 20 Multimodal Signals', fontsize=13, fontweight='bold', color=fig_color, pad=12)
plt.grid(True, linestyle=':', alpha=0.5, axis='x')

for bar in bars:
    width = bar.get_width()
    plt.text(width + 0.002, bar.get_y() + bar.get_height()/2, f'{width:.3f}',
             va='center', ha='left', fontsize=8.5, color='#334155', fontweight='bold')

plt.xlim(0, max(importances.values) * 1.16)
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, "feature_importance.png"), dpi=300)
plt.savefig(os.path.join(IMG_DIR, "feature_importance_shap.png"), dpi=300)
plt.close()

# 5. Radar / Bar Benchmark Comparison
with open(os.path.join(BASE, "model", "metrics.json"), "r") as f:
    m_data = json.load(f)

bench_df = pd.DataFrame(m_data["benchmark_comparison"]).T

plt.figure(figsize=(9, 5), dpi=300)
x = np.arange(len(bench_df))
width = 0.25

plt.bar(x - width, bench_df['roc_auc_mean'], width, label='ROC-AUC', color='#3B82F6')
plt.bar(x, bench_df['pr_auc_mean'], width, label='PR-AUC', color='#6366F1')
plt.bar(x + width, bench_df['f1_mean'], width, label='Fraud F1-Score', color='#10B981')

plt.ylabel('Score (5-Fold CV Mean)', fontsize=11, fontweight='bold', color=fig_color)
plt.title('5-Fold Cross-Validation Model Benchmark Comparison', fontsize=13, fontweight='bold', color=fig_color, pad=12)
plt.xticks(x, bench_df.index, rotation=15, ha='right', fontsize=9.5, fontweight='bold')
plt.ylim(0.70, 1.0)
plt.legend(loc='lower right', frameon=True, facecolor='#F8FAFC')
plt.grid(True, linestyle=':', alpha=0.5, axis='y')
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, "radar_model_benchmark.png"), dpi=300)
plt.close()

print("Successfully generated all publication-quality research figures in:", IMG_DIR)
