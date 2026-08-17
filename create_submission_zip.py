import os
import zipfile

BASE = os.path.dirname(os.path.abspath(__file__))

zip_names = [
    "Synthetic_Identity_Fraud_Detection_Research_Level_Submission.zip",
    "Synthetic_Identity_Fraud_Detection_Final_Submission.zip"
]

included_rel_paths = [
    "README.md",
    "requirements.txt",
    "data/generate_data.py",
    "data/synthetic_kyc_behavioral.csv",
    "model/train_model.py",
    "model/fraud_model.joblib",
    "model/feature_cols.joblib",
    "model/metrics.json",
    "model/research_metrics.json",
    "app/dashboard.py",
    "report/report.md",
    "report/SIMPLE_PROJECT_EXPLANATION.md",
    "report/SIMPLE_PROJECT_EXPLANATION.pdf",
    "report/SIMPLE_PROJECT_EXPLANATION.docx",
    "report/research_paper.pdf",
    "report/research_paper.docx",
    "report/presentation.pptx",
    "report/presentation.pdf",
    "report/generate_charts.py",
    "report/generate_research_paper.py",
    "report/generate_research_ppt.py",
    "report/generate_ppt_pdf.py",
    "report/images/roc_curve.png",
    "report/images/roc_curves_comparison.png",
    "report/images/pr_curves_comparison.png",
    "report/images/confusion_matrix.png",
    "report/images/feature_importance.png",
    "report/images/feature_importance_shap.png",
    "report/images/radar_model_benchmark.png",
]

def build_zip(zip_name, dest_dir):
    out_zip = os.path.join(dest_dir, zip_name)
    with zipfile.ZipFile(out_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for rel in included_rel_paths:
            abs_p = os.path.join(BASE, rel)
            if os.path.exists(abs_p):
                archive_name = os.path.join("fraud_project", rel)
                zipf.write(abs_p, archive_name)
                print(f"Added to {zip_name}: {rel}")
            else:
                print(f"Warning: file missing, skipped: {rel}")
    print(f"Created ZIP: {out_zip}\n")

for z_name in zip_names:
    build_zip(z_name, BASE)
    build_zip(z_name, os.path.dirname(BASE))

print("All Research-Level Submission ZIP Packages created successfully!")
