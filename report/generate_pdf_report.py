import os
import json
import pandas as pd
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, HRFlowable, KeepTogether, PageBreak
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY, TA_RIGHT

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPORT_DIR = os.path.join(BASE, "report")
IMG_DIR = os.path.join(REPORT_DIR, "images")
PDF_PATH = os.path.join(REPORT_DIR, "report.pdf")

doc = SimpleDocTemplate(
    PDF_PATH,
    pagesize=letter,
    rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=40
)

styles = getSampleStyleSheet()

# Custom styles
primary_color = colors.HexColor('#0F172A')   # Slate 900
secondary_color = colors.HexColor('#2563EB') # Blue 600
dark_text = colors.HexColor('#1E293B')       # Slate 800

title_style = ParagraphStyle(
    'DocTitle',
    parent=styles['Heading1'],
    fontName='Helvetica-Bold',
    fontSize=22,
    leading=26,
    textColor=primary_color,
    alignment=TA_LEFT,
    spaceAfter=6
)

subtitle_style = ParagraphStyle(
    'DocSubTitle',
    parent=styles['Normal'],
    fontName='Helvetica-Bold',
    fontSize=13,
    leading=16,
    textColor=secondary_color,
    spaceAfter=15
)

h1_style = ParagraphStyle(
    'H1',
    fontName='Helvetica-Bold',
    fontSize=15,
    leading=18,
    textColor=primary_color,
    spaceBefore=14,
    spaceAfter=8,
    keepWithNext=True
)

h2_style = ParagraphStyle(
    'H2',
    fontName='Helvetica-Bold',
    fontSize=12,
    leading=15,
    textColor=secondary_color,
    spaceBefore=10,
    spaceAfter=6,
    keepWithNext=True
)

body_style = ParagraphStyle(
    'Body',
    fontName='Helvetica',
    fontSize=10,
    leading=14,
    textColor=dark_text,
    alignment=TA_LEFT,
    spaceAfter=8
)

bullet_style = ParagraphStyle(
    'Bullet',
    parent=body_style,
    leftIndent=15,
    firstLineIndent=-10,
    spaceAfter=4
)

table_header_style = ParagraphStyle(
    'TableHeader',
    fontName='Helvetica-Bold',
    fontSize=9,
    leading=11,
    textColor=colors.white,
    alignment=TA_LEFT
)

table_body_style = ParagraphStyle(
    'TableBody',
    fontName='Helvetica',
    fontSize=9,
    leading=11,
    textColor=dark_text,
    alignment=TA_LEFT
)

story = []

# Title & Metadata
story.append(Paragraph("Synthetic Identity Fraud Detection in Digital Lending Onboarding", title_style))
story.append(Paragraph("KYC Verification & Behavioral Telemetry Machine Learning System", subtitle_style))
story.append(HRFlowable(width="100%", thickness=2, color=secondary_color, spaceAfter=15))

# Executive Summary
story.append(Paragraph("1. Executive Summary", h1_style))
summary_text = (
    "Digital lending platforms require automated risk assessment to approve loans rapidly without compromising "
    "security. <b>Synthetic Identity Fraud (SIF)</b> poses a critical threat because perpetrators assemble composite identities "
    "combining real credential fragments (e.g. valid PAN numbers) with fabricated contact details. Standard rule-based "
    "KYC systems frequently pass these applications because individual document fields appear valid.<br/><br/>"
    "This project develops an end-to-end Machine Learning pipeline combining <b>KYC consistency scores, identity freshness indicators, "
    "behavioral biometrics, and device/network telemetry</b> to detect synthetic identities in real time."
)
story.append(Paragraph(summary_text, body_style))

# Highlights Table
data_highlights = [
    [Paragraph("Key Performance Metric", table_header_style), Paragraph("Score Achieved", table_header_style), Paragraph("Benchmark Status", table_header_style)],
    [Paragraph("ROC-AUC", table_body_style), Paragraph("<b>0.8834</b>", table_body_style), Paragraph("Exceeds Target (>0.85)", table_body_style)],
    [Paragraph("Fraud Precision", table_body_style), Paragraph("<b>0.9500</b>", table_body_style), Paragraph("Low False Positives (Minimizes Friction)", table_body_style)],
    [Paragraph("Fraud Recall", table_body_style), Paragraph("<b>0.7500</b>", table_body_style), Paragraph("Catches 3/4 Synthetic Applicants", table_body_style)],
    [Paragraph("Fraud F1-Score", table_body_style), Paragraph("<b>0.8400</b>", table_body_style), Paragraph("Strong Imbalanced Performance", table_body_style)],
    [Paragraph("Overall Model Accuracy", table_body_style), Paragraph("<b>0.9600</b>", table_body_style), Paragraph("High Overall Reliability", table_body_style)],
]
t_high = Table(data_highlights, colWidths=[160, 120, 220])
t_high.setStyle(TableStyle([
    ('BACKGROUND', (0, 0), (-1, 0), primary_color),
    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
    ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#CBD5E1')),
    ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F8FAFC')]),
    ('TOPPADDING', (0, 0), (-1, -1), 6),
    ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
]))
story.append(t_high)
story.append(Spacer(1, 12))

# Problem Statement
story.append(Paragraph("2. Problem Statement & Risk Vectors", h1_style))
p_stmt = (
    "Synthetic identity fraudsters exploit traditional KYC gaps through two main strategies:<br/>"
    "<b>1. Fragmented Synthesis:</b> Combining a real citizen's PAN/SSN with a fabricated name or address.<br/>"
    "<b>2. Credit Building & Bust-Out:</b> Maintaining thin credit files over several months to increase credit limits before maxing out loan lines and abandoning the account.<br/><br/>"
    "Because synthetic applications do not immediately generate victim fraud complaints, machine learning models must look beyond basic static document checks and evaluate behavioral signals (form fill timing, keypress speed variance) and network application velocity."
)
story.append(Paragraph(p_stmt, body_style))

# Feature Architecture Table
story.append(Paragraph("3. Feature Dictionary (12 Core Signals)", h1_style))
feat_data = [
    [Paragraph("Category", table_header_style), Paragraph("Feature Name", table_header_style), Paragraph("Description", table_header_style)],
    [Paragraph("KYC Consistency", table_body_style), Paragraph("name_address_mismatch_score", table_body_style), Paragraph("Distance score between stated name/address & records.", table_body_style)],
    [Paragraph("KYC Consistency", table_body_style), Paragraph("dob_pan_mismatch", table_body_style), Paragraph("Binary flag if DOB mismatches PAN database.", table_body_style)],
    [Paragraph("KYC Consistency", table_body_style), Paragraph("document_reuse_count", table_body_style), Paragraph("Times document image fragments appeared previously.", table_body_style)],
    [Paragraph("Identity Freshness", table_body_style), Paragraph("phone_age_days", table_body_style), Paragraph("Age of registered mobile line in days.", table_body_style)],
    [Paragraph("Identity Freshness", table_body_style), Paragraph("email_age_days", table_body_style), Paragraph("Estimated age of applicant email account.", table_body_style)],
    [Paragraph("Identity Freshness", table_body_style), Paragraph("credit_bureau_hit", table_body_style), Paragraph("1 if credit bureau record exists; 0 for thin file.", table_body_style)],
    [Paragraph("Identity Freshness", table_body_style), Paragraph("social_footprint_score", table_body_style), Paragraph("Composite public digital footprint score.", table_body_style)],
    [Paragraph("Behavioral", table_body_style), Paragraph("session_fill_time_sec", table_body_style), Paragraph("Total form completion duration in seconds.", table_body_style)],
    [Paragraph("Behavioral", table_body_style), Paragraph("typing_speed_variance", table_body_style), Paragraph("Keypress speed variance (low = bot/copy-paste).", table_body_style)],
    [Paragraph("Device & Network", table_body_style), Paragraph("device_reuse_across_apps", table_body_style), Paragraph("Distinct identities on same device fingerprint.", table_body_style)],
    [Paragraph("Device & Network", table_body_style), Paragraph("application_velocity_24h", table_body_style), Paragraph("Applications submitted from same IP/device in 24h.", table_body_style)],
    [Paragraph("Device & Network", table_body_style), Paragraph("ip_geolocation_mismatch", table_body_style), Paragraph("Mismatch between IP location & stated address.", table_body_style)],
]
t_feat = Table(feat_data, colWidths=[110, 160, 230])
t_feat.setStyle(TableStyle([
    ('BACKGROUND', (0, 0), (-1, 0), primary_color),
    ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#CBD5E1')),
    ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F8FAFC')]),
    ('TOPPADDING', (0, 0), (-1, -1), 4),
    ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
]))
story.append(t_feat)
story.append(Spacer(1, 15))

# Model Results & Visual Charts
story.append(Paragraph("4. Machine Learning Model & Empirical Evaluation", h1_style))
model_desc = (
    "A <b>Balanced Random Forest Classifier</b> (300 trees, max depth 8, <code>class_weight='balanced'</code>) was trained on 8,000 synthetic applications. "
    "Class weighting adjusts for fraud imbalance (~15% positive class). Evaluation on an independent 20% test set (1,600 applications) yielded the following visual results:"
)
story.append(Paragraph(model_desc, body_style))
story.append(Spacer(1, 8))

# Images layout
img_roc = os.path.join(IMG_DIR, "roc_curve.png")
img_cm = os.path.join(IMG_DIR, "confusion_matrix.png")
img_fi = os.path.join(IMG_DIR, "feature_importance.png")

if os.path.exists(img_roc) and os.path.exists(img_cm):
    img_t = Table([
        [Image(img_roc, width=240, height=188), Image(img_cm, width=240, height=188)]
    ], colWidths=[250, 250])
    img_t.setStyle(TableStyle([('ALIGN', (0,0), (-1,-1), 'CENTER')]))
    story.append(img_t)
    story.append(Spacer(1, 10))

if os.path.exists(img_fi):
    story.append(Image(img_fi, width=480, height=240))
    story.append(Spacer(1, 12))

# Team Task Matrix
story.append(Paragraph("5. Team Task Allocation & Responsibilities", h1_style))
team_data = [
    [Paragraph("Role / Task Area", table_header_style), Paragraph("Key Responsibilities", table_header_style), Paragraph("Deliverables Created", table_header_style)],
    [Paragraph("Lead AI/ML Engineer", table_body_style), Paragraph("Model selection, cross-validation, hyperparameter tuning, metric saving.", table_body_style), Paragraph("train_model.py, fraud_model.joblib", table_body_style)],
    [Paragraph("Data & Feature Engineer", table_body_style), Paragraph("Dataset generation, feature distributions, realistic noise modeling.", table_body_style), Paragraph("generate_data.py, synthetic_kyc.csv", table_body_style)],
    [Paragraph("UI / App Engineer", table_body_style), Paragraph("Streamlit app development, preset application profiles, live scoring.", table_body_style), Paragraph("dashboard.py", table_body_style)],
    [Paragraph("Research & Documentation", table_body_style), Paragraph("Literature survey, technical report drafting, presentation slides.", table_body_style), Paragraph("report.md, report.pdf, presentation.pptx", table_body_style)],
]
t_team = Table(team_data, colWidths=[120, 200, 180])
t_team.setStyle(TableStyle([
    ('BACKGROUND', (0, 0), (-1, 0), primary_color),
    ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#CBD5E1')),
    ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F8FAFC')]),
    ('TOPPADDING', (0, 0), (-1, -1), 5),
    ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
]))
story.append(t_team)
story.append(Spacer(1, 15))

# Conclusion
story.append(Paragraph("6. Conclusion", h1_style))
conc_text = (
    "The Synthetic Identity Fraud Detection system achieves an ROC-AUC of <b>0.8834</b> and a Fraud Precision of <b>0.9500</b>, "
    "demonstrating that combining KYC consistency, identity freshness, and behavioral telemetry effectively isolates synthetic applicants. "
    "The accompanying Streamlit dashboard provides a production-ready demonstration for live digital lending onboarding pipelines."
)
story.append(Paragraph(conc_text, body_style))

doc.build(story)
print("Successfully generated PDF report at:", PDF_PATH)
