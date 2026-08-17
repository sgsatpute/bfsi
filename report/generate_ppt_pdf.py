import os
import json
from reportlab.lib.pagesizes import landscape, letter
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, HRFlowable, PageBreak
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPORT_DIR = os.path.join(BASE, "report")
IMG_DIR = os.path.join(REPORT_DIR, "images")
PDF_PATH = os.path.join(REPORT_DIR, "presentation.pdf")

doc = SimpleDocTemplate(
    PDF_PATH,
    pagesize=landscape(letter),
    rightMargin=30, leftMargin=30, topMargin=25, bottomMargin=25
)

styles = getSampleStyleSheet()
primary_color = colors.HexColor('#0F172A')   # Slate 900
secondary_color = colors.HexColor('#2563EB') # Blue 600
dark_text = colors.HexColor('#1E293B')       # Slate 800

slide_title_style = ParagraphStyle(
    'SlideTitle',
    fontName='Helvetica-Bold',
    fontSize=18,
    leading=22,
    textColor=primary_color,
    spaceAfter=4
)

slide_cat_style = ParagraphStyle(
    'SlideCat',
    fontName='Helvetica-Bold',
    fontSize=9,
    leading=11,
    textColor=secondary_color,
    spaceAfter=2
)

card_title_style = ParagraphStyle(
    'CardTitle',
    fontName='Helvetica-Bold',
    fontSize=12,
    leading=14,
    textColor=primary_color,
    spaceAfter=6
)

card_body_style = ParagraphStyle(
    'CardBody',
    fontName='Helvetica',
    fontSize=9.5,
    leading=13,
    textColor=dark_text,
    spaceAfter=4
)

story = []

def make_header(title_text):
    p_cat = Paragraph("BFSI / AI-ML CAPSTONE PROJECT PRESENTATION", slide_cat_style)
    p_t = Paragraph(title_text, slide_title_style)
    hr = HRFlowable(width="100%", thickness=1.5, color=secondary_color, spaceAfter=10)
    return [p_cat, p_t, hr]

# Slide 1: Title
story.append(Spacer(1, 40))
t_p1 = Paragraph("SYNTHETIC IDENTITY FRAUD DETECTION", ParagraphStyle('T1', fontName='Helvetica-Bold', fontSize=26, leading=30, textColor=primary_color, alignment=TA_CENTER))
t_p2 = Paragraph("Digital Lending Onboarding via KYC Verification & Behavioral Signals", ParagraphStyle('T2', fontName='Helvetica-Bold', fontSize=14, leading=18, textColor=secondary_color, alignment=TA_CENTER, spaceAfter=20))
t_p3 = Paragraph("Final Year Capstone Project — BFSI / AI-ML Domain<br/>Machine Learning Pipeline & Live Interactive Web Dashboard", ParagraphStyle('T3', fontName='Helvetica', fontSize=11, leading=15, textColor=dark_text, alignment=TA_CENTER))

story.append(t_p1)
story.append(Spacer(1, 10))
story.append(t_p2)
story.append(Spacer(1, 15))
story.append(t_p3)
story.append(PageBreak())

# Slide 2: Executive Summary & Problem
story.extend(make_header("Executive Summary & Problem Statement"))
c1_content = (
    "<b>The Threat: Synthetic Identity Fraud</b><br/><br/>"
    "• Combines real PII (valid PAN) with fake details (burner phone, fake address).<br/>"
    "• Fastest-growing financial crime category in digital lending onboarding.<br/>"
    "• Bypasses traditional static KYC checks because individual field queries return clean responses.<br/>"
    "• No individual victim exists initially to raise an alert (credit bust-out risk)."
)
c2_content = (
    "<b>Our Machine Learning Solution</b><br/><br/>"
    "• <b>Multimodal Risk Fusion:</b> Blends KYC consistency, identity freshness, behavioral biometrics, and device velocity.<br/>"
    "• <b>Machine Learning Engine:</b> Random Forest Classifier tuned with balanced class weights.<br/>"
    "• <b>Real-Time Scoring:</b> Live risk tiering (&lt;30% Low, 30-60% Medium, &gt;60% High).<br/>"
    "• <b>Streamlit Web App:</b> Interactive onboarding decision interface."
)
t2 = Table([[Paragraph(c1_content, card_body_style), Paragraph(c2_content, card_body_style)]], colWidths=[360, 360])
t2.setStyle(TableStyle([
    ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#F8FAFC')),
    ('BOX', (0,0), (-1,-1), 1, colors.HexColor('#E2E8F0')),
    ('VALIGN', (0,0), (-1,-1), 'TOP'),
    ('TOPPADDING', (0,0), (-1,-1), 12),
    ('BOTTOMPADDING', (0,0), (-1,-1), 12),
    ('LEFTPADDING', (0,0), (-1,-1), 12),
    ('RIGHTPADDING', (0,0), (-1,-1), 12),
]))
story.append(t2)
story.append(PageBreak())

# Slide 3: Feature Architecture
story.extend(make_header("Feature Engineering & 12 Core Signals"))
f1 = "<b>1. KYC Consistency</b><br/>• name_address_mismatch_score<br/>• dob_pan_mismatch<br/>• document_reuse_count"
f2 = "<b>2. Identity Freshness</b><br/>• phone_age_days<br/>• email_age_days<br/>• credit_bureau_hit<br/>• social_footprint_score"
f3 = "<b>3. Behavioral Telemetry</b><br/>• session_fill_time_sec<br/>• typing_speed_variance"
f4 = "<b>4. Device & Network Velocity</b><br/>• device_reuse_across_apps<br/>• application_velocity_24h<br/>• ip_geolocation_mismatch"

t3 = Table([
    [Paragraph(f1, card_body_style), Paragraph(f2, card_body_style)],
    [Paragraph(f3, card_body_style), Paragraph(f4, card_body_style)]
], colWidths=[360, 360])
t3.setStyle(TableStyle([
    ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#F8FAFC')),
    ('BOX', (0,0), (-1,-1), 1, colors.HexColor('#E2E8F0')),
    ('VALIGN', (0,0), (-1,-1), 'TOP'),
    ('TOPPADDING', (0,0), (-1,-1), 10),
    ('BOTTOMPADDING', (0,0), (-1,-1), 10),
    ('LEFTPADDING', (0,0), (-1,-1), 10),
    ('RIGHTPADDING', (0,0), (-1,-1), 10),
]))
story.append(t3)
story.append(PageBreak())

# Slide 4: Model Performance & Visuals
story.extend(make_header("Model Evaluation Results & Key Metrics"))
kpi_p = Paragraph("<b>ROC-AUC: 0.8834   |   Fraud Precision: 0.9500   |   Fraud Recall: 0.7500   |   Fraud F1: 0.8400</b>", ParagraphStyle('KPI', fontName='Helvetica-Bold', fontSize=12, textColor=secondary_color, alignment=TA_CENTER))
story.append(kpi_p)
story.append(Spacer(1, 10))

img_roc = os.path.join(IMG_DIR, "roc_curve.png")
img_cm = os.path.join(IMG_DIR, "confusion_matrix.png")
if os.path.exists(img_roc) and os.path.exists(img_cm):
    img_t = Table([[Image(img_roc, width=320, height=230), Image(img_cm, width=320, height=230)]], colWidths=[350, 350])
    img_t.setStyle(TableStyle([('ALIGN', (0,0), (-1,-1), 'CENTER')]))
    story.append(img_t)

story.append(PageBreak())

# Slide 5: Feature Importances
story.extend(make_header("Top Predictive Signals (Feature Importances)"))
img_fi = os.path.join(IMG_DIR, "feature_importance.png")
fi_desc = (
    "<b>Key Feature Insights:</b><br/><br/>"
    "1. <b>Email Age (0.232):</b> Freshly registered emails (&lt;10 days) heavily correlate with fraud.<br/>"
    "2. <b>Phone Number Age (0.220):</b> Burner lines purchased recently indicate identity creation.<br/>"
    "3. <b>Name/Address Mismatch (0.182):</b> Distance score catches spliced identity credentials.<br/>"
    "4. <b>Social Footprint (0.097):</b> Zero digital presence reflects synthetic profiles.<br/>"
    "5. <b>Session Fill Time (0.096):</b> Scripted/rushed form completion indicates automated bots."
)
if os.path.exists(img_fi):
    t5 = Table([[Paragraph(fi_desc, card_body_style), Image(img_fi, width=380, height=240)]], colWidths=[300, 420])
    t5.setStyle(TableStyle([('VALIGN', (0,0), (-1,-1), 'MIDDLE')]))
    story.append(t5)

doc.build(story)
print("Successfully generated PDF presentation deck at:", PDF_PATH)
