# 💡 Simple Guide: Synthetic Identity Fraud Detection

**Project Title:** Synthetic Identity Fraud Detection in Digital Lending Onboarding  
**GitHub Repository:** [https://github.com/sgsatpute/bfsi](https://github.com/sgsatpute/bfsi)

---

## 1. What is this project about? (In Plain English)
Imagine a bank offering instant online loans. Anyone can apply on their mobile phone and get money in 2 minutes.

A **fraudster** wants to steal loan money. But instead of stealing a real person's identity (which would cause the real person to complain to the police immediately), the fraudster creates a **"Frankenstein Fake Identity"**:
- **Real Part:** A valid PAN or tax number (purchased or leaked online).
- **Fake Part:** Fake name, fake residential address, brand new burner phone number, and newly created email address.

This fake person **does not exist in the real world**, but they apply for a loan online!

---

## 2. Why do traditional bank checks fail?
When the bank's system checks the PAN number, the government database says: *"Yes! This PAN number is valid!"*  
Because standard banks only check one field at a time, the fake person gets approved, takes the loan money, and disappears. Since no real person's name was used directly, no victim complains for months.

---

## 3. How does OUR AI system catch this fake person?
Our AI system acts like a **smart detective** looking at **4 different categories of clues (20 signals total)**:

1. **KYC Document Clues:** Checks if the name, address, DOB, and PAN actually match each other, or if the photo ID image was reused across previous loan apps.
2. **Identity Freshness Clues:** Is the phone number only 3 days old? Is the email registered yesterday? Does this person have zero credit history at the credit bureau? (Real adults usually have older phone numbers and email addresses).
3. **Behavioral Habits (How they fill the form):**
   - **Humans:** Type at normal speed, make typos, hit backspace, and pause while reading questions.
   - **Bots / Fraudsters:** Fill out 50 form questions in 10 seconds, copy-paste everything from a file, and make zero backspaces!
4. **Device & Network Clues:** Did 5 different loan applications come from the exact same laptop or same Wi-Fi IP address today?

---

## 4. How does the AI make a decision?
Our AI model (Random Forest Classifier) combines all 20 clues and outputs a **Fraud Risk Score (0% to 100%)**:

- 🟢 **Low Risk (< 30%):** Normal genuine customer → **Instant Auto-Approval**.
- 🟡 **Medium Risk (30% - 60%):** Slightly suspicious → **Ask for Video KYC / OTP verification**.
- 🔴 **High Risk (> 60%):** Confirmed fake identity → **Block Loan Application immediately!**

---

## 5. What are our final project results?
We tested our AI model on **10,000 application records**:
- **Accuracy / ROC-AUC:** **91.4%** (Extremely high accuracy!)
- **Precision:** **86.0%** (Very low false alarms — genuine customers are not bothered).
- **Recall:** **91.1%** (Catches 9 out of every 10 fraud attempts!).

---

## 6. How to explain this to your professor in 30 seconds:
> *"Sir, traditional KYC systems only verify static document numbers, so synthetic fake identities (real PAN + fake phone/address) bypass them easily. Our project uses Machine Learning on 20 signals—combining document consistency, phone/email age, behavioral biometrics (typing cadence, paste speed), and device velocity. Our model achieves 91.4% ROC-AUC accuracy and is fully deployed with an interactive Streamlit web dashboard for live risk scoring."*

---

## 7. What files are inside the submission?
- `data/synthetic_kyc_behavioral.csv`: 10,000 application records with 20 signals.
- `model/train_model.py`: AI model training & 5-fold cross-validation script.
- `app/dashboard.py`: Live interactive web app dashboard (run: `python -m streamlit run app/dashboard.py`).
- `report/research_paper.pdf`: IEEE-style formal academic research paper.
- `report/presentation.pptx`: 12-slide PowerPoint presentation deck.
