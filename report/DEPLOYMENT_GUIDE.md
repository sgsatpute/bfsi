# 🚀 Application Deployment Guide

**Project:** Multimodal Synthetic Identity Fraud Detection Engine  
**GitHub Repository:** [https://github.com/sgsatpute/bfsi](https://github.com/sgsatpute/bfsi)

---

## Option 1: Streamlit Community Cloud (Free Public URL — Recommended!)

Streamlit Community Cloud hosts your app directly from your GitHub repository for free in 2 minutes.

### Step-by-Step Instructions:
1. Go to **[share.streamlit.io](https://share.streamlit.io/)** and click **Log in with GitHub**.
2. Click **New app**.
3. Fill in the deployment details:
   - **Repository:** `sgsatpute/bfsi`
   - **Branch:** `main`
   - **Main file path:** `app/dashboard.py` (or `fraud_project/app/dashboard.py` if deployed from subfolder)
4. Click **Deploy!**
5. Streamlit will automatically install `requirements.txt` and launch your live public URL:  
   `https://bfsi-fraud-detection.streamlit.app`

---

## Option 2: Docker Container Deployment (Local or Cloud VM)

The repository includes a production `Dockerfile`.

### Build & Run Locally:
```bash
# 1. Build the Docker image
docker build -t bfsi-fraud-app .

# 2. Run the Docker container
docker run -p 8501:8501 bfsi-fraud-app
```
Access at: `http://localhost:8501`

---

## Option 3: Free Cloud Deployment on Hugging Face Spaces

1. Go to **[huggingface.co/spaces](https://huggingface.co/spaces)** and click **Create new Space**.
2. Select **Streamlit** as the Space SDK.
3. Upload project files or link your GitHub repository `sgsatpute/bfsi`.
4. Your app will build automatically and receive a free permanent URL.

---

## Option 4: Local & Local Network Presentation Deployment

### Run locally on your machine:
```bash
python -m streamlit run app/dashboard.py
```
- **Local URL:** `http://localhost:8501`
- **Network URL (for phone/tablet on same Wi-Fi):** `http://<your-local-ip>:8501`
