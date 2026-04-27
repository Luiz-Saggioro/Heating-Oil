# Energy Intelligence Dashboard

Deterministic commodity probability engine for **Heating Oil** and **WTI Crude Oil**.

## 🚀 Deploy to Streamlit Cloud (Free)

### Step 1 — Push to GitHub
```bash
git init
git add .
git commit -m "Initial Energy Intelligence Dashboard"
git remote add origin https://github.com/YOUR_USERNAME/energy-dashboard.git
git push -u origin main
```

### Step 2 — Deploy on Streamlit Cloud
1. Go to **[share.streamlit.io](https://share.streamlit.io)**
2. Sign in with GitHub
3. Click **"New app"**
4. Select your repository and set:
   - **Main file path:** `streamlit_app.py`
   - **Python version:** 3.11
5. Click **Deploy** — it's free!

Your app will be live at:  
`https://YOUR_USERNAME-energy-dashboard-streamlit-app-XXXX.streamlit.app`

---

## 📁 File Structure

```
├── streamlit_app.py      ← Main Streamlit application (run this)
├── ho_agent.py           ← Heating Oil probability engine
├── oil_agent_v2.py       ← WTI/Brent oil price agent
├── data_fetcher.py       ← Multi-source live price fetcher (Yahoo/Stooq/FRED)
├── requirements.txt      ← Python dependencies for Streamlit Cloud
├── .streamlit/
│   └── config.toml       ← Streamlit theme configuration
└── README.md
```

## 🔧 Run Locally

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Then open: http://localhost:8501

---

## 📊 Features

| Visual | Description |
|--------|-------------|
| ① KPI Snapshot | Live prices, margins, CI range |
| ② Price History + CI | Multi-band 80/90/95% forecast cone |
| ③ Probability Distribution | Log-normal + bootstrap ensemble by horizon |
| ④ Volatility Heatmap | Rolling 10-day annualised vol calendar |
| ⑤ Driver Analysis | SHAP-style feature importance |
| ⑥ Scenario Simulation | 5 macro scenarios, 14-day paths |
| ⑦ Regional Price Map | US geographic price distribution |
| ⑧ Profit/Cost Dashboard | Retail, margin, and breakeven over time |

All panels are **cross-filtered** — selecting a scenario, region, driver, or price bin in the sidebar updates every chart simultaneously.

---

## 📧 Contact
lsaggioro@potonmail.com
