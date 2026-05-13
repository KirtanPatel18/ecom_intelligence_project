# 🛒 E-Commerce 360° Intelligence System

> An end-to-end data intelligence pipeline built on **805,549 real wholesale transactions** —
> covering data cleaning, EDA, RFM analysis, K-Means segmentation, sales forecasting,
> churn prediction, and a product recommendation engine — deployed as a live Streamlit web app.

![Python](https://img.shields.io/badge/Python-3.10-blue?style=flat&logo=python)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-orange?style=flat&logo=scikit-learn)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red?style=flat&logo=streamlit)
![Pandas](https://img.shields.io/badge/Pandas-Data-green?style=flat&logo=pandas)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen?style=flat)

---

## 📌 Project Highlights

| | |
|---|---|
| 🧹 **Cleaned** | 805,549 real transaction records from 1M+ raw rows |
| 📊 **Analysed** | £17,743,429 in total revenue across 2 years |
| 👥 **Segmented** | 5,878 customers into 5 meaningful clusters |
| 📈 **Forecasted** | Daily sales with 65% accuracy using Random Forest |
| ⚠️ **Predicted** | Customer churn with 69% accuracy using Logistic Regression |
| 🎯 **Recommended** | Products using customer-based and item-based filtering |
| 🚀 **Deployed** | Full 6-page interactive Streamlit web application |

---

## 🗂️ Repository Structure

```
ecom_intelligence_project/
│
├── data/
│   └── processed/
│       ├── rfm_segments.csv           ← RFM scores for all customers
│       ├── customer_segments.csv      ← K-Means cluster assignments
│       ├── daily_revenue.csv          ← Daily aggregated revenue
│       └── churn_scores.csv           ← Churn probability per customer
│
├── notebooks/
│   ├── week1_cleaning.ipynb           ← Data cleaning
│   ├── week2_eda.ipynb                ← Exploratory data analysis
│   ├── week3_rfm_kpi.ipynb            ← RFM analysis & KPIs
│   ├── week4_segmentation.ipynb       ← K-Means clustering
│   ├── week5_forecasting.ipynb        ← Sales forecasting
│   ├── week6_churn.ipynb              ← Churn prediction
│   └── week7_recommendation.ipynb     ← Recommendation engine
│
├── models/
│   ├── forecast_model.pkl             ← Random Forest forecasting model
│   ├── churn_model.pkl                ← Logistic Regression churn model
│   └── churn_scaler.pkl               ← StandardScaler for churn model
│
├── outputs/
│   └── (all EDA and ML chart PNG files)
│
├── app/
│   ├── app.py                         ← Streamlit web application
│   └── requirements.txt               ← Python dependencies
│
├── .gitignore
└── README.md
```

---

## 📦 Dataset

| | |
|---|---|
| **Source** | [UCI Online Retail II — Kaggle](https://www.kaggle.com/datasets/mashlyn/online-retail-ii-uci) |
| **Period** | December 2009 — December 2011 |
| **Raw rows** | 1,067,371 |
| **After cleaning** | 805,549 |
| **Business type** | UK-based wholesale gift-ware retailer |

> ⚠️ The raw dataset is not included due to GitHub file size limits.
> Download it from Kaggle and place it at `data/raw/online_retail_II.csv`
> then run the notebooks in order to regenerate all processed files and models.

### Dataset Columns

| Column | Description |
|---|---|
| Invoice | Transaction ID. Starts with 'C' = cancellation |
| StockCode | Unique product code |
| Description | Product name |
| Quantity | Units per transaction |
| InvoiceDate | Date and time of transaction |
| Price | Unit price in GBP (£) |
| Customer ID | Unique customer identifier |
| Country | Customer's country |

---

## 🔬 Project Workflow

### 📁 Week 1 — Data Cleaning
- Removed missing Customer IDs, cancelled orders, negative quantities and prices
- Added `TotalPrice` column (Quantity × Price)
- Reduced dataset from **1,067,371 → 805,549 clean rows** (removed 24% dirty data)

### 📊 Week 2 — Exploratory Data Analysis
Built 8 charts covering monthly revenue, top products, top countries, peak hours, and order distribution.

**Key findings:**
- Average Order Value of **£479.95** confirms wholesale (not retail) buyer base
- **November 2010** was the best month — Christmas season spike
- **United Kingdom** accounts for 90%+ of total revenue
- Peak buying hours are **10am–12pm on weekdays**

### 📋 Week 3 — RFM Analysis & KPI Dashboard

Scored every customer on **Recency, Frequency, and Monetary** value (scale 1–4).

```
Total Revenue:           £17,743,429
Total Orders:            36,969
Unique Customers:        5,878
Average Order Value:     £479.95
Repeat Customer Rate:    72.4%      ← exceptionally high loyalty
Champions:               661
Lost Customers:          734
```

### 👥 Week 4 — Customer Segmentation (K-Means)

Used Elbow Method to select **K=5** and segmented all customers:

| Segment | Customers | Avg Recency | Avg Orders | Avg Spend |
|---|---|---|---|---|
| Ultra VIP Wholesalers | 4 | 3.5 days | 212 | £436,835 |
| Big Spenders | 24 | 22 days | 119 | £100,927 |
| Loyal Regulars | 383 | 28 days | 28 | £13,935 |
| Average Customers | 3,550 | 75 days | 5 | £1,912 |
| Lost / Inactive | 1,917 | 471 days | 2 | £755 |

### 📈 Week 5 — Sales Forecasting

- Aggregated daily revenue and engineered 7 features (lag, rolling average, day of week, etc.)
- Trained **Random Forest Regressor** with time-based 80/20 split

```
MAE:       £11,241
RMSE:      £18,158
Accuracy:  65.12%
```

> Note: 65% accuracy is realistic for volatile B2B wholesale data with extreme order spikes.

### ⚠️ Week 6 — Churn Prediction

- Defined churn as **no purchase in 90+ days**
- Removed Recency from features to **prevent data leakage**
- Handled class imbalance using **SMOTE**
- **Logistic Regression (69%) outperformed Random Forest (63%)**

> Key insight: Simpler model won because the decision boundary between churned and active customers was approximately linear after removing Recency.

### 🎯 Week 7 — Recommendation Engine

Built two types of collaborative filtering:

**Customer-based:** Find similar customers → recommend what they bought

**Item-based:** "Customers who bought X also bought Y"

Sample result — REGENCY CAKESTAND 3 TIER also bought:
1. ROSES REGENCY TEACUP AND SAUCER
2. GREEN REGENCY TEACUP AND SAUCER
3. PINK REGENCY TEACUP AND SAUCER
4. SET OF 3 REGENCY CAKE TINS
5. REGENCY TEAPOT ROSES

### 🚀 Week 8 — Streamlit Web App

Built a 6-page interactive web application:

| Page | Features |
|---|---|
| 📊 Overview | 5 KPI cards + monthly revenue + top countries + top products |
| 🔍 EDA | Revenue by day, hour, and top customers |
| 👥 Customer Segments | Pie chart + revenue by segment + details table |
| 📈 Sales Forecast | Actual vs predicted chart + accuracy metrics |
| ⚠️ Churn Predictor | Enter Customer ID → churn probability + risk level |
| 🎯 Recommendations | Customer-based + item-based product recommendations |

---

## ⚙️ How to Run Locally

**1. Clone the repository**
```bash
git clone https://github.com/KirtanPatel18/ecom_intelligence_project.git
cd ecom_intelligence_project
```

**2. Install dependencies**
```bash
pip install -r app/requirements.txt
```

**3. Download the dataset**

Download from [Kaggle](https://www.kaggle.com/datasets/mashlyn/online-retail-ii-uci) and place at:
```
data/raw/online_retail_II.csv
```

**4. Run notebooks in order**
```
notebooks/week1_cleaning.ipynb
notebooks/week2_eda.ipynb
notebooks/week3_rfm_kpi.ipynb
notebooks/week4_segmentation.ipynb
notebooks/week5_forecasting.ipynb
notebooks/week6_churn.ipynb
notebooks/week7_recommendation.ipynb
```

**5. Launch the Streamlit app**
```bash
cd app
python -m streamlit run app.py
```
App opens at: `http://localhost:8501`

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3 |
| Data Processing | Pandas, NumPy |
| Visualisation | Matplotlib, Seaborn, Plotly |
| Machine Learning | Scikit-learn, Imbalanced-learn |
| Web App | Streamlit |
| Environment | Jupyter Notebook |

---

## 💡 Key Learnings

**Data leakage is real** — Including Recency as a churn feature gave 100% accuracy (wrong). Removing it gave honest 69% (right).

**Simpler models can win** — Logistic Regression beat Random Forest because the decision boundary was linear.

**Domain knowledge matters** — Understanding this is a wholesale business explained the high AOV and repeat rate immediately.

**Real data is messy** — 26% of raw rows were removed during cleaning alone.

---

## 👤 About

**Name:** Kirtan Patel
**LinkedIn:** [*(add your LinkedIn URL)](https://www.linkedin.com/in/kirtan-patel-kp22143kp20/)*
**Email:** *kirtanpatel0888@gmail.com*
**Location:** Surat, Gujarat, India

> Built as a complete portfolio project covering Data Analytics, Business Analytics,
> Data Science, Machine Learning, and AI on a single real-world dataset.

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
