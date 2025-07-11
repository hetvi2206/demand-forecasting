
# 🏭 XDemand: Short-Term Demand Forecasting for Manufacturing Plants

**XDemand** is an AI-powered application built using **XGBoost** and **Streamlit** to forecast short-term product demand for manufacturing plants based on historical sales data.  
It uses time-series features, lag variables, and rolling statistics to provide accurate, explainable demand predictions for specific dates or date ranges.

---

## 📊 Features

- 📈 Forecast sales demand for a **single future date**.
- 📅 Predict demand for a **custom date range**.
- 🧠 XGBoost regression model trained on historical demand patterns.
- 📊 Visualize demand trends through interactive charts.
- 💾 Automatically saves trained models and feature lists for future use.
- 🔄 Uses lag and rolling window features for accurate time-series forecasting.

---

## 📦 Tech Stack

- **Python**
- **Pandas, NumPy, Matplotlib, Seaborn**
- **Scikit-learn**
- **XGBoost**
- **Streamlit**
- **Joblib**

---

## 📁 Project Structure

```
📦 demandpulse/
├── demand_forecast_notebook.ipynb   # Model training and EDA notebook
├── demandpulse_app.py               # Streamlit app for interactive forecasting
├── train.csv                        # Historical sales data
├── xgboost_demand_model.joblib      # Trained XGBoost model
├── model_features.joblib            # List of features used by the model
├── requirements.txt                 # Project dependencies
└── README.md                        # This documentation file
```

---

## 🚀 Getting Started

### 1️⃣ Install Dependencies
Create a virtual environment (recommended) and install required packages:

```bash
pip install -r requirements.txt
```

### 2️⃣ Train the Model

Open and run the Jupyter notebook:

```bash
jupyter notebook demand_forecast_notebook.ipynb
```

This will:
- Train an XGBoost model on `train.csv`
- Save `xgboost_demand_model.joblib` and `model_features.joblib` in your project directory

### 3️⃣ Run the Streamlit App

```bash
streamlit run demandpulse_app.py
```

The app will launch in your browser at `http://localhost:8501`

## 📌 Usage Instructions

1. Select a **Single Date** or **Date Range** from the sidebar.
2. Click **Predict Demand**.
3. View predicted sales numbers and interactive demand trend charts.
4. Adjust dates or retrain your model with new data when needed.

## 📚 Dataset Format (`train.csv`)

| Order Date | Sales  |
|------------|--------|
| 01/01/2024 | 2500.00|
| 02/01/2024 | 2400.50|

**Date Format:** `DD/MM/YYYY`

## 📌 Requirements

See [`requirements.txt`](requirements.txt)
