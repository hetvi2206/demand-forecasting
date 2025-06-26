# Streamlit Demand Forecasting Application

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import date, timedelta

# --- Streamlit UI Configuration ---
st.set_page_config(page_title="Demand Forecasting App", layout="centered")

# --- 1. Load Model and Features ---
try:
    model = joblib.load('xgboost_demand_model.joblib')
    features = joblib.load('model_features.joblib')
    st.sidebar.success("Model and features loaded successfully!")
except FileNotFoundError:
    st.sidebar.error("Error: Model files not found. Please run the Jupyter notebook first to train and save the model.")
    st.stop()

# Function to create features for prediction (consistent with notebook)
def create_features_for_prediction(df):
    df['year'] = df.index.year
    df['month'] = df.index.month
    df['day'] = df.index.day
    df['dayofweek'] = df.index.dayofweek
    df['dayofyear'] = df.index.dayofyear
    df['weekofyear'] = df.index.isocalendar().week.astype(int)
    df['quarter'] = df.index.quarter
    df['is_month_start'] = df.index.is_month_start.astype(int)
    df['is_month_end'] = df.index.is_month_end.astype(int)
    df['is_quarter_start'] = df.index.is_quarter_start.astype(int)
    df['is_quarter_end'] = df.index.is_quarter_end.astype(int)
    df['is_year_start'] = df.index.is_year_start.astype(int)
    df['is_year_end'] = df.index.is_year_end.astype(int)

    # Historical data for lag/rolling features
    try:
        temp_historical_df = pd.read_csv('train.csv')
        temp_historical_df.columns = temp_historical_df.columns.str.strip()
        temp_historical_df['Order Date'] = pd.to_datetime(temp_historical_df['Order Date'], format='%d/%m/%Y')
        temp_historical_df = temp_historical_df.set_index('Order Date').sort_index()

        if 'Sales' not in df.columns:
            df['Sales'] = np.nan

        df_combined = pd.concat([temp_historical_df['Sales'], df['Sales']])
        df_combined = df_combined[~df_combined.index.duplicated(keep='last')]

        df['sales_lag_1'] = df_combined.shift(1).loc[df.index]
        df['sales_lag_7'] = df_combined.shift(7).loc[df.index]
        df['sales_rolling_mean_7'] = df_combined.rolling(window=7).mean().shift(1).loc[df.index]
        df['sales_rolling_std_7'] = df_combined.rolling(window=7).std().shift(1).loc[df.index]

        for col in ['sales_lag_1', 'sales_lag_7', 'sales_rolling_mean_7', 'sales_rolling_std_7']:
            if df[col].isnull().any():
                df[col] = df[col].fillna(temp_historical_df['Sales'].mean())

    except Exception as e:
        st.sidebar.error(f"Error loading train.csv or processing features: {e}")
        for col in ['sales_lag_1', 'sales_lag_7', 'sales_rolling_mean_7', 'sales_rolling_std_7']:
            if col not in df.columns:
                df[col] = 0.0

    return df[features]

# --- Streamlit UI ---
st.title("🏭 Short-Term Demand Forecasting")
st.markdown("Use this app to predict future sales (demand) for your manufacturing plant using an XGBoost model.")

st.sidebar.header("Prediction Inputs")
prediction_type = st.sidebar.radio("Select Prediction Type:", ("Single Date", "Date Range"))

if prediction_type == "Single Date":
    predict_date = st.sidebar.date_input("Select a date for prediction:", min_value=date.today())

    if st.sidebar.button("Predict Demand"):
        if predict_date:
            st.subheader(f"Forecasting for: {predict_date.strftime('%Y-%m-%d')}")

            # Create a mini date range for context: 7 days prior to the selected date
            history_start_date = predict_date - timedelta(days=7)
            date_range = pd.date_range(start=history_start_date, end=predict_date, freq='D')
            future_df = pd.DataFrame({'Date': date_range})
            future_df['Date'] = pd.to_datetime(future_df['Date'])
            future_df = future_df.set_index('Date')

            # Create features for the entire date range
            future_features_full = create_features_for_prediction(future_df.copy())

            # Iterate through the range, predicting one by one (to update lag features)
            predictions = {}
            for current_date in date_range:
                X_current = future_features_full.loc[[current_date]]
                pred = model.predict(X_current)[0]
                predictions[current_date] = pred

                # Update sales value in df for lag/rolling feature computation
                future_df.loc[current_date, 'Sales'] = pred

                # Update lag/rolling features after each prediction
                combined_series = pd.concat([
                    pd.read_csv('train.csv', parse_dates=['Order Date'], dayfirst=True)
                    .set_index('Order Date')['Sales'],
                    future_df.loc[:current_date]['Sales']
                ])
                combined_series = combined_series[~combined_series.index.duplicated(keep='last')]

                # Update lag and rolling features for the remaining dates
                for next_date in date_range:
                    future_features_full.loc[next_date, 'sales_lag_1'] = combined_series.shift(1).loc[next_date] if next_date in combined_series.index else np.nan
                    future_features_full.loc[next_date, 'sales_lag_7'] = combined_series.shift(7).loc[next_date] if next_date in combined_series.index else np.nan
                    future_features_full.loc[next_date, 'sales_rolling_mean_7'] = combined_series.rolling(7).mean().shift(1).loc[next_date] if next_date in combined_series.index else np.nan
                    future_features_full.loc[next_date, 'sales_rolling_std_7'] = combined_series.rolling(7).std().shift(1).loc[next_date] if next_date in combined_series.index else np.nan

                # Fill any remaining NaNs
                for col in ['sales_lag_1', 'sales_lag_7', 'sales_rolling_mean_7', 'sales_rolling_std_7']:
                    future_features_full[col] = future_features_full[col].fillna(combined_series.mean())

            # Show final prediction for the selected date
            final_prediction = predictions[pd.Timestamp(predict_date)]
            st.metric(label="Predicted Sales (Demand)", value=f"{final_prediction:.2f}")
            st.success("Prediction complete with contextual history!")


        else:
            st.warning("Please select a date.")


elif prediction_type == "Date Range":
    start_date = st.sidebar.date_input("Start Date:", min_value=date.today())
    end_date = st.sidebar.date_input("End Date:", min_value=start_date + timedelta(days=1))

    if st.sidebar.button("Predict Demand for Range"):
        if start_date and start_date <= end_date:
            st.subheader(f"Forecasting from: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")

            date_range = pd.date_range(start=start_date, end=end_date, freq='D')
            future_df_range = pd.DataFrame({'Date': date_range})
            future_df_range = future_df_range.set_index('Date')

            future_features_range = create_features_for_prediction(future_df_range.copy())

            try:
                predictions_range = model.predict(future_features_range)
                forecast_df = pd.DataFrame({
                    'Date': date_range,
                    'Predicted Sales': predictions_range
                }).set_index('Date')

                st.dataframe(forecast_df.style.format({'Predicted Sales': '{:.2f}'}))
                st.line_chart(forecast_df['Predicted Sales'], use_container_width=True)

                st.success("Range prediction complete!")
            except Exception as e:
                st.error(f"Error during range prediction: {e}")
        else:
            st.warning("Please select a valid date range.")

st.markdown("---")
st.markdown("### How to Use:")
st.markdown("1. Run the Jupyter Notebook to train and save the model and features.")
st.markdown("2. Place `train.csv`, `xgboost_demand_model.joblib`, and `model_features.joblib` in the same directory.")
st.markdown("3. Run the app via `streamlit app.py`.")
st.markdown("4. Select prediction type and dates in the sidebar, and click 'Predict Demand'.")
