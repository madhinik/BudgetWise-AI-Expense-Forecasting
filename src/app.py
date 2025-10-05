import streamlit as st
import pandas as pd
import plotly.express as px

from preprocessing import load_data, create_time_features
from models import forecast_prophet, forecast_random_forest, forecast_xgboost
from forecasting import evaluate_forecast
from budget_optimizer import recommend_budget

# ----------------------------------------------------
# Streamlit page setup
# ----------------------------------------------------
st.set_page_config(layout="wide", page_title="💰 BudgetWise — Expense Forecaster")
st.title("💰 BudgetWise — Expense Forecaster (MVP)")

# ----------------------------------------------------
# File upload
# ----------------------------------------------------
uploaded = st.file_uploader("Upload your expenses CSV", type=["csv"])
if uploaded:
    df = load_data(uploaded)

    # Clean up missing categories
    df = df.dropna(subset=['category'])
    df = df[df['category'].astype(str).str.lower() != 'nan']

    st.write("### Raw data sample", df.head())

    # ----------------------------------------------------
    # Category normalization
    # ----------------------------------------------------
    category_map = {
        'food': 'food', 'fod': 'food', 'foodd': 'food', 'foods': 'food',
        'education': 'education', 'edu': 'education', 'educaton': 'education',
        'entertainment': 'entertainment', 'entertain': 'entertainment', 'entrtnmnt': 'entertainment',
        'rent': 'rent', 'rentt': 'rent', 'rnt': 'rent',
        'utilities': 'utilities', 'utility': 'utilities', 'utlities': 'utilities', 'utilties': 'utilities',
        'travel': 'travel', 'traval': 'travel', 'travl': 'travel',
        'health': 'health', 'helth': 'health',
        'saving': 'savings', 'savings': 'savings', 'salary': 'income', 'bonus': 'income',
        'misc': 'misc', 'other': 'misc', 'others': 'misc'
    }
    df['category_clean'] = df['category'].map(category_map).fillna(df['category'])

    # ----------------------------------------------------
    # Exploratory Data Analysis (EDA)
    # ----------------------------------------------------
    st.subheader("Exploratory Data Analysis")
    df_tf = create_time_features(df)

    # Category spending summary
    cat_counts = (
        df_tf.groupby('category_clean')['amount']
        .sum()
        .reset_index()
        .sort_values('amount', ascending=False)
    )
    st.write("#### Spending by category (total)")
    st.dataframe(cat_counts)

    # Pie chart for spending share
    fig = px.pie(cat_counts, names='category_clean', values='amount', title="Category spend share")
    st.plotly_chart(fig, use_container_width=True)

    # Time series plot of total expenses
    st.write("#### Time series (total)")
    ts = df_tf.groupby('date')['amount'].sum().reset_index()
    fig_ts = px.line(ts, x='date', y='amount', title='Total expenses over time')
    st.plotly_chart(fig_ts, use_container_width=True)

    # ----------------------------------------------------
    # Forecast Settings
    # ----------------------------------------------------
    st.sidebar.header("Forecast settings")
    horizon_days = st.sidebar.slider("Forecast horizon (days)", min_value=30, max_value=365, value=90, step=30)
    model_choice = st.sidebar.selectbox("Model (MVP)", ['Prophet', 'Random Forest', 'XGBoost'])

    # ----------------------------------------------------
    # Run Forecast
    # ----------------------------------------------------
    if st.sidebar.button("Run Forecast"):
        with st.spinner(f"Training {model_choice} model..."):
            if model_choice == 'Prophet':
                forecast_df = forecast_prophet(df, periods=horizon_days)
            elif model_choice == 'Random Forest':
                forecast_df = forecast_random_forest(df, periods=horizon_days)
            elif model_choice == 'XGBoost':
                forecast_df = forecast_xgboost(df, periods=horizon_days)

        st.success(f"Forecast ready ✅ ({model_choice})")

        # Plot forecast
        fig_f = px.line(forecast_df, x='ds', y='yhat', title="Forecast (yhat)")
        fig_f.add_scatter(
            x=forecast_df['ds'],
            y=forecast_df['yhat_upper'],
            mode='lines',
            name='upper',
            line=dict(width=0),
            showlegend=False
        )
        fig_f.add_scatter(
            x=forecast_df['ds'],
            y=forecast_df['yhat_lower'],
            mode='lines',
            name='lower',
            fill='tonexty',
            line=dict(width=0),
            showlegend=False
        )
        st.plotly_chart(fig_f, use_container_width=True)

        # ----------------------------------------------------
        # In-sample Accuracy
        # ----------------------------------------------------
        if st.sidebar.checkbox("Show in-sample accuracy (on history)"):
            in_sample = forecast_df[forecast_df['ds'] <= df['date'].max()]
            y_true = df.groupby('date')['amount'].sum().reindex(in_sample['ds']).fillna(0).values
            y_pred = in_sample['yhat'].values[:len(y_true)]
            metrics = evaluate_forecast(y_true, y_pred)
            st.write("### In-sample metrics", metrics)

        # ----------------------------------------------------
        # Budget Recommendation
        # ----------------------------------------------------
        st.subheader("Budget recommendation")
        income = st.number_input("Monthly income (currency)", min_value=0.0, value=5000.0)
        savings_target = st.slider("Desired savings (%)", 0.0, 0.8, 0.2, step=0.05)

        hist_share = df.groupby('category_clean')['amount'].sum() / df['amount'].sum()
        cat_forecast_df = pd.concat([
            pd.DataFrame({
                'ds': forecast_df['ds'],
                'category': cat,
                'yhat': forecast_df['yhat'] * share
            }) for cat, share in hist_share.items()
        ])

        out = recommend_budget(
            cat_forecast_df,
            periods=horizon_days,
            income=income,
            savings_target_pct=savings_target,
            category_col='category',
            amount_col='yhat'
        )

        # Clean recommendations (no NaN or zero)
        out['recommended_by_category'] = {
            k: v for k, v in out['recommended_by_category'].items()
            if k and pd.notna(v) and v > 0 and str(k).lower() != 'nan'
        }

        st.json(out)

        # ----------------------------------------------------
        # Download Forecast CSV
        # ----------------------------------------------------
        csv = forecast_df[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_csv(index=False).encode('utf-8')
        st.download_button(
            "Download forecast CSV",
            csv,
            file_name="forecast.csv",
            mime="text/csv"
        )
