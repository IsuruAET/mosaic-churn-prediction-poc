import streamlit as st
import pandas as pd
import requests
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.loader import fetch_customer_data

st.title("🧠 Customer Churn Predictor")

df = fetch_customer_data()
st.subheader("📊 Customer Order History")

# Display the dataframe
st.dataframe(df, use_container_width=True, height=300)

# Row selection with selectbox
index = st.selectbox("Select Customer Row", range(len(df)), format_func=lambda x: f"Row {x} - Customer Data")

selected = df.iloc[index]

st.subheader("🔍 Predict Churn Risk")

with st.form("predict_form"):
    days_since_last_purchase = st.number_input("Days Since Last Purchase", value=int(selected['days_since_last_purchase']))
    avg_orders_per_month = st.number_input("Avg Orders per Month", value=float(selected['avg_orders_per_month']))
    total_spent = st.number_input("Total Spent", value=float(selected['total_spent']))
    avg_spent_per_order = st.number_input("Avg Spent per Order", value=float(selected['avg_spent_per_order']))
    order_duration_months = st.number_input("Order Duration (Months)", value=int(selected['order_duration_months']))
    submit = st.form_submit_button("Predict")

if submit:
    input_data = {
        "days_since_last_purchase": days_since_last_purchase,
        "avg_orders_per_month": avg_orders_per_month,
        "total_spent": total_spent,
        "avg_spent_per_order": avg_spent_per_order,
        "order_duration_months": order_duration_months,
    }
    response = requests.post("http://localhost:8000/predict", json=input_data)
    result = response.json()
    
    prob = result['probability']
    if prob >= 0.75:
        st.error(f"🚨 Prediction: {result['risk']} (Churn Probability: {result['probability']})")
    elif prob >= 0.4:
        st.warning(f"⚠️ Prediction: {result['risk']} (Churn Probability: {result['probability']})")
    else:
        st.success(f"✅ Prediction: {result['risk']} (Churn Probability: {result['probability']})")
