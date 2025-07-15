import streamlit as st
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
    
    try:
        response = requests.post("http://localhost:8000/predict", json=input_data)
        result = response.json()
        
        if "error" in result:
            st.error(f"❌ Error: {result['error']}")
        else:
            # Display enhanced results
            churn_prob = result['churn_probability']
            risk_segment = result['risk_segment']
            top_reasons = result['top_3_reasons']
            
            # Risk visualization
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Churn Probability", f"{churn_prob}%")
            
            with col2:
                if risk_segment == "Safe":
                    st.success(f"✅ {risk_segment}")
                elif risk_segment == "Risky":
                    st.warning(f"⚠️ {risk_segment}")
                else:
                    st.error(f"🚨 {risk_segment}")
            
            # Top 3 contributing features
            st.subheader("🔍 Top 3 Contributing Factors")
            for i, reason in enumerate(top_reasons, 1):
                st.write(f"{i}. **{reason}**")
            
            # Progress bar for churn probability
            st.subheader("📊 Churn Risk Visualization")
            st.progress(churn_prob / 100)
            st.caption(f"Churn Probability: {churn_prob}%")
            
    except requests.exceptions.ConnectionError:
        st.error("❌ Cannot connect to API server. Please ensure the API is running on http://localhost:8000")
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
