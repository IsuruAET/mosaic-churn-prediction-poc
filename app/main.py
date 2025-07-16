import streamlit as st
import requests
import sys
import os
from dotenv import load_dotenv
from datetime import datetime

# Load environment variables from .env file
load_dotenv()

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.loader import fetch_customer_data

st.title("🧠 Customer Churn Predictor")

df = fetch_customer_data()
st.subheader("📊 Customer Order History")

# Display the dataframe with custom column names
df_display = df.copy()
df_display.columns = [
    'Customer ID',
    'Days Since Last Purchase', 
    'Avg Orders per Month',
    'Total Spent',
    'Avg Spent per Order',
    'Order Duration (Months)',
    'Churn Status'
]

st.dataframe(df_display, use_container_width=True, height=300)

# CSV Download functionality
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
csv_filename = f"Mosaic_Test_Customer_Order_History_{timestamp}.csv"

# Convert dataframe to CSV
csv_data = df.to_csv(index=False)

# Download button
st.download_button(
    label="📥 Download CSV",
    data=csv_data,
    file_name=csv_filename,
    mime="text/csv",
    help="Download the customer order history as a CSV file"
)

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
    
    with st.spinner("🤖 Analyzing customer data and generating predictions..."):
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
                ai_recommendations = result.get('ai_recommendations')
                
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
                
                # AI-Powered Recommendations Section
                st.subheader("🤖 AI-Powered Recommendations")
                
                if ai_recommendations:
                    # Overall Strategy
                    if 'overall_strategy' in ai_recommendations:
                        st.info(f"**Overall Strategy:** {ai_recommendations['overall_strategy']}")
                    
                    # Recommendations for each factor
                    if 'recommendations' in ai_recommendations:
                        for i, factor in enumerate(top_reasons, 1):
                            if factor in ai_recommendations['recommendations']:
                                factor_data = ai_recommendations['recommendations'][factor]
                                
                                # Create expandable section for each factor
                                with st.expander(f"🔍 Factor {i}: {factor.replace('_', ' ').title()}", expanded=True):
                                    col1, col2 = st.columns([2, 1])
                                    
                                    with col1:
                                        # Analysis
                                        st.markdown(f"**Analysis:** {factor_data.get('analysis', 'No analysis available')}")
                                        
                                        # Recommendations
                                        st.markdown("**Recommendations:**")
                                        for j, rec in enumerate(factor_data.get('recommendations', []), 1):
                                            st.markdown(f"• {rec}")
                                    
                                    with col2:
                                        # Priority indicator
                                        priority = factor_data.get('priority', 'medium')
                                        if priority == 'high':
                                            st.error("🔥 High Priority")
                                        elif priority == 'medium':
                                            st.warning("⚠️ Medium Priority")
                                        else:
                                            st.info("ℹ️ Low Priority")
                else:
                    # Fallback to basic display
                    st.subheader("🔍 Top 3 Contributing Factors")
                    for i, reason in enumerate(top_reasons, 1):
                        st.write(f"{i}. **{reason}**")
                    st.info("AI recommendations not available. Please check your OpenAI API key configuration.")
                
                # Progress bar for churn probability
                st.subheader("📊 Churn Risk Visualization")
                st.progress(churn_prob / 100)
                st.caption(f"Churn Probability: {churn_prob}%")
                
        except requests.exceptions.ConnectionError:
            st.error("❌ Cannot connect to API server. Please ensure the API is running on http://localhost:8000")
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
