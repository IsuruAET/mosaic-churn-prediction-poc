import pandas as pd
from sqlalchemy import create_engine

def fetch_customer_data():
    # Create SQLAlchemy engine for pandas compatibility
    engine = create_engine('mysql+mysqlconnector://root:root@localhost/mosaic_rfm_test_db')

    query = """
    SELECT 
        customer_id,
        DATEDIFF(CURDATE(), MAX(order_date)) AS days_since_last_purchase,
        COUNT(id) * 1.0 / GREATEST(TIMESTAMPDIFF(MONTH, MIN(order_date), MAX(order_date)), 1) AS avg_orders_per_month,
        SUM(order_amount) AS total_spent,
        AVG(order_amount) AS avg_spent_per_order,
        TIMESTAMPDIFF(MONTH, MIN(order_date), MAX(order_date)) AS order_duration_months,
        CASE 
            WHEN DATEDIFF(CURDATE(), MAX(order_date)) > 90 THEN 1
            ELSE 0
        END AS churn
    FROM order_detail
    GROUP BY customer_id;
    """

    df = pd.read_sql(query, engine)
    return df
