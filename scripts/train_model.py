# To Do: PYTHONPATH to be added to the .env file
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.loader import fetch_customer_data
from models.training import train_and_save_model

if __name__ == "__main__":
    df = fetch_customer_data()
    train_and_save_model(df)
    print("✅ Model trained and saved at models/churn_model.pkl")
