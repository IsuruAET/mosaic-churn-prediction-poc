from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()
model_data = joblib.load("models/churn_model.pkl")
model = model_data['model']

class CustomerInput(BaseModel):
    days_since_last_purchase: int
    avg_orders_per_month: float
    total_spent: float
    avg_spent_per_order: float
    order_duration_months: int

@app.post("/predict")
def predict_churn(data: CustomerInput):
    logger.info(f"Received prediction request with data: {data}")
    
    X = np.array([[data.days_since_last_purchase, data.avg_orders_per_month,
                   data.total_spent, data.avg_spent_per_order, data.order_duration_months]])
    
    logger.info(f"Input array shape: {X.shape}")
    logger.info(f"Input array: {X}")
    
    try:
        prob = model.predict_proba(X)[0][1]
        prediction = int(model.predict(X)[0])
        
        logger.info(f"Prediction: {prediction}, Probability: {prob}")
        
        if prob >= 0.75:
            risk = "High Risk"
        elif prob >= 0.4:
            risk = "Mid Risk"
        else:
            risk = "Low Risk"

        result = {
            "prediction": prediction,
            "risk": risk,
            "probability": round(prob, 2)
        }
        
        logger.info(f"Returning result: {result}")
        return result
        
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        return {"error": str(e)}

@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": model is not None}
