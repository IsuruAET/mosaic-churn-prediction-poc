from fastapi import FastAPI
from pydantic import BaseModel
import logging
import sys
from pathlib import Path

# Add parent directory to path to import training module
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
sys.path.append(str(parent_dir))

from models.training import predict_churn_risk

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

class CustomerInput(BaseModel):
    days_since_last_purchase: int
    avg_orders_per_month: float
    total_spent: float
    avg_spent_per_order: float
    order_duration_months: int

@app.post("/predict")
def predict_churn(data: CustomerInput):
    logger.info(f"Received prediction request with data: {data}")
    
    # Convert to dictionary format expected by predict_churn_risk
    input_data = {
        "days_since_last_purchase": data.days_since_last_purchase,
        "avg_orders_per_month": data.avg_orders_per_month,
        "total_spent": data.total_spent,
        "avg_spent_per_order": data.avg_spent_per_order,
        "order_duration_months": data.order_duration_months
    }
    
    try:
        # Use the enhanced prediction function
        result = predict_churn_risk(input_data)
        
        logger.info(f"Prediction result: {result}")
        return result
        
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        return {"error": str(e)}

@app.get("/health")
def health_check():
    return {"status": "healthy"}
