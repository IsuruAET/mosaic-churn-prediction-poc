# Customer Churn Prediction POC

A machine learning system for predicting customer churn with enhanced features including risk segmentation, feature importance analysis, and **AI-powered recommendations**.

## Features

- **Churn Prediction**: Predicts customer churn probability
- **Risk Segmentation**: Classifies customers as Safe/Risky/High Churn Risk
- **Feature Importance**: Shows top 3 contributing factors for each prediction
- **🤖 AI-Powered Recommendations**: OpenAI-powered actionable recommendations for each contributing factor
- **Model Insights**: Displays feature importance rankings and model performance
- **Interactive UI**: Streamlit-based web interface
- **REST API**: FastAPI-based prediction endpoint

## Setup

1. **Install Dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

2. **Configure OpenAI API** (for AI recommendations):

   Set your OpenAI API key as an environment variable:

   ```bash
   export OPENAI_API_KEY="your-api-key-here"
   ```

   Or create a `.env` file in the project root:

   ```
   OPENAI_API_KEY=your-api-key-here
   ```

3. **Train the Model** (if not already trained):

   ```bash
   python scripts/train_model.py
   ```

## Running the Application

### Option 1: Using Python Scripts (Recommended)

**Start the API Server**:

```bash
python run_api.py
```

The API will be available at: http://localhost:8000

**Start the Streamlit App** (in a new terminal):

```bash
python run_app.py
```

The app will be available at: http://localhost:8501

### Option 2: Using Batch Files (Windows)

**Start the API Server**:

```bash
start_api.bat
```

**Start the Streamlit App**:

```bash
start_app.bat
```

### Option 3: Manual Commands

**API Server**:

```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

**Streamlit App**:

```bash
streamlit run app/main.py --server.port 8501 --server.address 0.0.0.0
```

## API Endpoints

- `POST /predict` - Predict churn risk with AI recommendations
- `GET /health` - Health check
- `GET /model-info` - Model information and feature importance

## Usage

1. Open the Streamlit app in your browser
2. Select a customer from the dropdown
3. Adjust the input values if needed
4. Click "Predict" to get churn risk analysis
5. View the results including:
   - Churn probability percentage
   - Risk segmentation
   - **AI-powered recommendations for each contributing factor**
   - **Overall retention strategy**

## AI-Powered Recommendations

The system now includes OpenAI-powered recommendations that provide:

- **Factor Analysis**: Detailed analysis of why each factor contributes to churn
- **Actionable Recommendations**: Specific, business-focused actions for each factor
- **Priority Levels**: High/Medium/Low priority indicators
- **Overall Strategy**: Comprehensive retention strategy summary

### Example AI Recommendations

For a customer with high churn risk due to:

1. **Days since last purchase**:

   - Analysis: Customer hasn't made a purchase recently
   - Recommendations: Re-engagement campaigns, limited-time offers
   - Priority: High

2. **Average orders per month**:

   - Analysis: Low order frequency indicates declining engagement
   - Recommendations: Loyalty programs, automated reminders
   - Priority: Medium

3. **Total spent**:
   - Analysis: Low total spend suggests limited value perception
   - Recommendations: Upselling opportunities, premium bundles
   - Priority: Medium

## Model Features

The model uses the following features:

- Days since last purchase
- Average orders per month
- Total spent
- Average spent per order
- Order duration in months

## Troubleshooting

If you encounter path issues:

1. Ensure you're running commands from the project root directory
2. Use the provided startup scripts (`run_api.py`, `run_app.py`)
3. Check that the virtual environment is activated
4. Verify that `models/churn_model.pkl` exists (train the model first if needed)

**For AI Recommendations Issues**:

1. Verify your OpenAI API key is correctly set
2. Check that you have sufficient OpenAI API credits
3. The system will fall back to basic recommendations if AI is unavailable
