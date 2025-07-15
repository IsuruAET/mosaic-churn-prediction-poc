from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
import joblib
import pandas as pd

def train_and_save_model(df, test_size=0.2, random_state=42):
    # Validate required columns exist
    required_columns = [
        'days_since_last_purchase',
        'avg_orders_per_month', 
        'total_spent',
        'avg_spent_per_order',
        'order_duration_months',
        'churn'
    ]
    
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # Handle missing values
    df = df.dropna()
    
    # Prepare features and target
    feature_columns = required_columns[:-1]  # All except 'churn'
    X = df[feature_columns]
    y = df['churn']
    
    # Scale numeric features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Hyperparameter tuning with GridSearchCV
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    base_model = RandomForestClassifier(random_state=random_state)
    grid_search = GridSearchCV(
        base_model, param_grid, cv=5, scoring='accuracy', n_jobs=-1
    )
    grid_search.fit(X_train, y_train)
    
    # Get best model
    best_model = grid_search.best_estimator_
    
    # Evaluate model
    y_pred = best_model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)
    
    # Feature importance analysis
    feature_importances = best_model.feature_importances_
    importance_dict = dict(zip(feature_columns, feature_importances))
    
    # Print results
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
    print(f"Test accuracy: {accuracy:.4f}")
    print("\nConfusion Matrix:")
    print(conf_matrix)
    print("\nClassification Report:")
    print(class_report)
    print("\nFeature Importance:")
    for feature, importance in sorted(importance_dict.items(), key=lambda x: x[1], reverse=True):
        print(f"{feature}: {importance:.4f}")
    
    # Save model and metadata
    model_data = {
        'model': best_model,
        'feature_columns': feature_columns,
        'best_params': grid_search.best_params_,
        'test_accuracy': accuracy,
        'confusion_matrix': conf_matrix,
        'classification_report': class_report,
        'feature_importance': importance_dict,
        'scaler': scaler
    }
    
    joblib.dump(model_data, 'models/churn_model.pkl')
    
    return best_model, accuracy, conf_matrix, class_report, importance_dict

def predict_churn_risk(input_data: dict, model_data_path='models/churn_model.pkl'):
    """
    Predict churn probability, risk segment, and top 3 contributing features.
    
    Args:
        input_data: Dictionary with feature values
        model_data_path: Path to saved model data
    
    Returns:
        Dictionary with churn probability, risk segment, and top 3 reasons
    """
    # Load model data
    model_data = joblib.load(model_data_path)
    model = model_data['model']
    scaler = model_data['scaler']
    feature_columns = model_data['feature_columns']
    
    # Create DataFrame for input
    input_df = pd.DataFrame([input_data])
    
    # Ensure all required features are present
    missing_features = [col for col in feature_columns if col not in input_df.columns]
    if missing_features:
        raise ValueError(f"Missing features: {missing_features}")
    
    # Select only required features in correct order
    input_df = input_df[feature_columns]
    
    # Scale features
    input_scaled = scaler.transform(input_df)
    
    # Predict probability
    prob_churn = model.predict_proba(input_scaled)[0][1]
    
    # Risk classification
    if prob_churn < 0.3:
        risk = "Safe"
    elif prob_churn < 0.7:
        risk = "Risky"
    else:
        risk = "High Churn Risk"
    
    # Calculate feature contributions for this specific instance
    # Use permutation importance approach: remove each feature and see how prediction changes
    base_prediction = prob_churn
    feature_contributions = {}
    
    for i, feature in enumerate(feature_columns):
        # Create a copy of the input with this feature set to its mean value (neutral)
        modified_input = input_scaled.copy()
        modified_input[0, i] = 0  # Set to 0 (mean after scaling)
        
        # Get new prediction
        modified_prediction = model.predict_proba(modified_input)[0][1]
        
        # Contribution is the difference in prediction
        contribution = base_prediction - modified_prediction
        feature_contributions[feature] = contribution
    
    # Get top 3 features that contribute most to the current prediction
    # Sort by absolute contribution (we want the most impactful features regardless of direction)
    top_3_reasons = sorted(
        feature_contributions.items(), 
        key=lambda x: abs(x[1]), 
        reverse=True
    )[:3]
    
    return {
        "churn_probability": float(round(prob_churn*100, 2)),
        "risk_segment": risk,
        "top_3_reasons": [reason for reason, _ in top_3_reasons]
    }
