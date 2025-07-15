from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib

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
    
    # Prepare features and target
    feature_columns = required_columns[:-1]  # All except 'churn'
    X = df[feature_columns]
    y = df['churn']
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
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
    
    # Print results
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
    print(f"Test accuracy: {accuracy:.4f}")
    print("\nConfusion Matrix:")
    print(conf_matrix)
    print("\nClassification Report:")
    print(class_report)
    
    # Save model and metadata
    model_data = {
        'model': best_model,
        'feature_columns': feature_columns,
        'best_params': grid_search.best_params_,
        'test_accuracy': accuracy,
        'confusion_matrix': conf_matrix,
        'classification_report': class_report
    }
    
    joblib.dump(model_data, 'models/churn_model.pkl')
    
    return best_model, accuracy, conf_matrix, class_report
