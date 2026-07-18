import os
import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from preprocess import get_train_test_splits

def train_model():
    print("Fetching preprocessed data...")
    X_train, X_test, y_train, y_test = get_train_test_splits()
    
    print("Training Random Forest Regressor...")
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    
    # Evaluation
    y_pred = rf.predict(X_test)
    r2 = r2_score(y_test, y_pred) * 100
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    print(f"\nModel Training Metrics:")
    print(f"R2 Score: {r2:.2f}%")
    print(f"RMSE: {rmse:.4f}")
    
    # Save Model Locally (It will be ignored by Git automatically due to .gitignore)
    os.makedirs('../models', exist_ok=True)
    model_path = '../models/house_price_model.pkl'
    joblib.dump(rf, model_path)
    print(f"\nModel successfully saved locally at: {model_path}")

if __name__ == "__main__":
    train_model()