import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import precision_score, classification_report

def optimize_quant_model():
    print("Loading engineered features...")
    df = pd.read_csv("gold_features.csv", index_col=0, parse_dates=True)

    # 1. Setup Data
    exclude_cols = ['Signal', 'Target_Return', 'GLD', 'SP500', 'TNX', 'DXY', 'VIX']
    predictors = [c for c in df.columns if c not in exclude_cols]
    
    # Strict Split (Train on History, Validate on Recent Past)
    # We use 2010-2022 for Grid Search to find best params
    train_data = df.loc[df.index < "2023-01-01"]
    test_data = df.loc[df.index >= "2023-01-01"]
    
    X_train = train_data[predictors]
    y_train = train_data['Signal']
    X_test = test_data[predictors]
    y_test = test_data['Signal']

    print("\nStarting Grid Search (Finding Best Parameters)...")
    
    # Define the "Grid" of options to test
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, None],       # None = infinite depth (risk of overfitting)
        'min_samples_split': [20, 50, 100], # Higher = more conservative
        'max_features': ['sqrt', 'log2']    # How many features to look at per tree
    }

    # TimeSeriesSplit: Ensures we don't shuffle time (Train on Jan, Validate Feb)
    tscv = TimeSeriesSplit(n_splits=3)
    
    rf = RandomForestClassifier(random_state=42)
    
    # Run the search
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, 
                               cv=tscv, scoring='precision', n_jobs=-1, verbose=1)
    
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    print(f"\n[WINNER] Best Params: {grid_search.best_params_}")

    
    print("\nTesting 'High Confidence' Thresholds on 2023-Present Data...")
    
    # Get raw probabilities (e.g., 0.55, 0.42, 0.80) instead of just 0/1
    probs = best_model.predict_proba(X_test)[:, 1] # Probability of "Up" (Class 1)
    
    # Test different thresholds
    thresholds = [0.50, 0.55, 0.60]
    
    for thr in thresholds:
        # Create custom predictions based on threshold
        custom_preds = (probs >= thr).astype(int)
        
        # Calculate precision
        prec = precision_score(y_test, custom_preds, zero_division=0)
        n_trades = sum(custom_preds)
        
        print(f"Threshold > {thr:.2f} | Precision: {prec:.4f} | Total Trades: {n_trades}")

  
    print("\nOptimized Model Feature Importance:")
    importances = pd.Series(best_model.feature_importances_, index=predictors)
    print(importances.sort_values(ascending=False).head(3))

if __name__ == "__main__":
    optimize_quant_model()