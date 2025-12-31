import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, classification_report

def train_quant_model():
    print("Loading engineered features...")
    df = pd.read_csv("gold_features.csv", index_col=0, parse_dates=True)

    # 1. Define Predictors (X) and Target (y)
    exclude_cols = ['Signal', 'Target_Return', 'GLD', 'SP500', 'TNX', 'DXY', 'VIX']
    predictors = [c for c in df.columns if c not in exclude_cols]
    
    print(f"\nUsing {len(predictors)} Predictors: {predictors}")

    # 2. Strict Temporal Split (No Shuffling!)
    split_date = "2023-01-01"
    train = df.loc[df.index < split_date]
    test = df.loc[df.index >= split_date]

    X_train = train[predictors]
    y_train = train['Signal']
    X_test = test[predictors]
    y_test = test['Signal']

    print(f"\nTraining Range: {train.index.min().date()} to {train.index.max().date()}")
    print(f"Testing Range:  {test.index.min().date()} to {test.index.max().date()}")

    # 3. Initialize Random Forest
    model = RandomForestClassifier(n_estimators=200, min_samples_split=50, random_state=42)

    # 4. Train
    print("\nTraining Random Forest...")
    model.fit(X_train, y_train)

    # 5. Evaluate
    preds = model.predict(X_test)
    precision = precision_score(y_test, preds)
    
    print("\n" + "="*40)
    print(f"MODEL RESULTS (Test Set: {split_date}+)")
    print("="*40)
    print(f"Precision Score (Buy Signal): {precision:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, preds))

    # 6. Feature Importance (Text Output Only)
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1] # Sort descending

    print("\n" + "="*40)
    print("TOP DRIVERS OF GOLD PRICE (Feature Importance)")
    print("="*40)
    print(f"{'Rank':<5} | {'Feature':<15} | {'Importance':<10}")
    print("-" * 35)
    
    for i in range(len(predictors)):
        print(f"{i+1:<5} | {predictors[indices[i]]:<15} | {importances[indices[i]]:.4f}")

if __name__ == "__main__":
    train_quant_model()