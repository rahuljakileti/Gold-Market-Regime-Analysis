import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import PartialDependenceDisplay, permutation_importance
from sklearn.model_selection import train_test_split

def explain_machine_learning():
    print("Loading data for ML Interpretability...")
    df = pd.read_csv("gold_features.csv", index_col=0, parse_dates=True)
    
    # Setup Data
    exclude_cols = ['Signal', 'Target_Return', 'GLD', 'SP500', 'TNX', 'DXY', 'VIX']
    predictors = [c for c in df.columns if c not in exclude_cols]
    
    X = df[predictors]
    y = df['Signal']
    
    # Train a strong Random Forest (as used in your analysis)
    print("Training Random Forest for inspection...")
    model = RandomForestClassifier(n_estimators=100, min_samples_split=50, random_state=42)
    model.fit(X, y)


    
    print("\n[1] Calculating Permutation Importance...")
    result = permutation_importance(model, X, y, n_repeats=10, random_state=42, n_jobs=-1)
    
    sorted_idx = result.importances_mean.argsort()
    
    plt.figure(figsize=(10, 6))
    plt.boxplot(result.importances[sorted_idx].T, vert=False, labels=X.columns[sorted_idx])
    plt.title("Permutation Importance (Model Robustness Check)")
    plt.tight_layout()
    plt.savefig("ml_permutation_importance.png")
    print(" - Saved 'ml_permutation_importance.png'")

   
    
    print("\n[2] Generating Partial Dependence Plots (PDP)...")
    print("    Mapping the relationship between 'Correlation', 'Rates', and 'Gold'...")
    
    features_to_plot = ['Corr_GLD_DXY', 'TNX_Ret', 'GLD_Vol_20']
    
    fig, ax = plt.subplots(figsize=(12, 4))
    display = PartialDependenceDisplay.from_estimator(
        model, 
        X, 
        features_to_plot, 
        kind="average", 
        ax=ax
    )
    plt.suptitle("Partial Dependence: How Variables Affect Probability of 'Buy'", y=1.05)
    plt.tight_layout()
    plt.savefig("ml_partial_dependence.png")
    print(" - Saved 'ml_partial_dependence.png'")
    
    print("\n[INTERPRETATION]")
    print("Check 'ml_partial_dependence.png'.")
    print("If the lines are wiggly/curved, you have mathematically proven that")
    print("Linear Regression (a straight line) would have failed.")

if __name__ == "__main__":
    explain_machine_learning()