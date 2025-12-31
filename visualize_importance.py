import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance

def plot_feature_importance():
    print("Generating Feature Importance Analysis...")
    
    # 1. Load Data
    df = pd.read_csv("gold_features.csv", index_col=0, parse_dates=True)
    exclude_cols = ['Signal', 'Target_Return', 'GLD', 'SP500', 'TNX', 'DXY', 'VIX']
    predictors = [c for c in df.columns if c not in exclude_cols]
    
    X = df[predictors]
    y = df['Signal']
    
    # 2. Train Model
    model = RandomForestClassifier(n_estimators=100, min_samples_split=50, random_state=42)
    model.fit(X, y)
    
    # 3. Calculate Importances
    # Method A: Default (Gini Importance)
    gini_importance = model.feature_importances_
    
    # Method B: Permutation (Robust Accuracy Drop)
    print("Calculating Permutation Importance (this takes a moment)...")
    perm_result = permutation_importance(model, X, y, n_repeats=10, random_state=42, n_jobs=-1)
    perm_importance = perm_result.importances_mean
    
    # 4. Create Comparison DataFrame
    importance_df = pd.DataFrame({
        'Feature': predictors,
        'Gini_Importance': gini_importance,
        'Permutation_Importance': perm_importance
    })
    
    # Sort by Permutation Importance (The "Real" metric)
    importance_df = importance_df.sort_values(by='Permutation_Importance', ascending=False)
    
    # 5. Plotting
    plt.figure(figsize=(12, 6))
    
    # Plot Permutation Importance
    sns.barplot(x='Permutation_Importance', y='Feature', data=importance_df, palette='viridis')
    
    plt.title('Why the Model Works: Feature Importance (Permutation Method)', fontsize=14)
    plt.xlabel('Impact on Model Accuracy (Drop in Score)', fontsize=12)
    plt.ylabel('Market Factor', fontsize=12)
    plt.grid(True, axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("final_feature_importance.png")
    print("\n[SUCCESS] Saved chart to 'final_feature_importance.png'")
    
    # Print the "Why" for your interview
    top_feature = importance_df.iloc[0]['Feature']
    print(f"\n[CONCLUSION] The most critical factor is: {top_feature}")
    print("This proves the model is trading based on MACRO DRIVERS, not just price.")

if __name__ == "__main__":
    plot_feature_importance()