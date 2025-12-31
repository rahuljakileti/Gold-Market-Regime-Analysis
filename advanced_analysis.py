import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, accuracy_score
from sklearn.utils import resample
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

def run_advanced_analysis():
    print("Loading data for Comprehensive Analysis...")
    df = pd.read_csv("gold_features.csv", index_col=0, parse_dates=True)
    
    # Define Predictors
    exclude_cols = ['Signal', 'Target_Return', 'GLD', 'SP500', 'TNX', 'DXY', 'VIX']
    predictors = [c for c in df.columns if c not in exclude_cols]
    
    # Strict Temporal Split
    split_date = "2023-01-01"
    train = df.loc[df.index < split_date]
    test = df.loc[df.index >= split_date]
    
    X_train = train[predictors]
    y_train = train['Signal']
    X_test = test[predictors]
    y_test = test['Signal']
    
    # Standardize Data (Crucial for PCA, SVM, and Regression)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # ---------------------------------------------------------
    # PART 1: LINEAR ALGEBRA (PCA / Eigen-Decomposition)
    # ---------------------------------------------------------
    print("\n[1] Performing Linear Algebra Analysis (PCA)...")
    
    pca = PCA()
    pca.fit(X_train_scaled)
    explained_variance = np.cumsum(pca.explained_variance_ratio_)
    
    # Save Scree Plot
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(explained_variance) + 1), explained_variance, marker='o', linestyle='--', color='purple')
    plt.axhline(y=0.95, color='r', linestyle='-', label='95% Threshold')
    plt.title('PCA Scree Plot: Dimensionality of Market Signals')
    plt.xlabel('Number of Principal Components')
    plt.ylabel('Cumulative Variance Explained')
    plt.legend()
    plt.grid(True)
    plt.savefig("stat_pca_scree.png")
    
    n_components = np.argmax(explained_variance >= 0.95) + 1
    print(f" -> Insight: {n_components} Principal Components explain 95% of variance.")

    # ---------------------------------------------------------
    # PART 2: MODEL BENCHMARKING (The "Competition")
    # ---------------------------------------------------------
    print("\n[2] Benchmarking Models (RF vs. SVM vs. LogReg)...")
    
    # A. Random Forest (Ensemble / Logic)
    rf_model = RandomForestClassifier(n_estimators=100, min_samples_split=50, random_state=42)
    rf_model.fit(X_train, y_train) # RF doesn't strictly need scaling, but it's fine
    rf_pred = rf_model.predict(X_test)
    
    # B. Support Vector Machine (Geometric / High-Dim)
    svm_model = SVC(kernel='rbf', C=1.0, random_state=42)
    svm_model.fit(X_train_scaled, y_train)
    svm_pred = svm_model.predict(X_test_scaled)
    
    # C. Logistic Regression (Linear / Regularized)
    # penalty='l2' is Ridge Regularization
    lr_model = LogisticRegression(penalty='l2', C=1.0, random_state=42)
    lr_model.fit(X_train_scaled, y_train)
    lr_pred = lr_model.predict(X_test_scaled)

    # Print The Leaderboard
    print("\n" + "="*50)
    print(f"{'MODEL':<20} | {'PRECISION':<10} | {'ACCURACY':<10}")
    print("-" * 50)
    
    results = [
        ("SVM (RBF Kernel)", precision_score(y_test, svm_pred), accuracy_score(y_test, svm_pred)),
        ("Random Forest", precision_score(y_test, rf_pred), accuracy_score(y_test, rf_pred)),
        ("LogReg (L2 Reg)", precision_score(y_test, lr_pred), accuracy_score(y_test, lr_pred))
    ]
    
    # Sort by Precision (Highest First)
    results.sort(key=lambda x: x[1], reverse=True)
    
    for name, prec, acc in results:
        print(f"{name:<20} | {prec:.4f}     | {acc:.4f}")
    print("="*50)
    
    winner = results[0][0]
    print(f" -> Winner: {winner}")

    # ---------------------------------------------------------
    # PART 3: INFERENTIAL STATISTICS (Bootstrapping the Winner)
    # ---------------------------------------------------------
    print(f"\n[3] Calculating Confidence Intervals for {winner}...")
    
    # Select predictions based on the winner
    if "SVM" in winner:
        best_preds = svm_pred
    elif "Random Forest" in winner:
        best_preds = rf_pred
    else:
        best_preds = lr_pred

    n_iterations = 1000
    stats = []
    
    # Create DataFrame for resampling
    data_df = pd.DataFrame({'True': y_test.values, 'Pred': best_preds})
    
    for i in range(n_iterations):
        sample = resample(data_df, n_samples=len(data_df), random_state=i)
        score = precision_score(sample['True'], sample['Pred'], zero_division=0)
        stats.append(score)
    
    # 95% Confidence Interval
    lower = np.percentile(stats, 2.5)
    upper = np.percentile(stats, 97.5)
    
    print("\n" + "="*50)
    print("STATISTICAL VALIDATION REPORT")
    print("="*50)
    print(f"95% Confidence Interval: [{lower:.4f} - {upper:.4f}]")
    print("-" * 50)
    
    if lower > 0.50:
        print("[CONCLUSION] Result is STATISTICALLY SIGNIFICANT (Lower Bound > 50%).")
    else:
        print("[CONCLUSION] Result is promising but overlaps 50%.")
        print("             (Common in low-signal-to-noise financial data)")

if __name__ == "__main__":
    run_advanced_analysis()