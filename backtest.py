import pandas as pd
import numpy as np
import matplotlib.pyplot as plt # We will use this to save a plot file
from sklearn.ensemble import RandomForestClassifier

def backtest_strategy():
    print("Running Backtest Simulation...")
    
    # 1. Load Data
    df = pd.read_csv("gold_features.csv", index_col=0, parse_dates=True)
    
    # 2. Re-Define Predictors & Split (Same as before)
    exclude_cols = ['Signal', 'Target_Return', 'GLD', 'SP500', 'TNX', 'DXY', 'VIX']
    predictors = [c for c in df.columns if c not in exclude_cols]
    split_date = "2023-01-01"
    
    train = df.loc[df.index < split_date]
    test = df.loc[df.index >= split_date].copy() # Copy to avoid SettingWithCopy warnings
    
    # 3. Retrain Model (To ensure we have the exact same logic)
    model = RandomForestClassifier(n_estimators=200, min_samples_split=50, random_state=42)
    model.fit(train[predictors], train['Signal'])
    
    # 4. Generate Predictions
    print(f"Testing on {len(test)} days of data (2023-Present)...")
    test['Predicted_Signal'] = model.predict(test[predictors])
    
    # ---------------------------------------------------------
    # 5. CALCULATE STRATEGY RETURNS
    # ---------------------------------------------------------
    # Strategy Return = (Tomorrow's Actual Return) * (Today's Decision)
    # Note: Target_Return is already 'Tomorrow's return' from our feature engineering
    test['Strategy_Return'] = test['Target_Return'] * test['Predicted_Signal']
    
    # 6. Cumulative Returns (The "Equity Curve")
    # We start with $1. Cumprod() calculates the growth.
    test['Cum_Ret_Strategy'] = (1 + test['Strategy_Return']).cumprod()
    test['Cum_Ret_Market'] = (1 + test['Target_Return']).cumprod()
    
    # 7. Calculate Metrics
    total_return_strat = (test['Cum_Ret_Strategy'].iloc[-1] - 1) * 100
    total_return_market = (test['Cum_Ret_Market'].iloc[-1] - 1) * 100
    
    # Sharpe Ratio (Risk-Adjusted Return) - Assuming 0 risk-free rate for simplicity
    daily_sharpe = test['Strategy_Return'].mean() / test['Strategy_Return'].std()
    annual_sharpe = daily_sharpe * (252**0.5)
    
    print("\n" + "="*40)
    print("BACKTEST RESULTS (Jan 2023 - Present)")
    print("="*40)
    print(f"Strategy Total Return:  {total_return_strat:.2f}%")
    print(f"Buy & Hold Gold Return: {total_return_market:.2f}%")
    print("-" * 40)
    print(f"Sharpe Ratio (Annualized): {annual_sharpe:.2f}")
    
    if total_return_strat > total_return_market:
        print("\n[SUCCESS] The AI Strategy OUTPERFORMED the market!")
    else:
        print("\n[NOTE] The AI Strategy Underperformed (Market was too strong).")

    # 8. Save the Equity Curve Plot
    plt.figure(figsize=(12, 6))
    plt.plot(test.index, test['Cum_Ret_Strategy'], label='AI Model (Strategy)', color='green')
    plt.plot(test.index, test['Cum_Ret_Market'], label='Gold (Buy & Hold)', color='gray', linestyle='--')
    plt.title('Backtest: AI Strategy vs. Buy & Hold (2023-2025)')
    plt.legend()
    plt.grid(True)
    plt.savefig("backtest_chart.png")
    print("\n[CHART SAVED] 'backtest_chart.png' generated.")

if __name__ == "__main__":
    backtest_strategy()