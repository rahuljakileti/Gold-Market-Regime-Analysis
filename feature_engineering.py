import pandas as pd
import numpy as np

def generate_features():
    print("Loading raw data...")
    df = pd.read_csv("gold_market_regime_data.csv", index_col=0, parse_dates=True)

    # ---------------------------------------------------------
    # 1. LOG RETURNS (The "Stationary" Feature)
    # ---------------------------------------------------------
    # We calculate Log returns for ALL columns (GLD, TNX, DXY, etc.)
    # Formula: ln(Price_t / Price_t-1)
    print("Calculating Log Returns...")
    returns = np.log(df / df.shift(1))
    
    # Rename columns to indicate they are returns (e.g., 'GLD' -> 'GLD_Ret')
    returns.columns = [f"{col}_Ret" for col in returns.columns]

    # ---------------------------------------------------------
    # 2. ROLLING VOLATILITY (The "Risk" Feature)
    # ---------------------------------------------------------
    # How much did Gold fluctuate in the last 20 days (approx 1 trading month)?
    print("Calculating Rolling Volatility...")
    # standard deviation of returns * sqrt(252) annualizes it (standard quant practice)
    # We focus on Gold's volatility specifically
    df['GLD_Vol_20'] = returns['GLD_Ret'].rolling(window=20).std()

    # ---------------------------------------------------------
    # 3. MOMENTUM (The "Trend" Feature)
    # ---------------------------------------------------------
    # Simple Moving Average of Returns (is the recent trend positive?)
    df['GLD_MA_10'] = returns['GLD_Ret'].rolling(window=10).mean()

    # ---------------------------------------------------------
    # 4. MARKET REGIME (Correlation Feature)
    # ---------------------------------------------------------
    # Does Gold move OPPOSITE to the Dollar? (Correlation should be negative)
    # When this correlation breaks (becomes positive), it's a "Regime Change" signal.
    print("Calculating Dollar-Gold Correlation...")
    df['Corr_GLD_DXY'] = returns['GLD_Ret'].rolling(window=20).corr(returns['DXY_Ret'])

    # Merge Returns back into the main DataFrame
    df_features = pd.concat([df, returns], axis=1)

   
    
    df_features['Target_Return'] = df_features['GLD_Ret'].shift(-1) # The "Truth" for tomorrow
    df_features['Signal'] = (df_features['Target_Return'] > 0).astype(int) # 1 if Up, 0 if Down

 
    df_final = df_features.dropna()

    # Save
    output_file = "gold_features.csv"
    df_final.to_csv(output_file)
    
    print(f"\n[SUCCESS] Features engineered. Saved to: {output_file}")
    print(f"Features: {df_final.shape[1]} columns")
    print(f"Data Points: {len(df_final)} rows")
    print("\nSample Data (Last 5 days):")
    print(df_final[['GLD', 'GLD_Ret', 'Signal']].tail())

if __name__ == "__main__":
    generate_features()