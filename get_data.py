import yfinance as yf
import pandas as pd

def fetch_macro_data():
    print("Fetching generic market data...")
    
    tickers = {
        "GLD": "GLD",           # Target: Gold Price
        "SP500": "^GSPC",       # Economy: S&P 500
        "TNX": "^TNX",          # Rates: 10-Year Treasury Yield
        "DXY": "DX-Y.NYB",      # Currency: US Dollar Index
        "VIX": "^VIX"           # Sentiment: Volatility Index
    }

    # 1. Download ALL data first (don't slice yet)
    # auto_adjust=False ensures we get 'Adj Close' if available, or we handle it manually
    df = yf.download(list(tickers.values()), start="2010-01-01", end="2025-01-01", progress=True)

    # 2. Robust Column Selection
    # Check if 'Adj Close' is in the top level of columns
    if 'Adj Close' in df.columns.get_level_values(0):
        raw_data = df['Adj Close']
    elif 'Close' in df.columns.get_level_values(0):
        print("\n[NOTE] 'Adj Close' not found. Using 'Close' prices instead.")
        raw_data = df['Close']
    else:
        print("\n[ERROR] neither 'Adj Close' nor 'Close' found in data.")
        print("Columns returned:", df.columns)
        return

    # 3. Clean and Rename Columns
    # We strip any extra column levels if necessary to ensure it's just Tickers
    raw_data.columns = [c if isinstance(c, str) else c[0] for c in raw_data.columns]

    # Map the weird yfinance column names back to our clean names
    inv_tickers = {v: k for k, v in tickers.items()}
    
    # Rename matching columns
    raw_data = raw_data.rename(columns=inv_tickers)
    
    # Keep only the columns we asked for (removes any failed downloads)
    raw_data = raw_data[[k for k in tickers.keys() if k in raw_data.columns]]

    # 4. Handle Missing Data
    df_clean = raw_data.ffill().dropna()

    # 5. Save to CSV
    output_file = "gold_market_regime_data.csv"
    df_clean.to_csv(output_file)
    print(f"\n[SUCCESS] Dataset generated: {output_file}")
    print(f"Total Trading Days: {len(df_clean)}")
    print("\nFirst 5 rows:")
    print(df_clean.head())

if __name__ == "__main__":
    fetch_macro_data()