# download_data.py
import yfinance as yf
import pandas as pd
from pathlib import Path

# --- Configuration ---
TICKER = "AAPL"  # Example: Apple Inc.
START_DATE = "2018-01-01"
END_DATE = "2023-12-31" # Use several years for good backtest length
DATA_FILE = Path("data") / "historical_aapl.csv" # Store in a new 'data' folder
# --- -------------- ---

print(f"Downloading historical data for {TICKER}...")
ticker = yf.Ticker(TICKER)
hist_data = ticker.history(start=START_DATE, end=END_DATE)

# We primarily need the closing price
prices = hist_data[['Close']].copy()

# Basic check for enough data
if len(prices) < 100: # Need at least ~2 months for vol estimation + episode length
    raise ValueError("Not enough historical data downloaded.")

# Create the data directory if it doesn't exist
DATA_FILE.parent.mkdir(exist_ok=True)

# Save the closing prices
prices.to_csv(DATA_FILE)
print(f"Historical closing prices saved to {DATA_FILE}")