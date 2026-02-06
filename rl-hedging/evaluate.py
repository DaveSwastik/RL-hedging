#!/usr/bin/env python3
import os
import sys
import pandas as pd
import numpy as np
import torch
from pathlib import Path
from collections import deque
from tqdm import tqdm
import matplotlib.pyplot as plt

# Ensure repo root is importable
sys.path.insert(0, '.')

try:
    from src.agents.models import ERARL_Agent_V2
except ImportError:
    raise ImportError("Ensure the 'src' directory is in your PYTHONPATH.")

# ---- CONSTANTS ----
SEQ_LEN = 30
OBS_DIM = 6
TRADING_COST = 0.0002  # 2BP
OUTPUT_DIR = "./evaluation_outputs_v2"
MODEL_PATH = "models/era_rl_v2(5k-episodes).pth"
DATA_PATH = "data/historical_sp500_2016_2024.xlsx"

# ---------------- HELPERS ----------------

def build_single_obs(t, scaled_prices, vol, prev_pos):
    """Matches the 6-dim observation space: [time, price, vol, pos, avg, max]"""
    S0 = scaled_prices[0]
    p_t = scaled_prices[t]
    r_avg = np.mean(scaled_prices[:t+1]) / S0
    r_max = np.max(scaled_prices[:t+1]) / S0
    return [t/20.0, p_t/S0, vol, prev_pos, r_avg, r_max]

def read_price_file(path):
    """Loads data and calculates annualized realized volatility safely."""
    p = Path(path)
    df = pd.read_excel(p) if p.suffix in ['.xlsx', '.xls'] else pd.read_csv(p)
    
    # 1. Normalize column names to find Date and Close
    cols = list(df.columns)
    date_col = next((c for c in cols if 'date' in str(c).lower()), None)
    # Target specifically the S&P 500 Close column
    close_col = next((c for c in cols if 'close' in str(c).lower()), None)
    
    if not date_col or not close_col:
        raise KeyError(f"Required columns (Date/Close) not found. Available: {cols}")

    # 2. Extract ONLY the necessary columns to avoid Multi-Column Series errors
    df = df[[date_col, close_col]].copy()
    df.columns = ['Date', 'Close_^GSPC']
    
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)

    # 3. Calculate Volatility on the single 'Close_^GSPC' series
    # Using .squeeze() ensures we are working with a Series, not a DataFrame
    price_series = df['Close_^GSPC'].squeeze()
    
    df['LogRet'] = np.log(price_series / price_series.shift(1))
    
    # Calculate 20-day Rolling Annualized Volatility
    df['Volatility'] = df['LogRet'].rolling(window=20).std() * np.sqrt(252)
    
    # 4. Fill gaps (FFill for recent data, BFill for start of series)
    df['Volatility'] = df['Volatility'].ffill().bfill().fillna(0.20)
    
    return df

def monthly_episode_slices(df, start, end):
    mask = (df['Date'] >= pd.to_datetime(start)) & (df['Date'] <= pd.to_datetime(end))
    dfp = df.loc[mask].copy()
    dfp['YM'] = dfp['Date'].dt.to_period('M')
    for _, g in dfp.groupby('YM'):
        if len(g) >= 20:
            yield g.iloc[:20].copy().reset_index(drop=True)

# ---------------- EVALUATION ----------------

def evaluate_v2():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = torch.device("cpu")

    agent = ERARL_Agent_V2(obs_dim=OBS_DIM, hidden_dim=128)
    agent.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    agent.eval()

    df = read_price_file(DATA_PATH)
    phases = [
        ("Pre-COVID", "2016-01-01", "2018-12-31"),
        ("COVID",     "2019-01-01", "2021-12-31"),
        ("Post-COVID", "2022-01-01", "2024-12-31"),
    ]

    all_summaries = []
    all_step_logs = []

    for p_name, start_d, end_d in phases:
        print(f"Running Phase: {p_name}")
        episodes = list(monthly_episode_slices(df, start_d, end_d))

        for ep_df in tqdm(episodes):
            raw_prices = ep_df['Close_^GSPC'].values
            episode_vols = ep_df['Volatility'].values
            scaled_prices = (raw_prices / raw_prices[0]) * 100.0

            # 1. Initialize with starting volatility
            initial_obs = build_single_obs(0, scaled_prices, episode_vols[0], 0.0)
            obs_history = deque([torch.tensor(initial_obs) for _ in range(SEQ_LEN)], maxlen=SEQ_LEN)
            prev_pos = 0.0
            ep_cash_flow = 0.0
            ep_tx_total = 0.0

            for t in range(len(scaled_prices) - 1):
                # 2. Agent Decision based on history up to t
                obs_seq = torch.stack(list(obs_history)).unsqueeze(0).float()
                prev_h_tensor = torch.tensor([[prev_pos]]).float()

                with torch.no_grad():
                    diag = agent.get_diagnostics(obs_seq, prev_h_tensor)
                    new_pos = np.clip(diag['action_mean'], -2.0, 2.0)
                    mode = diag['mode']
                    xi = diag['xi']

                # 3. Accounting
                trade = new_pos - prev_pos
                tx = abs(trade) * scaled_prices[t] * TRADING_COST
                ep_tx_total += tx
                ep_cash_flow -= (trade * scaled_prices[t])

                all_step_logs.append({
                    'Date': ep_df['Date'].iloc[t],
                    'Phase': p_name,
                    'Close_^GSPC': scaled_prices[t],
                    'Hedge': new_pos,
                    'Regime': mode,
                    'Volatility': episode_vols[t],
                    'Xi': xi
                })

                # 4. Advance State to t+1 (Exactly one update per step)
                prev_pos = new_pos
                next_obs = build_single_obs(t+1, scaled_prices, episode_vols[t+1], prev_pos)
                obs_history.append(torch.tensor(next_obs))

            # Close position at Maturity
            final_price = scaled_prices[-1]
            ep_tx_total += abs(prev_pos) * final_price * TRADING_COST
            ep_cash_flow += prev_pos * final_price
            
            payoff = max(np.mean(scaled_prices) - 100.0, 0.0)
            net_pnl = payoff + ep_cash_flow - ep_tx_total

            all_summaries.append({
                'Month': ep_df['Date'].iloc[0].strftime('%Y-%m'),
                'Phase': p_name,
                'NetPnL': net_pnl,
                'Payoff': payoff,
                'HedgingPnL': ep_cash_flow,
                'TxCosts': ep_tx_total
            })

    # Save and Plot
    pd.DataFrame(all_summaries).to_csv(f"{OUTPUT_DIR}/episode_summary.csv", index=False)
    pd.DataFrame(all_step_logs).to_csv(f"{OUTPUT_DIR}/step_logs.csv", index=False)

    plt.figure(figsize=(10, 6))
    pd.DataFrame(all_summaries).boxplot(column='NetPnL', by='Phase')
    plt.title('Net PnL Distribution by Phase (ERA-RL V2)')
    plt.suptitle('')
    plt.savefig(f"{OUTPUT_DIR}/pnl_distribution.png")
    print(f"Evaluation complete. Results in {OUTPUT_DIR}")

if __name__ == "__main__":
    evaluate_v2()