#!/usr/bin/env python3
"""
evaluate_era_rl_v2.py
Optimized for ERA-RL V2 with sequence-based encoding and regime diagnostics.
"""


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


def build_single_obs(t, scaled_prices, prev_pos):
    """Matches the 6-dim observation space in the environment."""
    S0 = scaled_prices[0]
    p_t = scaled_prices[t]
    r_avg = np.mean(scaled_prices[:t+1]) / S0
    r_max = np.max(scaled_prices[:t+1]) / S0
    return [t/20.0, p_t/S0, 0.0, prev_pos, r_avg, r_max]


def read_price_file(path):
    """Generic loader for Excel/CSV."""
    p = Path(path)
    df = pd.read_excel(p) if p.suffix in ['.xlsx', '.xls'] else pd.read_csv(p)


    # --- Robust column normalization ---
    # Keep deliverables/outputs identical by canonicalizing to:
    # - 'Date'
    # - 'Close_^GSPC'
    cols = list(df.columns)
    lower_to_col = {str(c).strip().lower(): c for c in cols}


    # Date column
    date_col = None
    if 'date' in lower_to_col:
        date_col = lower_to_col['date']
    else:
        # fallback: first column containing 'date'
        for c in cols:
            if 'date' in str(c).strip().lower():
                date_col = c
                break
    if date_col is None:
        raise KeyError(f"Could not find a date column in file: {p}")


    # Close column (S&P 500 close); prefer exact match, then case-insensitive, then any 'close*'
    close_col = None
    if 'Close_^GSPC' in df.columns:
        close_col = 'Close_^GSPC'
    elif 'close_^gspc' in lower_to_col:
        close_col = lower_to_col['close_^gspc']
    else:
        for c in cols:
            if str(c).strip().lower().startswith('close'):
                close_col = c
                break
    if close_col is None:
        raise KeyError(
            "Could not find a close column (expected something like 'Close_^GSPC' or 'Close...')."
        )


    # Canonicalize names used everywhere else in this script
    rename_map = {}
    if date_col != 'Date':
        rename_map[date_col] = 'Date'
    if close_col != 'Close_^GSPC':
        rename_map[close_col] = 'Close_^GSPC'
    if rename_map:
        df = df.rename(columns=rename_map)


    df['Date'] = pd.to_datetime(df['Date'])
    return df.sort_values('Date').reset_index(drop=True)


def monthly_episode_slices(df, start, end):
    """Yields 20-day trading episodes for each month in range."""
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


    # Load Agent
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
            scaled_prices = (raw_prices / raw_prices[0]) * 100.0
           
            # Init Sequence History (Padding with first obs)
            initial_obs = build_single_obs(0, scaled_prices, 0.0)
            obs_history = deque([torch.tensor(initial_obs) for _ in range(SEQ_LEN)], maxlen=SEQ_LEN)
            prev_pos = 0.0
           
            ep_cash_flow = 0.0
            ep_tx_total = 0.0


            for t in range(len(scaled_prices) - 1):
                # Prepare Tensors
                obs_seq = torch.stack(list(obs_history)).unsqueeze(0).float()
                prev_h_tensor = torch.tensor([[prev_pos]]).float()


                with torch.no_grad():
                    diag = agent.get_diagnostics(obs_seq, prev_h_tensor)
                    # For evaluation, we use the mean action of the distribution
                    new_pos = np.clip(diag['action_mean'], -2.0, 2.0)
                    mode = diag['mode']
                    xi = diag['xi']


                # Accounting
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
                    'Xi': xi
                })


                # Update state for next step
                prev_pos = new_pos
                obs_history.append(torch.tensor(build_single_obs(t+1, scaled_prices, prev_pos)))


            # Close position at T
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


    # Save CSVs
    df_sum = pd.DataFrame(all_summaries)
    df_step = pd.DataFrame(all_step_logs)
    df_sum.to_csv(f"{OUTPUT_DIR}/episode_summary.csv", index=False)
    df_step.to_csv(f"{OUTPUT_DIR}/step_logs.csv", index=False)


    # Generate Visualization (Phase Comparison)
    plt.figure(figsize=(10, 6))
    df_sum.boxplot(column='NetPnL', by='Phase')
    plt.title('Net PnL Distribution by Phase (ERA-RL V2)')
    plt.suptitle('')
    plt.savefig(f"{OUTPUT_DIR}/pnl_distribution.png")
    print(f"Evaluation complete. Files saved in {OUTPUT_DIR}")


if __name__ == "__main__":
    evaluate_v2()

