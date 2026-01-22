#!/usr/bin/env python3
"""
evaluate_delta_hedger.py

Standalone evaluation script for Delta hedging baseline.
NO RL, NO MODEL LOADING.

Outputs:
 - evaluation_outputs_delta/episode_summary.csv
 - evaluation_outputs_delta/step_logs_delta.csv
"""

import os
import sys
import argparse
from pathlib import Path
import pandas as pd
import numpy as np

# make repo importable
sys.path.insert(0, '.')

from src.baselines.delta_hedge import DeltaHedger

# ================= CONFIG =================
HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE
DATA_PATH = str((REPO_ROOT / "data" / "historical_sp500_2016_2024.xlsx").resolve())
OUTPUT_DIR = "./evaluation_outputs_delta"

TRADING_COST = 0.001          # proportional transaction cost
ANNUAL_SIGMA = 0.20           # assumed constant vol for delta hedger
TRADING_DAYS_PER_YEAR = 252.0
EPISODE_LENGTH = 20           # trading days
STRIKE = 100.0                # ATM in scaled space
# =========================================


def read_price_excel(path):
    p = Path(path)
    # Handle Unix-style '/data/..' paths on Windows by treating them as repo-relative.
    if str(p).startswith(('/', '\\')) and not p.exists():
        p = REPO_ROOT / str(p).lstrip('/\\')
    if not p.is_absolute():
        p = (REPO_ROOT / p).resolve()

    if not p.exists():
        data_dir = REPO_ROOT / "data"
        hint = ""
        if data_dir.exists():
            hint = f"\nAvailable data files: {[x.name for x in sorted(data_dir.glob('*'))]}"
        raise FileNotFoundError(f"Data file not found: {p}{hint}")

    df = pd.read_excel(p, engine="openpyxl")
    cols = {c.lower(): c for c in df.columns}

    date_col = cols.get("date", None)
    if date_col is None:
        raise ValueError("Excel must contain a 'Date' column")

    price_col = cols.get("price", None)
    if price_col is None:
        price_col = cols.get("close", None)
    if price_col is None:
        # Handle Yahoo Finance-style columns like 'Adj Close_^GSPC' / 'Close_^GSPC'
        lowered = {str(c).strip().lower(): c for c in df.columns}
        for key in ("adj close", "adj_close"):
            for lc, orig in lowered.items():
                if lc.startswith(key) or (key in lc):
                    price_col = orig
                    break
            if price_col is not None:
                break
    if price_col is None:
        lowered = {str(c).strip().lower(): c for c in df.columns}
        for lc, orig in lowered.items():
            if "close" in lc and "volume" not in lc:
                price_col = orig
                break
    if price_col is None:
        raise ValueError("Excel must contain a price column (e.g. 'Close', 'Adj Close', 'Close_^GSPC')")

    df = df[[date_col, price_col]].copy()
    df.columns = ["Date", "Price"]
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    return df


def monthly_episodes(df, start, end):
    df = df[(df["Date"] >= start) & (df["Date"] <= end)].copy()
    df["ym"] = df["Date"].dt.to_period("M")

    for _, g in df.groupby("ym"):
        g = g.sort_values("Date").reset_index(drop=True)
        if len(g) >= EPISODE_LENGTH:
            yield g.iloc[:EPISODE_LENGTH].copy()


def evaluate_delta(data_path: str = DATA_PATH):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df = read_price_excel(data_path)

    phases = [
        (1, "2016-01-01", "2018-12-31"),
        (2, "2019-01-01", "2021-12-31"),
        (3, "2022-01-01", "2024-12-31"),
    ]

    episode_rows = []
    step_logs = []

    for phase_id, start, end in phases:
        for ep in monthly_episodes(df, start, end):
            dates = ep["Date"].values
            raw_prices = ep["Price"].values.astype(float)

            # -------- Scaling --------
            scale = 100.0 / raw_prices[0]
            prices = raw_prices * scale

            T = len(prices)
            dt = 1.0 / TRADING_DAYS_PER_YEAR
            maturity = T * dt

            option_spec = {
                "K": STRIKE,
                "maturity": maturity,
                "option_type": "call"
            }

            delta_hedger = DeltaHedger(
                option_spec=option_spec,
                trading_cost=TRADING_COST
            )

            # constant variance path
            v_path = np.full(T, ANNUAL_SIGMA ** 2)

            deltas = delta_hedger.get_positions(prices, v_path, dt)

            prev_pos = 0.0
            cash = 0.0
            total_tx_cost = 0.0

            for t in range(T - 1):
                price_t = prices[t]
                price_next = prices[t + 1]

                target_pos = float(deltas[t])
                trade = target_pos - prev_pos

                tx_cost = abs(trade) * price_t * TRADING_COST
                total_tx_cost += tx_cost

                cash -= trade * price_t
                pnl_hold = prev_pos * (price_next - price_t)

                step_logs.append({
                    "Phase": phase_id,
                    "EpisodeStart": dates[0],
                    "Date": dates[t],
                    "RawPrice": raw_prices[t],
                    "ScaledPrice": price_t,
                    "Delta": target_pos,
                    "ChangeInHedge": trade,
                    "TransactionCost": tx_cost,
                    "PnL_from_Hold": pnl_hold
                })

                prev_pos = target_pos

            # -------- Final liquidation --------
            final_price = prices[-1]
            liquidation_cost = abs(prev_pos) * final_price * TRADING_COST
            total_tx_cost += liquidation_cost
            cash += prev_pos * final_price

            # -------- Option payoff --------
            avg_price = prices.mean()
            option_payoff = max(avg_price - STRIKE, 0.0)

            hedging_pnl = cash
            net_pnl = option_payoff + hedging_pnl - total_tx_cost

            episode_rows.append({
                "EpisodeStart": dates[0],
                "EpisodeEnd": dates[-1],
                "Phase": phase_id,
                "OptionPayoff": option_payoff,
                "HedgingPnL": hedging_pnl,
                "TransactionCost": total_tx_cost,
                "NetPnL": net_pnl,
                "AvgScaledPrice": avg_price,
                "NumDays": T
            })

    # -------- Save outputs --------
    ep_df = pd.DataFrame(episode_rows)
    step_df = pd.DataFrame(step_logs)

    ep_df.to_csv(f"{OUTPUT_DIR}/episode_summary-delta.csv", index=False)
    step_df.to_csv(f"{OUTPUT_DIR}/step_logs-delta.csv", index=False)

    print("Delta hedger evaluation complete.")
    print(f"Saved to: {OUTPUT_DIR}")
    print(ep_df.head())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Delta hedging baseline")
    parser.add_argument("--data", default=DATA_PATH, help="Path to historical price Excel (.xlsx)")
    args = parser.parse_args()

    evaluate_delta(data_path=args.data)
