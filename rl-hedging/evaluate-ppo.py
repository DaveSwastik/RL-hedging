#!/usr/bin/env python3
"""evaluate-ppo.py

Standalone evaluation script for an SB3 PPO hedging agent on historical data.

Goal: match the same evaluation protocol/data split as:
- evaluate-delta.py (delta baseline)
- evaluate.py       (ERA-RL agent)

Outputs (default):
- evaluation_outputs_ppo/episode_summary-ppo.csv
- evaluation_outputs_ppo/step_logs-ppo.csv

Notes:
- We do NOT step the simulator env during evaluation. Instead we reproduce the
  same path-based bookkeeping as the delta / ERA scripts:
  - per-episode scaling so day-0 price is 100
  - self-financing hedge with proportional transaction costs
  - Asian option payoff uses the average price over the 20 trading days

Usage:
  python evaluate-ppo.py --model models/sb3_ppo_heston_20d.zip
  python evaluate-ppo.py --data data/historical_sp500_2016_2024.xlsx
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# make repo importable
sys.path.insert(0, ".")

try:
    from stable_baselines3 import PPO
except ImportError as e:  # pragma: no cover
    raise SystemExit(
        "stable_baselines3 is not installed. Install with: poetry add stable-baselines3"
    ) from e


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE

DEFAULT_DATA_PATH = str((REPO_ROOT / "data" / "historical_sp500_2016_2024.xlsx").resolve())
DEFAULT_MODEL_PATH = str((REPO_ROOT / "models" / "sb3_ppo_heston_20d.zip").resolve())
DEFAULT_OUTPUT_DIR = "./evaluation_outputs_ppo"


def _resolve_input_path(path: str | Path) -> Path:
    p = Path(path)

    # Handle Unix-style '/data/..' paths on Windows by treating them as repo-relative.
    if str(p).startswith(("/", "\\")) and not p.exists():
        p = REPO_ROOT / str(p).lstrip("/\\")

    if not p.is_absolute():
        p = (REPO_ROOT / p).resolve()

    return p


def read_price_excel(path: str | Path) -> pd.DataFrame:
    p = _resolve_input_path(path)
    if not p.exists():
        data_dir = REPO_ROOT / "data"
        hint = ""
        if data_dir.exists():
            hint = f"\nAvailable data files: {[x.name for x in sorted(data_dir.glob('*'))]}"
        raise FileNotFoundError(f"Data file not found: {p}{hint}")

    df = pd.read_excel(p, engine="openpyxl")
    cols = {str(c).strip().lower(): c for c in df.columns}

    date_col = cols.get("date")
    if date_col is None:
        raise ValueError("Excel must contain a 'Date' column")

    price_col = cols.get("price") or cols.get("close")
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

    out = df[[date_col, price_col]].copy()
    out.columns = ["Date", "Price"]
    out["Date"] = pd.to_datetime(out["Date"])
    out = out.sort_values("Date").reset_index(drop=True)
    return out


def monthly_episodes(df: pd.DataFrame, start: str, end: str, episode_length: int):
    df = df[(df["Date"] >= start) & (df["Date"] <= end)].copy()
    df["ym"] = df["Date"].dt.to_period("M")

    for _, g in df.groupby("ym"):
        g = g.sort_values("Date").reset_index(drop=True)
        if len(g) >= episode_length:
            yield g.iloc[:episode_length].copy()


def evaluate_ppo(
    model_path: str,
    data_path: str,
    output_dir: str,
    trading_cost: float,
    annual_sigma: float,
    episode_length: int,
    strike: float,
):
    os.makedirs(output_dir, exist_ok=True)

    model_resolved = _resolve_input_path(model_path)
    if not model_resolved.exists():
        models_dir = REPO_ROOT / "models"
        hint = ""
        if models_dir.exists():
            hint = f"\nAvailable model files: {[p.name for p in sorted(models_dir.glob('*.zip'))]}"
        raise FileNotFoundError(f"Model file not found: {model_resolved}{hint}")

    model = PPO.load(str(model_resolved), device="cpu")

    df = read_price_excel(data_path)

    phases = [
        (1, "2016-01-01", "2018-12-31"),
        (2, "2019-01-01", "2021-12-31"),
        (3, "2022-01-01", "2024-12-31"),
    ]

    episode_rows: list[dict] = []
    step_logs: list[dict] = []

    for phase_id, start, end in phases:
        for ep in monthly_episodes(df, start, end, episode_length=episode_length):
            dates = ep["Date"].values
            raw_prices = ep["Price"].values.astype(float)

            # Scale so Day-0 is 100 (match delta/ERA scripts)
            scale = 100.0 / raw_prices[0]
            prices = raw_prices * scale

            T = len(prices)

            prev_pos = 0.0
            cash = 0.0
            total_tx_cost = 0.0

            running_sum = float(prices[0])
            running_max = float(prices[0])

            for t in range(T - 1):
                price_t = float(prices[t])
                price_next = float(prices[t + 1])

                # Build observation consistent with HedgingEnv:
                # [time_norm, spot/S0, vol, pos, running_avg/S0, running_max/S0]
                time_norm = t / float(episode_length)
                S0 = 100.0
                spot_norm = price_t / S0
                running_avg = running_sum / (t + 1)

                obs = np.array(
                    [
                        time_norm,
                        spot_norm,
                        float(annual_sigma),
                        float(prev_pos),
                        float(running_avg / S0),
                        float(running_max / S0),
                    ],
                    dtype=np.float32,
                )

                action, _ = model.predict(obs, deterministic=True)
                # action is shape (1,) for Box(1,)
                target_pos = float(np.clip(float(action[0]), -2.0, 2.0))

                trade = target_pos - prev_pos
                tx_cost = abs(trade) * price_t * trading_cost
                total_tx_cost += tx_cost

                cash -= trade * price_t
                pnl_hold = prev_pos * (price_next - price_t)

                step_logs.append(
                    {
                        "Phase": phase_id,
                        "EpisodeStart": dates[0],
                        "Date": dates[t],
                        "RawPrice": raw_prices[t],
                        "ScaledPrice": price_t,
                        "Delta": target_pos,
                        "ChangeInHedge": trade,
                        "TransactionCost": tx_cost,
                        "PnL_from_Hold": pnl_hold,
                    }
                )

                prev_pos = target_pos
                running_sum += price_next
                running_max = max(running_max, price_next)

            # Final liquidation
            final_price = float(prices[-1])
            liquidation_cost = abs(prev_pos) * final_price * trading_cost
            total_tx_cost += liquidation_cost
            cash += prev_pos * final_price

            # Option payoff (Asian on scaled path)
            avg_price = float(np.mean(prices))
            option_payoff = max(avg_price - strike, 0.0)

            hedging_pnl = float(cash)
            net_pnl = float(option_payoff + hedging_pnl - total_tx_cost)

            episode_rows.append(
                {
                    "EpisodeStart": dates[0],
                    "EpisodeEnd": dates[-1],
                    "Phase": phase_id,
                    "OptionPayoff": option_payoff,
                    "HedgingPnL": hedging_pnl,
                    "TransactionCost": total_tx_cost,
                    "NetPnL": net_pnl,
                    "AvgScaledPrice": avg_price,
                    "NumDays": int(T),
                }
            )

    ep_df = pd.DataFrame(episode_rows)
    step_df = pd.DataFrame(step_logs)

    ep_df.to_csv(f"{output_dir}/episode_summary-ppo.csv", index=False)
    step_df.to_csv(f"{output_dir}/step_logs-ppo.csv", index=False)

    print("PPO (SB3) evaluation complete.")
    print(f"Saved to: {output_dir}")
    print(ep_df.head())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate SB3 PPO hedger on historical data")
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH, help="Path to SB3 PPO .zip")
    parser.add_argument("--data", default=DEFAULT_DATA_PATH, help="Path to historical price Excel (.xlsx)")
    parser.add_argument("--out", default=DEFAULT_OUTPUT_DIR, help="Output directory")
    parser.add_argument("--tcost", type=float, default=0.001, help="Proportional transaction cost")
    parser.add_argument("--annual-sigma", type=float, default=0.20, help="Constant volatility feature during eval")
    parser.add_argument("--episode-length", type=int, default=20, help="Trading days per episode")
    parser.add_argument("--strike", type=float, default=100.0, help="Strike in scaled space")

    args = parser.parse_args()

    evaluate_ppo(
        model_path=args.model,
        data_path=args.data,
        output_dir=args.out,
        trading_cost=args.tcost,
        annual_sigma=args.annual_sigma,
        episode_length=args.episode_length,
        strike=args.strike,
    )
