#!/usr/bin/env python3
"""
evaluate_era_rl.py

Evaluation script for pre-trained ERA-RL hedging agent.

Outputs:
 - episode_summary.csv  (episode-level PnL summary)
 - step_logs.csv        (detailed day-by-day logs)

Assumptions / notes:
 - Excel file has a date column named 'Date' (or 'date') and a price column named 'Price' (or 'Close').
 - Each calendar month is an episode; for each month we take the first 20 trading days as the episode.
 - If a month has fewer than 20 trading days, that month is skipped.
 - Per-episode scaling: Day-0 price scaled to 100; all prices in episode scaled proportionally.
 - Model actor output is squashed in (-1,1). We map to action range [-2,2] by action*2 (matches env.ActionBox in repo).
 - Default trading cost is a fraction (e.g. 0.001 = 0.1%). Change TRADING_COST as needed.
"""

import os
import sys
from pathlib import Path
import argparse
import pandas as pd
import numpy as np
import torch
from datetime import datetime

# ensure repo root is importable (adjust if your repo root differs)
sys.path.insert(0, '.')

# ---- CONFIG ----
HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE

# Prefer repo-relative defaults that exist in this workspace.
MODEL_PATH = str((REPO_ROOT / "models" / "era_rl_agent.pth").resolve())
DATA_PATH = str((REPO_ROOT / "data" / "historical_sp500_2016_2024.xlsx").resolve())
TRADING_COST = 0.001   # proportional cost per trade (changeable)
OUTPUT_DIR = "./evaluation_outputs"
OBS_DIM = 6            # model expects 6-dim observations (time_norm, spot/S0, vol, pos, running_avg/S0, running_max/S0)
ACTION_SCALE = 2.0     # map model tanh output (-1,1) to env action space (-2,2)

# ---- Imports from your repo (models / env) ----
# ERARL_Agent_V2 (model architecture) is in your repo's models file.
try:
    from src.agents.models import ERARL_Agent_V2
except Exception:
    # fallback: try loading from top-level models.py if present
    try:
        from models import ERARL_Agent_V2  # if file is models.py in repo root
    except Exception as e:
        raise ImportError("Couldn't import ERARL_Agent_V2 from src.agents.models or models.py. "
                          "Make sure your PYTHONPATH includes repo root.") from e

# ---------------- helper funcs ----------------
def _resolve_input_path(path: str | Path) -> Path:
    """Resolve path robustly on Windows.

    - Treat absolute paths normally.
    - If a user passes something like '/data/foo.csv' on Windows, interpret it as
      repo-relative rather than 'C:\\data\\foo.csv'.
    - Otherwise, resolve relative to the repo root.
    """
    p = Path(path)

    # Special-case Unix-style absolute paths on Windows (e.g. '/data/x.xlsx').
    # Path('/data/x.xlsx') is absolute but points to drive root; almost never intended here.
    if str(p).startswith(('/', '\\')) and not p.exists():
        p2 = REPO_ROOT / str(p).lstrip('/\\')
        return p2

    if p.is_absolute():
        return p
    return (REPO_ROOT / p).resolve()


def _normalize_date_series(s: pd.Series) -> pd.Series:
    dt = pd.to_datetime(s, errors='coerce', utc=False)
    # If timezone-aware, drop tz for easier comparisons/grouping.
    try:
        if getattr(dt.dt, "tz", None) is not None:
            dt = dt.dt.tz_convert(None)
    except Exception:
        pass
    return dt


def _infer_date_price_columns(df: pd.DataFrame) -> tuple[str, str]:
    cols = {str(c).strip().lower(): c for c in df.columns}

    date_col = None
    for candidate in ('date', 'datetime', 'timestamp', 'time'):
        if candidate in cols:
            date_col = cols[candidate]
            break
    if date_col is None:
        for c in df.columns:
            if np.issubdtype(df[c].dtype, np.datetime64):
                date_col = c
                break
    if date_col is None:
        raise ValueError("Couldn't find a date column. Expected something like 'Date'.")

    price_col = None
    for candidate in ('price', 'close', 'adj close', 'adj_close', 'last', 'value'):
        if candidate in cols:
            price_col = cols[candidate]
            break
    if price_col is None:
        numeric_cols = [c for c in df.columns if np.issubdtype(df[c].dtype, np.number)]
        numeric_cols = [c for c in numeric_cols if c != date_col]
        if len(numeric_cols) == 0:
            raise ValueError("Couldn't find a numeric price column. Expected something like 'Close'.")
        price_col = numeric_cols[0]

    return str(date_col), str(price_col)


def read_price_file(path: str | Path) -> pd.DataFrame:
    """Load historical prices from .xlsx/.xls or .csv and normalize to ['Date','Price'] columns."""
    resolved = _resolve_input_path(path)
    if not resolved.exists():
        data_dir = REPO_ROOT / "data"
        models_dir = REPO_ROOT / "models"
        hint = []
        if data_dir.exists():
            hint.append(f"Available data files: {[p.name for p in sorted(data_dir.glob('*'))]}")
        if models_dir.exists():
            hint.append(f"Available model files: {[p.name for p in sorted(models_dir.glob('*.pth'))]}")
        hint_text = ("\n" + "\n".join(hint)) if hint else ""
        raise FileNotFoundError(f"Data file not found: {resolved}{hint_text}")

    suffix = resolved.suffix.lower()
    if suffix in ('.xlsx', '.xls'):
        df = pd.read_excel(resolved, engine='openpyxl')
    elif suffix == '.csv':
        df = pd.read_csv(resolved)
    else:
        raise ValueError(f"Unsupported data format: {resolved.suffix}. Use .csv or .xlsx")

    date_col, price_col = _infer_date_price_columns(df)
    out = df[[date_col, price_col]].copy()
    out.columns = ['Date', 'Price']
    out['Date'] = _normalize_date_series(out['Date'])
    out = out.dropna(subset=['Date', 'Price'])
    out = out.sort_values('Date').reset_index(drop=True)
    return out

def monthly_episode_slices(df, start_date, end_date):
    """Yield (phase_label, month_start_date, month_df) for months between start/end inclusive.
    Each month_df contains the rows for that calendar month. We will take the first 20 trading days.
    """
    mask = (df['Date'] >= pd.to_datetime(start_date)) & (df['Date'] <= pd.to_datetime(end_date))
    dfp = df.loc[mask].copy()
    if dfp.empty:
        return

    dfp['year_month'] = dfp['Date'].dt.to_period('M')
    grouped = dfp.groupby('year_month', sort=True)
    for ym, g in grouped:
        g = g.sort_values('Date').reset_index(drop=True)
        if len(g) >= 20:
            episode_df = g.iloc[:20].copy().reset_index(drop=True)
            yield episode_df
        else:
            # skip months with less than 20 trading days
            continue

def build_obs_sequence(prices_scaled):
    """Given a list/array of scaled prices for an episode up to current time index,
    build the per-timestep obs sequence as used in your env (6-dim).
    Returns np.array shape (T, OBS_DIM).
    Features per timestep:
      0: time_norm = t / T
      1: spot / S0  (S0 is episode_day0 scaled -> 100, so this becomes price/100)
      2: volatility (unknown in historical data) -> set to 0.0
      3: position (we'll set at time t the previous position; placeholder 0 for building sequence)
      4: running_avg / S0
      5: running_max / S0
    Note: position will be set at each step by the main loop; here we construct time-invariant fields and will
    override 'pos' column when making the actual per-step obs sequence.
    """
    T = len(prices_scaled)
    S0 = prices_scaled[0]
    seq = []
    running_sum = 0.0
    running_max = -np.inf
    for t in range(T):
        p = float(prices_scaled[t])
        running_sum += p
        running_max = max(running_max, p)
        time_norm = t / 20.0  # since episodes are 20 trading days
        spot_norm = p / S0
        vol = 0.0
        running_avg = (running_sum / (t + 1)) / S0
        running_max_norm = running_max / S0
        # placeholder pos=0 (we will set actual previous pos during inference)
        seq.append([time_norm, spot_norm, vol, 0.0, running_avg, running_max_norm])
    return np.array(seq, dtype=np.float32)

# ---------------- main evaluation loop ----------------
def evaluate(model_path, data_path, trading_cost=TRADING_COST, output_dir=OUTPUT_DIR):
    os.makedirs(output_dir, exist_ok=True)

    # load data
    df = read_price_file(data_path)

    # phase definitions
    phases = [
        ("Phase 1 (Pre-COVID)", "2016-01-01", "2018-12-31"),
        ("Phase 2 (COVID)",     "2019-01-01", "2021-12-31"),
        ("Phase 3 (Post-COVID)", "2022-01-01", "2024-12-31"),
    ]

    # load model weights into architecture
    device = torch.device("cpu")
    # instantiate agent (obs_dim = 6)
    agent = ERARL_Agent_V2(obs_dim=OBS_DIM)
    # load weights
    model_resolved = _resolve_input_path(model_path)
    if not model_resolved.exists():
        models_dir = REPO_ROOT / "models"
        hint = ""
        if models_dir.exists():
            hint = f"\nAvailable model files: {[p.name for p in sorted(models_dir.glob('*.pth'))]}"
        raise FileNotFoundError(f"Model file not found: {model_resolved}{hint}")

    ckpt = torch.load(model_resolved, map_location=device)
    # try to load state_dict or full model
    if isinstance(ckpt, dict) and 'state_dict' in ckpt:
        state = ckpt['state_dict']
    else:
        state = ckpt
    try:
        agent.load_state_dict(state)
    except Exception:
        # if the checkpoint already was a model (rare), attempt to use it directly
        try:
            agent = ckpt
        except Exception as e:
            raise RuntimeError("Failed to load model state_dict into ERARL_Agent_V2.") from e
    agent.to(device)
    agent.eval()

    all_episode_summaries = []
    all_step_logs = []

    # iterate phases
    for phase_idx, (phase_name, start_d, end_d) in enumerate(phases, start=1):
        for episode_df in monthly_episode_slices(df, start_d, end_d):
            ep_start = episode_df['Date'].iloc[0]
            ep_end = episode_df['Date'].iloc[-1]
            # raw prices
            raw_prices = episode_df['Price'].values.astype(float)

            # scale so Day0 -> 100
            scale_factor = 100.0 / raw_prices[0]
            scaled_prices = raw_prices * scale_factor

            # compute obs_sequence templates for entire episode
            seq_template = build_obs_sequence(scaled_prices)  # shape (T, OBS_DIM)

            # bookkeeping for the episode
            T = len(scaled_prices)  # should be 20
            prev_pos = 0.0
            cash_no_tx = 0.0         # bookkeeping without transaction costs
            total_tx_cost = 0.0
            # For logging and hedging_pnl calculation
            step_records = []

            # We'll iterate day by day: at day t, agent sees obs_sequence up to t (including t),
            # outputs hedge ratio which becomes position for next interval (t -> t+1)
            # Follow convention: at first step t=0, prev_pos=0
            running_sum_for_avg = scaled_prices[0]
            running_max = scaled_prices[0]

            for t in range(T-1):  # we iterate 0..T-2 to have price_{t+1} available
                date = episode_df['Date'].iloc[t]
                raw_price = raw_prices[t]
                scaled_price = float(scaled_prices[t])

                # build obs_sequence up to t (length t+1), and set position feature to prev_pos
                seq = seq_template[:(t+1)].copy()  # shape (t+1, OBS_DIM)
                seq[:, 3] = prev_pos  # set position field to prev_pos for all timesteps
                seq_tensor = torch.tensor(seq[np.newaxis, :, :], dtype=torch.float32, device=device)  # [1, T_seq, OBS_DIM]
                prev_hedge_tensor = torch.tensor([[prev_pos]], dtype=torch.float32, device=device)

                with torch.no_grad():
                    dist, risk_out = agent(seq_tensor, prev_hedge_tensor)
                    # get deterministic action = mode
                    try:
                        action_t = float(dist.mode().item())
                    except Exception:
                        # fallback: sample
                        action_t = float(dist.sample().item())

                # map action from (-1,1) to env action space (-2,2)
                new_pos = float(np.clip(action_t * ACTION_SCALE, -2.0, 2.0))
                trade_amount = new_pos - prev_pos

                # transaction cost for this trade at current price (use scaled price)
                tx_cost = abs(trade_amount) * scaled_price * trading_cost
                total_tx_cost += tx_cost

                # update cash_no_tx and cash_with_tx bookkeeping
                cash_no_tx -= trade_amount * scaled_price   # cash ignoring tx costs
                # (we separately record tx costs in total_tx_cost)

                # price move to next day
                next_scaled_price = float(scaled_prices[t+1])
                dS = next_scaled_price - scaled_price

                # hedging PnL contribution (position held over the interval is prev_pos)
                pnl_hold = prev_pos * dS

                # record step
                step_records.append({
                    'Phase': phase_idx,
                    'Episode_start': ep_start,
                    'Date': date,
                    'RawPrice': raw_price,
                    'ScaledPrice': scaled_price,
                    'AgentActionRaw': action_t,
                    'HedgeRatio': new_pos,
                    'ChangeInHedge': trade_amount,
                    'TransactionCost': tx_cost,
                    'PnlFromHolding': pnl_hold,
                    'RunningAvgSoFar': running_sum_for_avg / (t+1),
                    'RunningMaxSoFar': running_max
                })

                # update running stats
                running_sum_for_avg += next_scaled_price
                running_max = max(running_max, next_scaled_price)

                # apply trade: update prev_pos for next interval
                prev_pos = new_pos

            # After loop, handle final step liquidation at final price (T-1 -> T)
            final_price = float(scaled_prices[-1])
            # Liquidation cost for unwinding final position (as in your env)
            liquidation_tx = abs(prev_pos) * final_price * trading_cost
            total_tx_cost += liquidation_tx
            # cash_no_tx final (we didn't subtract tx costs there)
            cash_no_tx += prev_pos * final_price  # receive proceeds from selling position
            # hedging_pnl excluding tx costs:
            hedging_pnl_excl_tx = cash_no_tx
            # hedging_pnl_including_tx = hedging_pnl_excl_tx - total_tx_cost
            # Option payoff (Asian) uses average of scaled prices over 20 days
            avg_scaled = float(np.mean(scaled_prices))
            option_payoff = max(avg_scaled - 100.0, 0.0)

            # Final net PnL following your formula:
            # Episode PnL = Option Payoff + Hedging PnL − Transaction Costs
            final_net_pnl = option_payoff + hedging_pnl_excl_tx - total_tx_cost

            ep_summary = {
                'EpisodeStart': ep_start,
                'EpisodeEnd': ep_end,
                'Phase': phase_idx,
                'OptionPayoff': option_payoff,
                'HedgingPnL_excl_tx': hedging_pnl_excl_tx,
                'TransactionCost': total_tx_cost,
                'NetPnL': final_net_pnl,
                'AvgScaledPrice': avg_scaled,
                'NumDays': T
            }
            all_episode_summaries.append(ep_summary)
            all_step_logs.extend(step_records)

    # build outputs
    df_episodes = pd.DataFrame(all_episode_summaries)
    df_steps = pd.DataFrame(all_step_logs)

    # Save CSVs
    ep_path = os.path.join(output_dir, "episode_summary.csv")
    steps_path = os.path.join(output_dir, "step_logs.csv")
    df_episodes.to_csv(ep_path, index=False)
    df_steps.to_csv(steps_path, index=False)

    print(f"Saved episode summary to: {ep_path}")
    print(f"Saved step logs to: {steps_path}")
    print("\nEpisode summary (first 10 rows):")
    print(df_episodes.head(10))
    return df_episodes, df_steps

# ---------------- CLI ----------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate ERA-RL agent on historical data (monthly episodes)")
    parser.add_argument("--model", default=MODEL_PATH, help="Path to model .pth")
    parser.add_argument("--data", default=DATA_PATH, help="Path to historical excel file")
    parser.add_argument("--tcost", type=float, default=TRADING_COST, help="Proportional trading cost")
    parser.add_argument("--out", default=OUTPUT_DIR, help="Output directory for CSVs")
    args = parser.parse_args()

    evaluate(model_path=args.model, data_path=args.data, trading_cost=args.tcost, output_dir=args.out)
