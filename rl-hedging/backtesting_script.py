# evaluation_script.py (Now for Backtesting)
import pandas as pd
import yaml
import torch
from tqdm import tqdm
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Import NEW HistoricalEnv and other necessary components
from src.envs.historical_env import HistoricalEnv # <-- USE THIS ENV
from src.analysis.metrics import calculate_summary_stats # Keep using this for stats
from src.agents.models import InterpretableHedger
from src.baselines.delta_hedge import DeltaHedger
from stable_baselines3 import PPO
from src.utils.payoffs import PAYOFF_FUNCTIONS

def load_config(path='src/configs/default.yaml'):
    with open(path, 'r') as f:
        return yaml.safe_load(f)



# --- BACKTESTING EVALUATION FUNCTIONS ---
# These now instantiate HistoricalEnv

def backtest_rl_agent(model_path, config_path, hist_data_path, n_episodes=500):
    cfg = load_config(config_path)
    # Instantiate HistoricalEnv
    env = HistoricalEnv(hist_data_path, cfg['option'], cfg['simulator'], cfg['environment']['trading_cost'])
    model = PPO.load(model_path)

    results = []
    for i in tqdm(range(n_episodes), desc="Backtesting PPO Agent"):
        obs, _ = env.reset(seed=i) # Use seed for reproducible episode selection
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        results.append(info)
    return pd.DataFrame(results)

def backtest_custom_agent(model_path, config_path, hist_data_path, n_episodes=500):
    cfg = load_config(config_path)
    # Instantiate HistoricalEnv
    env = HistoricalEnv(hist_data_path, cfg['option'], cfg['simulator'], cfg['environment']['trading_cost'])
    obs_dim = env.observation_space.shape[0]
    model = InterpretableHedger(obs_dim=obs_dim)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    results = []
    for i in tqdm(range(n_episodes), desc="Backtesting Custom Agent"):
        obs, _ = env.reset(seed=i)
        obs_history = [torch.from_numpy(obs).float()]
        done = False
        while not done:
            with torch.no_grad():
                obs_sequence = torch.stack(obs_history).unsqueeze(0)
                action_tensor, _ = model(obs_sequence)
                action = action_tensor.detach().numpy().flatten()
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            obs_history.append(torch.from_numpy(obs).float())
        info['pnl'] = info['hedging_error'] + info['payoff']
        results.append(info)
    return pd.DataFrame(results)

def backtest_delta_hedger(config_path, hist_data_path, n_episodes=500):
    cfg = load_config(config_path)
    # Instantiate HistoricalEnv just to easily get paths via reset()
    env = HistoricalEnv(hist_data_path, cfg['option'], cfg['simulator'], cfg['environment']['trading_cost'])
    hedger = DeltaHedger(cfg['option'], cfg['environment']['trading_cost'])

    results = []
    for i in tqdm(range(n_episodes), desc="Backtesting Delta Hedger"):
        env.reset(seed=i) # Generates S_path and estimates v_path_est
        S_path = env.S_path
        # Use the estimated variance from the env for consistency
        # Convert daily variance back to annualized vol for BS delta
        # Note: DeltaHedger expects volatility (std dev), not variance
        v_path_for_bs = env.v_path_est * env.dt # Approx daily variance
        vol_path_for_bs = np.sqrt(np.maximum(v_path_for_bs, 1e-8)) # Daily std dev

        # The DeltaHedger needs annualized volatility
        # We pass the estimated daily std dev and dt. Let _bs_delta handle time scaling.
        # OR: Modify DeltaHedger to accept variance path? Simpler to just use vol.
        # We need an annualized vol estimate for the BS formula.
        # Let's approximate using the average estimated daily vol
        avg_daily_vol = np.mean(vol_path_for_bs[vol_path_for_bs > 1e-6]) # Avoid zeros
        annualized_vol = avg_daily_vol * np.sqrt(252) # Simple approximation

        # Re-run the BS delta calc step-by-step using a *single* average vol
        # (More accurate would be to use the time-varying vol_path_for_bs in _bs_delta)
        # For simplicity, let's modify DeltaHedger's evaluate slightly
        pnl = hedger.evaluate_with_vol_path(S_path, vol_path_for_bs, env.dt) # Need to add this method

        payoff = PAYOFF_FUNCTIONS[cfg['option']['type']](S_path, cfg['option']['K'])
        results.append({'pnl': pnl, 'payoff': payoff, 'hedging_error': pnl - payoff})
    return pd.DataFrame(results)

# --- Need to update DeltaHedger class to accept vol_path ---
# Temporarily add this method here for simplicity, or modify the class file
def evaluate_with_vol_path(self: DeltaHedger, S_path, daily_vol_path, dt):
    """Evaluates using a pre-computed path of daily volatilities."""
    n_steps = len(S_path) - 1
    position = 0.0
    cash = 0.0
    annual_sqrt_dt = np.sqrt(dt * 252) # Scaling factor if needed, but BS formula uses T-t

    for t_idx in range(n_steps):
        t_rem = self.T - (t_idx * dt) # T is option maturity in years
        S_t = S_path[t_idx]
        # Use the estimated daily vol, convert to approx annualized for BS formula
        sigma_t_annual = daily_vol_path[t_idx] * np.sqrt(252)

        target_delta = self._bs_delta(S_t, self.K, t_idx * dt, self.T, 0.0, sigma_t_annual)

        trade_amount = target_delta - position
        cash -= trade_amount * S_t
        cash -= abs(trade_amount) * S_t * self.trading_cost
        position = target_delta

    final_price = S_path[-1]
    cash += position * final_price
    cash -= abs(position) * final_price * self.trading_cost

    return cash
# Monkey-patch the method onto the class for this run
DeltaHedger.evaluate_with_vol_path = evaluate_with_vol_path
# --- End of DeltaHedger modification ---


def plot_error_histograms(results_dict, save_path):
    plt.figure(figsize=(12, 7))
    sns.set_style("whitegrid")
    for name, df in results_dict.items():
        sns.histplot(df['hedging_error'], kde=True, label=name, alpha=0.6, bins=50)
    plt.title('Backtest: Distribution of Hedging Errors (500 Episodes)', fontsize=16)
    plt.xlabel('Hedging Error (Final P&L - Payoff)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.legend()
    plt.axvline(0, color='k', linestyle='--', alpha=0.7)
    plt.savefig(save_path)
    print(f"Backtest error distribution plot saved to {save_path}")

def plot_single_episode_behavior_backtest(config_path, hist_data_path, ppo_path, custom_path, seed, save_path):
    print(f"\nGenerating single-episode backtest behavior plot for seed {seed}...")
    cfg = load_config(config_path)
    env = HistoricalEnv(hist_data_path, cfg['option'], cfg['simulator'], cfg['environment']['trading_cost'])
    obs, _ = env.reset(seed=seed)
    S_path = env.S_path
    v_path_est = env.v_path_est # Use estimated variance path
    t_steps = np.arange(len(S_path))

    # Get DH positions using estimated vol
    dh = DeltaHedger(cfg['option'], cfg['environment']['trading_cost'])
    daily_vol_path = np.sqrt(np.maximum(v_path_est * env.dt, 1e-8))
    dh_positions = dh.get_positions_from_vol_path(S_path, daily_vol_path, env.dt) # Need to add this method

    # Get PPO positions
    ppo_model = PPO.load(ppo_path)
    ppo_positions = []
    obs, _ = env.reset(seed=seed)
    done = False
    while not done:
        action, _ = ppo_model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = env.step(action)
        ppo_positions.append(action[0])
        done = terminated or truncated

    # Get Custom positions
    custom_model = InterpretableHedger(obs_dim=env.observation_space.shape[0])
    custom_model.load_state_dict(torch.load(custom_path))
    custom_model.eval()
    custom_positions = []
    obs, _ = env.reset(seed=seed)
    obs_history = [torch.from_numpy(obs).float()]
    done = False
    while not done:
        with torch.no_grad():
            obs_sequence = torch.stack(obs_history).unsqueeze(0)
            action_tensor, _ = custom_model(obs_sequence)
            action = action_tensor.detach().numpy().flatten()
        obs, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        obs_history.append(torch.from_numpy(obs).float())
        custom_positions.append(action[0])

    # Plotting
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    ax1.plot(t_steps, S_path, label='Stock Price (Historical)', color='black')
    ax1.set_title(f'Backtest: Agent Hedging Behavior (Episode Seed {seed})', fontsize=16)
    ax1.set_ylabel('Stock Price ($)', fontsize=12)
    ax1.legend(loc='upper left')
    ax1.grid(True)
    ax2.plot(t_steps[:-1], dh_positions[:-1], label='Delta Hedger', linestyle='--', alpha=0.9)
    ax2.plot(t_steps[:-1], ppo_positions, label='PPO Agent', linestyle='-', alpha=0.8)
    ax2.plot(t_steps[:-1], custom_positions, label='Custom (CVaR) Agent', linestyle='-', alpha=0.8)
    ax2.set_xlabel('Time Step (Day)', fontsize=12)
    ax2.set_ylabel('Hedge Position (Units of Stock)', fontsize=12)
    ax2.legend(loc='upper left')
    ax2.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Backtest episode behavior plot saved to {save_path}")

# --- Need to add get_positions_from_vol_path to DeltaHedger ---
def get_positions_from_vol_path(self: DeltaHedger, S_path, daily_vol_path, dt):
    """Gets positions using a pre-computed daily volatility path."""
    n_steps = len(S_path) - 1
    positions = []
    for t_idx in range(n_steps):
        t_rem = self.T - (t_idx * dt)
        S_t = S_path[t_idx]
        sigma_t_annual = daily_vol_path[t_idx] * np.sqrt(252) # Approx annualized vol
        target_delta = self._bs_delta(S_t, self.K, t_idx * dt, self.T, 0.0, sigma_t_annual)
        positions.append(target_delta)
    positions.append(positions[-1]) # Hold final position
    return np.array(positions)
# Monkey-patch
DeltaHedger.get_positions_from_vol_path = get_positions_from_vol_path
# --- End of DeltaHedger modification ---

if __name__ == "__main__":
    # --- Configuration ---
    CONFIG_PATH = 'src/configs/default.yaml'
    HIST_DATA_PATH = 'data/historical_aapl.csv' # Path to your downloaded data
    PPO_MODEL_PATH = 'models/ppo_hedge'
    CUSTOM_MODEL_PATH = 'models/custom_hedger.pth'
    RESULTS_DIR = Path('results_backtest') # Save backtest results separately

    RESULTS_DIR.mkdir(exist_ok=True)

    # --- Run All Backtests ---
    print("Backtesting PPO Agent on Historical Data...")
    ppo_results_bt = backtest_rl_agent(PPO_MODEL_PATH, CONFIG_PATH, HIST_DATA_PATH, n_episodes=500)

    print("\nBacktesting Delta Hedger on Historical Data...")
    dh_results_bt = backtest_delta_hedger(CONFIG_PATH, HIST_DATA_PATH, n_episodes=500)

    print("\nBacktesting Custom Risk-Aware Agent on Historical Data...")
    custom_results_bt = backtest_custom_agent(CUSTOM_MODEL_PATH, CONFIG_PATH, HIST_DATA_PATH, n_episodes=500)

    # --- Save Detailed Backtest Results ---
    ppo_results_bt.to_csv(RESULTS_DIR / 'ppo_results_backtest.csv', index=False)
    dh_results_bt.to_csv(RESULTS_DIR / 'dh_results_backtest.csv', index=False)
    custom_results_bt.to_csv(RESULTS_DIR / 'custom_results_backtest.csv', index=False)
    print(f"\nDetailed backtest results saved in '{RESULTS_DIR}'.")

    # --- Calculate and Display Final Backtest Summary ---
    ppo_stats_bt = calculate_summary_stats(ppo_results_bt, "PPO Agent (Backtest)")
    dh_stats_bt = calculate_summary_stats(dh_results_bt, "Delta Hedger (Backtest)")
    custom_stats_bt = calculate_summary_stats(custom_results_bt, "Custom Agent (Backtest)")

    final_comparison_bt = pd.DataFrame([ppo_stats_bt, dh_stats_bt, custom_stats_bt])

    print("\n--- Final Backtest Performance Comparison ---")
    print(final_comparison_bt)

    # --- Generate and Save Backtest Plots ---
    results_dict_bt = {
        "Delta Hedger": dh_results_bt,
        "PPO Agent": ppo_results_bt,
        "Custom Agent (CVaR)": custom_results_bt
    }
    plot_error_histograms(results_dict_bt, RESULTS_DIR / 'backtest_error_distribution.png')
    plot_single_episode_behavior_backtest(
        CONFIG_PATH,
        HIST_DATA_PATH,
        PPO_MODEL_PATH,
        CUSTOM_MODEL_PATH,
        seed=250,  # Use the same seed for consistency
        save_path=RESULTS_DIR / 'backtest_episode_behavior_seed250.png'
    )