# run_evaluation_suite.py
import torch
import pandas as pd
import numpy as np
import yaml
from tqdm import tqdm
from pathlib import Path
from stable_baselines3 import PPO

from src.envs.hedging_env import HedgingEnv
from src.agents.models import InterpretableHedger
from src.baselines.delta_hedge import DeltaHedger
from src.utils.payoffs import PAYOFF_FUNCTIONS

def load_config(path='src/configs/default.yaml'):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

# --- Helper function for Global Stats ---
def calculate_global_stats(df, policy_name):
    errors = df['hedging_error']
    pnl = df['pnl']
    trades = df['num_trades']
    stats = {
        "Policy": policy_name,
        "Mean P&L": pnl.mean(),
        "Std Dev P&L": pnl.std(),
        "Mean Error": errors.mean(),
        "RMSE": np.sqrt((errors**2).mean()),
        "Mean Abs Error": errors.abs().mean(),
        "95% VaR": errors.quantile(0.05),
        "95% CVaR": errors[errors <= errors.quantile(0.05)].mean(),
        "Avg Trades": trades.mean()
    }
    return stats

# --- Agent-specific Evaluation Functions ---
@torch.inference_mode()
def evaluate_custom_agent(actor_path, config, device, n_episodes):
    env = HedgingEnv(config['simulator'], config['option'], config['environment']['trading_cost'])
    obs_dim = env.observation_space.shape[0]
    actor = InterpretableHedger(obs_dim=obs_dim, hidden_dim=64).to(device)
    actor.load_state_dict(torch.load(actor_path, map_location=device))
    actor.eval()
    
    results = []
    for i in tqdm(range(n_episodes), desc="Evaluating Custom Agent"):
        obs, info = env.reset(seed=i + 50000) # Use test seeds
        obs_t = torch.from_numpy(obs).to(device=device, dtype=torch.float32).view(1, 1, -1)
        done = False
        h_a = None
        num_trades = 0
        last_pos = 0.0
        while not done:
            dist, h_a = actor.step(obs_t, h_a)
            a = torch.tanh(dist.base.mean)
            a_np = a.detach().cpu().numpy().flatten()
            
            if abs(a_np[0] - last_pos) > 1e-6:
                num_trades += 1
            last_pos = a_np[0]
            
            obs, _, terminated, truncated, info = env.step(a_np)
            done = terminated or truncated
            obs_t = torch.from_numpy(obs).to(device=device, dtype=torch.float32).view(1, 1, -1)
            if done:
                info['num_trades'] = num_trades
                results.append(info)
    return pd.DataFrame(results)

def evaluate_ppo_agent(model_path, config, device, n_episodes):
    env = HedgingEnv(config['simulator'], config['option'], config['environment']['trading_cost'])
    model = PPO.load(model_path, device=device)
    
    results = []
    for i in tqdm(range(n_episodes), desc="Evaluating PPO Agent"):
        obs, info = env.reset(seed=i + 50000)
        done = False
        num_trades = 0
        last_pos = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            if abs(action[0] - last_pos) > 1e-6:
                num_trades += 1
            last_pos = action[0]
            
            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            if done:
                info['num_trades'] = num_trades
                results.append(info)
    return pd.DataFrame(results)

def evaluate_delta_hedger(config, n_episodes):
    env = HedgingEnv(config['simulator'], config['option'], config['environment']['trading_cost'])
    hedger = DeltaHedger(config['option'], config['environment']['trading_cost'])
    payoff_fn = PAYOFF_FUNCTIONS[config['option']['type']]
    
    results = []
    for i in tqdm(range(n_episodes), desc="Evaluating Delta Hedger"):
        obs, info = env.reset(seed=i + 50000)
        S_path, v_path = env.S_path, env.v_path
        
        # Get positions and count trades
        positions = hedger.get_positions(S_path, v_path, env.sim_params['dt'])
        num_trades = np.count_nonzero(np.diff(np.insert(positions, 0, 0)))
        
        pnl = hedger.evaluate(S_path, v_path, env.sim_params['dt'])
        payoff = payoff_fn(S_path, env.K, env.option_type_name)
        
        info['pnl'] = pnl
        info['payoff'] = payoff
        info['hedging_error'] = pnl - payoff
        info['num_trades'] = num_trades
        results.append(info)
    return pd.DataFrame(results)

def run_evaluation(config_path, ppo_path, custom_path, n_episodes=500):
    cfg = load_config(config_path)
    device = torch.device("cpu")
    
    # --- Run all evaluations ---
    ppo_results = evaluate_ppo_agent(ppo_path, cfg, device, n_episodes)
    custom_results = evaluate_custom_agent(custom_path, cfg, device, n_episodes)
    dh_results = evaluate_delta_hedger(cfg, n_episodes)
    
    # --- 1. Global Metrics Table ---
    ppo_stats = calculate_global_stats(ppo_results, "PPO Agent")
    custom_stats = calculate_global_stats(custom_results, "Custom Agent (A2C)")
    dh_stats = calculate_global_stats(dh_results, "Delta Hedger")
    
    global_df = pd.DataFrame([dh_stats, ppo_stats, custom_stats])
    print("\n--- [Global Performance Metrics] ---")
    print(global_df.to_string())
    
    # --- 2. Per-Moneyness Analysis ---
    all_results = pd.concat([
        dh_results.assign(Policy="Delta Hedger"),
        ppo_results.assign(Policy="PPO Agent"),
        custom_results.assign(Policy="Custom Agent (A2C)")
    ])
    
    bins = [0.5, 0.8, 0.95, 1.05, 1.2, 1.5]
    labels = ["Deep OTM", "OTM", "ATM", "ITM", "Deep ITM"]
    all_results['Moneyness Bin'] = pd.cut(all_results['moneyness_start'], bins=bins, labels=labels, right=False)
    
    print("\n--- [Performance by Moneyness Bin (RMSE)] ---")
    bin_analysis = all_results.groupby(['Policy', 'Moneyness Bin'])['hedging_error'].agg(
        RMSE=lambda x: np.sqrt(np.mean(x**2))
    ).reset_index()
    print(bin_analysis.pivot(index='Moneyness Bin', columns='Policy', values='RMSE').to_string())

    # --- 3. Save Raw Data for Bootstrapping ---
    RESULTS_DIR = Path('results_suite')
    RESULTS_DIR.mkdir(exist_ok=True)
    all_results.to_csv(RESULTS_DIR / 'all_agent_errors.csv', index=False)
    print(f"\nRaw results saved to {RESULTS_DIR / 'all_agent_errors.csv'}")
    print("You can now use this CSV for bootstrapping CIs and p-values.")

if __name__ == "__main__":
    CONFIG_PATH = 'src/configs/default.yaml'
    # Use the new "best" model from training
    CUSTOM_MODEL_PATH = 'models/custom_a2c_hedger_cpu_hiddenstep_best.pth'
    PPO_MODEL_PATH = 'models/ppo_hedge_randomized.zip' # Use the new randomized PPO
    
    run_evaluation(CONFIG_PATH, PPO_MODEL_PATH, CUSTOM_MODEL_PATH)