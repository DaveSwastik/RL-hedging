# src/analysis/metrics.py
import numpy as np
import pandas as pd
from tqdm import tqdm
import yaml

from src.envs.hedging_env import HedgingEnv
from src.baselines.delta_hedge import DeltaHedger
from stable_baselines3 import PPO
from src.utils.payoffs import PAYOFF_FUNCTIONS

def evaluate_rl_agent(model_path, config_path, n_episodes=1000):
    """Evaluates a trained RL agent."""
    cfg = yaml.safe_load(open(config_path, 'r'))
    env = HedgingEnv(cfg['simulator'], cfg['option'], cfg['environment']['trading_cost'])
    model = PPO.load(model_path)
    
    results = []
    for i in tqdm(range(n_episodes), desc="Evaluating RL Agent"):
        obs, _ = env.reset(seed=i)
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        results.append(info)
        
    return pd.DataFrame(results)

def evaluate_delta_hedger(config_path, n_episodes=1000):
    """Evaluates the baseline delta hedger."""
    cfg = yaml.safe_load(open(config_path, 'r'))
    simulator = HedgingEnv(cfg['simulator'], cfg['option']).simulator
    hedger = DeltaHedger(cfg['option'], cfg['environment']['trading_cost'])
    
    results = []
    for i in tqdm(range(n_episodes), desc="Evaluating Delta Hedger"):
        S, v, _ = simulator.simulate(n_paths=1, seed=i)
        pnl = hedger.evaluate(S[0], v[0], simulator.dt)
        payoff = PAYOFF_FUNCTIONS[cfg['option']['type']](S[0], cfg['option']['K'])
        results.append({'pnl': pnl, 'payoff': payoff, 'hedging_error': pnl - payoff})
        
    return pd.DataFrame(results)

def calculate_summary_stats(df_results, policy_name):
    """Computes VaR, CVaR, and other summary statistics."""
    errors = df_results['hedging_error']
    stats = {
        "Policy": policy_name,
        "Mean P&L": df_results['pnl'].mean(),
        "Std Dev P&L": df_results['pnl'].std(),
        "Mean Error": errors.mean(),
        "RMSE": np.sqrt((errors**2).mean()),
        "95% VaR": errors.quantile(0.05),
        "95% CVaR": errors[errors <= errors.quantile(0.05)].mean(),
    }
    return stats