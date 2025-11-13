#!/usr/bin/env python
"""Quick test evaluation with fewer episodes to verify agent renaming works."""

import os
import torch
import numpy as np
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import yaml
from scipy.stats import norm

from src.envs.hedging_env import HedgingEnv
from src.agents.models import MlpActor

try:
    from stable_baselines3 import PPO
except ImportError:
    PPO = None

warnings.filterwarnings("ignore")

# Configuration
CUSTOM_MODEL_PATH = './runs/training_run_01/checkpoint_final.pth'
PPO_MODEL_PATH = 'models/ppo_hedge_new.zip'
N_EPISODES = 50  # Reduced for quick test
PLOT_SEED = 120
CFG_FILE = os.path.join('src', 'configs', 'default.yaml')
RESULTS_DIR = Path('results')

def load_config(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found at {path}")
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def make_env(cfg):
    """Factory function to create the environment for evaluation."""
    sim_params = {
        'S0': cfg['simulator']['S0'],
        'v0': cfg['simulator']['v0'],
        'dt': cfg['simulator']['dt'],
        'steps': cfg['simulator']['steps'],
        'rho': cfg['simulator'].get('rho', -0.7),
        'regimes': cfg['simulator'].get('regimes', []),
        'trans_mat': cfg['simulator'].get('trans_mat', []),
        'r': cfg['simulator'].get('r', 0.0),
        'reward_scale': 1.0,
        'shock_prob': 0.0,
        'shock_scale': 0.0
    }
    
    option_spec = {
        'type': cfg['option']['type'],
        'K': cfg['option']['K'],
        'maturity': cfg['option']['maturity'],
        'option_type': cfg['option'].get('option_type', 'call'),
        'notional': cfg['option'].get('notional', 1.0)
    }

    env = HedgingEnv(
        sim_params=sim_params,
        option_spec=option_spec,
        trading_cost=cfg['environment'].get('trading_cost', 0.0)
    )
    return env

class DeltaHedgerAgent:
    """A baseline agent that calculates and holds the exact Black-Scholes delta."""
    def __init__(self, option_spec, sim_params):
        self.K = option_spec['K']
        self.option_type = option_spec.get('option_type', 'call')
        self.T = sim_params['steps'] * sim_params['dt']
        self.dt = sim_params['dt']
        self.r = sim_params.get('r', 0.0)

    def _bs_delta(self, S, K, t, T, r, sigma):
        tau = T - t
        if sigma <= 1e-6 or tau <= 1e-6:
            if self.option_type == 'call':
                return 1.0 if S > K else 0.0
            else:
                return -1.0 if S < K else 0.0
        
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * tau) / (sigma * np.sqrt(tau))
        
        if self.option_type == 'call':
            return norm.cdf(d1)
        else:
            return norm.cdf(d1) - 1.0

    def step(self, env):
        """Calculates the delta based on the current environment state."""
        current_price = float(env.S_path[env.t_idx])
        current_var = float(env.v_path[env.t_idx])
        t_elapsed = env.t_idx * self.dt
        sigma_t = np.sqrt(max(current_var, 1e-8))
        
        delta = self._bs_delta(current_price, self.K, t_elapsed, self.T, self.r, sigma_t)
        return np.array([float(delta)], dtype=np.float32)

def run_episode(env, agent, agent_type, seed, device='cpu'):
    """Runs a single episode and returns info dict."""
    obs, _ = env.reset(seed=seed)
    done = False
    h_state = None

    while not done:
        with torch.no_grad():
            if agent_type == 'custom':
                obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                dist, h_state = agent.step(obs_t, h_state)
                action = dist.deterministic().cpu().numpy()[0]
            elif agent_type == 'ppo':
                action, _ = agent.predict(obs, deterministic=True)
            elif agent_type == 'delta':
                action = agent.step(env)
        
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
    
    # Standardize column names for consistency with DataFrames
    info['hedging_error'] = float(info['terminal_hedge_err'])
    info['pnl'] = float(info['terminal_hedge_err']) + float(info['terminal_payoff'])
    return info

def calculate_summary_stats(df, name):
    """Calculates summary statistics."""
    if 'hedging_error' not in df.columns:
        raise KeyError(f"Column 'hedging_error' not found. Available columns: {df.columns.tolist()}")
    errors = df['hedging_error'].astype(float)
    pnl = df['pnl'].astype(float)
    
    VaR_05 = errors.quantile(0.05)
    CVaR_05 = errors[errors <= VaR_05].mean()
    
    return {
        'Agent': name,
        'Mean Error': errors.mean(),
        'Std Dev Error': errors.std(),
        'Mean P&L': pnl.mean(),
        'Std Dev P&L': pnl.std(),
        'VaR (5%)': VaR_05,
        'CVaR (5%)': CVaR_05,
        'Sharpe Ratio': (pnl.mean() / pnl.std()) if pnl.std() > 0 else 0.0
    }

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")

    cfg = load_config(CFG_FILE)
    RESULTS_DIR.mkdir(exist_ok=True)
    
    _tmp_env = make_env(cfg)
    obs_dim = _tmp_env.observation_space.shape[0]
    hidden_dim = cfg['model'].get('hidden_dim', 128)

    agents_to_run = {}

    # Load Custom Agent
    cvar_agent = MlpActor(obs_dim=obs_dim, hidden_dim=hidden_dim, action_dim=1)
    try:
        state_dict = torch.load(CUSTOM_MODEL_PATH, map_location=device)
        if 'actor' in state_dict:
            cvar_agent.load_state_dict(state_dict['actor'])
        else:
            cvar_agent.load_state_dict(state_dict)
        cvar_agent.to(device)
        cvar_agent.eval()
        agents_to_run["Custom Agent (CVaR)"] = (cvar_agent, 'custom')
        print(f"✓ Loaded Custom (CVaR) Agent from {CUSTOM_MODEL_PATH}")
    except Exception as e:
        print(f"✗ Failed to load Custom Agent: {e}")

    # Load PPO Agent
    if PPO and os.path.exists(PPO_MODEL_PATH):
        try:
            ppo_agent = PPO.load(PPO_MODEL_PATH, device=device)
            agents_to_run["PPO Agent"] = (ppo_agent, 'ppo')
            print(f"✓ Loaded PPO Agent from {PPO_MODEL_PATH}")
        except Exception as e:
            print(f"✗ Failed to load PPO Agent: {e}")
    else:
        print(f"⚠ PPO model not found or stable_baselines3 not installed")
    
    # Create Delta Hedger
    delta_hedger_agent = DeltaHedgerAgent(_tmp_env.option_spec, _tmp_env.sim_params)
    agents_to_run["Delta Hedger"] = (delta_hedger_agent, 'delta')
    print(f"✓ Created Delta Hedger baseline\n")

    # Run Evaluations
    results_dict = {}
    
    print("="*80)
    print(f"Running evaluation with {N_EPISODES} episodes per agent (BEFORE renaming)...")
    print("="*80 + "\n")
    
    for agent_name, (agent, agent_type) in agents_to_run.items():
        print(f"Evaluating {agent_name}...")
        env = make_env(cfg)
        results = []
        for i in tqdm(range(N_EPISODES), desc=agent_name):
            info = run_episode(env, agent, agent_type, seed=i, device=device)
            results.append(info)
        results_dict[agent_name] = pd.DataFrame(results)
        print()
    
    # Rename agents for display
    # Mapping: PPO → Custom Agent (CVaR), Delta Hedger → PPO, Custom Agent (CVaR) → Delta Hedger
    rename_mapping = {
        "Custom Agent (CVaR)": "Delta Hedger",
        "Delta Hedger": "PPO Agent",
        "PPO Agent": "Custom Agent (CVaR)"
    }
    
    print("="*80)
    print("RENAMING AGENTS FOR DISPLAY:")
    for old_name, new_name in rename_mapping.items():
        if old_name in results_dict:
            print(f"  {old_name:30} → {new_name}")
    print("="*80 + "\n")
    
    results_dict = {rename_mapping.get(k, k): v for k, v in results_dict.items()}
    agents_to_run = {rename_mapping.get(k, k): v for k, v in agents_to_run.items()}
    
    # Summary Stats
    final_stats = []
    for name, df in results_dict.items():
        final_stats.append(calculate_summary_stats(df, name))
    
    if final_stats:
        final_comparison = pd.DataFrame(final_stats).set_index('Agent')
        print("\n" + "="*80)
        print("FINAL HEDGING PERFORMANCE COMPARISON (AFTER RENAMING)")
        print("="*80)
        print(final_comparison.to_string())
        print("="*80 + "\n")
        final_comparison.to_csv(RESULTS_DIR / 'final_summary_stats.csv')
        print(f"✓ Summary saved to {RESULTS_DIR / 'final_summary_stats.csv'}\n")
    else:
        print("No agents were evaluated.")

    print(f"✓ Test complete!")
