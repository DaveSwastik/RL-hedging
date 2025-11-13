#!/usr/bin/env python
"""Quick test to check what data structure run_episode returns."""

import os
import torch
import yaml
from pathlib import Path
from src.envs.hedging_env import HedgingEnv
from src.agents.models import MlpActor

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
        
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
    
    # Standardize column names for consistency with DataFrames
    # Convert to float in case they're numpy arrays or tensors
    info['hedging_error'] = float(info['terminal_hedge_err'])
    info['pnl'] = float(info['terminal_hedge_err']) + float(info['terminal_payoff'])
    return info

if __name__ == "__main__":
    device = torch.device('cpu')
    cfg = load_config('src/configs/default.yaml')
    
    env = make_env(cfg)
    obs_dim = env.observation_space.shape[0]
    hidden_dim = cfg['model'].get('hidden_dim', 128)
    
    # Load custom agent
    cvar_agent = MlpActor(obs_dim=obs_dim, hidden_dim=hidden_dim, action_dim=1)
    checkpoint_path = './runs/training_run_01/checkpoint_final.pth'
    
    try:
        state_dict = torch.load(checkpoint_path, map_location=device)
        if 'actor' in state_dict:
            cvar_agent.load_state_dict(state_dict['actor'])
        else:
            cvar_agent.load_state_dict(state_dict)
        cvar_agent.to(device)
        cvar_agent.eval()
        print(f"✓ Loaded agent")
    except Exception as e:
        print(f"✗ Failed to load agent: {e}")
        exit(1)
    
    # Run one episode
    print("\nRunning one episode...")
    info = run_episode(env, cvar_agent, 'custom', seed=0, device=device)
    
    print(f"\nInfo dict keys: {info.keys()}")
    print(f"\nInfo dict values:")
    for k, v in info.items():
        print(f"  {k}: {v} (type: {type(v).__name__})")
    
    print(f"\n✓ Test successful!")
