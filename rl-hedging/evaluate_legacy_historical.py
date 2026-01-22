#!/usr/bin/env python3
"""evaluate.py

Evaluation script (env-based) modeled after evaluation_script.py, but:
- No PPO dependency
- Evaluates ERARL_Agent_V2 (src/agents/models.py) vs DeltaHedger baseline
- Uses HedgingEnv + YAML config (same simulator/payoff conventions as training)

Outputs (by default into ./results):
- era_rl_results.csv
- dh_results.csv
- final_summary_stats.csv
- hedging_error_distribution.png
- episode_behavior_seed<seed>.png (optional)

Example:
  python evaluate.py --config-path src/configs/default.yaml --model-path models/era_rl_agent.pth --n-episodes 500
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch
import yaml
from tqdm import tqdm

import matplotlib.pyplot as plt

from src.envs.hedging_env import HedgingEnv
from src.baselines.delta_hedge import DeltaHedger
from src.utils.payoffs import PAYOFF_FUNCTIONS
from src.agents.models import ERARL_Agent_V2


def load_config(path: str) -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    try:
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def _extract_terminal_metrics(info: dict) -> Dict[str, float]:
    """Normalize terminal info to {'pnl','payoff','hedging_error'}."""
    payoff = info.get('terminal_payoff', info.get('payoff', None))
    pnl = info.get('terminal_pnl', info.get('pnl', None))
    hedging_error = info.get('terminal_hedge_err', info.get('hedging_error', None))

    if pnl is None and payoff is None and hedging_error is None:
        # last-resort: some envs put these under different names
        # If nothing exists, return NaNs.
        return {'pnl': float('nan'), 'payoff': float('nan'), 'hedging_error': float('nan')}

    if hedging_error is None and pnl is not None and payoff is not None:
        hedging_error = float(pnl) - float(payoff)
    if pnl is None and payoff is not None and hedging_error is not None:
        pnl = float(hedging_error) + float(payoff)

    return {'pnl': float(pnl), 'payoff': float(payoff), 'hedging_error': float(hedging_error)}


def calculate_summary_stats(df_results: pd.DataFrame, policy_name: str) -> dict:
    """Match the style of src/analysis/metrics.py without importing PPO."""
    errors = df_results['hedging_error'].astype(float)
    pnl = df_results['pnl'].astype(float)

    q05 = errors.quantile(0.05)
    cvar = errors[errors <= q05].mean() if np.isfinite(q05) else float('nan')

    return {
        'Policy': policy_name,
        'Mean P&L': pnl.mean(),
        'Std Dev P&L': pnl.std(),
        'Mean Error': errors.mean(),
        'RMSE': float(np.sqrt((errors ** 2).mean())),
        '95% VaR': float(q05),
        '95% CVaR': float(cvar),
    }


def evaluate_delta_hedger(config_path: str, n_episodes: int, seed_offset: int = 0) -> pd.DataFrame:
    cfg = load_config(config_path)

    # Use the simulator directly for fast evaluation (same as src/analysis/metrics.py)
    simulator = HedgingEnv(cfg['simulator'], cfg['option']).simulator
    hedger = DeltaHedger(cfg['option'], cfg['environment']['trading_cost'])

    payoff_fn = PAYOFF_FUNCTIONS[cfg['option']['type']]
    K = cfg['option']['K']
    option_type = cfg['option'].get('option_type', 'call')

    rows = []
    for i in tqdm(range(n_episodes), desc='Evaluating Delta Hedger'):
        S, v, _ = simulator.simulate(n_paths=1, seed=int(seed_offset + i))
        pnl = float(hedger.evaluate(S[0], v[0], simulator.dt))
        payoff = float(payoff_fn(S[0], K, option_type=option_type))
        rows.append({'pnl': pnl, 'payoff': payoff, 'hedging_error': pnl - payoff})

    return pd.DataFrame(rows)


def evaluate_era_rl_agent(
    model_path: str,
    config_path: str,
    n_episodes: int,
    device: str = 'cpu',
    seed_offset: int = 0,
    deterministic: bool = True,
) -> pd.DataFrame:
    cfg = load_config(config_path)
    env = HedgingEnv(cfg['simulator'], cfg['option'], cfg['environment']['trading_cost'])

    obs_dim = int(env.observation_space.shape[0])
    model = ERARL_Agent_V2(obs_dim=obs_dim)

    state = torch.load(model_path, map_location='cpu')
    if isinstance(state, dict) and ('state_dict' in state or 'model_state_dict' in state):
        key = 'state_dict' if 'state_dict' in state else 'model_state_dict'
        state = state[key]
    model.load_state_dict(state)

    if device.startswith('cuda') and torch.cuda.is_available():
        torch_device = torch.device(device)
    else:
        torch_device = torch.device('cpu')

    model.to(torch_device)
    model.eval()

    rows = []
    for i in tqdm(range(n_episodes), desc='Evaluating ERA-RL V2'):
        obs, _ = env.reset(seed=int(seed_offset + i))

        obs_history = [torch.from_numpy(obs).float().to(torch_device)]
        prev_hedge = torch.tensor([[0.0]], dtype=torch.float32, device=torch_device)

        xi_steps = []
        unc_steps = []
        mode_steps = []

        done = False
        info = {}
        while not done:
            with torch.no_grad():
                obs_seq = torch.stack(obs_history).unsqueeze(0)  # [1, T, obs_dim]
                dist, risk_out = model(obs_seq, prev_hedge)

                if deterministic and hasattr(dist, 'mode'):
                    action_t = dist.mode()
                else:
                    action_t = dist.sample() if hasattr(dist, 'sample') else dist

                action_val = float(action_t.detach().cpu().view(-1)[0].item())

                # diagnostics
                try:
                    xi_steps.append(float(risk_out['xi'].detach().cpu().view(-1)[0].item()))
                    unc_steps.append(float(risk_out['uncertainty'].detach().cpu().view(-1)[0].item()))
                except Exception:
                    pass
                try:
                    mode_steps.append(int(torch.argmax(dist.cat_dist.logits, dim=-1).detach().cpu().item()))
                except Exception:
                    pass

            action = np.array([action_val], dtype=np.float32)
            obs, _, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)

            obs_history.append(torch.from_numpy(obs).float().to(torch_device))
            prev_hedge = torch.tensor([[action_val]], dtype=torch.float32, device=torch_device)

        terminal = _extract_terminal_metrics(info)

        row = {
            **terminal,
            'xi_mean': float(np.mean(xi_steps)) if len(xi_steps) else np.nan,
            'xi_max': float(np.max(xi_steps)) if len(xi_steps) else np.nan,
            'uncertainty_mean': float(np.mean(unc_steps)) if len(unc_steps) else np.nan,
        }

        # Add simple mode counts (0/1/2) if present
        if len(mode_steps):
            for m in [0, 1, 2]:
                row[f'mode_{m}_frac'] = float(np.mean(np.array(mode_steps) == m))
        rows.append(row)

    return pd.DataFrame(rows)


def plot_error_histograms(results_dict: Dict[str, pd.DataFrame], save_path: Path) -> None:
    plt.figure(figsize=(12, 7))

    for name, df in results_dict.items():
        x = df['hedging_error'].astype(float).dropna().values
        if len(x) == 0:
            continue
        plt.hist(x, bins=60, density=True, alpha=0.45, label=name)

    plt.title('Distribution of Hedging Errors', fontsize=16)
    plt.xlabel('Hedging Error (Final P&L - Payoff)', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.legend()
    plt.axvline(0, color='k', linestyle='--', alpha=0.7)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=200)
    plt.close()


def plot_single_episode_behavior(
    config_path: str,
    model_path: str,
    seed: int,
    save_path: Path,
    device: str = 'cpu',
) -> None:
    cfg = load_config(config_path)

    env = HedgingEnv(cfg['simulator'], cfg['option'], cfg['environment']['trading_cost'])
    obs, _ = env.reset(seed=seed)

    S_path = np.array(env.S_path, dtype=float)
    v_path = np.array(env.v_path, dtype=float)
    t_steps = np.arange(len(S_path))

    # Delta hedger positions on same path
    dh = DeltaHedger(cfg['option'], cfg['environment']['trading_cost'])
    dh_positions = dh.get_positions(S_path, v_path, cfg['simulator']['dt'])

    # ERA-RL positions on the same seed/path (reset again)
    obs_dim = int(env.observation_space.shape[0])
    model = ERARL_Agent_V2(obs_dim=obs_dim)
    state = torch.load(model_path, map_location='cpu')
    if isinstance(state, dict) and ('state_dict' in state or 'model_state_dict' in state):
        key = 'state_dict' if 'state_dict' in state else 'model_state_dict'
        state = state[key]
    model.load_state_dict(state)

    if device.startswith('cuda') and torch.cuda.is_available():
        torch_device = torch.device(device)
    else:
        torch_device = torch.device('cpu')

    model.to(torch_device)
    model.eval()

    era_positions = []
    obs, _ = env.reset(seed=seed)
    obs_history = [torch.from_numpy(obs).float().to(torch_device)]
    prev_hedge = torch.tensor([[0.0]], dtype=torch.float32, device=torch_device)

    done = False
    while not done:
        with torch.no_grad():
            obs_seq = torch.stack(obs_history).unsqueeze(0)
            dist, _ = model(obs_seq, prev_hedge)
            action_val = float(dist.mode().detach().cpu().view(-1)[0].item())

        obs, _, terminated, truncated, _ = env.step(np.array([action_val], dtype=np.float32))
        done = bool(terminated or truncated)
        era_positions.append(action_val)
        obs_history.append(torch.from_numpy(obs).float().to(torch_device))
        prev_hedge = torch.tensor([[action_val]], dtype=torch.float32, device=torch_device)

    # Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    ax1.plot(t_steps, S_path, label='Stock Price', color='black')
    ax1.set_title(f'Agent Hedging Behavior (Episode Seed {seed})', fontsize=16)
    ax1.set_ylabel('Stock Price', fontsize=12)
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)

    ax2.plot(t_steps[:-1], dh_positions[:-1], label='Delta Hedger', linestyle='--', alpha=0.9)
    ax2.plot(t_steps[:-1], era_positions, label='ERA-RL V2', linestyle='-', alpha=0.85)
    ax2.set_xlabel('Time Step', fontsize=12)
    ax2.set_ylabel('Hedge Position (Units of Stock)', fontsize=12)
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Evaluate ERARL_Agent_V2 vs Delta Hedger (no PPO).')
    p.add_argument('--config-path', type=str, default='src/configs/default.yaml')
    p.add_argument('--model-path', type=str, default='models/era_rl_agent.pth')
    p.add_argument('--n-episodes', type=int, default=500)
    p.add_argument('--results-dir', type=str, default='results')
    p.add_argument('--seed', type=int, default=0, help='Seed offset for episode generation.')
    p.add_argument('--device', type=str, default='cpu')
    p.add_argument('--behavior-seed', type=int, default=250, help='Seed for single-episode behavior plot.')
    p.add_argument('--no-behavior-plot', action='store_true', help='Skip single-episode behavior plot.')
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    era_path = Path(args.model_path)
    if not era_path.exists():
        raise FileNotFoundError(f'Model not found: {era_path}')

    print('Evaluating Delta Hedger...')
    dh_results = evaluate_delta_hedger(args.config_path, n_episodes=args.n_episodes, seed_offset=args.seed)

    print('\nEvaluating ERA-RL V2 Agent...')
    era_results = evaluate_era_rl_agent(
        str(era_path),
        args.config_path,
        n_episodes=args.n_episodes,
        device=args.device,
        seed_offset=args.seed,
        deterministic=True,
    )

    dh_results.to_csv(results_dir / 'dh_results.csv', index=False)
    era_results.to_csv(results_dir / 'era_rl_results.csv', index=False)

    dh_stats = calculate_summary_stats(dh_results, 'Delta Hedger')
    era_stats = calculate_summary_stats(era_results, 'ERA-RL V2')

    final_comparison = pd.DataFrame([dh_stats, era_stats])
    final_comparison.to_csv(results_dir / 'final_summary_stats.csv', index=False)

    print('\n--- Final Hedging Performance Comparison ---')
    print(final_comparison)

    plot_error_histograms(
        {'Delta Hedger': dh_results, 'ERA-RL V2': era_results},
        results_dir / 'hedging_error_distribution.png',
    )

    if not args.no_behavior_plot:
        plot_single_episode_behavior(
            args.config_path,
            str(era_path),
            seed=int(args.behavior_seed),
            save_path=results_dir / f'episode_behavior_seed{int(args.behavior_seed)}.png',
            device=args.device,
        )


if __name__ == '__main__':
    main()
