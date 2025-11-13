# run_evaluation_improved.py
import os
import torch
import numpy as np
import pprint
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import yaml
from scipy.stats import norm

# --- Import from your src ---
from src.envs.hedging_env import HedgingEnv
from src.agents.models import MlpActor
from src.utils.bs import bs_price_and_delta

# --- Import Baselines ---
try:
    from stable_baselines3 import PPO
except ImportError:
    print("Warning: stable_baselines3 not installed. PPO agent will be skipped.")
    PPO = None

warnings.filterwarnings("ignore")

# -------------------------------------------------------------------
# --- 1. CONFIGURATION ---
# -------------------------------------------------------------------

CUSTOM_MODEL_PATH = './runs/training_run_01/checkpoint_final.pth'
PPO_MODEL_PATH = 'models/ppo_hedge_new.zip'
N_EPISODES = 500
PLOT_SEED = 300
CFG_FILE = os.path.join('src', 'configs', 'default.yaml')
RESULTS_DIR = Path('results')

# -------------------------------------------------------------------

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

# -------------------------------------------------------------------
# --- 2. DELTA HEDGER ---
# -------------------------------------------------------------------
class DeltaHedgerAgent:
    """Baseline: Black-Scholes delta hedger."""
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
        """Calculates the delta."""
        current_price = float(env.S_path[env.t_idx])
        current_var = float(env.v_path[env.t_idx])
        t_elapsed = env.t_idx * self.dt
        sigma_t = np.sqrt(max(current_var, 1e-8))
        
        delta = self._bs_delta(current_price, self.K, t_elapsed, self.T, self.r, sigma_t)
        return np.array([float(delta)], dtype=np.float32)

# -------------------------------------------------------------------
# --- 3. EVALUATION ---
# -------------------------------------------------------------------

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
    # Convert to float in case they're numpy arrays or tensors
    info['hedging_error'] = float(info['terminal_hedge_err'])
    info['pnl'] = float(info['terminal_hedge_err']) + float(info['terminal_payoff'])
    return info

def calculate_summary_stats(df, name):
    """Calculates summary statistics."""
    # Ensure columns exist and are numeric
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

# -------------------------------------------------------------------
# --- 4. PLOTTING FUNCTIONS (IMPROVED) ---
# -------------------------------------------------------------------

def plot_error_boxplot(results_dict, save_path):
    """Box plot: Better visualization of distributions than overlapped histograms."""
    print(f"Generating error box plot...")
    
    data_for_plot = []
    for agent_name, df in results_dict.items():
        for error in df['hedging_error']:
            data_for_plot.append({'Agent': agent_name, 'Hedging Error': error})
    
    df_plot = pd.DataFrame(data_for_plot)
    
    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")
    sns.boxplot(data=df_plot, x='Agent', y='Hedging Error', palette="Set2")
    
    # Add perfect hedge line
    plt.axhline(0, color='r', linestyle='--', linewidth=2, label='Perfect Hedge', alpha=0.7)
    
    plt.title('Distribution of Hedging Errors (Box Plot)', fontsize=14, fontweight='bold')
    plt.ylabel('Hedging Error', fontsize=12)
    plt.xlabel('Agent', fontsize=12)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Box plot saved to {save_path}")

def plot_cvar_comparison(results_dict, save_path):
    """Bar chart: VaR vs CVaR for each agent (CRITICAL for CVaR-trained model)."""
    print(f"Generating CVaR comparison plot...")
    
    agents = []
    vars_05 = []
    cvars_05 = []
    
    for name, df in results_dict.items():
        errors = df['hedging_error']
        var_05 = errors.quantile(0.05)
        cvar_05 = errors[errors <= var_05].mean()
        
        agents.append(name)
        vars_05.append(var_05)
        cvars_05.append(cvar_05)
    
    x = np.arange(len(agents))
    width = 0.35
    
    plt.figure(figsize=(10, 6))
    bars1 = plt.bar(x - width/2, vars_05, width, label='VaR (5%)', alpha=0.8, color='steelblue')
    bars2 = plt.bar(x + width/2, cvars_05, width, label='CVaR (5%)', alpha=0.8, color='darkorange')
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.4f}', ha='center', va='bottom', fontsize=9)
    
    plt.xlabel('Agent', fontsize=12)
    plt.ylabel('Loss (Lower is Better)', fontsize=12)
    plt.title('Tail Risk Comparison: VaR vs CVaR (5%)', fontsize=14, fontweight='bold')
    plt.xticks(x, agents, rotation=45, ha='right')
    plt.legend(fontsize=11)
    plt.axhline(0, color='k', linestyle=':', alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"CVaR comparison plot saved to {save_path}")

def plot_pnl_distribution(results_dict, save_path):
    """Histogram: Final P&L distribution (not just errors!)."""
    print(f"Generating P&L distribution plot...")
    
    plt.figure(figsize=(12, 6))
    sns.set_style("whitegrid")
    
    for agent_name, df in results_dict.items():
        plt.hist(df['pnl'], bins=40, alpha=0.6, label=agent_name, edgecolor='black')
    
    plt.axvline(0, color='r', linestyle='--', linewidth=2, label='Break-even', alpha=0.8)
    plt.xlabel('Final P&L', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Distribution of Final P&L Across Episodes', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"P&L distribution plot saved to {save_path}")

def plot_mean_vs_std(results_dict, save_path):
    """Scatter: Risk-return tradeoff (low left = best)."""
    print(f"Generating risk-return plot...")
    
    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")
    
    colors = {'Delta Hedger': 'blue', 'PPO Agent': 'orange', 'Custom Agent (CVaR)': 'green'}
    
    for agent_name, df in results_dict.items():
        mean_err = df['hedging_error'].mean()
        std_err = df['hedging_error'].std()
        color = colors.get(agent_name, 'gray')
        plt.scatter(std_err, mean_err, s=300, label=agent_name, alpha=0.7, color=color, edgecolor='black', linewidth=2)
        
        # Add label
        plt.annotate(agent_name, (std_err, mean_err), xytext=(5, 5), textcoords='offset points', fontsize=10)
    
    plt.xlabel('Std Dev of Hedging Error (Risk)', fontsize=12, fontweight='bold')
    plt.ylabel('Mean Hedging Error (Bias)', fontsize=12, fontweight='bold')
    plt.title('Risk-Return Tradeoff (Lower-Left is Better)', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11, loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.axhline(0, color='k', linestyle='--', alpha=0.3)
    plt.axvline(0, color='k', linestyle='--', alpha=0.3)
    
    # Shade lower-left quadrant (best region)
    plt.axhspan(-np.inf, 0, alpha=0.05, color='green', label='Desired Region')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Risk-return plot saved to {save_path}")

def plot_error_stability(results_dict, save_path):
    """Line plot: Rolling average of errors over episodes."""
    print(f"Generating stability plot...")
    
    plt.figure(figsize=(12, 6))
    sns.set_style("whitegrid")
    
    for agent_name, df in results_dict.items():
        errors = df['hedging_error'].values
        rolling_mean = pd.Series(errors).rolling(window=50, center=True).mean()
        plt.plot(rolling_mean, label=agent_name, linewidth=2.5, alpha=0.8)
    
    plt.xlabel('Episode Number', fontsize=12)
    plt.ylabel('Hedging Error (50-episode rolling avg)', fontsize=12)
    plt.title('Agent Performance Stability Over Episodes', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.axhline(0, color='r', linestyle='--', alpha=0.5, label='Perfect Hedge')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Stability plot saved to {save_path}")

def plot_episode_behavior(agents, cfg, seed, save_path, device):
    """Positions and cumulative P&L for a single episode."""
    print(f"\nGenerating single-episode analysis (seed {seed})...")
    
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(3, 1, hspace=0.4)
    
    ax_price = fig.add_subplot(gs[0])
    ax_position = fig.add_subplot(gs[1], sharex=ax_price)
    ax_pnl = fig.add_subplot(gs[2], sharex=ax_price)
    
    # Get market path
    env_ref = make_env(cfg)
    obs_ref, _ = env_ref.reset(seed=seed)
    S_path = env_ref.S_path.copy()
    t_steps = np.arange(len(S_path))
    
    # Plot stock price
    ax_price.plot(t_steps, S_path, label='Stock Price', color='black', linewidth=2, alpha=0.8)
    ax_price.set_ylabel('Stock Price ($)', fontsize=11, fontweight='bold')
    ax_price.set_title(f'Agent Behavior Analysis (Episode Seed {seed})', fontsize=14, fontweight='bold')
    ax_price.legend(loc='upper left', fontsize=10)
    ax_price.grid(True, alpha=0.3)
    
    # Run each agent and collect positions + P&L
    colors = {'Delta Hedger': 'blue', 'PPO Agent': 'orange', 'Custom Agent (CVaR)': 'green'}
    
    for agent_name, (agent, agent_type) in agents.items():
        env_sim = make_env(cfg)
        obs_sim, _ = env_sim.reset(seed=seed)
        positions = []
        cumulative_pnl = [0.0]
        done = False
        h_state = None
        
        for t in range(env_sim.T):
            with torch.no_grad():
                if agent_type == 'custom':
                    obs_t = torch.as_tensor(obs_sim, dtype=torch.float32, device=device).unsqueeze(0)
                    dist, h_state = agent.step(obs_t, h_state)
                    action = dist.deterministic().cpu().numpy()[0]
                elif agent_type == 'ppo':
                    action, _ = agent.predict(obs_sim, deterministic=True)
                elif agent_type == 'delta':
                    action = agent.step(env_sim)
            
            obs_sim, reward, terminated, truncated, info = env_sim.step(action)
            positions.append(action[0])
            cumulative_pnl.append(cumulative_pnl[-1] + reward)
            done = terminated or truncated
            if done:
                break
        
        # Plot positions
        color = colors.get(agent_name, 'gray')
        ax_position.plot(range(len(positions)), positions, label=agent_name, 
                        linewidth=2, marker='o', markersize=3, alpha=0.7, color=color)
        
        # Plot cumulative P&L
        ax_pnl.plot(range(len(cumulative_pnl)), cumulative_pnl, label=agent_name, 
                   linewidth=2, marker='o', markersize=3, alpha=0.7, color=color)
    
    ax_position.set_ylabel('Hedge Position (Units)', fontsize=11, fontweight='bold')
    ax_position.set_xlabel('Time Step', fontsize=11, fontweight='bold')
    ax_position.legend(loc='upper left', fontsize=10)
    ax_position.grid(True, alpha=0.3)
    ax_position.axhline(0, color='k', linestyle='--', alpha=0.3)
    
    ax_pnl.set_ylabel('Cumulative P&L', fontsize=11, fontweight='bold')
    ax_pnl.set_xlabel('Time Step', fontsize=11, fontweight='bold')
    ax_pnl.legend(loc='upper left', fontsize=10)
    ax_pnl.grid(True, alpha=0.3)
    ax_pnl.axhline(0, color='r', linestyle='--', alpha=0.5, linewidth=1.5)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Episode analysis plot saved to {save_path}")

# -------------------------------------------------------------------
# --- 5. MAIN ---
# -------------------------------------------------------------------
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
    
    results_dict = {rename_mapping.get(k, k): v for k, v in results_dict.items()}
    agents_to_run = {rename_mapping.get(k, k): v for k, v in agents_to_run.items()}
    
    # Summary Stats
    final_stats = []
    for name, df in results_dict.items():
        final_stats.append(calculate_summary_stats(df, name))
    
    if final_stats:
        final_comparison = pd.DataFrame(final_stats).set_index('Agent')
        print("\n" + "="*80)
        print("FINAL HEDGING PERFORMANCE COMPARISON")
        print("="*80)
        print(final_comparison.to_string())
        print("="*80 + "\n")
        final_comparison.to_csv(RESULTS_DIR / 'final_summary_stats.csv')
        print(f"Summary saved to {RESULTS_DIR / 'final_summary_stats.csv'}\n")

    # Generate All Plots
    if results_dict:
        print("Generating plots...")
        plot_error_boxplot(results_dict, RESULTS_DIR / 'hedging_error_boxplot.png')
        plot_cvar_comparison(results_dict, RESULTS_DIR / 'cvar_comparison.png')
        plot_pnl_distribution(results_dict, RESULTS_DIR / 'pnl_distribution.png')
        plot_mean_vs_std(results_dict, RESULTS_DIR / 'risk_return_tradeoff.png')
        plot_error_stability(results_dict, RESULTS_DIR / 'performance_stability.png')
        plot_episode_behavior(agents_to_run, cfg, PLOT_SEED, 
                             RESULTS_DIR / f'episode_analysis_seed{PLOT_SEED}.png', device)
        print(f"\n✓ All plots saved to {RESULTS_DIR}/")
    else:
        print("No results to plot.")

    print(f"\n{'='*80}")
    print(f"Evaluation complete! Results saved in '{RESULTS_DIR}' directory.")
    print(f"{'='*80}")
