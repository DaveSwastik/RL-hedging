# run_evaluation.py
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
from src.agents.models import MlpActor # Your trained model
from src.utils.bs import bs_price_and_delta

# --- Import Baselines ---
try:
    from stable_baselines3 import PPO
except ImportError:
    print("Warning: stable_baselines3 not installed. PPO agent will be skipped.")
    PPO = None

warnings.filterwarnings("ignore")

# -------------------------------------------------------------------
# --- 1. CONFIGURATION: SET THESE PARAMETERS ---
# -------------------------------------------------------------------

# Point this to the model checkpoint you saved during training
# e.g., './runs/training_run_01/checkpoint_4900.pth'
CUSTOM_MODEL_PATH = './runs/training_run_01/checkpoint_4900.pth' # <-- UPDATE THIS

# --- *** UPDATED PATH *** ---
# Path to your NEW PPO agent
PPO_MODEL_PATH = 'models/ppo_hedge_new.zip' 
# --- *** END UPDATE *** ---

# Set the number of episodes to run for the distribution plot
N_EPISODES = 500

# Set the specific episode seed to plot for the time-series graph
PLOT_SEED = 120 # (This matches your sample graph)

# Set the path to your config file
CFG_FILE = os.path.join('src', 'configs', 'default.yaml')

# Output directory for plots and CSVs
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
        'reward_scale': 1.0, # Not used in eval, but good to have
        
        # --- IMPORTANT: Turn OFF synthetic shocks for evaluation ---
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
# --- 2. BASELINE AGENT: Delta Hedger ---
# -------------------------------------------------------------------
class DeltaHedgerAgent:
    """A baseline agent that calculates and holds the exact Black-Scholes delta."""
    def __init__(self, option_spec, sim_params):
        self.K = option_spec['K']
        self.option_type = option_spec.get('option_type', 'call')
        self.T = sim_params['steps'] * sim_params['dt'] # Total maturity in years
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
        else: # put
            return norm.cdf(d1) - 1.0

    def step(self, env_state):
        """Calculates the delta based on the *current* environment state."""
        current_price = float(env_state.S_path[env_state.t_idx])
        current_var = float(env_state.v_path[env_state.t_idx])
        t_elapsed = env_state.t_idx * self.dt
        sigma_t = np.sqrt(max(current_var, 1e-8))
        
        delta = self._bs_delta(current_price, self.K, t_elapsed, self.T, self.r, sigma_t)
        return np.array([float(delta)], dtype=np.float32)

# -------------------------------------------------------------------
# --- 3. EVALUATION RUNNERS ---
# -------------------------------------------------------------------

def run_episode(env, agent, agent_type, seed, device='cpu'):
    """Runs a single episode for a given agent and returns the final info dict."""
    obs, _ = env.reset(seed=seed)
    done = False
    h_state = None # Hidden state (None for MLPs)

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
    
    # Store final info dict
    info['pnl'] = info['terminal_hedge_err'] + info['terminal_payoff']
    return info

def calculate_summary_stats(df, name):
    """Calculates summary statistics for the results dataframe."""
    errors = df['hedging_error']
    pnl = df['pnl']
    
    VaR_05 = errors.quantile(0.05)
    CVaR_05 = errors[errors <= VaR_05].mean()
    
    return {
        'Agent': name,
        'Mean Error': errors.mean(),
        'Std Dev Error': errors.std(),
        'Mean P&L': pnl.mean(),
        'Std Dev P&L': pnl.std(),
        'VaR (5%)': VaR_05,
        'CVaR (5%)': CVaR_05
    }

# -------------------------------------------------------------------
# --- 4. PLOTTING FUNCTIONS (to match your samples) ---
# -------------------------------------------------------------------

def plot_error_histograms(results_dict, save_path):
    """Plots a single histogram comparing the hedging error distributions."""
    print(f"Generating error distribution plot...")
    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")
    
    data = []
    palette = {}
    
    if "Delta Hedger" in results_dict:
        palette["Delta Hedger"] = "tab:blue"
        for err in results_dict["Delta Hedger"]['hedging_error']:
            data.append({'Agent': "Delta Hedger", 'Hedging Error': err})
            
    if "PPO Agent" in results_dict:
        palette["PPO Agent"] = "tab:orange"
        for err in results_dict["PPO Agent"]['hedging_error']:
            data.append({'Agent': "PPO Agent", 'Hedging Error': err})
            
    if "Custom Agent (CVaR)" in results_dict:
        palette["Custom Agent (CVaR)"] = "tab:green"
        for err in results_dict["Custom Agent (CVaR)"]['hedging_error']:
            data.append({'Agent': "Custom Agent (CVaR)", 'Hedging Error': err})

    if not data:
        print("No data to plot for histogram.")
        return

    df_plot = pd.DataFrame(data)

    sns.histplot(df_plot, x='Hedging Error', hue='Agent', palette=palette, 
                 multiple='layer', kde=True, stat="frequency", bins=50, element="step")
    
    # Add VaR lines
    for name, df in results_dict.items():
        var_05 = df['hedging_error'].quantile(0.05)
        plt.axvline(var_05, color=palette[name], linestyle='--', 
                    label=f'{name} 5% VaR: {var_05:.2f}')
        
    plt.title(f'Distribution of Hedging Errors ({N_EPISODES} Episodes)', fontsize=16)
    plt.xlabel('Hedging Error (Final P&L - Payoff)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.legend()
    plt.axvline(0, color='k', linestyle=':', alpha=0.7, label='Perfect Hedge')
    plt.savefig(save_path)
    print(f"Error distribution plot saved to {save_path}")

def plot_single_episode_behavior(agents, cfg, seed, save_path, device):
    """
    Runs a single episode for all agents on the same market path
    and plots their hedging positions.
    """
    print(f"\nGenerating single-episode behavior plot for seed {seed}...")
    
    # 1. Get the single market path
    env = make_env(cfg)
    obs, _ = env.reset(seed=seed)
    S_path = env.S_path.copy()
    t_steps = np.arange(len(S_path))
    
    all_positions = {}

    # Run each agent
    for name, (agent, agent_type) in agents.items():
        env_sim = make_env(cfg)
        obs_sim, _ = env_sim.reset(seed=seed)
        positions = []
        done = False
        h_state = None
        
        for t in range(env_sim.T): # Iterate up to T steps
            with torch.no_grad():
                if agent_type == 'custom':
                    obs_t = torch.as_tensor(obs_sim, dtype=torch.float32, device=device).unsqueeze(0)
                    dist, h_state = agent.step(obs_t, h_state)
                    action = dist.deterministic().cpu().numpy()[0]
                elif agent_type == 'ppo':
                    action, _ = agent.predict(obs_sim, deterministic=True)
                elif agent_type == 'delta':
                    action = agent.step(env_sim)
            
            obs_sim, _, terminated, truncated, _ = env_sim.step(action)
            positions.append(action[0])
            done = terminated or truncated
            if done:
                break
        
        all_positions[name] = positions

    # 5. Create the plots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True, 
                                  gridspec_kw={'height_ratios': [1, 2]})
    
    # Subplot 1: Stock Price
    ax1.plot(t_steps, S_path, label='Stock Price', color='k', alpha=0.8)
    ax1.set_title(f'Agent Hedging Behavior (Episode Seed {seed})', fontsize=16)
    ax1.set_ylabel('Stock Price ($)', fontsize=12)
    ax1.legend(loc='upper left')
    ax1.grid(True)

    # Subplot 2: Agent Positions
    plot_data = []
    for agent_name, positions in all_positions.items():
        # Add a final "0" position for liquidation at T
        pos_with_liq = positions + [0.0]
        # Ensure we plot for T+1 total time steps
        for t, pos in enumerate(pos_with_liq):
             if t >= len(t_steps): continue # Should not happen, but safeguard
             plot_data.append({'Time Step': t_steps[t], 'Agent': agent_name, 'Hedge Position': pos})
    
    df_plot = pd.DataFrame(plot_data)

    palette = {
        "Delta Hedger": "tab:blue",
        "PPO Agent": "tab:orange",
        "Custom Agent (CVaR)": "tab:green"
    }
    dashes = {
        "Delta Hedger": (2, 2),
        "PPO Agent": (5, 2),
        "Custom Agent (CVaR)": ""
    }

    sns.lineplot(data=df_plot, x='Time Step', y='Hedge Position', hue='Agent', 
                 style='Agent', dashes=dashes, palette=palette, ax=ax2)
    
    ax2.set_xlabel('Time Step', fontsize=12)
    ax2.set_ylabel('Hedge Position (Units of Stock)', fontsize=12)
    ax2.legend(loc='upper left')
    ax2.grid(True)
    ax2.set_xlim(left=0, right=len(t_steps)-1) # Ensure x-axis is aligned

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path)
    print(f"Episode behavior plot saved to {save_path}")

# -------------------------------------------------------------------
# --- 5. MAIN EXECUTION BLOCK ---
# -------------------------------------------------------------------
if __name__ == "__main__":
    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load config
    cfg = load_config(CFG_FILE)
    
    # Create results directory
    RESULTS_DIR.mkdir(exist_ok=True)
    
    # Build a temporary env to get dimensions
    _tmp_env = make_env(cfg)
    obs_dim = _tmp_env.observation_space.shape[0]
    hidden_dim = cfg['model'].get('hidden_dim', 128)

    # This dictionary will hold our loaded agents
    agents_to_run = {}

    # --- 1. Load Your Trained CVaR Agent ---
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
        print(f"Successfully loaded Custom (CVaR) Agent from {CUSTOM_MODEL_PATH}")
    except FileNotFoundError:
        print(f"*** WARNING: Custom model not found at {CUSTOM_MODEL_PATH}. Skipping. ***")
    except Exception as e:
        print(f"*** WARNING: Failed to load Custom Agent. Error: {e} ***")

    # --- 2. Load Baseline PPO Agent (Optional) ---
    ppo_agent = None
    if PPO and os.path.exists(PPO_MODEL_PATH):
        try:
            ppo_agent = PPO.load(PPO_MODEL_PATH, device=device)
            agents_to_run["PPO Agent"] = (ppo_agent, 'ppo')
            print(f"Successfully loaded PPO Agent from {PPO_MODEL_PATH}")
        except Exception as e:
            print(f"*** WARNING: Failed to load PPO Agent. Error: {e} ***")
    else:
        print(f"PPO model not found at {PPO_MODEL_PATH} or stable_baselines3 not installed. Skipping.")
            
    # --- 3. Create the Delta Hedger ---
    delta_hedger_agent = DeltaHedgerAgent(_tmp_env.option_spec, _tmp_env.sim_params)
    agents_to_run["Delta Hedger"] = (delta_hedger_agent, 'delta')

    # --- 4. Run All Evaluations ---
    results_dict = {}
    
    for agent_name, (agent, agent_type) in agents_to_run.items():
        print(f"\nEvaluating {agent_name}...")
        env = make_env(cfg)
        results = []
        for i in tqdm(range(N_EPISODES)):
            info = run_episode(env, agent, agent_type, seed=i, device=device)
            results.append(info)
        results_dict[agent_name] = pd.DataFrame(results)
        
    # --- 5. Calculate and Display Final Summary ---
    final_stats = []
    for name, df in results_dict.items():
        final_stats.append(calculate_summary_stats(df, name))
    
    if final_stats:
        final_comparison = pd.DataFrame(final_stats).set_index('Agent')
        print("\n--- Final Hedging Performance Comparison ---")
        print(final_comparison.to_markdown(floatfmt=".4f"))
        final_comparison.to_csv(RESULTS_DIR / 'final_summary_stats.csv')
    else:
        print("\nNo agents were evaluated. No stats to show.")

    # --- 6. Generate and Save New Plots ---
    if results_dict:
        plot_error_histograms(results_dict, RESULTS_DIR / 'hedging_error_distribution.png')
        
        plot_single_episode_behavior(
            agents=agents_to_run,
            cfg=cfg,
            seed=PLOT_SEED,
            save_path=RESULTS_DIR / f'episode_behavior_seed{PLOT_SEED}.png',
            device=device
        )
    else:
        print("No results to plot.")

    print(f"\nEvaluation script finished. All results saved in '{RESULTS_DIR}' directory.")
    # plt.show() # Uncomment if running interactively