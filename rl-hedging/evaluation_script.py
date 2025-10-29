# evaluation_script.py
import pandas as pd
import yaml
import torch
from tqdm import tqdm
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Import all evaluation functions and models
from src.analysis.metrics import evaluate_rl_agent, evaluate_delta_hedger, calculate_summary_stats
from src.envs.hedging_env import HedgingEnv
from src.agents.models import InterpretableHedger # Your new model
from src.baselines.delta_hedge import DeltaHedger
from stable_baselines3 import PPO

def load_config(path='src/configs/default.yaml'):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def evaluate_custom_agent(model_path, config_path, n_episodes=500):
    """Evaluates the custom-trained InterpretableHedger."""
    cfg = yaml.safe_load(open(config_path, 'r'))
    env = HedgingEnv(cfg['simulator'], cfg['option'], cfg['environment']['trading_cost'])
    
    obs_dim = env.observation_space.shape[0]
    model = InterpretableHedger(obs_dim=obs_dim)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    results = []
    for i in tqdm(range(n_episodes), desc="Evaluating Custom Agent"):
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

def plot_error_histograms(results_dict, save_path):
    """
    Plots a single histogram comparing the hedging error distributions
    for all agents.
    """
    plt.figure(figsize=(12, 7))
    sns.set_style("whitegrid")
    
    for name, df in results_dict.items():
        sns.histplot(df['hedging_error'], kde=True, label=name, alpha=0.6, bins=50)
        
    plt.title('Distribution of Hedging Errors (500 Episodes)', fontsize=16)
    plt.xlabel('Hedging Error (Final P&L - Payoff)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.legend()
    plt.axvline(0, color='k', linestyle='--', alpha=0.7)
    plt.savefig(save_path)
    print(f"Error distribution plot saved to {save_path}")

def plot_single_episode_behavior(config_path, ppo_path, custom_path, seed, save_path):
    """
    Runs a single episode for all three agents on the same market path
    and plots their hedging positions over time.
    """
    print(f"\nGenerating single-episode behavior plot for seed {seed}...")
    cfg = load_config(config_path)
    
    # 1. Get the single market path from the environment
    env = HedgingEnv(cfg['simulator'], cfg['option'], cfg['environment']['trading_cost'])
    obs, _ = env.reset(seed=seed)
    
    S_path = env.S_path
    v_path = env.v_path
    t_steps = np.arange(len(S_path))

    # 2. Get Delta Hedger positions
    dh = DeltaHedger(cfg['option'], cfg['environment']['trading_cost'])
    dh_positions = dh.get_positions(S_path, v_path, cfg['simulator']['dt'])

    # 3. Get PPO Agent positions
    ppo_model = PPO.load(ppo_path)
    ppo_positions = []
    obs, _ = env.reset(seed=seed)
    done = False
    while not done:
        action, _ = ppo_model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = env.step(action)
        ppo_positions.append(action[0])
        done = terminated or truncated

    # 4. Get Custom Agent positions
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

    # 5. Create the plots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    
    # Subplot 1: Stock Price
    ax1.plot(t_steps, S_path, label='Stock Price', color='black')
    ax1.set_title(f'Agent Hedging Behavior (Episode Seed {seed})', fontsize=16)
    ax1.set_ylabel('Stock Price ($)', fontsize=12)
    ax1.legend(loc='upper left')
    ax1.grid(True)

    # Subplot 2: Agent Positions
    ax2.plot(t_steps[:-1], dh_positions[:-1], label='Delta Hedger', linestyle='--', alpha=0.9)
    ax2.plot(t_steps[:-1], ppo_positions, label='PPO Agent', linestyle='-', alpha=0.8)
    ax2.plot(t_steps[:-1], custom_positions, label='Custom (CVaR) Agent', linestyle='-', alpha=0.8)
    ax2.set_xlabel('Time Step', fontsize=12)
    ax2.set_ylabel('Hedge Position (Units of Stock)', fontsize=12)
    ax2.legend(loc='upper left')
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Episode behavior plot saved to {save_path}")

if __name__ == "__main__":
    # --- Configuration ---
    CONFIG_PATH = 'src/configs/default.yaml'
    PPO_MODEL_PATH = 'models/ppo_hedge' # Note: no .zip extension
    CUSTOM_MODEL_PATH = 'models/custom_hedger.pth'
    RESULTS_DIR = Path('results')
    
    # Create results directory if it doesn't exist
    RESULTS_DIR.mkdir(exist_ok=True)

    # --- Run All Evaluations ---
    print("Evaluating PPO Agent...")
    ppo_results = evaluate_rl_agent(PPO_MODEL_PATH, CONFIG_PATH, n_episodes=500)

    print("\nEvaluating Delta Hedger...")
    dh_results = evaluate_delta_hedger(CONFIG_PATH, n_episodes=500)

    print("\nEvaluating Custom Risk-Aware Agent...")
    custom_results = evaluate_custom_agent(CUSTOM_MODEL_PATH, CONFIG_PATH, n_episodes=500)

    # --- Save Detailed Results to CSV ---
    ppo_results.to_csv(RESULTS_DIR / 'ppo_results.csv', index=False)
    dh_results.to_csv(RESULTS_DIR / 'dh_results.csv', index=False)
    custom_results.to_csv(RESULTS_DIR / 'custom_results.csv', index=False)
    print(f"\nDetailed results for each strategy saved in the '{RESULTS_DIR}' directory.")

    # --- Calculate and Display Final Summary ---
    ppo_stats = calculate_summary_stats(ppo_results, "PPO Agent")
    dh_stats = calculate_summary_stats(dh_results, "Delta Hedger")
    custom_stats = calculate_summary_stats(custom_results, "Custom Agent (CVaR)")

    final_comparison = pd.DataFrame([ppo_stats, dh_stats, custom_stats])

    print("\n--- Final Hedging Performance Comparison ---")
    print(final_comparison)

    # --- Generate and Save New Plots ---
    results_dict = {
        "Delta Hedger": dh_results,
        "PPO Agent": ppo_results,
        "Custom Agent (CVaR)": custom_results
    }
    
    # 1. Plot histogram of errors
    plot_error_histograms(results_dict, RESULTS_DIR / 'hedging_error_distribution.png')
    
    # 2. Plot step-by-step behavior for one episode (e.g., seed 42)
    plot_single_episode_behavior(
        CONFIG_PATH,
        PPO_MODEL_PATH,
        CUSTOM_MODEL_PATH,
        seed=250,
        save_path=RESULTS_DIR / 'episode_behavior_seed250.png'
    )