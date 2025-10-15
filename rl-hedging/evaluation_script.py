# evaluation_script.py
import pandas as pd
import yaml
import torch
from tqdm import tqdm
from pathlib import Path

# Import all evaluation functions and models
from src.analysis.metrics import evaluate_rl_agent, evaluate_delta_hedger, calculate_summary_stats
from src.envs.hedging_env import HedgingEnv
from src.agents.models import InterpretableHedger # Your new model

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

if __name__ == "__main__":
    # --- Configuration ---
    CONFIG_PATH = 'src/configs/default.yaml'
    PPO_MODEL_PATH = 'models/ppo_hedge' #
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