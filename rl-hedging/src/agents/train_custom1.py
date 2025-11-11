# src/agents/train_custom1.py
import torch
import torch.optim as optim
import numpy as np
import yaml
from tqdm import tqdm
from pathlib import Path

from src.envs.hedging_env import HedgingEnv
from src.agents.models import InterpretableHedger
from src.baselines.delta_hedge import DeltaHedger
from src.utils.payoffs import PAYOFF_FUNCTIONS

def load_config(path='src/configs/default.yaml'):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def calculate_sharpe_loss(hedging_errors: torch.Tensor) -> torch.Tensor:
    """
    Calculates the loss as the negative Sharpe Ratio of the hedging errors.
    We want to MAXIMIZE Sharpe, so we MINIMIZE negative Sharpe.
    """
    epsilon = 1e-8
    mean_error = torch.mean(hedging_errors)
    std_error = torch.std(hedging_errors) + epsilon
    sharpe_ratio = mean_error / std_error
    return -sharpe_ratio

def get_target_deltas(env: HedgingEnv, cfg: dict) -> np.ndarray:
    """Helper to get the baseline delta-hedging path for pre-training."""
    dh = DeltaHedger(cfg['option'], 0.0) 
    safe_v_path = np.maximum(env.v_path, 1e-8)
    deltas = dh.get_positions(env.S_path, safe_v_path, cfg['simulator']['dt'])
    return deltas

def train(config_path: str = 'src/configs/default.yaml', save_path: str = 'models/custom_sharpe_hedger.pth'):
    """
    Trains the custom InterpretableHedger with a two-phase strategy:
    1. Pre-training: Mimic the classical Delta Hedger (inspired by Nian et al., 2021).
    2. Fine-tuning: Maximize the Sharpe Ratio of hedging errors.
    """
    cfg = load_config(config_path)
    
    # --- Setup ---
    learning_rate = float(cfg['training']['lr'])
    env = HedgingEnv(cfg['simulator'], cfg['option'], cfg['environment']['trading_cost'])
    obs_dim = env.observation_space.shape[0]

    model = InterpretableHedger(obs_dim=obs_dim)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # --- Phase 1: Pre-training (Mimicking Delta Hedger) ---
    print("--- Phase 1: Pre-training (Mimicking Delta Hedger) ---")
    n_pretrain_episodes = 5000 
    pretrain_batch_size = 64

    for i in range(n_pretrain_episodes // pretrain_batch_size):
        batch_actions = []
        batch_targets = []
        
        # --- FIX 2: Set leave=True ---
        for _ in tqdm(range(pretrain_batch_size), desc=f"Pre-train Batch {i+1}/{n_pretrain_episodes//pretrain_batch_size}", leave=True):
            obs, _ = env.reset()
            obs_history = [torch.from_numpy(obs).float()]
            target_deltas = get_target_deltas(env, cfg)
            
            episode_actions = []
            done = False
            t = 0
            while not done:
                obs_sequence = torch.stack(obs_history).unsqueeze(0)
                action_tensor, _ = model(obs_sequence)
                episode_actions.append(action_tensor)
                obs, _, terminated, truncated, _ = env.step(action_tensor.detach().numpy().flatten())
                done = terminated or truncated
                obs_history.append(torch.from_numpy(obs).float())
                t += 1
            
            batch_actions.append(torch.cat(episode_actions).squeeze())
            batch_targets.append(torch.tensor(target_deltas[:-1], dtype=torch.float32))

        actions_tensor = torch.cat(batch_actions)
        targets_tensor = torch.cat(batch_targets)
        
        pretrain_loss = torch.mean((actions_tensor - targets_tensor)**2) 

        optimizer.zero_grad()
        pretrain_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        if (i+1) % 10 == 0: 
             print(f"Pre-train Batch {i+1}: MSE Loss vs Delta = {pretrain_loss.item():.6f}")

    print(f"Pre-training complete.")

    # --- Phase 2: Fine-Tuning (Optimizing Sharpe Ratio) ---
    print("\n--- Phase 2: Fine-Tuning (Optimizing Sharpe Ratio) ---")
    n_finetune_episodes = 50000 
    
    # --- FIX 1: Set batch_size to 128 ---
    batch_size = 128 # Reduced to prevent memory overflow

    print(f"Using Negative Sharpe Ratio Loss.")
    print(f"Running for {n_finetune_episodes} episodes...")

    for i in range(n_finetune_episodes // batch_size):
        batch_hedging_errors = []

        # --- FIX 2: Set leave=True ---
        for _ in tqdm(range(batch_size), desc=f"Fine-tune Batch {i+1}/{n_finetune_episodes//batch_size}", leave=True):
            obs, _ = env.reset()
            obs_history = [torch.from_numpy(obs).float()]
            action_history = [] 

            done = False
            while not done:
                obs_sequence = torch.stack(obs_history).unsqueeze(0)
                action_tensor, _ = model(obs_sequence) 
                action_history.append(action_tensor)
                obs, _, terminated, truncated, _ = env.step(action_tensor.detach().numpy().flatten())
                done = terminated or truncated
                obs_history.append(torch.from_numpy(obs).float())

            # --- Differentiable P&L Calculation ---
            actions = torch.cat(action_history).squeeze()
            prices = torch.tensor(env.S_path, dtype=torch.float32, device=actions.device) 
            trading_cost = cfg['environment']['trading_cost']
            positions = actions
            prev_positions = torch.cat([torch.tensor([0.0], device=actions.device), positions[:-1]])
            trades = positions - prev_positions
            trade_pnl = -torch.sum(trades * prices[:-1])
            final_pnl = trade_pnl + positions[-1] * prices[-1]
            total_costs = torch.sum(torch.abs(trades) * prices[:-1] * trading_cost)
            net_pnl = final_pnl - total_costs
            payoff_fn = PAYOFF_FUNCTIONS[cfg['option']['type']]
            payoff = payoff_fn(env.S_path, cfg['option']['K'])
            payoff_tensor = torch.tensor(payoff, dtype=torch.float32, device=actions.device)
            hedging_error = net_pnl - payoff_tensor
            batch_hedging_errors.append(hedging_error)

        # --- Calculate Sharpe Loss and Update Model ---
        errors_tensor = torch.stack(batch_hedging_errors)
        loss = calculate_sharpe_loss(errors_tensor)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) 
        optimizer.step()

        if (i+1) % 10 == 0:
            print(f"Batch {i+1}: Negative Sharpe Loss = {loss.item():.4f}")
    
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"--- Training Complete ---\nModel saved to {save_path}")

if __name__ == "__main__":
    train()