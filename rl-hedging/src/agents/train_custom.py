# src/agents/train_custom.py
import torch
import torch.optim as optim
import numpy as np
import yaml
from tqdm import tqdm
from pathlib import Path

from src.envs.hedging_env import HedgingEnv
from src.agents.models import InterpretableHedger
from src.utils.payoffs import PAYOFF_FUNCTIONS

def load_config(path='src/configs/default.yaml'):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def calculate_cvar_loss(hedging_errors: torch.Tensor, alpha: float = 0.95) -> torch.Tensor:
    """Calculates the Conditional Value at Risk (CVaR) of the hedging losses."""
    # We want to minimize large positive errors (losses)
    losses = hedging_errors
    var_index = int(alpha * len(losses))
    
    if var_index < len(losses):
        sorted_losses, _ = torch.sort(losses)
        var = sorted_losses[var_index]
        cvar = sorted_losses[var_index:].mean()
        return cvar
    else: # Handle case with few samples
        return losses.mean()

def train(config_path: str = 'src/configs/default.yaml', save_path: str = 'models/custom_hedger.pth'):
    """Trains the custom InterpretableHedger with a CVaR loss function."""
    cfg = load_config(config_path)
    
    # --- Setup ---
    env = HedgingEnv(cfg['simulator'], cfg['option'], cfg['environment']['trading_cost'])
    obs_dim = env.observation_space.shape[0]
    
    model = InterpretableHedger(obs_dim=obs_dim)
    optimizer = optim.Adam(model.parameters(), lr=float(cfg['training']['lr']))
    
    n_episodes = 2000
    batch_size = 256 # Reduced batch size for memory efficiency
    
    print("--- Starting Custom Training for Interpretable Hedger ---")
    
    for i in range(n_episodes // batch_size):
        batch_hedging_errors = []
        
        # --- Collect a batch of episodes ---
        for _ in tqdm(range(batch_size), desc=f"Batch {i+1}/{n_episodes//batch_size}"):
            obs, _ = env.reset()
            obs_history = [torch.from_numpy(obs).float()]
            action_history = []
            
            done = False
            while not done:
                obs_sequence = torch.stack(obs_history).unsqueeze(0)
                
                # Get action from the model (DO NOT DETACH)
                action_tensor, _ = model(obs_sequence)
                action_history.append(action_tensor)
                
                # Step the environment with a detached action
                obs, _, terminated, truncated, _ = env.step(action_tensor.detach().numpy().flatten())
                done = terminated or truncated
                obs_history.append(torch.from_numpy(obs).float())

            # --- Differentiable P&L Calculation ---
            actions = torch.cat(action_history).squeeze() # Shape: (T,)
            prices = torch.tensor(env.S_path, dtype=torch.float32)
            trading_cost = cfg['environment']['trading_cost']

            # Reconstruct the hedging P&L differentiably
            positions = actions
            prev_positions = torch.cat([torch.tensor([0.0]), positions[:-1]])
            trades = positions - prev_positions

            # P&L from trading the underlying asset
            trade_pnl = -torch.sum(trades * prices[:-1])
            
            # Final liquidation of position
            final_pnl = trade_pnl + positions[-1] * prices[-1]
            
            # Transaction costs
            total_costs = torch.sum(torch.abs(trades) * prices[:-1] * trading_cost)
            
            net_pnl = final_pnl - total_costs
            
            # Calculate option payoff
            payoff_fn = PAYOFF_FUNCTIONS[cfg['option']['type']]
            payoff = payoff_fn(env.S_path, cfg['option']['K'])
            
            # Differentiable hedging error
            hedging_error = net_pnl - payoff
            batch_hedging_errors.append(hedging_error)

        # --- Calculate CVaR Loss and Update Model ---
        errors_tensor = torch.stack(batch_hedging_errors)
        
        loss = calculate_cvar_loss(errors_tensor)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print(f"Batch {i+1}: CVaR (95%) Loss = {loss.item():.4f}")
    
    # Save the trained model
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"--- Training Complete ---\nModel saved to {save_path}")

if __name__ == "__main__":
    train()