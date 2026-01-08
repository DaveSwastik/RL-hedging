# src/agents/train_custom.py
import torch
import torch.optim as optim
import numpy as np
import yaml
from tqdm import tqdm
from pathlib import Path

from src.envs.hedging_env import HedgingEnv
from src.agents.models import InterpretableHedger
from src.baselines.delta_hedge import DeltaHedger # Needed for pre-training targets
from src.utils.payoffs import PAYOFF_FUNCTIONS

def load_config(path='src/configs/default.yaml'):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def calculate_cvar(hedging_errors: torch.Tensor, alpha: float = 0.95) -> torch.Tensor:
    """Calculates the Conditional Value at Risk (CVaR) of the hedging losses."""
    losses = hedging_errors # Focus on positive errors (losses)
    var_index = int(alpha * len(losses))
    cvar_loss = torch.tensor(0.0, device=hedging_errors.device) # Default if no tail
    if var_index < len(losses):
        sorted_losses, _ = torch.sort(losses)
        # CVaR is the mean of losses greater than or equal to VaR
        cvar_loss = sorted_losses[var_index:].mean()
    # If not enough samples for tail or all errors are negative, return mean
    if torch.isnan(cvar_loss) or len(losses) <= (1/(1-alpha)):
         return losses.mean()
    return cvar_loss


def calculate_hybrid_cvar_mse_loss(hedging_errors: torch.Tensor, alpha: float = 0.95, mse_weight: float = 0.3):
    """
    Calculates a hybrid loss combining CVaR (tail risk) and MSE (average error).
    Adjust mse_weight to balance risk aversion vs. average accuracy.
    """
    # 1. CVaR Loss (Tail Risk)
    cvar_loss = calculate_cvar(hedging_errors, alpha)

    # 2. MSE Loss (Average Error - encourage centering around zero)
    mse_loss = torch.mean(hedging_errors**2)

    # 3. Hybrid Loss (Adjust weighting here)
    return (1.0 - mse_weight) * cvar_loss + mse_weight * mse_loss

def get_target_deltas(env: HedgingEnv, cfg: dict) -> np.ndarray:
    """Helper to get the baseline delta-hedging path for pre-training."""
    dh = DeltaHedger(cfg['option'], 0.0) # Use zero cost for target deltas
    # Ensure volatility used in BS delta is non-zero
    safe_v_path = np.maximum(env.v_path, 1e-8)
    deltas = dh.get_positions(env.S_path, safe_v_path, cfg['simulator']['dt'])
    return deltas

def train(config_path: str = 'src/configs/default.yaml', save_path: str = 'models/custom_hedger.pth', theta: float = None, rho: float = None, kappa: float = None):
    """
    Trains the custom InterpretableHedger with a two-phase strategy:
    1. Pre-training: Mimic the classical Delta Hedger (inspired by Nian et al., 2021).
    2. Fine-tuning: Optimize a hybrid CVaR + MSE loss.
    """
    cfg = load_config(config_path)

    # --- Parameter Overrides for Sweep ---
    if rho is not None:
        cfg['simulator']['rho'] = float(rho)
        print(f"Overriding rho to {rho}")
    
    if theta is not None or kappa is not None:
        for regime in cfg['simulator']['regimes']:
            if theta is not None:
                regime['theta'] = float(theta)
            if kappa is not None:
                regime['kappa'] = float(kappa)
        print(f"Overriding regimes: theta={theta}, kappa={kappa}")

    # --- Setup ---
    # Use a slightly higher LR for potentially faster convergence with hybrid loss
    learning_rate = float(cfg['training']['lr']) * 1.5
    env = HedgingEnv(cfg['simulator'], cfg['option'], cfg['environment']['trading_cost'])
    obs_dim = env.observation_space.shape[0]

    model = InterpretableHedger(obs_dim=obs_dim)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # --- Phase 1: Pre-training (Mimicking Delta Hedger) ---
    print("--- Phase 1: Pre-training (Mimicking Delta Hedger) ---")
    n_pretrain_episodes = 500 # Number of episodes for pre-training
    pretrain_batch_size = 64

    for i in range(n_pretrain_episodes // pretrain_batch_size):
        batch_actions = []
        batch_targets = []
        
        for _ in tqdm(range(pretrain_batch_size), desc=f"Pre-train Batch {i+1}/{n_pretrain_episodes//pretrain_batch_size}"):
            obs, _ = env.reset()
            obs_history = [torch.from_numpy(obs).float()]
            target_deltas = get_target_deltas(env, cfg) # Get BS deltas for this path
            
            episode_actions = []
            done = False
            t = 0
            while not done:
                obs_sequence = torch.stack(obs_history).unsqueeze(0)
                action_tensor, _ = model(obs_sequence) # Get model's action
                episode_actions.append(action_tensor)

                # Step env using a detached action (env doesn't need grads)
                obs, _, terminated, truncated, _ = env.step(action_tensor.detach().numpy().flatten())
                done = terminated or truncated
                obs_history.append(torch.from_numpy(obs).float())
                t += 1
            
            # Store actions and targets for the batch
            batch_actions.append(torch.cat(episode_actions).squeeze())
            # Match length - BS deltas are for t=0 to T-1
            batch_targets.append(torch.tensor(target_deltas[:-1], dtype=torch.float32))

        # Calculate pre-training loss for the batch
        actions_tensor = torch.cat(batch_actions)
        targets_tensor = torch.cat(batch_targets)
        
        pretrain_loss = torch.mean((actions_tensor - targets_tensor)**2) # MSE Loss

        # Update model
        optimizer.zero_grad()
        pretrain_loss.backward()
        # Optional: Gradient Clipping can help stabilize RNN training
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        if (i+1) % 5 == 0: # Print loss occasionally
             print(f"Pre-train Batch {i+1}: MSE Loss vs Delta = {pretrain_loss.item():.6f}")


    print(f"Pre-training complete.")

    # --- Phase 2: Fine-Tuning (Optimizing Hybrid CVaR + MSE Loss) ---
    print("\n--- Phase 2: Fine-Tuning (Optimizing Hybrid CVaR + MSE Loss) ---")
    n_finetune_episodes = 2500 # Increased episodes for fine-tuning
    batch_size = 256 # Keep batch size reasonable
    mse_weight = 0.3 # <--- ADJUST THIS WEIGHT (0.0 to 1.0) to balance goals

    print(f"Using Hybrid Loss with MSE Weight: {mse_weight}")

    for i in range(n_finetune_episodes // batch_size):
        batch_hedging_errors = []

        for _ in tqdm(range(batch_size), desc=f"Fine-tune Batch {i+1}/{n_finetune_episodes//batch_size}"):
            obs, _ = env.reset()
            obs_history = [torch.from_numpy(obs).float()]
            action_history = [] # Store actions with gradients attached

            done = False
            while not done:
                obs_sequence = torch.stack(obs_history).unsqueeze(0)
                action_tensor, _ = model(obs_sequence) # Action retains gradient info
                action_history.append(action_tensor)

                # Step env using detached action
                obs, _, terminated, truncated, _ = env.step(action_tensor.detach().numpy().flatten())
                done = terminated or truncated
                obs_history.append(torch.from_numpy(obs).float())

            # --- Differentiable P&L Calculation (Same as before) ---
            actions = torch.cat(action_history).squeeze()
            prices = torch.tensor(env.S_path, dtype=torch.float32, device=actions.device) # Ensure device match
            trading_cost = cfg['environment']['trading_cost']

            positions = actions
            # Ensure prev_positions is on the same device
            prev_positions = torch.cat([torch.tensor([0.0], device=actions.device), positions[:-1]])
            trades = positions - prev_positions

            # Ensure prices match device for calculations
            trade_pnl = -torch.sum(trades * prices[:-1])
            final_pnl = trade_pnl + positions[-1] * prices[-1]
            total_costs = torch.sum(torch.abs(trades) * prices[:-1] * trading_cost)
            net_pnl = final_pnl - total_costs

            payoff_fn = PAYOFF_FUNCTIONS[cfg['option']['type']]
            payoff = payoff_fn(env.S_path, cfg['option']['K'])
            # Ensure payoff is a tensor on the correct device
            payoff_tensor = torch.tensor(payoff, dtype=torch.float32, device=actions.device)

            hedging_error = net_pnl - payoff_tensor
            batch_hedging_errors.append(hedging_error)

        # --- Calculate Hybrid Loss and Update Model ---
        errors_tensor = torch.stack(batch_hedging_errors)
        loss = calculate_hybrid_cvar_mse_loss(errors_tensor, alpha=0.95, mse_weight=mse_weight)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # Clip gradients
        optimizer.step()

        print(f"Batch {i+1}: Hybrid Loss = {loss.item():.4f}")

    # Save the final trained model
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"--- Training Complete ---\nModel saved to {save_path}")

if __name__ == "__main__":
    train()