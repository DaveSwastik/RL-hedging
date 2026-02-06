# src/agents/train_custom.py
import torch
import torch.optim as optim
import numpy as np
import yaml
from tqdm import tqdm
from pathlib import Path
import math

from src.envs.hedging_env import HedgingEnv
from src.agents.models import ERARL_Agent_V2
from src.baselines.delta_hedge import DeltaHedger # Needed for pre-training targets
from src.utils.payoffs import PAYOFF_FUNCTIONS

def load_config(path='src/configs/default.yaml'):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def calculate_cvar(hedging_losses: torch.Tensor, alpha: float = 0.95) -> torch.Tensor:
    """Stable CVaR (expected shortfall) estimate over the batch of hedging_losses."""
    losses = hedging_losses.view(-1)
    if losses.numel() == 0:
        return torch.tensor(0.0, device=hedging_losses.device)
    # VaR at alpha
    try:
        var = torch.quantile(losses, torch.tensor(alpha, device=losses.device))
    except Exception:
        # Fallback for very old torch versions
        sorted_losses, _ = torch.sort(losses)
        var_idx = int(math.floor(alpha * len(sorted_losses)))
        var = sorted_losses[ min(max(var_idx, 0), len(sorted_losses)-1) ]
    tail = losses[losses >= var]
    if tail.numel() == 0:
        # If no samples in tail (small batch), return mean as fallback
        return losses.mean()
    return tail.mean()

def calculate_hybrid_cvar_mse_loss(hedging_losses: torch.Tensor, alpha: float = 0.95, mse_weight: float = 0.3):
    """
    Hybrid loss combining CVaR (tail risk) and MSE (average error).
    Keep mse_weight in [0.0, 1.0].
    """
    cvar_loss = calculate_cvar(hedging_losses, alpha)
    mse_loss = torch.mean(hedging_losses**2)
    return (1.0 - mse_weight) * cvar_loss + mse_weight * mse_loss

def get_target_deltas(env: HedgingEnv, cfg: dict) -> np.ndarray:
    """Helper to get the baseline delta-hedging path for pre-training and shaped loss."""
    dh = DeltaHedger(cfg['option'], 0.0) # zero cost for target deltas
    safe_v_path = np.maximum(env.v_path, 1e-8)
    deltas = dh.get_positions(env.S_path, safe_v_path, cfg['simulator']['dt'])
    return deltas

def train(config_path: str = 'src/configs/default.yaml', save_path: str = 'models/custom_hedger.pth'):
    cfg = load_config(config_path)

    # hyperparams (tweak in config or here)
    learning_rate = float(cfg['training']['lr']) * 1.5
    env = HedgingEnv(cfg['simulator'], cfg['option'], cfg['environment']['trading_cost'])
    obs_dim = env.observation_space.shape[0]

    model = ERARL_Agent_V2(obs_dim=obs_dim)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training knobs (can move into YAML if you want)
    # Discount for per-step shaped loss
    gamma = cfg['training'].get('discount_factor', 0.99)
    mse_weight = cfg['training'].get('mse_weight', 0.3)
    alpha = cfg['training'].get('cvar_alpha', 0.95)
    entropy_coef = cfg['training'].get('entropy_coef', 1e-3)
    grad_accum_steps = cfg['training'].get('grad_accum_steps', 1)

    # --- Phase 1: Pre-training (Mimicking Delta Hedger) ---
    print("--- Phase 1: Pre-training (Mimicking Delta Hedger) ---")
    n_pretrain_episodes = cfg['training'].get('n_pretrain_episodes', 500)
    pretrain_batch_size = cfg['training'].get('pretrain_batch_size', 64)

    for i in range(max(1, n_pretrain_episodes // pretrain_batch_size)):
        batch_actions = []
        batch_targets = []
        for _ in tqdm(range(pretrain_batch_size), desc=f"Pre-train Batch {i+1}/{max(1, n_pretrain_episodes//pretrain_batch_size)}"):
            obs, _ = env.reset()
            obs_history = [torch.from_numpy(obs).float()]
            target_deltas = get_target_deltas(env, cfg) # BS deltas for path

            prev_hedge = torch.zeros(1, 1, device=next(model.parameters()).device)  # start flat position

            episode_actions = []
            done = False
            t = 0
            while not done:
                obs_sequence = torch.stack(obs_history).unsqueeze(0)
                with torch.no_grad():
                    dist, _ = model(obs_sequence, prev_hedge)
                    action_tensor = dist.mode()  # stable target-mimic action

                prev_hedge = action_tensor.detach()
                episode_actions.append(action_tensor)

                # Step env using detached action (env doesn't need grads)
                obs, _, terminated, truncated, _ = env.step(action_tensor.detach().numpy().flatten())
                done = terminated or truncated
                obs_history.append(torch.from_numpy(obs).float())
                t += 1

            batch_actions.append(torch.cat(episode_actions).squeeze())
            # align lengths: BS deltas are usually length T -> take up to T-1 to match actions
            batch_targets.append(
    torch.tensor(target_deltas[:-1], dtype=torch.float32,
                 device=actions_tensor.device)
)

        actions_tensor = torch.cat(batch_actions)
        targets_tensor = torch.cat(batch_targets)
        pretrain_loss = torch.mean((actions_tensor - targets_tensor)**2)

        optimizer.zero_grad()
        pretrain_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if (i+1) % 5 == 0:
            print(f"Pre-train Batch {i+1}: MSE Loss vs Delta = {pretrain_loss.item():.6f}")

    print("Pre-training complete.")

    # --- Phase 2: Fine-Tuning (Optimizing Hybrid CVaR + MSE Loss) ---
    print("\n--- Phase 2: Fine-Tuning (Optimizing Hybrid CVaR + MSE Loss) ---")         
    n_finetune_episodes = cfg['training'].get('n_finetune_episodes', 2500)                                           #### Training length
    batch_size = cfg['training'].get('finetune_batch_size', 256)

    print(f"Using Hybrid Loss with MSE Weight: {mse_weight}, CVaR alpha: {alpha}, discount: {gamma}")

    n_batches = max(1, n_finetune_episodes // batch_size)
    for i in range(n_batches):
        batch_hedging_errors = []
        batch_entropy_terms = []  # collect entropy terms (if available) to average per batch

        for b in tqdm(range(batch_size), desc=f"Fine-tune Batch {i+1}/{n_batches}"):
            obs, _ = env.reset()
            obs_history = [torch.from_numpy(obs).float()]
            action_history = [] # actions WITH grad
            risk_out_history = []
            entropy_history = []
            # try to get BS deltas for shaped per-step loss
            bs_deltas = get_target_deltas(env, cfg) # numpy array
            done = False
            t = 0
            prev_hedge = torch.zeros(1, 1, device=next(model.parameters()).device)  # start flat position
            while not done:
                obs_sequence = torch.stack(obs_history).unsqueeze(0)
                # ===== Correct API usage for ERARL_Agent_V2 =====
                dist, risk_out = model(obs_sequence, prev_hedge)

                # Reparameterized sample (keeps gradients)
                action_tensor = dist.rsample()

                # Policy statistics
                # log_prob = dist.log_prob(action_tensor)                        For PPO/A2C
                entropy = dist.entropy()

                # Track previous hedge for next step
                prev_hedge = action_tensor.detach()

                action_history.append(action_tensor)
                risk_out_history.append(risk_out)
                if entropy is not None:
                    entropy_history.append(entropy)

                # step env with detached action (env non-differentiable)
                obs, _, terminated, truncated, _ = env.step(action_tensor.detach().numpy().flatten())
                done = terminated or truncated
                obs_history.append(torch.from_numpy(obs).float())
                t += 1

            # Episode-mean entropy (if available)
            if len(entropy_history) > 0:
                entropy_terms = []
                for ent in entropy_history:
                    entropy_terms.append(ent.mean() if ent.dim() > 0 else ent)
                entropy = torch.stack(entropy_terms).mean()
            else:
                entropy = None

            # --- Per-step shaped loss (dense signal) ---
            positions = torch.cat(action_history, dim = 0)           # shape: [T_steps]
            device = positions.device if positions.numel() > 0 else next(model.parameters()).device

            prices_np = env.S_path  # numpy array shape [T]
            prices = torch.tensor(prices_np, dtype=torch.float32, device=device)
            trading_cost = float(cfg['environment']['trading_cost'])

            # prev positions (initially zero)
            prev_positions = torch.cat([torch.tensor([0.0], device=device), positions[:-1]])
            trades = positions - prev_positions

            # If bs_deltas available, use them for delta-mismatch term
            # Align bs_deltas length: bs_deltas typically length T -> use up to T-1 to match positions
            try:
                bs_tensor = torch.tensor(bs_deltas[:-1], dtype=torch.float32, device=device)
                use_bs = True
            except Exception:
                bs_tensor = None
                use_bs = False

            step_losses = []
            T_steps = positions.shape[0]
            for tt in range(T_steps):
                pos_t = positions[tt]
                trade_t = trades[tt]
                price_t = prices[tt] if tt < prices.shape[0] else prices[-1]

                # trading cost
                trade_cost = torch.abs(trade_t) * price_t * trading_cost

                if use_bs and tt < bs_tensor.shape[0]:
                    # delta mismatch squared
                    delta_mismatch = pos_t - bs_tensor[tt]
                    step_loss = delta_mismatch**2 + trade_cost
                else:
                    # fallback: penalize immediate negative incremental pnl + trade_cost
                    incremental_pnl = - trade_t * price_t
                    # penalize losses but don't reward positive pnl here (keep signal focused)
                    incremental_penalty = torch.relu(-incremental_pnl)
                    step_loss = incremental_penalty + trade_cost

                # Add your GPD-based risk penalty
                if tt < len(risk_out_history) and hasattr(model, 'compute_risk_penalty'):
                    risk_penalty = model.compute_risk_penalty(risk_out_history[tt])
                    step_loss = step_loss + 0.1 * risk_penalty.squeeze()

                step_losses.append(step_loss)

            # Discounted sum
            discounted = [ (gamma**tt) * step_losses[tt] for tt in range(len(step_losses)) ]
            episode_loss = torch.stack(discounted).sum()
            batch_hedging_errors.append(episode_loss)

            # We already computed entropy above
            entropy_val = entropy.detach() if entropy is not None else None

            if entropy_val is not None:
                if entropy_val.dim() > 0:
                    batch_entropy_terms.append(entropy_val.mean())
                else:
                    batch_entropy_terms.append(entropy_val)

        # --- Compute hybrid loss (CVaR + MSE) across the batch ---
        errors_tensor = torch.stack(batch_hedging_errors)
        base_loss = calculate_hybrid_cvar_mse_loss(errors_tensor, alpha=alpha, mse_weight=mse_weight)

        # entropy term (if collected)
        if len(batch_entropy_terms) > 0:
            entropy_mean = torch.stack(batch_entropy_terms).mean()
            entropy_term = - entropy_coef * entropy_mean
            total_loss = base_loss + entropy_term
        else:
            total_loss = base_loss

        # --- Backprop with optional gradient accumulation ---
        # If grad_accum_steps > 1, you can divide the loss accordingly and step every grad_accum_steps
        total_loss_to_backprop = total_loss / float(grad_accum_steps)
        optimizer.zero_grad()
        total_loss_to_backprop.backward()
        # If using accumulation, looped stepping would be needed; here we assume one call per effective batch
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        print(f"Batch {i+1}/{n_batches}: Hybrid Loss = {total_loss.item():.6f} (base {base_loss.item():.6f})")

    # Save final model
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"--- Training Complete ---\nModel saved to {save_path}")

if __name__ == "__main__":
    train()
