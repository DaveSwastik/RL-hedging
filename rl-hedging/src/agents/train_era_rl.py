# train_era_rl.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import yaml
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from collections import deque
import matplotlib.pyplot as plt

# Import your custom environment and architecture
# Ensure src/envs/hedging_env.py and src/agents/models.py exist
from src.envs.hedging_env import HedgingEnv 
from src.agents.models import ERARL_Agent_V2
from src.baselines.delta_hedge import DeltaHedger
from src.utils.bs import bs_greeks

# -------------------------------------------------------------------
# 1. CONFIGURATION & UTILS
# -------------------------------------------------------------------
def load_config(path='src/configs/default.yaml'):
    with open(path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def generalized_quantile_huber_loss(quantiles, target, kappa=1.0):
    """
    Robust distributional loss for the Body Critic.
    quantiles: [Batch, N_Quantiles]
    target:    [Batch, 1] (Cumulative Returns)
    """
    batch_size, n_quantiles = quantiles.shape
    
    # Create tau vector (quantile levels: 0.05, ..., 0.95)
    # Ensure device consistency
    taus = torch.linspace(0.0, 1.0, n_quantiles + 2, device=quantiles.device)[1:-1]
    taus = taus.unsqueeze(0).expand(batch_size, n_quantiles)
    
    # Expand target to match quantiles shape
    target_expanded = target.expand(batch_size, n_quantiles)
    
    # Huber Loss logic
    pairwise_diff = target_expanded - quantiles
    abs_diff = torch.abs(pairwise_diff)
    huber_mask = (abs_diff <= kappa).float()
    
    squared_loss = 0.5 * pairwise_diff.pow(2)
    linear_loss = kappa * (abs_diff - 0.5 * kappa)
    element_loss = huber_mask * squared_loss + (1 - huber_mask) * linear_loss
    
    # Quantile weighting
    # Loss = |tau - I(target < q)| * element_loss
    loss = torch.abs(taus - (pairwise_diff.detach() < 0).float()) * element_loss
    
    return loss.mean()

def gpd_negative_log_likelihood(tail_returns, u, sigma, xi):
    """
    NLL for the Tail Critic (GPD).
    Assumes inputs are samples where return < u (left tail).
    We model the exceedance: y = u - return (which is positive)
    """
    # Clamp parameters for stability
    xi = torch.clamp(xi, -0.49, 0.49)
    sigma = torch.clamp(sigma, min=1e-6)
    
    # Exceedance (assuming we look at losses or negative returns)
    y = (u - tail_returns) 
    
    # Normalize
    z = y / sigma
    
    # Log-likelihood: -log(sigma) - (1 + 1/xi) * log(1 + xi*z)
    term1 = torch.log(sigma)
    term2 = (1 + 1/xi) * torch.log1p(xi * z + 1e-8)
    
    nll = term1 + term2
    return nll.mean()

def inject_adversarial_shock(env, severity=0.20):
    """Injects synthetic market crash."""
    env.S_t = env.S_t * (1.0 - severity)
    env.V_t = env.V_t + 0.20 # Volatility spike


def save_final_dashboard(history, save_path="training_mastery_report.png"):
    """Plots Price, Action vs Delta, and Internal Risk Metrics."""
    fig, axes = plt.subplots(4, 1, figsize=(12, 16), sharex=True)

    # 1. Price Path (Regime Context)
    axes[0].plot(history['prices'], color='black', label='Asset Price')
    axes[0].set_title('Market Price Dynamics')
    axes[0].set_ylabel('Price')
    axes[0].legend(loc='upper left')
    axes[0].grid(True, alpha=0.3)

    # 2. Hedging: Agent vs BS Delta
    axes[1].plot(history['deltas'], color='grey', linestyle='--', label='BS Delta (Benchmark)', alpha=0.7)
    axes[1].plot(history['actions'], color='blue', linewidth=2, label='ERA-RL Action')
    axes[1].set_title('Hedging Behavior: Agent vs Benchmark')
    axes[1].set_ylabel('Hedge Ratio')
    axes[1].legend(loc='upper left')
    axes[1].grid(True, alpha=0.3)

    # 3. The "Brain": Regime Mode & Uncertainty
    ax3 = axes[2]
    ax3.step(range(len(history['modes'])), history['modes'], color='orange', where='post', label='Mixture Mode')
    ax3.set_yticks([0, 1, 2])
    ax3.set_yticklabels(['Normal', 'Defensive', 'Panic'])
    ax3.set_ylabel('Agent Regime')
    ax3.set_title('Agent Internal Regime Detection')

    # Overlay Uncertainty on right axis
    ax3b = ax3.twinx()
    ax3b.fill_between(
        range(len(history['uncertainty'])),
        0,
        history['uncertainty'],
        color='red',
        alpha=0.1,
        label='Critic Uncertainty',
    )
    ax3b.set_ylabel('Uncertainty')

    # 4. Tail Risk Perception (Xi)
    axes[3].plot(history['xis'], color='purple', label='Tail Index (Xi)')
    axes[3].axhline(0, color='black', linewidth=0.5)
    axes[3].set_title('Perceived Tail Risk (Xi > 0 means Fat Tail)')
    axes[3].set_ylabel('Xi Parameter')
    axes[3].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"--- Dashboard Saved to {save_path} ---")

# -------------------------------------------------------------------
# 2. MAIN TRAINING LOOP
# -------------------------------------------------------------------

def train_era_rl(config_path='src/configs/default.yaml', save_path='models/era_rl_agent.pth'):
    cfg = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Starting ERA-RL Training on {device} ---")

    # 1. Init Env & Agent
    env = HedgingEnv(cfg['simulator'], cfg['option'], cfg['environment']['trading_cost'])
    obs_dim = env.observation_space.shape[0]
    
    agent = ERARL_Agent_V2(obs_dim=obs_dim).to(device)
    
    # 2. Optimizers (Time-Scale Separation)
    # Fast Clock: Encoder, Actor, Body Critic
    optimizer_main = optim.Adam([
        {'params': agent.encoder.parameters(), 'lr': 3e-4},
        {'params': agent.actor.parameters(), 'lr': 1e-4},
        {'params': agent.critic.body_net.parameters(), 'lr': 3e-4},
        {'params': agent.critic.uncertainty_net.parameters(), 'lr': 3e-4}
    ])
    
    # Slow Clock: Tail Critic
    optimizer_tail = optim.Adam(agent.critic.tail_net.parameters(), lr=5e-5)

    # 3. Training Schedule
    n_episodes = 5000
    phase_1_end = int(0.2 * n_episodes)
    phase_2_end = int(0.5 * n_episodes)
    
    # Tail Buffers & Gates
    k_min = 32
    tail_update_freq = 10
    # Store: (latent_state, gru_history, return_val)
    tail_buffer = deque(maxlen=2000) 
    
    delta_hedger = DeltaHedger(cfg['option'])

    # --- NEW: Initialize Data Log ---
    training_data_log = []

    pbar = tqdm(range(n_episodes))
    for episode in pbar:
        
        # --- A. Maturity Logic ---
        if episode < phase_1_end:
            phase = "ANCHOR"
            shock_prob = 0.0
            lambda_cvar = 0.0
        elif episode < phase_2_end:
            phase = "DISCOVERY"
            shock_prob = 0.05
            progress = (episode - phase_1_end) / (phase_2_end - phase_1_end)
            lambda_cvar = 0.1 * progress
        else:
            phase = "MASTERY"
            progress = (episode - phase_2_end) / (n_episodes - phase_2_end)
            shock_prob = 0.05 + (0.15 * progress)
            lambda_cvar = 0.5

        # --- B. Rollout (Loop 1) ---
        obs, _ = env.reset()
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device)
        obs_history = [obs_tensor]
        prev_hedge = torch.tensor([[0.0]], device=device)
        
        # Storage for Episode Update
        episode_log_probs = []
        episode_entropies = []
        episode_quantiles = []
        episode_rewards = []
        episode_values = []
        episode_latents = []     # For tail buffer
        episode_gru_hists = []   # For tail buffer
        episode_us = []          # To check exceedances
        
        done = False
        t = 0
        last_mode = 0
        
        while not done:
            # Track whether a synthetic shock was triggered this step (for logging)
            shock_active = 0
            if t > 5 and (np.random.rand() < shock_prob):
                shock_active = 1
                inject_adversarial_shock(env)

            # Snapshot current market state (state at which action is chosen)
            price_t = float(env.S_t)
            vol_t = float(env.V_t)
            
            # Manual Forward Pass to capture intermediate tensors
            obs_seq = torch.stack(obs_history).unsqueeze(0) # [1, T, D]
            
            # 1. Encoder
            latent, gru_history = agent.encoder(obs_seq, prev_hedge)
            
            # 2. Critic
            risk_out = agent.critic(latent, gru_history)
            
            # 3. Actor
            dist = agent.actor(latent, risk_out)
            action = dist.sample()

            # Track current dominant regime for logging
            last_mode = torch.argmax(dist.cat_dist.logits, dim=-1).item()

            # Execute
            action_val = float(action.item())
            next_obs, reward, terminated, truncated, _ = env.step(np.array([action_val], dtype=np.float32))
            done = terminated or truncated
            
            # Reward Shaping
            uncertainty = risk_out['uncertainty'].item()
            smoothness_penalty = abs(action.item() - prev_hedge.item()) * 0.01
            # Relax penalty if unsure/risky
            reward -= smoothness_penalty * (1.0 - uncertainty)

            # --- NEW: Log Step Data ---
            training_data_log.append({
                "Episode": int(episode),
                "Step": int(t),
                "Phase": phase,
                "Price": price_t,
                "Volatility": vol_t,
                "Action": action_val,
                "Reward": float(reward),
                "Regime_Mode": int(last_mode),  # 0=Normal, 1=Defensive, 2=Panic
                "Xi_Tail_Index": float(risk_out['xi'].item()),
                "Uncertainty": float(risk_out['uncertainty'].item()),
                "Shock_Active": int(shock_active),
            })
            # --------------------------

            # Store TENSORS for Gradient Graph
            episode_log_probs.append(dist.log_prob(action))
            episode_entropies.append(dist.entropy())
            episode_quantiles.append(risk_out['quantiles']) # Keep graph!
            episode_latents.append(latent.detach())         # Detach for buffer
            episode_gru_hists.append(gru_history.detach())  # Detach for buffer
            episode_us.append(risk_out['u'].detach().item())
            
            # Store Scalar for Reward Calculation
            episode_rewards.append(reward)
            
            # Use median quantile as value estimate
            median_idx = agent.critic.num_quantiles // 2
            episode_values.append(risk_out['quantiles'][0, median_idx])

            # Update State
            obs_history.append(torch.tensor(next_obs, dtype=torch.float32, device=device))
            prev_hedge = action.detach()
            t += 1

        # --- C. Compute Returns (Target for Critic) ---
        # Calculate Discounted Returns (Monte Carlo or GAE)
        # Using simple discounted return for distributional target
        gamma = 0.99
        returns = []
        R = 0
        for r in reversed(episode_rewards):
            R = r + gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns, dtype=torch.float32, device=device).unsqueeze(1) # [T, 1]

        # --- D. Populate Tail Buffer ---
        # Identify steps where Return < u (Tail Event)
        # FIX: Pad GRU history tensors to consistent length for batching
        target_seq_len = int(cfg['simulator']['steps']) + 1
        
        for i in range(len(returns)):
            ret_val = returns[i].item()
            u_val = episode_us[i]
            if ret_val < u_val:
                h = episode_gru_hists[i]
                # h shape is [1, seq_len, hidden_dim]
                # We need to pad seq_len to target_seq_len
                pad_size = target_seq_len - h.shape[1]
                
                if pad_size > 0:
                    # Pad dimension 1 (sequence length)
                    # pad tuple is (last_dim_left, last_dim_right, 2nd_last_left, 2nd_last_right)
                    h = torch.nn.functional.pad(h, (0, 0, 0, pad_size))
                elif pad_size < 0:
                    # Truncate if somehow larger (defensive)
                    h = h[:, :target_seq_len, :]

                # Store (Latent, Padded_GRU_Hist, Return)
                tail_buffer.append((episode_latents[i], h, returns[i]))

        # --- E. Main Update (Loop 2) ---
        optimizer_main.zero_grad()
        
        # 1. Body Loss (Train quantiles against Returns)
        # Stack quantiles: [T, 1, N] -> [T, N]
        quantiles_stack = torch.cat(episode_quantiles, dim=0)
        body_loss = generalized_quantile_huber_loss(quantiles_stack, returns)
        
        # 2. Actor Loss
        # Compute Advantages
        values_stack = torch.stack(episode_values).squeeze()
        advantages = returns.squeeze() - values_stack.detach() # Baseline subtraction
        
        # --- FIX: Advantage Normalization (Prevents NaN during Shocks) ---
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Compute CVaR Penalty (Detached)
        # We assume actor "saw" the risk in forward pass.
        
        # Implementation of L = -log_prob * (A - lambda * CVaR)
        # Penalty = lambda * abs(min(0, u))
        
        cvar_proxy = torch.tensor(episode_us, device=device)
        # If u is negative (loss), it's a risk.
        risk_penalty = lambda_cvar * torch.abs(torch.minimum(cvar_proxy, torch.tensor(0.0)))
        
        adjusted_advantage = advantages - risk_penalty
        
        log_probs_stack = torch.stack(episode_log_probs).squeeze()
        entropies_stack = torch.stack(episode_entropies).squeeze()
        
        actor_loss = -(log_probs_stack * adjusted_advantage.detach()).mean()
        entropy_loss = -0.01 * entropies_stack.mean()
        
        total_main_loss = body_loss + actor_loss + entropy_loss
        
        # --- FIX: NaN Guard (Safety Check) ---
        if torch.isnan(total_main_loss):
            print(f"Warning: NaN loss detected at Episode {episode}. Skipping update.")
        else:
            total_main_loss.backward()
            # Clip & Step (Slightly reduced clip value for safety)
            nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
            optimizer_main.step()

        # --- F. Tail Update (Loop 3) ---
        if phase != "ANCHOR" and episode % tail_update_freq == 0:
            if len(tail_buffer) >= k_min:
                optimizer_tail.zero_grad()
                
                # 1. Sample Batch from Buffer
                indices = np.random.choice(len(tail_buffer), k_min, replace=False)
                batch_latents = torch.cat([tail_buffer[i][0] for i in indices])
                
                # batch_grus now works because we padded them in Section D
                batch_grus = torch.cat([tail_buffer[i][1] for i in indices])
                
                # Use torch.stack for returns to get [Batch, 1] shape
                batch_returns = torch.stack([tail_buffer[i][2] for i in indices])
                
                # 2. Re-run Tail Network Only
                raw_tail = agent.critic.tail_net(batch_latents)
                
                # Robust Softplus manually (match model logic)
                sigma = torch.nn.functional.softplus(raw_tail[:, 0:1]) + 1e-4
                xi = torch.tanh(raw_tail[:, 1:2]) * 0.5
                
                # Get u (We need u to compute GPD loss)
                with torch.no_grad():
                     q_out = agent.critic.body_net(batch_latents)
                     # Monotonic layer ensures sorted. u is first col.
                     u_batch = q_out[:, 0:1]

                # 3. Compute GPD Loss
                tail_loss = gpd_negative_log_likelihood(batch_returns, u_batch, sigma, xi)
                
                # NaN check for tail
                if not torch.isnan(tail_loss):
                    tail_loss.backward()
                    optimizer_tail.step()
        
        # Logging
        if episode % 50 == 0:
            # Extract live metrics from the last step's risk_out
            curr_xi = risk_out['xi'].item()
            curr_unc = risk_out['uncertainty'].item()
            desc = f"{phase} | R: {np.sum(episode_rewards):.2f} | Xi: {curr_xi:.3f} | Mode: {last_mode} | Unc: {curr_unc:.3f}"
            pbar.set_description(desc)

    # 4. Save
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(agent.state_dict(), save_path)

    # --- FINAL VISUALIZATION EPISODE ---
    print("\nRunning Final Diagnostic Episode for Dashboard...")
    agent.eval()

    # Setup containers
    hist_prices, hist_deltas, hist_actions = [], [], []
    hist_modes, hist_uncertainty, hist_xis = [], [], []

    obs, _ = env.reset()
    obs_history = [torch.tensor(obs, dtype=torch.float32, device=device)]
    prev_hedge = torch.tensor([[0.0]], device=device)

    # Inject a guaranteed shock at step 10 to see reaction
    shock_trigger_step = 10

    done = False
    t_step = 0
    dt = env.sim_params.get('dt', 1 / 252)

    while not done:
        # Force a shock for the plot
        if t_step == shock_trigger_step:
            inject_adversarial_shock(env, severity=0.15)  # 15% drop

        obs_seq = torch.stack(obs_history).unsqueeze(0)

        # 1. Get Agent Diagnostics
        diag = agent.get_diagnostics(obs_seq, prev_hedge)
        action = float(diag['action_mean'])  # deterministic mean for plot

        # 2. Get BS Benchmark Greeks
        tau = max(env.T - env.t_idx, 0) * dt
        _, bs_delta, _, _ = bs_greeks(
            env.S_t,
            cfg['option']['K'],
            cfg['simulator'].get('r', 0.0),
            0.0,
            np.sqrt(max(env.V_t, 1e-12)),
            tau,
            cfg['option'].get('option_type', 'call'),
        )

        # 3. Store
        hist_prices.append(float(env.S_t))
        hist_deltas.append(float(bs_delta))
        hist_actions.append(action)
        hist_modes.append(int(diag['mode']))
        hist_uncertainty.append(float(diag['uncertainty']))
        hist_xis.append(float(diag['xi']))

        # Step
        next_obs, _, terminated, truncated, _ = env.step(np.array([action], dtype=np.float32))
        done = terminated or truncated

        obs_history.append(torch.tensor(next_obs, dtype=torch.float32, device=device))
        prev_hedge = torch.tensor([[action]], device=device)
        t_step += 1

    # Generate Plot
    dashboard_data = {
        'prices': hist_prices,
        'deltas': hist_deltas,
        'actions': hist_actions,
        'modes': hist_modes,
        'uncertainty': hist_uncertainty,
        'xis': hist_xis,
    }
    save_final_dashboard(dashboard_data)

    # --- NEW: Save Training Data to Excel ---
    print("Saving training data to Excel (this may take a moment)...")
    df_log = pd.DataFrame(training_data_log)
    excel_path = "training_data_log.xlsx"
    try:
        df_log.to_excel(excel_path, index=False)
        print(f"Training data saved to {excel_path}")
    except ImportError as e:
        # Common cause: missing optional Excel writer dependency (e.g., openpyxl)
        csv_fallback = excel_path.replace('.xlsx', '.csv')
        df_log.to_csv(csv_fallback, index=False)
        print(f"Excel writer missing ({e}). Saved CSV instead: {csv_fallback}")

    print("Training Complete.")

if __name__ == "__main__":
    train_era_rl()