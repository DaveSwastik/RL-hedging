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

# --- IMPORTS (Adjust paths as necessary) ---
from src.envs.hedging_env import HedgingEnv 
from src.agents.models import ERARL_Agent_V2
from src.utils.bs import bs_greeks

# -------------------------------------------------------------------
# 1. UTILS & CONFIG
# -------------------------------------------------------------------

def load_config(path='src/configs/default.yaml'):
    with open(path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def generalized_quantile_huber_loss(quantiles, target, kappa=1.0):
    """Robust distributional loss (Huber-Pinball)."""
    batch_size, n_quantiles = quantiles.shape
    taus = torch.linspace(0.05, 0.95, n_quantiles, device=quantiles.device).view(1, n_quantiles)
    
    # Target shape needs to broadcast: [B, 1] vs [B, N]
    diff = target - quantiles
    abs_diff = diff.abs()
    huber = torch.where(
        abs_diff <= kappa,
        0.5 * diff.pow(2),
        kappa * (abs_diff - 0.5 * kappa),
    )
    indicator = (diff > 0).float() # Standard quantile indicator
    weights = torch.abs(taus - indicator)
    loss = (weights * huber).mean()
    return loss

def save_dashboard(history, save_path="training_mastery_report.png"):
    """Visualizes the Agent's performance vs BS."""
    fig, axes = plt.subplots(4, 1, figsize=(12, 16), sharex=True)

    # 1. Price
    axes[0].plot(history['prices'], color='black', label='Asset Price')
    axes[0].set_title('Market Price Dynamics')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 2. Actions
    axes[1].plot(history['deltas'], color='grey', linestyle='--', label='BS Delta', alpha=0.7)
    axes[1].plot(history['actions'], color='blue', linewidth=2, label='ERA-RL Action')
    axes[1].set_title('Hedging Behavior')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # 3. Regime
    ax3 = axes[2]
    ax3.step(range(len(history['modes'])), history['modes'], color='orange', where='post')
    ax3.set_yticks([0, 1, 2])
    ax3.set_yticklabels(['Normal', 'Defensive', 'Panic'])
    ax3.set_title('Regime Detection')
    
    # 4. Tail Risk
    axes[3].plot(history['xis'], color='purple', label='Tail Index (Xi)')
    axes[3].axhline(0, color='black', linewidth=0.5)
    axes[3].set_title('Perceived Tail Risk')
    axes[3].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)

# -------------------------------------------------------------------
# 2. MAIN TRAINING LOOP
# -------------------------------------------------------------------

def train_era_rl(config_path='src/configs/default.yaml', save_path='models/era_rl_v2(5k-episodes).pth'):
    cfg = load_config(config_path)
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")  # Force CPU for debugging
    print(f"--- Starting ERA-RL V2 Training on {device} ---")

    # 1. Environment & Agent
    real_cost = cfg['environment'].get('trading_cost', 0.0002)      # 2BP
    env = HedgingEnv(cfg['simulator'], cfg['option'], trading_cost=real_cost)
    
    obs_dim = env.observation_space.shape[0]
    agent = ERARL_Agent_V2(obs_dim=obs_dim, hidden_dim=128).to(device)

    # 2. Optimizers
    # Fast Clock (Body, Actor, Encoder)
    optimizer_main = optim.Adam([
        {'params': agent.encoder.parameters(), 'lr': 3e-4},
        {'params': agent.actor.parameters(), 'lr': 1e-4}, 
        {'params': agent.critic.body_net.parameters(), 'lr': 3e-4},
        {'params': agent.critic.uncertainty_net.parameters(), 'lr': 3e-4}
    ])
    # Slow Clock (Tail GPD)
    optimizer_tail = optim.Adam(agent.critic.tail_net.parameters(), lr=5e-5)

    # 3. Training Config
    n_episodes = 5000
    seq_len = 30 
    
    # Phases
    p1 = 0.15
    p2 = 0.60

    phase_1_end = int(p1 * n_episodes) 
    phase_2_end = int((p1 + p2) * n_episodes) 
    
    # Buffers
    tail_buffer = deque(maxlen=2000)
    training_log = [] 
    
    # --- PATCH 1: Episode Summaries Initialization ---
    episode_summaries = []   # collect per-episode totals
    # -------------------------------------------------

    pbar = tqdm(range(n_episodes))
    
    for episode in pbar:
        # --- PHASE LOGIC ---
        if episode < phase_1_end:
            phase = "FORCED_ENTRY"
            cost_multiplier = 0.0 
            entropy_coef = 0.05
            shock_prob = 0.0
        elif episode < phase_2_end:
            phase = "STABILIZATION"
            cost_multiplier = 1.0 
            entropy_coef = 0.01
            shock_prob = 0.05
        else:
            phase = "MASTERY"
            cost_multiplier = 1.0
            entropy_coef = 0.005
            shock_prob = 0.15 

        # --- ROLLOUT ---
        obs, _ = env.reset()
        episode_shock_triggered = False # Ensure only one massive shock per ep

        
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device)
        obs_history = deque([obs_tensor for _ in range(seq_len)], maxlen=seq_len)
        
        prev_hedge = torch.tensor([[0.0]], device=device)
        
        # Episode Storage
        ep_log_probs = []
        ep_entropies = []
        ep_quantiles = []
        ep_rewards = []
        ep_values = []
        ep_latents = []  
        ep_grus = []     
        ep_us = []       
        
        done = False
        t = 0
        
        while not done:
            # 1. Inject Shock 
            is_shock = False
            if phase == "MASTERY" and not episode_shock_triggered:
                # Allow shock any time between 20% and 80% of the option life
                if t > (env.n_steps * 0.2) and np.random.rand() < (shock_prob / 10): 
                    env.S_t = env.S_t * 0.90 
                    env.V_t = env.V_t + 0.15
                    is_shock = True
                    episode_shock_triggered = True # Ensure only one massive shock per ep

            # 2. Prepare Input
            obs_seq = torch.stack(list(obs_history)).unsqueeze(0)
            
            # 3. Forward Pass
            latent, gru_hist = agent.encoder(obs_seq, prev_hedge)
            risk_out = agent.critic(latent, gru_hist)
            # Calculate CVaR (Expected Shortfall) at the 5% tail
            cvar_val = agent.compute_risk_penalty(risk_out, alpha=0.05).item()

            dist = agent.actor(latent, risk_out)
            
            # Sample Action
            action, mode_idx = dist.sample_with_mode()
            action_val = action.item()

            # --- Optional: capture regime probabilities for analysis ---
            with torch.no_grad():
                mode_probs = torch.softmax(dist.cat_dist.logits, dim=-1).cpu().numpy()[0]
            
            # 4. Step Environment
            next_obs, raw_reward, terminated, truncated, info = env.step(np.array([action_val]))
            
            # Cost Annealing
            actual_trade = abs(action_val - prev_hedge.item()) * env.S_t
            implied_cost = actual_trade * real_cost
            adjusted_reward = raw_reward + (implied_cost * (1.0 - cost_multiplier))
            
            done = terminated or truncated
            
            # 5. Log Data (Placeholders for summary stats added later)
            training_log.append({
                "Episode": episode,
                "Step": t,
                "Phase": phase,
                "Price": env.S_t,
                "Action": action_val,
                "Reward": adjusted_reward,
                "Cost_Mult": cost_multiplier,
                "Mode": mode_idx.item(),

                # --- Regime ---
            
                "Prob_Mode0": float(mode_probs[0]),
                "Prob_Mode1": float(mode_probs[1]),
                "Prob_Mode2": float(mode_probs[2]),

                "Xi": risk_out['xi'].item(),
                "Uncertainty": risk_out['uncertainty'].item(),
                "Shock": is_shock,

                "CVaR_5pct": cvar_val,          # added for benchmarking

                "EpisodeTotalPnl": np.nan,      # Placeholder
                "EpisodeAvgPnlPerStep": np.nan, # Placeholder
                "EpisodeLen": 0                 # Placeholder
            })
                
            
            # 6. Store Tensors
            ep_log_probs.append(dist.log_prob(action))
            ep_entropies.append(dist.entropy())
            ep_quantiles.append(risk_out['quantiles'])
            ep_latents.append(latent.detach()) 
            ep_grus.append(gru_hist.detach())
            ep_us.append(risk_out['u'].detach())
            ep_rewards.append(adjusted_reward)
            ep_values.append(risk_out['quantiles'][0, agent.critic.num_quantiles // 2])
            
            # Update State
            obs_history.append(torch.tensor(next_obs, dtype=torch.float32, device=device))
            
            # --- PATCH 2: Stable prev_hedge shape ---
            prev_hedge = action.detach().view(1, 1)
            # ----------------------------------------
            
            t += 1
            
        # --- PATCH 4: Episode Summaries & Backfill ---
        episode_total_pnl = float(info.get('terminal_pnl', np.sum(ep_rewards)))
        episode_avg_pnl_per_step = float(np.mean(ep_rewards)) if len(ep_rewards) > 0 else 0.0
        episode_len = t

        episode_summaries.append({
            'Episode': episode,
            'Phase': phase,
            'EpisodeTotalPnl': episode_total_pnl,
            'EpisodeAvgPnlPerStep': episode_avg_pnl_per_step,
            'EpisodeLen': episode_len
        })

        # Backfill current episode rows in training_log
        start_idx = len(training_log) - episode_len
        for row_idx in range(max(0, start_idx), len(training_log)):
            training_log[row_idx]['EpisodeTotalPnl'] = episode_total_pnl
            training_log[row_idx]['EpisodeAvgPnlPerStep'] = episode_avg_pnl_per_step
            training_log[row_idx]['EpisodeLen'] = episode_len
        # -----------------------------------------------------

        # --- UPDATE STEP ---
        
        # A. Calculate Returns
        gamma = 0.99
        returns = []
        R = 0
        for r in reversed(ep_rewards):
            R = r + gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns, dtype=torch.float32, device=device).unsqueeze(1)
        
        # B. Tail Buffer Population
        for i in range(len(returns)):
            if returns[i] < ep_us[i]:
                # --- PATCH 3: Normalize Tail Buffer Items ---
                _latent = ep_latents[i].detach().squeeze(0)   # -> [H]
                _gru = ep_grus[i].detach().squeeze(0)         # -> [T, H]
                _ret = returns[i].detach().squeeze()          # -> scalar
                tail_buffer.append((_latent, _gru, float(_ret)))
                # --------------------------------------------
                
        # C. Main Gradient Update
        optimizer_main.zero_grad()
        
        # Body Loss
        quantiles_stack = torch.cat(ep_quantiles, dim=0)
        body_loss = generalized_quantile_huber_loss(quantiles_stack, returns)
        
        # Actor Loss
        values_stack = torch.stack(ep_values).squeeze()
        adv = returns.squeeze() - values_stack.detach()
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        
        log_probs_stack = torch.stack(ep_log_probs).squeeze()
        entropy_stack = torch.stack(ep_entropies).squeeze()
        
        actor_loss = -(log_probs_stack * adv).mean()
        ent_loss = -entropy_coef * entropy_stack.mean()
        
        loss = body_loss + actor_loss + ent_loss
        loss.backward()
        nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
        optimizer_main.step()
        
        # D. Tail Gradient Update (GPD)
        # --- PATCH 5: Robust Tail Batching & Stable GPD Loss ---
        if len(tail_buffer) > 64 and episode % 10 == 0:
            optimizer_tail.zero_grad()
            indices = np.random.choice(len(tail_buffer), 64, replace=False)

            # Reconstruct batch from buffer
            b_latents = torch.stack([tail_buffer[i][0] for i in indices], dim=0).to(device)   # [B, H]
            b_grus    = torch.stack([tail_buffer[i][1] for i in indices], dim=0).to(device)   # [B, T, H]

            # returns were stored as scalars → create clean tensor once
            b_returns = torch.tensor(
                [tail_buffer[i][2] for i in indices],
                dtype=torch.float32,
                device=device
            ).view(-1, 1)  # [B,1]

            # Forward Tail Head
            raw_tail = agent.critic.tail_net(b_latents)
            sigma = torch.nn.functional.softplus(raw_tail[:, 0:1]) + 1e-6
            xi = torch.tanh(raw_tail[:, 1:2]) * 0.5

            # Recalculate u using critic (no grad)
            with torch.no_grad():
                risk_current = agent.critic(b_latents, b_grus)
                u_current = risk_current['u']  # [B,1]

            # Exceedances (positive by construction)
            exceedances = (u_current - b_returns).clamp(min=1e-6)  # [B,1]

            # Stable GPD NLL
            eps = 1e-6
            xi_clamped = xi.clamp(min=-0.999, max=0.999)
            sigma_clamped = sigma.clamp(min=1e-6)

            z = exceedances / sigma_clamped
            t_gpd = 1.0 + xi_clamped * z
            t_gpd = t_gpd.clamp(min=1e-12)

            small = 1e-4
            abs_xi = xi_clamped.abs()

            nll = torch.where(
                abs_xi > small,
                torch.log(sigma_clamped) + (1.0 + 1.0/xi_clamped) * torch.log(t_gpd),
                torch.log(sigma_clamped) + z   # limiting exponential case for xi -> 0
            )

            tail_loss = nll.mean()
            tail_loss.backward()
            optimizer_tail.step()
        # -------------------------------------------------------

        # Update progress bar
        if episode % 10 == 0:
            pbar.set_description(f"Ep {episode} | R: {np.sum(ep_rewards):.2f} | Phase: {phase}")

    # --- SAVE ---
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(agent.state_dict(), save_path)
    print(f"Model saved to {save_path}")

    # --- EXPORT LOGS ---
    print("Exporting training logs...")
    # 1. Step-level log
    df_log = pd.DataFrame(training_log)
    df_log.to_csv("training_log_v2(5k-episodes).csv", index=False)
    
    # 2. Episode-level summary
    df_summary = pd.DataFrame(episode_summaries)
    df_summary.to_csv("episode_summaries(5k-episodes).csv", index=False)
    print("Logs saved: training_log_v2(5k-episodes).csv, episode_summaries(5k-episodes).csv")

    # --- DASHBOARD ---
    print("Generating final dashboard...")
    agent.eval()
    obs, _ = env.reset()
    obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device)
    obs_history = deque([obs_tensor for _ in range(seq_len)], maxlen=seq_len)
    prev_hedge = torch.tensor([[0.0]], device=device)
    
    hist = {'prices': [], 'deltas': [], 'actions': [], 'modes': [], 'xis': []}
    
    done = False
    while not done:
        obs_seq = torch.stack(list(obs_history)).unsqueeze(0)
        
        diag = agent.get_diagnostics(obs_seq, prev_hedge)
        
        tau = env.T - env.t_idx * env.dt
        _, delta, _, _ = bs_greeks(env.S_t, cfg['option']['K'], 0, 0, env.V_t, tau, 'call')
        
        hist['prices'].append(env.S_t)
        hist['deltas'].append(delta)
        hist['actions'].append(diag['action_mean'])
        hist['modes'].append(diag['mode'])
        hist['xis'].append(diag['xi'])
        
        next_obs, _, term, trunc, _ = env.step(np.array([diag['action_mean']]))
        obs_history.append(torch.tensor(next_obs, dtype=torch.float32, device=device))
        prev_hedge = torch.tensor([[diag['action_mean']]], device=device)
        done = term or trunc
        
    save_dashboard(hist)
    print("Done.")

if __name__ == "__main__":
    train_era_rl()