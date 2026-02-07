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

    # # 3. Regime
    # ax3 = axes[2]
    # ax3.step(range(len(history['modes'])), history['modes'], color='orange', where='post')
    # ax3.set_yticks([0, 1, 2])
    # ax3.set_yticklabels(['Normal', 'Defensive', 'Panic'])
    # ax3.set_title('Regime Detection')
    
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


def empirical_cvar(values, alpha=0.05):
    """Lower-tail CVaR for PnL (more negative = worse)."""
    if len(values) == 0:
        return 0.0
    q = np.quantile(values, alpha)
    tail = [v for v in values if v <= q]
    return float(np.mean(tail)) if tail else float(q)

def train_era_rl(
    config_path='src/configs/default.yaml',
    save_path='models/era_rl_v2(test6(5k)).pth',
    # n_episodes_override: int | None = None,
):
    cfg = load_config(config_path)
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")  # Force CPU for debugging
    print(f"--- Starting ERA-RL V2 Training on {device} ---")

    # 1. Environment & Agent
    real_cost = cfg['environment'].get('trading_cost', 0.0002)      # 2BP
    env = HedgingEnv(cfg['simulator'], cfg['option'], trading_cost=real_cost)
    
    # We append running average A_t (normalized) as an extra feature
    obs_dim = env.observation_space.shape[0] + 1
    agent = ERARL_Agent_V2(obs_dim=obs_dim, hidden_dim=128).to(device)

    # 2. Optimizers
    # Fast Clock (Body, Actor, Encoder)
    optimizer_main = optim.Adam([
        {'params': agent.encoder.parameters(), 'lr': 3e-4},
        {'params': agent.actor.parameters(), 'lr': 1e-4}, 
        {'params': agent.critic.body_net.parameters(), 'lr': 3e-4},
        {'params': agent.critic.quantile_layer.parameters(), 'lr': 3e-4},
        {'params': agent.critic.attention_score.parameters(), 'lr': 3e-4},
        {'params': agent.critic.uncertainty_net.parameters(), 'lr': 3e-4}
    ])

    # Critic-only optimizer for optional pretraining (keeps actor frozen)
    optimizer_critic = optim.Adam([
        {'params': agent.encoder.parameters(), 'lr': 3e-4},
        {'params': agent.critic.body_net.parameters(), 'lr': 3e-4},
        {'params': agent.critic.quantile_layer.parameters(), 'lr': 3e-4},
        {'params': agent.critic.attention_score.parameters(), 'lr': 3e-4},
        {'params': agent.critic.uncertainty_net.parameters(), 'lr': 3e-4},
    ])
    # Slow Clock (Tail GPD)
    optimizer_tail = optim.Adam(agent.critic.tail_net.parameters(), lr=5e-5)

    # 3. Training Config
    n_episodes = 5000          # full training (use smaller for smoke tests)
    seq_len = 30

    # # QUICK TEST: you can override to e.g. 200 for fast runs
    # if n_episodes_override is not None:
    #     n_episodes = int(n_episodes_override)
    
    # Phases
    p1 = 0.15
    p2 = 0.60

    phase_1_end = int(p1 * n_episodes) 
    phase_2_end = int((p1 + p2) * n_episodes) 

    # Tail / critic pretrain & imitation warm-start
    EPISODES_PRETRAIN_CRITIC = 1000   # if >0, train critic/tail only for these episodes
    EPISODES_IMITATE = 2000           # warm-start actor to follow BS; decay after
    IMITATION_START_WEIGHT = 0.8     # WAS 0.4: Stronger initial guidance
    IMITATION_END_WEIGHT = 0.0       # decays to this over EPISODES_IMITATE

    # Relative CVaR objective hyperparams (tuned for better PnL balance)
    # NOTE: the training objective is now replication-loss based (not PnL-vs-BS)
    LAMBDA_CVAR_NEW = 0.3    # weight on critic CVaR term
    LAMBDA_TURN = 0.1        # turnover penalty
    LAMBDA_REG = 0.01        # L2 on action
    KAPPA = 30.0             # terminal replication penalty weight
    
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
            entropy_coef = 0.2
            shock_prob = 0.0
        elif episode < phase_2_end:
            phase = "STABILIZATION"
            cost_multiplier = 1.0 
            entropy_coef = 0.05
            shock_prob = 0.05
        else:
            phase = "MASTERY"
            cost_multiplier = 1.0
            # Linearly decay entropy to stop random guessing
            progress = (episode - phase_2_end) / max(1, (n_episodes - phase_2_end))
            entropy_coef = max(0.005, 0.05 * (1.0 - progress))
            shock_prob = 0.15

        # --- ROLLOUT ---
        obs, _ = env.reset()
        episode_shock_triggered = False # Ensure only one massive shock per ep


        # --- Running average A_t (for Asian/path dependence) ---
        # If env doesn't provide A_t, we compute and append it.
        running_sum = float(env.S_t)
        running_count = 1
        notional = max(1.0, cfg['option'].get('notional', cfg['option'].get('K', 1.0)))
        A_t = running_sum / running_count

        obs_with_A = np.concatenate([np.asarray(obs, dtype=np.float32), np.asarray([A_t / notional], dtype=np.float32)])
        obs_tensor = torch.tensor(obs_with_A, dtype=torch.float32, device=device)
        obs_history = deque([obs_tensor for _ in range(seq_len)], maxlen=seq_len)
        
        # stable prev_hedge shapes (actor position)
        prev_hedge = torch.tensor([[0.0]], device=device)

        # BS baseline bookkeeping (scalar floats) - diagnostics / warm-start only
        bs_prev_hedge = 0.0
        prev_V = env.V_t  # variance state (used to form sigma for BS delta)
        
        # Episode Storage
        ep_log_probs = []
        ep_entropies = []
        ep_quantiles = []
        # ep_rewards = []
        ep_values = []
        ep_latents = []  
        ep_grus = []     
        ep_us = []       
        ep_actions = []
        ep_price_rets = []
        ep_mode_probs = []
        ep_bs_deltas = []

        # Raw hedged PnL stream + reward stream
        ep_rl_raw_rewards = []   # raw hedged pnl (delta_pnl)
        ep_rewards_adj = []      # replication-loss reward
        ep_norm_losses = []      # normalized positive loss magnitudes (for tail buffer)
        
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
            
            # Sample Action (MixtureSigmoidNormal -> returns in (0,1))
            action_sample, mode_idx = dist.sample_with_mode()  # already in (0,1)
            action_val = float(np.clip(float(action_sample.item()) + np.random.normal(scale=0.02), 0.0, 1.0))

            # For log-prob evaluation, pass the [0,1] action
            action_for_logprob = torch.tensor([[action_val]], dtype=torch.float32, device=device)

            # --- Optional: capture regime probabilities for analysis ---
            with torch.no_grad():
                mode_probs = torch.softmax(dist.cat_dist.logits, dim=-1).cpu().numpy()[0]
            ep_mode_probs.append(mode_probs)
            
            # 4. Step Environment + BS baseline + CVaR objective
            prev_price = env.S_t
            prev_V = env.V_t  # variance before step

            # BS delta computed at the same state/time as the RL decision
            tau = max(env.T - env.t_idx, 0) * env.dt
            sigma = float(np.sqrt(max(prev_V, 1e-8)))
            _, bs_delta, _, _ = bs_greeks(prev_price, cfg['option']['K'], 0, 0, sigma, tau, 'call')
            ep_bs_deltas.append(float(bs_delta))

            next_obs, raw_reward, terminated, truncated, info = env.step(np.array([action_val]))

            # Store prev_price in info for downstream logging/analysis
            info['prev_price'] = prev_price

            # -------------------------------------------------------------
            # Replication-loss reward (NO PnL-vs-BS objective)
            # compute discrete hedged pnl (seller of 1 option, holding a_prev units of underlying)
            dV = float(info.get('d_option', 0.0))
            dS = float(env.S_t - prev_price)
            trade_size = float(abs(action_val - prev_hedge.item()))
            tx_cost = float(real_cost * trade_size * env.S_t)

            # discrete hedged portfolio change (Delta Pi)
            delta_pnl = float((-dV) + (prev_hedge.item() * dS) - tx_cost)
            L_t = float(-delta_pnl)  # loss = -PnL (positive = bad)
            normL = float(L_t / notional)

            # CVaR estimate from critic (parametric tail head)
            # risk_out already computed above (before stepping env)
            cvar_val = agent.compute_risk_penalty(risk_out, alpha=0.05).item()

            reward = - (normL ** 2) \
                     - (LAMBDA_CVAR_NEW * cvar_val) \
                     - (LAMBDA_TURN * trade_size) \
                     - (LAMBDA_REG * (action_val ** 2))

            # Terminal replication penalty (apply to final step reward)
            terminal_rep_err = np.nan
            terminal_penalty = 0.0
            if terminated or truncated:
                accumulated_hedged_pnl = float(np.sum(ep_rl_raw_rewards) + delta_pnl)
                payoff = float(info.get('payoff', 0.0))
                terminal_rep_err = float(payoff - accumulated_hedged_pnl)
                terminal_penalty = float(-KAPPA * (terminal_rep_err / notional) ** 2)
                reward = float(reward + terminal_penalty)

            # Diagnostics baseline (BS delta hedge) - NOT used in reward
            bs_trade = abs(bs_delta - bs_prev_hedge) * prev_price
            bs_tx_cost = bs_trade * real_cost
            bs_pnl = (-dV) + (bs_prev_hedge * dS) - bs_tx_cost
            

            # Optional: keep annealing of explicit transaction refund if desired
            # implied_cost = abs(action_val - prev_hedge.item()) * env.S_t * real_cost
            # adjusted_reward += implied_cost * (1.0 - cost_multiplier)

            # update BS bookkeeping for next tick
            bs_prev_hedge = float(bs_delta)

            # Price return retained for diagnostics
            price_ret = (env.S_t - info.get('prev_price', env.S_t)) / (info.get('prev_price', env.S_t) + 1e-8)
            
            done = terminated or truncated
            
            # 5. Log Data (Placeholders for summary stats added later)
            training_log.append({
                "Episode": episode,
                "Step": t,
                "Phase": phase,
                "Price": env.S_t,
                "PrevPrice": float(info.get('prev_price', env.S_t)),
                "PriceRet": float(price_ret),
                "Action": action_val,
                "Reward": float(reward),
                "Cost_Mult": cost_multiplier,
                "Mode": mode_idx.item(),

                # --- Regime ---
            
                "Prob_Mode0": float(mode_probs[0]),
                "Prob_Mode1": float(mode_probs[1]),
                "Prob_Mode2": float(mode_probs[2]),

                "Xi": risk_out['xi'].item(),
                "Sigma": risk_out['sigma'].item(),
                "Uncertainty": risk_out['uncertainty'].item(),
                "Shock": is_shock,

                "CVaR_5pct": cvar_val,          # added for benchmarking

                # Replication-loss diagnostics
                "Loss_t": float(L_t),
                "NormLoss": float(normL),
                "TradeSize": float(trade_size),
                "TxCost": float(tx_cost),
                "TerminalRepErr": float(terminal_rep_err) if np.isfinite(terminal_rep_err) else np.nan,
                "TerminalPenalty": float(terminal_penalty),

                # Baseline diagnostics
                "RL_PnL": float(delta_pnl),
                "BS_PnL": float(bs_pnl),
                "BS_Delta": float(bs_delta),

                "EpisodeTotalPnl": np.nan,      # Placeholder
                "EpisodeAvgPnlPerStep": np.nan, # Placeholder
                "EpisodeLen": 0,                # Placeholder

                # Empirical CVaR episode metrics (backfilled at end)
                "Empirical_RL_CVaR": np.nan,
                "Empirical_BS_CVaR": np.nan,
                "Delta_CVaR_RL_minus_BS": np.nan,
                "Critic_CVaR_Estimate": np.nan,

                # Critic calibration / tail diagnostics (backfilled at end)
                "Empirical_RL_Q5": np.nan,
                "Critic_u": np.nan,
                "TailFillFrac": np.nan,
                "MeanXi": np.nan,
                "MeanSigma": np.nan,

                # Episode-level diagnostics (backfilled at end)
                "ActionStd": np.nan,
                "CorrActionRet": np.nan,
                "ModeEntropy": np.nan,
            })
                
            
            # 6. Store Tensors
            ep_log_probs.append(dist.log_prob(action_for_logprob))
            ep_entropies.append(dist.entropy())
            ep_quantiles.append(risk_out['quantiles'])
            ep_latents.append(latent.detach()) 
            ep_grus.append(gru_hist.detach())
            ep_us.append(float(risk_out['u'].item()))
            ep_rl_raw_rewards.append(float(delta_pnl))
            ep_rewards_adj.append(float(reward))
            ep_norm_losses.append(float(max(normL, 0.0)))
            ep_values.append(risk_out['quantiles'][0, agent.critic.num_quantiles // 2])
            ep_actions.append(action_val)
            ep_price_rets.append(float(price_ret))
            
            # Update State
            running_sum += float(env.S_t)
            running_count += 1
            A_t = running_sum / running_count
            next_obs_with_A = np.concatenate([
                np.asarray(next_obs, dtype=np.float32),
                np.asarray([A_t / notional], dtype=np.float32),
            ])
            obs_history.append(torch.tensor(next_obs_with_A, dtype=torch.float32, device=device))
            
            # --- PATCH 2: Stable prev_hedge shape ---
            prev_hedge = torch.tensor([[action_val]], dtype=torch.float32, device=device)
            # ----------------------------------------
            
            t += 1
            
        # --- PATCH 4: Episode Summaries & Backfill ---
        episode_total_pnl = float(info.get('terminal_pnl', np.sum(ep_rl_raw_rewards)))
        episode_avg_pnl_per_step = float(np.mean(ep_rl_raw_rewards)) if len(ep_rl_raw_rewards) > 0 else 0.0
        episode_len = t

        # terminal replication error (logged at last step if provided)
        terminal_rep_err_ep = float(info.get('payoff', 0.0) - episode_total_pnl) if episode_len > 0 else 0.0

        # --- Patch 5: Episode-end diagnostics ---
        action_std = float(np.std(ep_actions)) if len(ep_actions) > 0 else 0.0
        if len(ep_actions) > 1 and np.std(ep_price_rets) > 1e-12 and np.std(ep_actions) > 1e-12:
            corr_action_ret = float(np.corrcoef(np.array(ep_actions), np.array(ep_price_rets))[0, 1])
            if not np.isfinite(corr_action_ret):
                corr_action_ret = 0.0
        else:
            corr_action_ret = 0.0

        if len(ep_mode_probs) > 0:
            avg_mode_probs = np.mean(np.stack(ep_mode_probs, axis=0), axis=0)
            mode_entropy = float(-(avg_mode_probs * np.log(avg_mode_probs + 1e-8)).sum())
        else:
            mode_entropy = 0.0

        # ----- Empirical CVaR calculation from this episode -----
        ep_rows = training_log[-episode_len:] if episode_len > 0 else []
        ep_rl_pnls = [row.get("RL_PnL", 0.0) for row in ep_rows]
        ep_bs_pnls = [row.get("BS_PnL", 0.0) for row in ep_rows]

        emp_rl_cvar = empirical_cvar(ep_rl_pnls, alpha=0.05)
        emp_bs_cvar = empirical_cvar(ep_bs_pnls, alpha=0.05)
        delta_cvar = emp_rl_cvar - emp_bs_cvar

        critic_cvar_estimate = float(
            np.mean([r.get("CVaR_5pct", 0.0) for r in ep_rows])
        ) if episode_len > 0 else 0.0
        # --------------------------------------------------------

        # --- Critic calibration diagnostics ---
        emp_rl_q5 = float(np.quantile(ep_rl_pnls, 0.05)) if len(ep_rl_pnls) > 0 else 0.0
        critic_u = float(np.mean(ep_us)) if len(ep_us) > 0 else 0.0
        mean_xi = float(np.mean([r.get('Xi', 0.0) for r in ep_rows])) if episode_len > 0 else 0.0
        mean_sigma = float(np.mean([r.get('Sigma', 0.0) for r in ep_rows])) if episode_len > 0 else 0.0

        episode_summaries.append({
            'Episode': episode,
            'Phase': phase,
            'EpisodeTotalPnl': episode_total_pnl,
            'EpisodeAvgPnlPerStep': episode_avg_pnl_per_step,
            'EpisodeLen': episode_len,

            'TerminalRepErr': terminal_rep_err_ep,

            # --- NEW METRICS ---
            'Empirical_RL_CVaR': emp_rl_cvar,
            'Empirical_BS_CVaR': emp_bs_cvar,
            'Delta_CVaR_RL_minus_BS': delta_cvar,
            'Critic_CVaR_Estimate': critic_cvar_estimate,
            # -------------------

            # --- Calibration / tail diagnostics ---
            'Empirical_RL_Q5': emp_rl_q5,
            'Critic_u': critic_u,
            'MeanXi': mean_xi,
            'MeanSigma': mean_sigma,
            # -------------------

            'ActionStd': action_std,
            'CorrActionRet': corr_action_ret,
            'ModeEntropy': mode_entropy,
        })

        # Backfill current episode rows in training_log
        start_idx = len(training_log) - episode_len
        for row_idx in range(max(0, start_idx), len(training_log)):
            training_log[row_idx]['EpisodeTotalPnl'] = episode_total_pnl
            training_log[row_idx]['EpisodeAvgPnlPerStep'] = episode_avg_pnl_per_step
            training_log[row_idx]['EpisodeLen'] = episode_len

            training_log[row_idx]['Empirical_RL_CVaR'] = emp_rl_cvar
            training_log[row_idx]['Empirical_BS_CVaR'] = emp_bs_cvar
            training_log[row_idx]['Delta_CVaR_RL_minus_BS'] = delta_cvar
            training_log[row_idx]['Critic_CVaR_Estimate'] = critic_cvar_estimate

            training_log[row_idx]['Empirical_RL_Q5'] = emp_rl_q5
            training_log[row_idx]['Critic_u'] = critic_u
            training_log[row_idx]['MeanXi'] = mean_xi
            training_log[row_idx]['MeanSigma'] = mean_sigma

            training_log[row_idx]['ActionStd'] = action_std
            training_log[row_idx]['CorrActionRet'] = corr_action_ret
            training_log[row_idx]['ModeEntropy'] = mode_entropy
        # -----------------------------------------------------

        # --- UPDATE STEP ---
        
        # A. Calculate Returns
        gamma = 0.99

        # returns_raw: used for critic targets and tail fitting
        returns_raw = []
        Rr = 0.0
        for r in reversed(ep_rl_raw_rewards):
            Rr = r + gamma * Rr
            returns_raw.insert(0, Rr)
        returns_raw = torch.tensor(returns_raw, dtype=torch.float32, device=device).unsqueeze(1)

        # returns_adj: used for actor advantage (penalized objective)
        returns_adj = []
        Ra = 0.0
        for r in reversed(ep_rewards_adj):
            Ra = r + gamma * Ra
            returns_adj.insert(0, Ra)
        returns_adj = torch.tensor(returns_adj, dtype=torch.float32, device=device).unsqueeze(1)

        # B. Tail Buffer Population (robust): add worst 20% (at least 3) returns_raw
        tail_added = 0
        if len(ep_norm_losses) > 0:
            k = max(3, int(0.20 * len(ep_norm_losses)))
            worst_idx = np.argsort(np.asarray(ep_norm_losses))[-k:]
            for i in worst_idx:
                _latent = ep_latents[int(i)].detach().squeeze(0)  # [H]
                _gru = ep_grus[int(i)].detach().squeeze(0)        # [T,H]
                # store normalized positive loss magnitude for GPD fitting
                _loss_mag = float(ep_norm_losses[int(i)])
                tail_buffer.append((_latent, _gru, _loss_mag))
                tail_added += 1

        tail_fill_frac = float(tail_added / max(1, episode_len))

        # backfill tail fill fraction into episode rows
        for row_idx in range(max(0, start_idx), len(training_log)):
            training_log[row_idx]['TailFillFrac'] = tail_fill_frac
                
        # C. Main Gradient Update
        if episode < EPISODES_PRETRAIN_CRITIC:
            # --- Critic-only pretraining: fit quantiles (and encoder representation) ---
            optimizer_critic.zero_grad()

            quantiles_stack = torch.cat(ep_quantiles, dim=0)
            # PnL-to-Loss mapping (EVT models right tail; finance risk is left tail)
            # Quantiles learn the distribution of LOSSES where Loss = -PnL
            target_loss = -returns_raw.detach()
            body_loss = generalized_quantile_huber_loss(quantiles_stack, target_loss)

            body_loss.backward()
            nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
            optimizer_critic.step()
        else:
            # Normal joint update (actor + critic body)
            optimizer_main.zero_grad()

            quantiles_stack = torch.cat(ep_quantiles, dim=0)
            # Quantiles learn the distribution of LOSSES where Loss = -PnL
            target_loss = -returns_raw.detach()
            body_loss = generalized_quantile_huber_loss(quantiles_stack, target_loss)

            values_stack = torch.stack(ep_values).squeeze()
            # ep_values come from the critic quantiles; after the Loss mapping they are in loss-space.
            # Convert to a PnL-like proxy so the advantage has the correct directionality.
            values_pnl_proxy = -values_stack.detach()
            adv = returns_adj.squeeze() - values_pnl_proxy

            # ROBUST ADVANTAGE SCALING
            if adv.std() > 1e-4:
                adv = (adv - adv.mean()) / (adv.std() + 1e-8)
            else:
                adv = adv - adv.mean()

            log_probs_stack = torch.stack(ep_log_probs).squeeze()
            entropy_stack = torch.stack(ep_entropies).squeeze()

            # Actor (policy) loss (REINFORCE-style)
            actor_loss = -(log_probs_stack * adv).mean()
            ent_loss = -entropy_coef * entropy_stack.mean()



            # === Imitation warm-start (MSE to BS delta) ===
            imit_loss = torch.tensor(0.0, device=device)
            imit_weight = 0.0
            if episode < EPISODES_IMITATE and len(ep_actions) > 0 and len(ep_bs_deltas) == len(ep_actions):
                # actions and BS deltas are both in [0,1] space
                actor_actions_01 = torch.tensor(ep_actions, dtype=torch.float32, device=device)
                bs_deltas_t = torch.tensor(
                    ep_bs_deltas,
                    dtype=torch.float32,
                    device=device,
                )
                imit_loss = nn.functional.mse_loss(actor_actions_01, bs_deltas_t)

                frac = max(0.0, 1.0 - (episode / EPISODES_IMITATE))
                imit_weight = IMITATION_END_WEIGHT + (IMITATION_START_WEIGHT - IMITATION_END_WEIGHT) * frac

            mode_probs = torch.softmax(dist.cat_dist.logits, dim=-1)
            # Encourage diversity (higher entropy) across mixture modes
            diversity_bonus = 0.01 * (mode_probs * torch.log(mode_probs + 1e-8)).sum(dim=-1).mean()

            loss = body_loss + actor_loss + ent_loss + (imit_weight * imit_loss) + diversity_bonus
            loss.backward()
            nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
            optimizer_main.step()
        
        # D. Tail Gradient Update (GPD)
        # --- TAIL UPDATE PATCH ---
        if len(tail_buffer) > 64 and episode % 1 == 0:  # Update every episode
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
            # 1. Synchronize Xi scaling with models.py
            xi = torch.tanh(raw_tail[:, 1:2]) * 0.5

            # Recalculate u using critic (no grad)
            with torch.no_grad():
                risk_current = agent.critic(b_latents, b_grus)
                # u is now trained in LOSS space (see target_loss above)
                u_loss_threshold = risk_current['u']  # [B,1]

            # 2. Correct Exceedance Calculation (for positive losses)
            # exceedances = Loss_Magnitude - Loss_Threshold
            exceedances = (b_returns - u_loss_threshold).clamp(min=1e-4)  # [B,1]

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

            # 3. Add L2 Reg to keep parameters from hitting boundaries
            l2_reg = 1e-4 * sum(p.pow(2).sum() for p in agent.critic.tail_net.parameters())

            tail_loss = nll.mean() + l2_reg
            tail_loss.backward()
            optimizer_tail.step()
        # -------------------------------------------------------

        # Update progress bar
        if episode % 10 == 0:
            pbar.set_description(f"Ep {episode} | R: {np.sum(ep_rewards_adj):.2f} | Phase: {phase}")

    # --- SAVE ---
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(agent.state_dict(), save_path)
    print(f"Model saved to {save_path}")

    # --- EXPORT LOGS ---
    print("Exporting training logs...")
    # 1. Step-level log
    df_log = pd.DataFrame(training_log)
    df_log.to_csv("training_log_v2(test6(5k)).csv", index=False)
    
    # 2. Episode-level summary
    df_summary = pd.DataFrame(episode_summaries)
    df_summary.to_csv("episode_summaries(test6(5k)).csv", index=False)
    print("Logs saved: training_log_v2(test6(5k)).csv, episode_summaries(test6(5k)).csv")

    # --- Quick summaries / smoke-test diagnostics ---
    try:
        print("Summary (last 100 episodes):")
        recent = df_log[df_log['Episode'] >= max(0, n_episodes - 100)]
        n_eps_recent = max(1, int(recent['Episode'].nunique()))
        print("Mean RL_PnL:", float(recent['RL_PnL'].sum() / n_eps_recent) if 'RL_PnL' in recent else float('nan'))
        print("Mean BS_PnL:", float(recent['BS_PnL'].sum() / n_eps_recent) if 'BS_PnL' in recent else float('nan'))
        print("Mean Advantage:", float(recent['Advantage'].sum() / n_eps_recent) if 'Advantage' in recent else float('nan'))
        print("Mean CVaR:", float(recent['CVaR_5pct'].mean()) if 'CVaR_5pct' in recent else float('nan'))
        print("Empirical RL CVaR:", float(recent['Empirical_RL_CVaR'].mean()))
        print("Empirical BS CVaR:", float(recent['Empirical_BS_CVaR'].mean()))
        print("Δ CVaR (RL - BS):", float(recent['Delta_CVaR_RL_minus_BS'].mean()))
        print("Critic CVaR estimate:", float(recent['CVaR_5pct'].mean()))

        if len(df_summary) > 0:
            last = df_summary.tail(1)
            print("Diagnostics (last episode):")
            print("ActionStd:", float(last['ActionStd'].iloc[0]) if 'ActionStd' in last else float('nan'))
            print("CorrActionRet:", float(last['CorrActionRet'].iloc[0]) if 'CorrActionRet' in last else float('nan'))
            print("ModeEntropy:", float(last['ModeEntropy'].iloc[0]) if 'ModeEntropy' in last else float('nan'))
    except Exception as e:
        print("Summary printing failed:", repr(e))

# --- DASHBOARD & MASTERY REPORT ---
    print("Generating final dashboard...")
    agent.eval()

    # ============================================================
    # 1) Rollout one episode for behavior visualization
    # ============================================================
    obs, _ = env.reset()
    
    # Recalculate A_t setup for rollout
    notional = max(1.0, cfg['option'].get('notional', cfg['option'].get('K', 1.0)))
    running_sum = float(env.S_t)
    running_count = 1
    A_t = running_sum / running_count

    obs_with_A = np.concatenate([
        np.asarray(obs, dtype=np.float32),
        np.asarray([A_t / notional], dtype=np.float32)
    ])

    obs_tensor = torch.tensor(obs_with_A, dtype=torch.float32, device=device)
    obs_history = deque([obs_tensor for _ in range(seq_len)], maxlen=seq_len)
    prev_hedge = torch.tensor([[0.0]], device=device)

    hist = {'prices': [], 'deltas': [], 'actions': [], 'modes': [], 'xis': []}

    done = False
    while not done:
        obs_seq = torch.stack(list(obs_history)).unsqueeze(0)
        diag = agent.get_diagnostics(obs_seq, prev_hedge)

        # BS Delta for comparison
        tau = env.T - env.t_idx * env.dt
        _, delta, _, _ = bs_greeks(
            env.S_t, cfg['option']['K'], 0, 0, env.V_t, tau, 'call'
        )

        hist['prices'].append(env.S_t)
        hist['deltas'].append(delta)
        # MixtureSigmoidNormal actions are already in [0,1]
        hist['actions'].append(float(np.clip(diag['sampled_action'], 0.0, 1.0)))
        hist['modes'].append(diag['mode'])
        hist['xis'].append(diag['xi'])

        action_eval = float(np.clip(diag['action_mean'], 0.0, 1.0))
        next_obs, _, term, trunc, _ = env.step(np.array([action_eval]))

        # Update running average
        running_sum += float(env.S_t)
        running_count += 1
        A_t = running_sum / running_count

        next_obs_with_A = np.concatenate([
            np.asarray(next_obs, dtype=np.float32),
            np.asarray([A_t / notional], dtype=np.float32),
        ])

        obs_history.append(torch.tensor(next_obs_with_A, dtype=torch.float32, device=device))
        prev_hedge = torch.tensor([[action_eval]], dtype=torch.float32, device=device)
        done = term or trunc

    # ============================================================
    # 2) BUILD VERTICAL MASTERY REPORT (Fixed Layout & Files)
    # ============================================================
    try:
        # FIX 1: Use the EXACT filenames you saved earlier
        log_df = pd.read_csv("training_log_v2(test6(5k)).csv")
        sum_df = pd.read_csv("episode_summaries(test6(5k)).csv")

        # Prepare Data
        if 'EpisodeTotalPnl' in sum_df.columns:
            ep_pnls = sum_df['EpisodeTotalPnl'].values
        else:
            ep_pnls = log_df.groupby('Episode')['RL_PnL'].sum().values

        window = 100
        ma = pd.Series(ep_pnls).rolling(window, min_periods=1).mean().values
        
        critic_cvar = sum_df['Critic_CVaR_Estimate'].values
        empirical_cvar_series = sum_df['Empirical_RL_CVaR'].values
        action_std = sum_df['ActionStd'].values

        # Rollout Data
        prices = np.array(hist['prices'])
        actions = np.array(hist['actions'])
        deltas = np.array(hist['deltas'])
        steps = np.arange(len(prices))

        # FIX 2: Vertical Stack (4 Rows, 1 Column)
        fig, axes = plt.subplots(4, 1, figsize=(12, 18))  # Taller figure
        
        # --- Panel 1: PnL History ---
        axes[0].plot(ep_pnls, alpha=0.3, label='Raw RL PnL', color='blue')
        axes[0].plot(ma, color='red', linewidth=2, label=f'{window}-Ep Moving Avg')
        axes[0].set_title('RL Agent Profit/Loss Convergence')
        axes[0].set_ylabel('Total PnL')
        axes[0].legend(loc='upper left')
        axes[0].grid(True, alpha=0.3)

        # --- Panel 2: The "Delusion Gap" ---
        axes[1].plot(critic_cvar, color='green', label='Critic Estimate', linewidth=1.5)
        axes[1].plot(empirical_cvar_series, color='orange', label='Empirical Reality', linewidth=1.5, alpha=0.8)
        axes[1].set_title('Risk Perception: Critic CVaR vs. Actual CVaR')
        axes[1].set_ylabel('CVaR (Loss Magnitude)')
        axes[1].legend(loc='upper right')
        axes[1].grid(True, alpha=0.3)

        # --- Panel 3: Confidence (Action Std) ---
        axes[2].plot(action_std, color='purple', label='Action Std Dev')
        axes[2].set_title('Agent Confidence (Exploration Decay)')
        axes[2].set_ylabel('Std Dev')
        axes[2].legend(loc='upper right')
        axes[2].grid(True, alpha=0.3)

        # --- Panel 4: Final Episode Behavior (Dual Axis) ---
        ax4 = axes[3]
        ax4_twin = ax4.twinx()
        
        # Left Axis: Price
        ax4.plot(steps, prices, color='black', label='Asset Price', linewidth=1.5)
        ax4.set_ylabel('Price', color='black')
        ax4.set_xlabel('Step (Final Episode)')
        
        # Right Axis: Actions vs Delta
        ax4_twin.plot(steps, deltas, linestyle='--', color='gray', label='BS Delta', linewidth=2)
        ax4_twin.plot(steps, actions, color='blue', label='RL Action', linewidth=2)
        ax4_twin.set_ylabel('Hedge Ratio', color='blue')
        ax4_twin.set_ylim(-0.1, 1.1)  # Fix limits to show [0,1] clearly

        # Combined Legend
        lines1, labels1 = ax4.get_legend_handles_labels()
        lines2, labels2 = ax4_twin.get_legend_handles_labels()
        ax4.legend(lines1 + lines2, labels1 + labels2, loc='upper center', ncol=3)
        ax4.set_title('Final Policy: RL Hedge vs. Black-Scholes Delta')
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig("mastery_report_vertical.png", dpi=160)
        print("Mastery report saved to: mastery_report_vertical.png")

    except Exception as e:
        print(f"Could not build mastery report: {e}")
        import traceback
        traceback.print_exc()

    print("Done.")


if __name__ == "__main__":
    train_era_rl()