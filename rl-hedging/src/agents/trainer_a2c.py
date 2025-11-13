# src/agents/trainer_a2c.py
import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import csv 
from collections import namedtuple
import time
from tqdm import tqdm


from src.agents.cvar_ctitic import (
    ExDrlCritic,
    pinball_loss,  # kept for compatibility
    gpd_log_likelihood_loss_vectorized,
    estimate_cvar_from_gpd,
    generalized_quantile_huber_loss,
    estimate_noise_disparity
)

# Utilities
def maybe_mkdir(path):
    os.makedirs(path, exist_ok=True)

Rollout = namedtuple('Rollout', ['obs', 'actions', 'log_probs', 'rewards', 'values', 'dones'])

class SimpleRolloutBuffer:
    def __init__(self, capacity, obs_shape, device='cpu'):
        self.capacity = capacity
        self.device = device
        self.obs_buf = torch.zeros((capacity,)+obs_shape, dtype=torch.float32, device=device)
        self.act_buf = torch.zeros((capacity,1), dtype=torch.float32, device=device)
        self.logp_buf = torch.zeros((capacity,), dtype=torch.float32, device=device)
        self.rew_buf = torch.zeros((capacity,), dtype=torch.float32, device=device)
        self.val_buf = torch.zeros((capacity,), dtype=torch.float32, device=device)
        self.done_buf = torch.zeros((capacity,), dtype=torch.bool, device=device)
        self.ptr = 0

    def add(self, obs, action, logp, reward, value, done):
        idx = self.ptr
        self.obs_buf[idx].copy_(torch.as_tensor(obs, dtype=torch.float32, device=self.device))
        self.act_buf[idx].copy_(torch.as_tensor(action, dtype=torch.float32, device=self.device))
        self.logp_buf[idx] = float(logp)
        self.rew_buf[idx] = float(reward)
        self.val_buf[idx] = float(value)
        self.done_buf[idx] = bool(done)
        self.ptr += 1

    def reset(self):
        self.ptr = 0

    def size(self):
        return self.ptr

    def get(self):
        # The buffer must be full for the A2C update logic
        assert self.ptr == self.capacity, f"Buffer not full: {self.ptr}/{self.capacity}"
        return Rollout(
            obs=self.obs_buf.clone(),
            actions=self.act_buf.clone(),
            log_probs=self.logp_buf.clone(),
            rewards=self.rew_buf.clone(),
            values=self.val_buf.clone(),
            dones=self.done_buf.clone()
        )

# GAE
def compute_gae(rewards, values, dones, last_value, gamma=0.995, lam=0.95):
    T = rewards.shape[0]
    adv = torch.zeros(T, dtype=torch.float32, device=rewards.device)
    last_gae = 0.0
    for t in reversed(range(T)):
        mask = 1.0 - dones[t].float()
        if t == T - 1:
            next_values = last_value
        else:
            next_values = values[t + 1]
        delta = rewards[t] + gamma * next_values * mask - values[t]
        last_gae = delta + gamma * lam * mask * last_gae
        adv[t] = last_gae
    returns = adv + values
    return adv, returns

def critic_params(trainer):
    ps = list(trainer.critic.parameters())
    if trainer.quantile_critic is not None:
        ps += list(trainer.quantile_critic.parameters())
    return ps

# Trainer
class A2CTrainer:
    def __init__(
        self,
        env_fn,
        actor: nn.Module,
        critic: nn.Module,
        quantile_critic: ExDrlCritic = None,
        device='cpu',
        rollout_length=50,
        num_envs=16,
        lr_actor=1e-5,
        lr_critic=1e-4,
        gamma=0.995,
        lam=0.95,
        entropy_coef=0.01,
        value_coef=0.5,
        cvar_lambda=1.0,
        quantile_weight=1.0,
        gpd_loss_weight=1.0,
        max_grad_norm=0.5,
        advantage_clip=10.0,
        clip_adv_by_std=False,
        eval_fn=None,
        logdir='./runs/a2c',
        cvar_alpha=0.05,
    ):
        self.device = device
        self.envs = [env_fn() for _ in range(num_envs)]
        self.num_envs = num_envs
        self.actor = actor.to(device)
        self.critic = critic.to(device)
        self.quantile_critic = quantile_critic.to(device) if quantile_critic is not None else None

        self.rollout_length = rollout_length
        self.gamma = gamma
        self.lam = lam
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.cvar_lambda = cvar_lambda
        self.quantile_weight = quantile_weight
        self.gpd_loss_weight = gpd_loss_weight
        self.max_grad_norm = max_grad_norm
        self.advantage_clip = advantage_clip
        self.clip_adv_by_std = clip_adv_by_std
        self.eval_fn = eval_fn
        self.logdir = logdir
        maybe_mkdir(logdir)
        self.cvar_alpha = cvar_alpha
        self.verbose = False  # Set to True for detailed debug output

        # optimizers
        self.opt_actor = torch.optim.Adam(self.actor.parameters(), lr=lr_actor, betas=(0.9, 0.999), eps=1e-8)
        critic_params_list = list(self.critic.parameters())
        if self.quantile_critic is not None:
            critic_params_list += list(self.quantile_critic.parameters())
        self.opt_critic = torch.optim.Adam(critic_params_list, lr=lr_critic, betas=(0.9, 0.999), eps=1e-8)

        # buffer
        obs_shape = self.envs[0].observation_space.shape
        self.buffer = SimpleRolloutBuffer(rollout_length * num_envs, obs_shape, device=device)
        
        # Persistent states for stateful collector
        self.current_obs_t = [torch.as_tensor(e.reset()[0], dtype=torch.float32, device=self.device) for e in self.envs]
        self.h_states = [None for _ in range(self.num_envs)]


    def collect_rollouts(self):
        """
        Stateful collector. Collects N*T samples, handling resets internally.
        """
        obs_t = self.current_obs_t
        h_states = self.h_states
        
        for t in range(self.rollout_length):
            for i in range(self.num_envs):
                o = obs_t[i].unsqueeze(0)  # [1, obs_dim]
                with torch.no_grad():
                    dist, _ = self.actor.step(o, h_states[i])
                    
                    action_tensor = dist.sample() # Shape [1, 1]
                    action_np = action_tensor.cpu().numpy()[0] # Shape (1,)
                        
                    logp = dist.log_prob(action_tensor).cpu().item() 
                    value, _ = self.critic.step(o, h_states[i])

                next_o, reward, terminated, truncated, info = self.envs[i].step(action_np) # Pass numpy action
                done = bool(terminated or truncated)

                # Store the [1,] numpy action
                self.buffer.add(o.squeeze(0).cpu().numpy(), action_np, logp, reward, float(value), done)

                # update hidden state
                _, h_next = self.actor.step(o, h_states[i])
                h_states[i] = h_next

                if done:
                    no, _ = self.envs[i].reset()
                    obs_t[i] = torch.as_tensor(no, dtype=torch.float32, device=self.device)
                    h_states[i] = None # Reset hidden state
                else:
                    obs_t[i] = torch.as_tensor(next_o, dtype=torch.float32, device=self.device)

        # Store the persistent obs and hidden states
        self.current_obs_t = obs_t
        self.h_states = h_states

        last_values = []
        for i in range(self.num_envs):
            o = obs_t[i].unsqueeze(0)
            with torch.no_grad():
                v, _ = self.critic.step(o, h_states[i])
            last_values.append(float(v))

        rollout = self.buffer.get()
        return rollout, last_values


    def update(self, rollout: Rollout, last_values):
        N = self.num_envs
        T = self.rollout_length
        B = rollout.obs.shape[0]

        # --- GAE: Reshape, Loop, Flatten ---
        # 1. Reshape the flat buffer (which is [N*T, ...]) into [N, T, ...]
        try:
            obs = rollout.obs.view(N, T, *rollout.obs.shape[1:])
            actions = rollout.actions.view(N, T, *rollout.actions.shape[1:])
            rewards = rollout.rewards.view(N, T)
            values = rollout.values.view(N, T)
            dones = rollout.dones.view(N, T)
        except RuntimeError as e:
            print(f"Buffer/Reshape Error: {e}. Buffer size {B}, expected {N*T}.")
            self.buffer.reset()
            return {}

        # 2. Compute GAE per environment
        all_adv = []
        all_ret = []
        for i in range(N):
            last_v = torch.tensor(last_values[i], device=self.device, dtype=torch.float32)
            adv_i, ret_i = compute_gae(rewards[i], values[i], dones[i], last_v, gamma=self.gamma, lam=self.lam)
            all_adv.append(adv_i)
            all_ret.append(ret_i)
        
        # 3. Flatten *after* GAE calculation
        adv = torch.cat(all_adv).view(-1)
        returns = torch.cat(all_ret).view(-1)
        
        # 4. Flatten obs/actions for the update
        obs_flat = obs.view(-1, *obs.shape[2:]) # [800, 6]
        acts_flat = actions.view(-1, *actions.shape[2:]) # [800, 1]
        # --- END GAE ---

        # normalize advantages
        adv_mean = adv.mean()
        adv_std = adv.std(unbiased=False) + 1e-8
        adv = (adv - adv_mean) / adv_std
        
        if self.clip_adv_by_std:
            adv = torch.clamp(adv, -self.advantage_clip / adv_std, self.advantage_clip / adv_std)
        else:
            adv = torch.clamp(adv, -self.advantage_clip, self.advantage_clip)

        # Actor forward
        dists, _ = self.actor.forward(obs_flat)
        
        # `dists` is [800, 1] and `acts_flat` is [800, 1]. No squeeze needed.
        new_logps = dists.log_prob(acts_flat)

        entropy = dists.entropy().mean()
        actor_loss = - (new_logps * adv).mean()  # A2C loss

        # compute CVaR penalty using hybrid critic (if present)
        cvar_pen = 0.0
        loss_quantile = torch.tensor(0.0, device=self.device)
        loss_gpd = torch.tensor(0.0, device=self.device)
        cvar_est_mean = 0.0
        b_est = torch.tensor(0.0, device=self.device)
        
        stats_u, stats_scale, stats_shape, stats_tail_pct = 0.0, 0.0, 0.0, 0.0

        if self.quantile_critic is not None:
            # forward quantile critic on obs_flat (state_ctx)
            u, body_q, scale, shape = self.quantile_critic(obs_flat)  # u:[B,1]
            cvar_est = estimate_cvar_from_gpd(u, scale, shape, alpha=self.cvar_alpha)  # [B]
            cvar_pen = - self.cvar_lambda * cvar_est.mean()
            cvar_est_mean = float(cvar_est.mean().item())
            
            stats_u = float(u.mean().item())
            stats_scale = float(scale.mean().item())
            stats_shape = float(shape.mean().item())

        total_actor_loss = actor_loss - self.entropy_coef * entropy + cvar_pen

        # actor update
        self.opt_actor.zero_grad()
        total_actor_loss.backward(retain_graph=True)
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        self.opt_actor.step()

        # Critic (value) update
        preds = self.critic.forward(obs_flat)
        value_loss = F.mse_loss(preds, returns)
        total_critic_loss = value_loss

        if self.quantile_critic is not None:
            detached_returns = returns.detach()  # [B]
            detached_values = preds.detach()    # for b estimation

            u_flat = u.view(-1).detach()  # detached threshold
            
            # --- IMPROVED: Fallback to percentile if u_flat is miscalibrated ---
            # Check if the quantile critic's u makes sense; if not, use empirical percentile
            tail_percentile = self.cvar_alpha  # e.g., 0.05 for 5% tail
            u_empirical = torch.quantile(detached_returns, tail_percentile)
            
            # Use quantile critic's u only if it's within reasonable bounds
            # (i.e., between min and max of returns, roughly)
            u_min = detached_returns.min()
            u_max = detached_returns.max()
            u_mean = detached_returns.mean()
            
            # Check if u_flat is "reasonable" (somewhere in the distribution)
            u_is_reasonable = (u_flat.mean() >= u_min) and (u_flat.mean() <= u_max)
            
            if not u_is_reasonable:
                if self.verbose:
                    print(f"\n    [WARNING] u_flat={u_flat.mean():.4f} outside return range [{u_min:.4f}, {u_max:.4f}]")
                    print(f"    [FALLBACK] Using empirical {tail_percentile*100:.1f}%-percentile: {u_empirical:.4f}")
                u_flat = torch.ones_like(u_flat) * u_empirical
            
            is_tail = (detached_returns < u_flat)
            
            # --- Debug output (only if verbose) ---
            if self.verbose:
                print(f"\n=== GPD Loss Debug (Update) ===")
                print(f"    detached_returns: min={detached_returns.min():.6f}, max={detached_returns.max():.6f}, mean={detached_returns.mean():.6f}, std={detached_returns.std():.6f}")
                print(f"    u_flat (threshold): min={u_flat.min():.6f}, max={u_flat.max():.6f}, mean={u_flat.mean():.6f}")
                print(f"    tail_count: {is_tail.sum().item()} / {is_tail.numel()} ({100*is_tail.float().mean():.2f}%)")
                print(f"    u compared to return stats:")
                print(f"      - returns < u_min? {(detached_returns < u_flat.min()).sum().item()} samples")
                print(f"      - returns < u_mean? {(detached_returns < u_flat.mean()).sum().item()} samples")
                print(f"      - returns < u_max? {(detached_returns < u_flat.max()).sum().item()} samples")

            is_body = ~is_tail
            
            stats_tail_pct = float(is_tail.float().mean().item())

            # Generalized Quantile Huber Loss for body
            b_est = estimate_noise_disparity(detached_returns, detached_values)
            try:
                b_est = b_est.reshape(())
            except Exception:
                b_est = b_est.view(-1)[0:1]

            if is_body.any():
                mask_body = is_body
                loss_quantile = generalized_quantile_huber_loss(body_q, detached_returns, b_est, mask=mask_body)
            else:
                loss_quantile = torch.tensor(0.0, device=self.device)

            # GPD loss on tail using vectorized per-sample params
            tail_idx = is_tail.nonzero(as_tuple=False).view(-1)
            if tail_idx.numel() > 0:
                u_tail = u_flat[tail_idx]                # [K], detached
                r_tail = detached_returns[tail_idx]      # [K]
                y_tail = (u_tail - r_tail).clamp(min=1e-12)  # [K]

                if self.verbose and loss_gpd.item() == 0.0: 
                    print(f"\n--- TAIL EVENT DETECTED ---")
                    print(f"    Num tail samples: {tail_idx.numel()} (out of {B})")
                    print(f"    Sample y_tail (exceedances): {y_tail.cpu().numpy()[:5]}")
                    print(f"    Mean u: {u_tail.mean().item():.4f}, Mean r_tail: {r_tail.mean().item():.4f}\n")
                    
                shape_tail = shape.view(-1, 1)[tail_idx]  # [K,1]
                scale_tail = scale.view(-1, 1)[tail_idx]  # [K,1]

                loss_gpd = gpd_log_likelihood_loss_vectorized(shape_tail, scale_tail, y_tail)
            else:
                loss_gpd = torch.tensor(0.0, device=self.device)

            total_critic_loss = total_critic_loss + self.quantile_weight * loss_quantile + self.gpd_loss_weight * loss_gpd

        # critic update
        self.opt_critic.zero_grad()
        total_critic_loss.backward()
        nn.utils.clip_grad_norm_(critic_params(self), self.max_grad_norm)
        self.opt_critic.step()

        # reset buffer
        self.buffer.reset()

        stats = {
            'actor_loss': float(total_actor_loss.item()),
            'value_loss': float(value_loss.item()),
            'gqhl_loss': float(loss_quantile.item()),
            'gpd_loss': float(loss_gpd.item()),
            'estimated_b': float(b_est.item()),
            'entropy': float(entropy.item()),
            'cvar_pen': float(cvar_pen.item()),
            'cvar_est_mean': float(cvar_est_mean),
            'tail_percent': stats_tail_pct,
            'mean_u': stats_u,
            'mean_gpd_scale': stats_scale,
            'mean_gpd_shape': stats_shape
        }
        return stats


def train_loop(trainer: A2CTrainer,
               total_updates: int = 1000,
               eval_interval: int = 50,
               save_interval: int = 50,
               csv_path: str = None):
    """
    Runs training for `total_updates` updates with clean tqdm progress bar.
    """
    maybe_mkdir(trainer.logdir)
    writer = None
    csvfile = None
    if csv_path is not None:
        os.makedirs(os.path.dirname(csv_path) or '.', exist_ok=True)
        csvfile = open(csv_path, 'w', newline='')
        writer = None

    logs = []
    start_time = time.time()
    
    # Running aggregators for metrics display
    tail_events_count = 0
    gpd_loss_sum = 0
    value_loss_sum = 0
    actor_loss_sum = 0
    update_count = 0
    
    with tqdm(total=total_updates, desc="Training", unit="update") as pbar:
        for update in range(total_updates):
            # anneal cvar_lambda over first 500 updates
            if update < 500:
                trainer.cvar_lambda = 0.1 + 0.9 * (update / 500.0)
            else:
                trainer.cvar_lambda = 1.0

            rollout, last_values = trainer.collect_rollouts()
            
            # Check if buffer was empty or not full, skip update if so
            if rollout.obs.shape[0] < trainer.num_envs * trainer.rollout_length:
                pbar.update(1)
                continue
                
            stats = trainer.update(rollout, last_values)
            if not stats: # Check if update was skipped
                pbar.update(1)
                continue
                
            stats['update'] = update
            stats['time_elapsed_s'] = time.time() - start_time
            logs.append(stats)

            # CSV write
            if writer is None and csvfile is not None:
                writer = csv.DictWriter(csvfile, fieldnames=sorted(stats.keys()))
                writer.writeheader()
            if writer is not None:
                writer.writerow(stats)
                csvfile.flush()

            # Accumulate metrics
            tail_events_count += int(stats.get('tail_percent', 0) * trainer.num_envs * trainer.rollout_length / 100)
            gpd_loss_sum += stats.get('gpd_loss', 0)
            value_loss_sum += stats.get('value_loss', 0)
            actor_loss_sum += stats.get('actor_loss', 0)
            update_count += 1

            # optional evaluation
            if update % eval_interval == 0 and trainer.eval_fn is not None:
                trainer.eval_fn(trainer.actor)

            # Save final model after every update (overwrites the same file)
            checkpoint = {
                'actor': trainer.actor.state_dict(),
                'critic': trainer.critic.state_dict(),
            }
            if trainer.quantile_critic is not None:
                checkpoint['quantile_critic'] = trainer.quantile_critic.state_dict()
            
            final_path = os.path.join(trainer.logdir, 'checkpoint_final.pth')
            torch.save(checkpoint, final_path)

            # Update progress bar with metrics every 10%
            progress_pct = (update + 1) / total_updates
            if progress_pct == 1.0 or (update + 1) % max(1, total_updates // 10) == 0:
                avg_actor_loss = actor_loss_sum / max(1, update_count)
                avg_value_loss = value_loss_sum / max(1, update_count)
                avg_gpd_loss = gpd_loss_sum / max(1, update_count)
                
                metrics_str = (
                    f"Actor Loss: {avg_actor_loss:.4f} | "
                    f"Value Loss: {avg_value_loss:.4f} | "
                    f"GPD Loss: {avg_gpd_loss:.4f} | "
                    f"Tail Events: {tail_events_count:,}"
                )
                pbar.set_postfix_str(metrics_str)
                
                # Reset accumulators for next 10%
                if progress_pct < 1.0:
                    tail_events_count = 0
                    gpd_loss_sum = 0
                    value_loss_sum = 0
                    actor_loss_sum = 0
                    update_count = 0
            
            pbar.update(1)

    if csvfile is not None:
        csvfile.close()
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE ✓")
    print("="*80)
    print(f"Total updates: {total_updates}")
    print(f"Time elapsed: {(time.time() - start_time) / 3600:.2f} hours")
    print(f"Model saved: {os.path.join(trainer.logdir, 'checkpoint_final.pth')}")
    print(f"Log saved: {csv_path}")
    print("="*80)
    
    return logs