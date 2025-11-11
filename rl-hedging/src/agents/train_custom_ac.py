# src/agents/train_custom_ac.py
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import yaml
from tqdm import tqdm
from pathlib import Path

from src.envs.hedging_env import HedgingEnv 
from src.agents.models import InterpretableHedger, Critic
from src.utils.payoffs import PAYOFF_FUNCTIONS 

def load_config(path='src/configs/default.yaml'):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def explained_variance(y, y_pred):
    """ Calculates explained variance (1 - Var(residual) / Var(y)) """
    var_y = torch.var(y)
    return 1 - torch.var(y - y_pred) / (var_y + 1e-8)

@torch.inference_mode()
def evaluate(actor, env, device, n_eval_episodes=10):
    actor.eval()
    total_abs_error = 0.0
    for _ in range(n_eval_episodes):
        obs, _ = env.reset()
        obs_t = torch.from_numpy(obs).to(device=device, dtype=torch.float32).view(1, 1, -1)
        done = False
        h_a = None
        while not done:
            dist, h_a = actor.step(obs_t, h_a)
            a = torch.tanh(dist.base.mean) 
            a_np = a.detach().cpu().numpy().flatten()
            
            obs, _, terminated, truncated, info = env.step(a_np)
            done = terminated or truncated
            obs_t = torch.from_numpy(obs).to(device=device, dtype=torch.float32).view(1, 1, -1)
            if done:
                total_abs_error += abs(info['pnl'] - info['payoff'])
    actor.train()
    return total_abs_error / n_eval_episodes

def train(config_path: str = 'src/configs/default.yaml',
          save_path: str = 'models/custom_a2c_hedger_cpu_hiddenstep.pth'):
    
    cfg = load_config(config_path)

    torch.manual_seed(42)
    np.random.seed(42)

    device = torch.device("cpu")
    print(f"Using device: {device}")
    try:
        torch.set_num_threads(12)  
    except Exception:
        pass

    # --- FIX: Pass trade_penalty from config to env ---
    env_kwargs = {
        'sim_params': cfg['simulator'],
        'option_spec': cfg['option'],
        'trading_cost': cfg['environment']['trading_cost'],
        'trade_penalty': cfg['environment'].get('trade_penalty', 0.02) # Default 0.02
    }
    env = HedgingEnv(**env_kwargs)
    # --- END FIX ---
    
    obs_dim = env.observation_space.shape[0]

    actor = InterpretableHedger(obs_dim=obs_dim, hidden_dim=64).to(device)
    critic = Critic(obs_dim=obs_dim, hidden_dim=64).to(device)

    actor_lr = 1e-5
    critic_lr = 3e-4
    actor_opt = optim.AdamW(actor.parameters(), lr=actor_lr, weight_decay=1e-4)
    critic_opt = optim.AdamW(critic.parameters(), lr=critic_lr, weight_decay=1e-4)

    episodes = cfg['training'].get('episodes', 20000) 
    gamma = cfg['training'].get('gamma', 0.99)
    lam = cfg['training'].get('gae_lambda', 0.95) 
    ent_coef = 0.01
    max_steps = cfg['simulator']['steps'] + 5 

    sched_a = optim.lr_scheduler.CosineAnnealingLR(actor_opt, T_max=episodes, eta_min=actor_lr*0.1)
    sched_c = optim.lr_scheduler.CosineAnnealingLR(critic_opt, T_max=episodes, eta_min=critic_lr*0.1)

    best_eval = float("inf")
    save_best = Path(str(save_path).replace(".pth", "_best.pth"))

    print(f"--- A2C+GAE (CPU, hidden-step) --- Episodes={episodes} max_steps={max_steps} hidden=64")
    print(f"--- Hyperparams: ActorLR={actor_lr} | CriticLR={critic_lr} | EntropyCoeff={ent_coef} ---")

    for ep in tqdm(range(episodes), desc="Training"):
        obs, _ = env.reset(seed=ep)
        obs_t = torch.from_numpy(obs).to(device=device, dtype=torch.float32).view(1, 1, -1)

        h_a = None
        h_c = None
        logps, vals, rews, dns, ents = [], [], [], [], []
        done = False
        steps = 0
        
        # --- FIX: Initialize error and turnover ---
        episode_final_error = 0.0 
        turnover = 0.0
        prev_pos = 0.0
        # --- END FIX ---

        while not done and steps < max_steps:
            dist, h_a = actor.step(obs_t, h_a)
            v, h_c = critic.step(obs_t, h_c)
            a = dist.rsample() 
            a_np = a.detach().numpy().flatten()
            
            # --- FIX: Track turnover ---
            new_pos_scalar = float(a_np[0])
            turnover += abs(new_pos_scalar - prev_pos)
            prev_pos = new_pos_scalar
            # --- END FIX ---

            obs, env_reward, terminated, truncated, info = env.step(a_np)
            done = terminated or truncated
            
            logps.append(dist.log_prob(a).squeeze())
            vals.append(v.squeeze())
            ents.append(dist.entropy().squeeze())
            dns.append(torch.tensor(float(done)))
            rews.append(torch.tensor(float(env_reward), dtype=torch.float32))
            
            if done:
                episode_final_error = info.get('hedging_error', 0.0)
            
            obs_t = torch.from_numpy(obs).to(device=device, dtype=torch.float32).view(1, 1, -1)
            steps += 1

        logps = torch.stack(logps)       
        vals  = torch.stack(vals)        
        rews  = torch.stack(rews)        
        dns   = torch.stack(dns)         
        ents  = torch.stack(ents)        

        # ---- GAE(λ) ----
        vals_detached = vals.detach() 
        next_v = torch.zeros(1, dtype=vals_detached.dtype)
        adv = torch.zeros_like(rews)
        gae = 0.0
        for t in reversed(range(len(rews))):
            delta = rews[t] + gamma * next_v * (1.0 - dns[t]) - vals_detached[t]
            gae = delta + gamma * lam * (1.0 - dns[t]) * gae
            adv[t] = gae
            next_v = vals_detached[t]
            
        ret = adv + vals_detached 
        
        adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)
        adv = adv.clamp(-3.0, 3.0) # Stricter clipping

        # losses
        actor_loss = -(logps * adv.detach()).mean() 
        entropy = ents.mean()
        total_actor_loss = actor_loss - ent_coef * entropy

        critic_loss = F.smooth_l1_loss(vals, ret) # Huber Loss

        # update
        actor_opt.zero_grad()
        total_actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
        actor_opt.step()

        critic_opt.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
        critic_opt.step()

        sched_a.step()
        sched_c.step()

        if (ep + 1) % 100 == 0:
            ep_r_sum = rews.sum().item()
            ev = explained_variance(ret, vals).item()
            # --- FIX: Log turnover and entropy ---
            print(f"Ep {ep+1}: Act {actor_loss.item():.4f} | Crt {critic_loss.item():.4f} | R_Sum {ep_r_sum:.3f} | EV {ev:.2f} | |err| {abs(episode_final_error):.4f} | Turn {turnover:.2f} | Ent {entropy.item():.3f}")

        if (ep + 1) % 1000 == 0:
            eval_err = evaluate(actor, env, device)
            if eval_err < best_eval:
                best_eval = eval_err
                Path(save_best).parent.mkdir(parents=True, exist_ok=True)
                torch.save(actor.state_dict(), save_best)
                print(f"*** New best |eval err| = {eval_err:.4f} ***")

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(actor.state_dict(), save_path)
    print(f"--- Training Complete (CPU hidden-step) ---\nSaved to {save_path}")
    
if __name__ == "__main__":
    train()