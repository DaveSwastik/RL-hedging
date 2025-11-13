# run_training.py
import os
import torch
import numpy as np
import pprint
import warnings
warnings.filterwarnings("ignore")

from src.agents.trainer_a2c import A2CTrainer, train_loop
from src.agents.models import MlpActor, MlpCritic
from src.agents.cvar_ctitic import ExDrlCritic
from src.envs.hedging_env import HedgingEnv
import yaml

# -------------------------
# Load config
# -------------------------
cfg_file = os.path.join('src', 'configs', 'default.yaml')
if os.path.exists(cfg_file):
    with open(cfg_file, 'r') as f:
        cfg = yaml.safe_load(f)
else:
    # Fallback config (if default.yaml is missing)
    cfg = {
        'simulator': {
            'S0': 100.0, 'v0': 0.04, 'dt': 0.004, 'steps': 20, 'rho': -0.7,
            'regimes': [{'name':'Low Vol','theta':0.04,'kappa':3.0,'sigma_v':0.2,'mu':0.05}],
            'trans_mat': [[1.0]]
        },
        'option': {'type':'Asian','K':100.0,'maturity':0.08,'option_type':'call'},
        'environment': {'trading_cost': 0.001},
        'model': {'hidden_dim':128}
    }

# env factory
def make_env():
    # --- simulator setup ---
    sim_params = {
        'S0': cfg['simulator']['S0'],
        'v0': cfg['simulator']['v0'],
        'dt': cfg['simulator']['dt'],
        'steps': cfg['simulator']['steps'],
        'rho': cfg['simulator'].get('rho', -0.7),
        'regimes': cfg['simulator'].get('regimes', []),
        'trans_mat': cfg['simulator'].get('trans_mat', []),
        'r': cfg['simulator'].get('r', 0.0),
        
        # --- Reward Scale: Amplify small per-step P&L for learning ---
        # Per-step P&L is typically 1e-5 to 1e-4. Without scaling, agent receives
        # tiny, noisy rewards. Using 100-1000 amplifies signal.
        'reward_scale': cfg['simulator'].get('reward_scale', 100.0),
        
        # --- Shocks: KEEP ON for tail event training ---
        # CVaR/GPD loss needs tail samples. Shocks guarantee tail events during training.
        # Use realistic frequency (5%) and severity.
        'shock_prob': cfg['simulator'].get('shock_prob', 0.05),   # 5% of paths
        'shock_scale': cfg['simulator'].get('shock_scale', 1.0)   # Moderate severity
    }
    
    option_spec = {
        'type': cfg['option']['type'],
        'K': cfg['option']['K'],
        'maturity': cfg['option']['maturity'],
        'option_type': cfg['option'].get('option_type', 'call'),
        'notional': cfg['option'].get('notional', 1.0)
    }

    env = HedgingEnv(
        sim_params=sim_params,
        option_spec=option_spec,
        trading_cost=cfg['environment'].get('trading_cost', 0.0)
    )
    return env


# -------------------------
# Training settings
# -------------------------
n_envs = 16
rollout_length = 50
device = 'cpu'
hidden_dim = cfg['model'].get('hidden_dim', 128)

# --- *** CHANGE 3: Set full training length & log path *** ---
total_updates = 5000  # Start with 5k, increase later if needed
logdir = './runs/training_run_01'
csv_path = os.path.join(logdir, 'training_log.csv')


# Build a temporary env to infer shapes
_tmp_env = make_env()
obs_dim = _tmp_env.observation_space.shape[0]

# instantiate MLP models
actor = MlpActor(obs_dim=obs_dim, hidden_dim=hidden_dim, action_dim=1, state_dependent_std=False)
critic = MlpCritic(obs_dim=obs_dim, hidden_dim=hidden_dim)
quantile_critic = ExDrlCritic(state_dim=obs_dim, hidden_dim=hidden_dim, n_body_quantiles=32)

# trainer
trainer = A2CTrainer(
    env_fn=make_env,
    actor=actor,
    critic=critic,
    quantile_critic=quantile_critic,
    device=device,
    rollout_length=rollout_length,
    num_envs=n_envs,
    lr_actor=1e-4,              # Increased: faster learning early
    lr_critic=5e-4,             # Increased: critic needs estimates early
    gamma=0.995,
    lam=0.95,
    entropy_coef=0.05,          # Increased: more exploration early
    value_coef=0.5,
    cvar_lambda=1.0,            # Annealed by train_loop (0.1 -> 1.0 over first 500 updates)
    quantile_weight=1.0,        # Weight for quantile body loss
    gpd_loss_weight=1.0,        # Weight for GPD tail loss
    max_grad_norm=0.5,
    advantage_clip=10.0,
    clip_adv_by_std=False,
    logdir=logdir,
    cvar_alpha=0.05             # 5% tail for CVaR
)

# --- *** This makes the script runnable *** ---
if __name__ == "__main__":
    print(f"Starting training run: {total_updates} updates")
    print(f"Logging to: {csv_path}")
    print(f"With shocks: shock_prob=0.05, shock_scale=1.0")
    print(f"Reward scale: 100.0x")
    print(f"\n✓ Live model will be saved to: {logdir}/checkpoint_final.pth")
    print(f"  (single file, updated after every training update)\n")
    
    train_loop(
        trainer, 
        total_updates=total_updates, 
        eval_interval=100, 
        save_interval=100,        # Save more frequently
        csv_path=csv_path
    )
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"✓ Final model saved in: {logdir}/checkpoint_final.pth")
    print(f"✓ Training log saved in: {csv_path}")
    print(f"✓ Total updates completed: {total_updates}")
    print(f"\nTo evaluate the trained model:")
    print(f"  python load_and_eval.py")
    print("="*60)