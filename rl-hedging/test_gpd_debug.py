"""
Quick debug test to see GPD loss computation
"""
import torch
import os
import sys
sys.path.insert(0, '.')

from src.agents.trainer_a2c import A2CTrainer
from src.agents.models import MlpActor, MlpCritic
from src.agents.cvar_ctitic import ExDrlCritic
from src.envs.hedging_env import HedgingEnv
import yaml

cfg_file = 'src/configs/default.yaml'
if os.path.exists(cfg_file):
    with open(cfg_file, 'r') as f:
        cfg = yaml.safe_load(f)
else:
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

def make_env():
    sim_params = {
        'S0': cfg['simulator']['S0'],
        'v0': cfg['simulator']['v0'],
        'dt': cfg['simulator']['dt'],
        'steps': cfg['simulator']['steps'],
        'rho': cfg['simulator'].get('rho', -0.7),
        'regimes': cfg['simulator'].get('regimes', []),
        'trans_mat': cfg['simulator'].get('trans_mat', []),
        'r': cfg['simulator'].get('r', 0.0),
        'reward_scale': cfg['simulator'].get('reward_scale', 1e4),
    }
    option_spec = {
        'type': cfg['option']['type'],
        'K': cfg['option']['K'],
        'maturity': cfg['option']['maturity'],
        'option_type': cfg['option'].get('option_type', 'call'),
        'notional': cfg['option'].get('notional', 1.0)
    }
    env = HedgingEnv(sim_params=sim_params, option_spec=option_spec, trading_cost=cfg['environment'].get('trading_cost', 0.0))
    return env

_tmp_env = make_env()
obs_dim = _tmp_env.observation_space.shape[0]
hidden_dim = cfg['model'].get('hidden_dim', 128)

actor = MlpActor(obs_dim=obs_dim, hidden_dim=hidden_dim, action_dim=1, state_dependent_std=False)
critic = MlpCritic(obs_dim=obs_dim, hidden_dim=hidden_dim)
quantile_critic = ExDrlCritic(state_dim=obs_dim, hidden_dim=hidden_dim, n_body_quantiles=16)

trainer = A2CTrainer(
    env_fn=make_env,
    actor=actor,
    critic=critic,
    quantile_critic=quantile_critic,
    device='cpu',
    rollout_length=10,  # SMALL for speed
    num_envs=2,  # SMALL for speed
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
    cvar_alpha=0.05
)

print("Collecting rollouts...")
rollout, last_values = trainer.collect_rollouts()
print(f"Rollout shape: {rollout.obs.shape}")

print("\nRunning update (watch for debug output)...")
stats = trainer.update(rollout, last_values)

print("\n" + "="*60)
print("FINAL STATS:")
print("="*60)
print(f"gpd_loss: {stats['gpd_loss']:.6f}")
print(f"tail_percent: {stats['tail_percent']:.6f}")
print(f"mean_u: {stats['mean_u']:.6f}")
print(f"gqhl_loss: {stats['gqhl_loss']:.6f}")
