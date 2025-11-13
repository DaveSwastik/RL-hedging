# run_smoke.py
import os
import torch
import numpy as np
import pprint
import warnings
warnings.filterwarnings("ignore")

# ensure src/ is importable when running from repo root
# (if you run from a different cwd, uncomment and adjust PYTHONPATH)
# import sys
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))



from src.agents.trainer_a2c import A2CTrainer, train_loop
from src.agents.models import MlpActor, MlpCritic
from src.agents.cvar_ctitic import ExDrlCritic
from src.envs.hedging_env import HedgingEnv
import yaml

# -------------------------
# Small helper: load config (or fallback)
# -------------------------
cfg_file = os.path.join('src', 'configs', 'default.yaml')
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
        'environment': {'trading_cost': 0.001, 'reward_fn':'squared_error'},
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
        'reward_scale': cfg['simulator'].get('reward_scale', 1e4),
         # --- add synthetic shocks for tail testing (debug only) ---
        'shock_prob': 1,  # 100% of paths
        'shock_scale': 1.0  # exp(-1.0) ~ 63% price drop
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
# Smoke test settings
# -------------------------
n_envs = 16
rollout_length = 50
device = 'cpu'

# Build a temporary env to infer shapes
_tmp_env = make_env()
obs_dim = _tmp_env.observation_space.shape[0]
hidden_dim = cfg['model'].get('hidden_dim', 128)

# instantiate MLP models (compatible with batch-based trainer)
actor = MlpActor(obs_dim=obs_dim, hidden_dim=hidden_dim, action_dim=1, state_dependent_std=False)
critic = MlpCritic(obs_dim=obs_dim, hidden_dim=hidden_dim)
quantile_critic = ExDrlCritic(state_dim=obs_dim, hidden_dim=hidden_dim, n_body_quantiles=16)

# trainer
trainer = A2CTrainer(
    env_fn=make_env,
    actor=actor,
    critic=critic,
    quantile_critic=quantile_critic,
    device=device,
    rollout_length=rollout_length,
    num_envs=n_envs,
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
    logdir='./runs/smoke',
    cvar_alpha=0.05
)

print("=== Running smoke test: collect_rollouts() ===")
rollout, last_values = trainer.collect_rollouts()
print("Buffer size:", rollout.obs.shape)
print("Sample obs[0]:", rollout.obs[0])
print("Sample rewards[0:5]:", rollout.rewards[:5].cpu().numpy())

print("\n=== Running smoke test: trainer.update() ===")
stats = trainer.update(rollout, last_values)
print("\n=== Update stats ===")
pprint.pprint(stats)

# Sanity checks
def sanity_checks(stats, actor, critic, quantile_critic):
    ok = True
    # NaN checks
    for k,v in stats.items():
        if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
            print(f"BAD: stat {k} is NaN/Inf -> {v}")
            ok = False

    # forward pass shapes for MLP models
    with torch.no_grad():
        dummy_obs_flat = torch.as_tensor(np.zeros((1, obs_dim), dtype=np.float32))
        # actor: forward expects [B, obs_dim]
        adist, ctx = actor.forward(dummy_obs_flat)
        v = critic.forward(dummy_obs_flat)
        # quantile_critic expects [B, state_dim]
        u, body_q, scale, shape = quantile_critic(dummy_obs_flat)

        print("Actor dist mean shape:", adist.mean.shape if hasattr(adist, 'mean') else 'n/a')
        print("Critic value shape:", v.shape)
        print("Hybrid critic outputs shapes: u", u.shape, "body_q", body_q.shape, "scale", scale.shape, "shape", shape.shape)

    return ok

ok = sanity_checks(stats, actor, critic, quantile_critic)
print("\nSanity checks passed?" , ok)
 