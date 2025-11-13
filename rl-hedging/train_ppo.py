# train_ppo.py
import os
import torch
import numpy as np
import pprint
import warnings
import yaml
from functools import partial

# --- Import from your src ---
from src.envs.hedging_env import HedgingEnv
from src.utils.bs import bs_price_and_delta

# --- Import Baselines ---
try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.utils import set_random_seed
except ImportError:
    print("Error: stable_baselines3 not installed.")
    print("Please run: poetry add stable-baselines3-contrib")
    exit()

warnings.filterwarnings("ignore")

# -------------------------------------------------------------------
# --- 1. CONFIGURATION ---
# -------------------------------------------------------------------

# Set the number of parallel environments to run
N_ENVS = 16

# This is the main knob for training time. 10,000 is very fast (< 5 mins).
# For a full run, you would use 1,000,000 or more.
TOTAL_TIMESTEPS = 10_000 

# Path to your config file
CFG_FILE = os.path.join('src', 'configs', 'default.yaml')

# Where to save the new PPO model
SAVE_PATH = "models/ppo_hedge_new.zip"

# -------------------------------------------------------------------

def load_config(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found at {path}")
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def make_env(cfg):
    """Factory function to create the environment for evaluation."""
    sim_params = {
        'S0': cfg['simulator']['S0'],
        'v0': cfg['simulator']['v0'],
        'dt': cfg['simulator']['dt'],
        
        # --- *** THIS IS THE FIX *** ---
        # Changed cfg('simulator') to cfg['simulator']
        'steps': cfg['simulator'].get('steps', 20), 
        # --- *** END FIX *** ---
        
        'rho': cfg['simulator'].get('rho', -0.7),
        'regimes': cfg['simulator'].get('regimes', []),
        'trans_mat': cfg['simulator'].get('trans_mat', []),
        'r': cfg['simulator'].get('r', 0.0),
        
        # --- Reward Scale ---
        'reward_scale': cfg['simulator'].get('reward_scale', 100.0),
        
        # --- Shocks ---
        'shock_prob': 0.0, 
        'shock_scale': 0.0
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

# -------------------------------------------------------------------
# --- 2. MAIN TRAINING BLOCK ---
# -------------------------------------------------------------------
if __name__ == "__main__":
    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load config
    cfg = load_config(CFG_FILE)
    
    # Create the vectorized environment
    # SB3's make_vec_env handles running N_ENVS in parallel
    print(f"Creating {N_ENVS} parallel environments...")
    env_fn = partial(make_env, cfg=cfg)
    vec_env = make_vec_env(env_fn, n_envs=N_ENVS)

    # --- PPO Model Configuration ---
    # n_steps * n_envs = total samples per update.
    # 100 * 16 = 1600 samples. This is a good, fast batch size.
    # Your episode length is 20, so 100 steps = 5 episodes per env.
    model = PPO(
        "MlpPolicy",
        vec_env,
        n_steps=100,         # Rollout length per env
        batch_size=64,         # Mini-batch size for updates
        n_epochs=10,           # How many epochs to train on each rollout
        gamma=0.995,
        gae_lambda=0.95,
        ent_coef=0.01,
        learning_rate=3e-4,    # PPO typically uses a larger LR than A2C
        verbose=1,
        device=device
    )

    # --- Run the Training ---
    print(f"Starting PPO training for {TOTAL_TIMESTEPS} total timesteps...")
    print(f"This will be very fast (approx. {TOTAL_TIMESTEPS // (N_ENVS * model.n_steps)} updates).")
    
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        progress_bar=True
    )
    
    # --- Save the Model ---
    os.makedirs("models", exist_ok=True)
    model.save(SAVE_PATH)
    
    print("\n--- PPO Training Complete ---")
    print(f"Model saved to: {SAVE_PATH}")
    
    vec_env.close()# train_ppo.py
import os
import torch
import numpy as np
import pprint
import warnings
import yaml
from functools import partial

# --- Import from your src ---
from src.envs.hedging_env import HedgingEnv
from src.utils.bs import bs_price_and_delta

# --- Import Baselines ---
try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.utils import set_random_seed
except ImportError:
    print("Error: stable_baselines3 not installed.")
    print("Please run: poetry add stable-baselines3-contrib")
    exit()

warnings.filterwarnings("ignore")

# -------------------------------------------------------------------
# --- 1. CONFIGURATION ---
# -------------------------------------------------------------------

# Set the number of parallel environments to run
N_ENVS = 16

# This is the main knob for training time. 10,000 is very fast (< 5 mins).
# For a full run, you would use 1,000,000 or more.
TOTAL_TIMESTEPS = 10_000 

# Path to your config file
CFG_FILE = os.path.join('src', 'configs', 'default.yaml')

# Where to save the new PPO model
SAVE_PATH = "models/ppo_hedge_new.zip"

# -------------------------------------------------------------------

def load_config(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found at {path}")
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def make_env(cfg):
    """Factory function to create the environment for evaluation."""
    sim_params = {
        'S0': cfg['simulator']['S0'],
        'v0': cfg['simulator']['v0'],
        'dt': cfg['simulator']['dt'],
        
        # --- *** THIS IS THE FIX *** ---
        # Changed cfg('simulator') to cfg['simulator']
        'steps': cfg['simulator'].get('steps', 20), 
        # --- *** END FIX *** ---
        
        'rho': cfg['simulator'].get('rho', -0.7),
        'regimes': cfg['simulator'].get('regimes', []),
        'trans_mat': cfg['simulator'].get('trans_mat', []),
        'r': cfg['simulator'].get('r', 0.0),
        
        # --- Reward Scale ---
        'reward_scale': cfg['simulator'].get('reward_scale', 100.0),
        
        # --- Shocks ---
        'shock_prob': 0.0, 
        'shock_scale': 0.0
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

# -------------------------------------------------------------------
# --- 2. MAIN TRAINING BLOCK ---
# -------------------------------------------------------------------
if __name__ == "__main__":
    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load config
    cfg = load_config(CFG_FILE)
    
    # Create the vectorized environment
    # SB3's make_vec_env handles running N_ENVS in parallel
    print(f"Creating {N_ENVS} parallel environments...")
    env_fn = partial(make_env, cfg=cfg)
    vec_env = make_vec_env(env_fn, n_envs=N_ENVS)

    # --- PPO Model Configuration ---
    # n_steps * n_envs = total samples per update.
    # 100 * 16 = 1600 samples. This is a good, fast batch size.
    # Your episode length is 20, so 100 steps = 5 episodes per env.
    model = PPO(
        "MlpPolicy",
        vec_env,
        n_steps=100,         # Rollout length per env
        batch_size=64,         # Mini-batch size for updates
        n_epochs=10,           # How many epochs to train on each rollout
        gamma=0.995,
        gae_lambda=0.95,
        ent_coef=0.01,
        learning_rate=3e-4,    # PPO typically uses a larger LR than A2C
        verbose=1,
        device=device
    )

    # --- Run the Training ---
    print(f"Starting PPO training for {TOTAL_TIMESTEPS} total timesteps...")
    print(f"This will be very fast (approx. {TOTAL_TIMESTEPS // (N_ENVS * model.n_steps)} updates).")
    
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        progress_bar=True
    )
    
    # --- Save the Model ---
    os.makedirs("models", exist_ok=True)
    model.save(SAVE_PATH)
    
    print("\n--- PPO Training Complete ---")
    print(f"Model saved to: {SAVE_PATH}")
    
    vec_env.close()