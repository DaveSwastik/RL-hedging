# src/agents/train.py
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from pathlib import Path
import torch as th

from src.envs.hedging_env import HedgingEnv

def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def train(config_path: str = 'src/configs/default.yaml', save_path: str = 'models/ppo_hedge.zip'):
    """Trains the PPO agent based on the provided configuration."""
    cfg = load_config(config_path)
    
    # Environment setup
    env_kwargs = {
        'sim_params': cfg['simulator'],
        'option_spec': cfg['option'],
        'trading_cost': cfg['environment']['trading_cost']
    }
    
    vec_env = make_vec_env(HedgingEnv, n_envs=cfg['training']['n_envs'], env_kwargs=env_kwargs)

    # Agent setup
    policy_kwargs = dict(activation_fn=th.nn.Tanh, net_arch=[dict(pi=[256, 256], vf=[256, 256])])
    
    model = PPO(
        policy=cfg['training']['policy'],
        env=vec_env,
        verbose=1,
        batch_size=cfg['training']['batch_size'],
        n_epochs=cfg['training']['n_epochs'],
        learning_rate=cfg['training']['lr'],
        gamma=cfg['training']['gamma'],
        gae_lambda=cfg['training']['gae_lambda'],
        clip_range=cfg['training']['clip_range'],
        tensorboard_log="./logs/ppo_hedging_tensorboard/",
        policy_kwargs=policy_kwargs
    )
    
    print("--- Starting Training ---")
    model.learn(
        total_timesteps=cfg['training']['timesteps'],
        log_interval=cfg['training']['log_interval']
    )
    
    # Save the model
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    model.save(save_path)
    print(f"--- Training Complete --- \nModel saved to {save_path}")

if __name__ == "__main__":
    train()