#!/usr/bin/env python3
"""train_sb3_ppo_heston.py

Train a PPO hedging agent using Stable-Baselines3 on the repo's Heston simulator
(`src.simulators.regime_heston.RegimeHestonSimulator`) via `src.envs.hedging_env.HedgingEnv`.

User intent:
- Train for 1000 episodes
- Each episode is 20 trading days

In SB3, training length is specified in timesteps, so we set:
  total_timesteps = episodes * episode_length

Output:
- Model zip under ./models/
- Monitor logs under ./runs/

Usage:
  python train_sb3_ppo_heston.py
  python train_sb3_ppo_heston.py --timesteps 20000 --seed 42
"""

from __future__ import annotations

import argparse
import os
from functools import partial
from pathlib import Path

import torch
import yaml

from src.envs.hedging_env import HedgingEnv

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.monitor import Monitor
except ImportError as e:  # pragma: no cover
    raise SystemExit(
        "stable_baselines3 is not installed. Install with: poetry add stable-baselines3"
    ) from e


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE

DEFAULT_CFG = REPO_ROOT / "src" / "configs" / "default.yaml"
DEFAULT_SAVE_PATH = REPO_ROOT / "models" / "sb3_ppo_heston_20d.zip"
DEFAULT_LOG_DIR = REPO_ROOT / "runs" / "sb3_ppo_heston_20d"


def load_config(path: str | Path) -> dict:
    p = Path(path)
    if not p.is_absolute():
        p = (REPO_ROOT / p).resolve()
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {p}")
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _build_env(cfg: dict, episode_steps: int, seed: int | None = None):
    sim_cfg = cfg["simulator"]

    sim_params = {
        "S0": float(sim_cfg["S0"]),
        "v0": float(sim_cfg["v0"]),
        "dt": float(sim_cfg["dt"]),
        "steps": int(episode_steps),
        "rho": float(sim_cfg.get("rho", -0.7)),
        "regimes": sim_cfg.get("regimes", []),
        "trans_mat": sim_cfg.get("trans_mat", []),
        "r": float(sim_cfg.get("r", 0.0)),
        "reward_scale": sim_cfg.get("reward_scale", 1.0),
        "shock_prob": 0.0,
        "shock_scale": 0.0,
    }

    opt_cfg = cfg["option"]
    option_spec = {
        "type": opt_cfg["type"],
        "K": float(opt_cfg["K"]),
        "maturity": float(opt_cfg["maturity"]),
        "option_type": opt_cfg.get("option_type", "call"),
        "notional": float(opt_cfg.get("notional", 1.0)),
    }

    env = HedgingEnv(
        sim_params=sim_params,
        option_spec=option_spec,
        trading_cost=float(cfg.get("environment", {}).get("trading_cost", 0.0)),
    )

    # SB3 expects a Gymnasium env; Monitor adds episode stats.
    env = Monitor(env)

    # Ensure deterministic reset seeding if desired.
    if seed is not None:
        env.reset(seed=seed)

    return env


def main():
    parser = argparse.ArgumentParser(description="Train SB3 PPO on Heston simulator (20-day episodes)")
    parser.add_argument("--config", default=str(DEFAULT_CFG), help="Path to YAML config")
    parser.add_argument("--episodes", type=int, default=1000, help="Number of episodes")
    parser.add_argument("--episode-steps", type=int, default=20, help="Steps (trading days) per episode")
    parser.add_argument(
        "--timesteps",
        type=int,
        default=None,
        help="Override total timesteps (default = episodes * episode_steps)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--save", default=str(DEFAULT_SAVE_PATH), help="Where to save the SB3 model (.zip)")
    parser.add_argument("--logdir", default=str(DEFAULT_LOG_DIR), help="Directory for monitor logs")
    parser.add_argument("--n-envs", type=int, default=1, help="Parallel envs (keep 1 to match 'episodes' literally)")

    # PPO hyperparams (lightweight defaults)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.995)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--ent-coef", type=float, default=0.0)
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument(
        "--n-steps",
        type=int,
        default=200,
        help="Rollout steps per env before update (recommend multiple of episode_steps)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Minibatch size (must be <= n_steps*n_envs)",
    )

    args = parser.parse_args()

    cfg = load_config(args.config)

    total_timesteps = args.timesteps if args.timesteps is not None else args.episodes * args.episode_steps

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Training timesteps: {total_timesteps} (episodes={args.episodes}, steps={args.episode_steps})")

    logdir = Path(args.logdir)
    logdir.mkdir(parents=True, exist_ok=True)

    env_fn = partial(_build_env, cfg=cfg, episode_steps=args.episode_steps, seed=args.seed)
    vec_env = make_vec_env(env_fn, n_envs=args.n_envs, seed=args.seed, monitor_dir=str(logdir))

    # Safety: ensure SB3 constraints
    max_batch = args.n_steps * args.n_envs
    if args.batch_size > max_batch:
        raise ValueError(f"batch_size={args.batch_size} must be <= n_steps*n_envs={max_batch}")

    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        ent_coef=args.ent_coef,
        verbose=1,
        device=device,
        seed=args.seed,
    )

    model.learn(total_timesteps=total_timesteps, progress_bar=True)

    save_path = Path(args.save)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(save_path))

    vec_env.close()

    print("\nPPO training complete.")
    print(f"Saved model: {save_path}")
    print(f"Logs: {logdir}")


if __name__ == "__main__":
    main()
