# ERA-RL Model, Training, and Evaluation

## Environment
- Regime-switching Heston simulator with Asian-call default payoff, proportional trading costs, optional trade penalty; random S0 in [0.5K, 1.5K]. See [src/envs/hedging_env.py](src/envs/hedging_env.py).
- Observation (dim=9): normalized time, moneyness, volatility, current position, running average/max, time-to-maturity, BS delta/vega. Action: continuous hedge ratio in [-1, 1].
- Reward: negative squared hedging error (scaled by strike) minus transaction cost and trade penalty; terminal info returns P&L, payoff, hedging error.

## Models
- CVaR A2C: `MlpActor` + `MlpCritic` with auxiliary `ExDrlCritic` for quantile/GPD tail loss. Trained via `A2CTrainer` with GAE, entropy regularization, gradient clipping, CVaR alpha 0.05. See [src/agents/trainer_a2c.py](src/agents/trainer_a2c.py) and [run_training.py](run_training.py).
- PPO baseline: Stable-Baselines3 PPO MLP policy; short 10k-timestep run for comparison. See [train_ppo.py](train_ppo.py).
- Delta hedge baseline: Black–Scholes delta strategy with costs; embedded in eval scripts.

## Training
- Config: [src/configs/default.yaml](src/configs/default.yaml) defines regime-Heston params, Asian call (K=100, maturity=steps×dt), trading_cost, and simulator reward_scale/shocks defaults.
- CVaR A2C run ([run_training.py](run_training.py)):
  - n_envs 16, rollout_length 50, total_updates 5000, device CPU, hidden_dim from config (default 128).
  - lr_actor 1e-4, lr_critic 5e-4, gamma 0.995, lam 0.95, entropy_coef 0.05, value_coef 0.5, cvar_lambda annealed to 1.0, quantile/gpd loss weights =1, max_grad_norm 0.5, advantage_clip 10.
  - Simulator extras: reward_scale 100, shock_prob 0.05, shock_scale 1.0 to expose tail events.
  - Outputs checkpoints and CSV to runs/training_run_01/ (e.g., checkpoint_final.pth, training_log.csv).
- PPO run ([train_ppo.py](train_ppo.py)):
  - n_envs 16, n_steps 100, batch_size 64, n_epochs 10, gamma 0.995, gae_lambda 0.95, ent_coef 0.01, lr 3e-4, total_timesteps 10k; saves models/ppo_hedge_new.zip.

## Evaluation
- Improved suite ([run_evaluation_improved.py](run_evaluation_improved.py)):
  - Evaluates Custom CVaR (runs/training_run_01/checkpoint_final.pth), PPO (models/ppo_hedge_new.zip), and Delta over 500 episodes (seeds 0–499).
  - Metrics: hedging error, P&L, VaR/CVaR (5%), Sharpe; plots boxplots, CVaR bars, P&L histograms, risk-return scatter, stability (rolling errors), and single-episode positions/P&L (seed 300). Writes results to results/.
- Standard suite ([run_evaluation.py](run_evaluation.py)):
  - Similar agents; 500 episodes; outputs summary CSV and plots hedging-error histogram plus single-episode position plot (seed 120) to results/.

## How to run
1) Train CVaR A2C: `python run_training.py`
2) Train PPO baseline: `python train_ppo.py`
3) Evaluate (full plots): `python run_evaluation_improved.py`
4) Quick eval: `python run_evaluation.py`