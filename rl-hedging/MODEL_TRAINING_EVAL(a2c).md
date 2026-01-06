# Model, Training, and Evaluation Summary

## Environment and Simulator
- Environment: [src/envs/hedging_env.py](src/envs/hedging_env.py) wraps a regime-switching Heston simulator with Asian-call default payoff, proportional trading costs, optional trade-penalty, and random initial spot in [0.5K, 1.5K].
- Observation: time features, moneyness, spot volatility, running average/max (for path dependence), current position, and Black-Scholes delta/vega helpers (dim=9). Action is a continuous hedge ratio in [-1, 1].
- Reward: negative squared hedging error per step (scaled by strike) minus transaction cost and trade penalty; terminal info includes P&L, payoff, and hedging error.

## Models
- PPO baseline: Stable-Baselines3 PPO with Tanh MLP policy (256-256 for actor/critic), vectorized envs via `make_vec_env`. See [src/agents/train.py](src/agents/train.py).
- PPO randomized variant: Same architecture but trained on the same HedgingEnv with CPU device; logs to `logs/ppo_randomized_tensorboard`. See [src/agents/train_ppo_randomized.py](src/agents/train_ppo_randomized.py).
- Custom A2C (interpretable hedger): Actor-Critic with custom `InterpretableHedger` + `Critic`, trained with GAE(λ), entropy regularization, gradient clipping, and cosine LR schedules. CPU-only, hidden size 64. See [src/agents/train_custom_ac.py](src/agents/train_custom_ac.py).
- Baseline delta hedge: Black-Scholes delta strategy with trading costs for comparison. See [src/baselines/delta_hedge.py](src/baselines/delta_hedge.py).

## Training Strategy
- Configuration: [src/configs/default.yaml](src/configs/default.yaml) sets regime-Heston params, Asian call (K=100, maturity=50×dt), trading_cost=0.001, and PPO hyperparams (16 envs, batch_size 64, n_epochs 10, lr 3e-4, gamma 0.99, gae_lambda 0.95, clip_range 0.2, 100k timesteps, log_interval 1).
- PPO baseline: Run `python src/agents/train.py` to train and save `models/ppo_hedge.zip`; uses tensorboard logging under `logs/ppo_hedging_tensorboard`.
- PPO randomized: Run `python src/agents/train_ppo_randomized.py`; saves `models/ppo_hedge_randomized.zip`; same hyperparams, CPU device, same env kwargs.
- Custom A2C: Run `python src/agents/train_custom_ac.py`; 20k episodes, AdamW (actor lr 1e-5, critic lr 3e-4), entropy coeff 0.01, stricter advantage clipping, turnover tracking, eval checkpointing to `models/custom_a2c_hedger_cpu_hiddenstep_best.pth`.

## Evaluation Strategy
- Synthetic evaluation (paths from Regime-Heston):
  - [evaluation_script.py](evaluation_script.py): evaluates PPO, Delta hedger, and custom agent over 500 episodes; saves per-episode CSVs (`ppo_results.csv`, `dh_results.csv`, `custom_results.csv`) and plots hedging error histograms and single-episode position traces.
  - [run_evaluation_suite.py](run_evaluation_suite.py): deeper evaluation with CPU devices and test seeds offset (seed+50,000); computes global stats (mean/σ P&L, mean error, RMSE, VaR, CVaR, avg trades), per-moneyness RMSE bins, and exports `results_suite/all_agent_errors.csv` for bootstrapping.

- Historical backtesting:
  - Backtest section in [evaluation_script.py](evaluation_script.py) (HistoricalEnv) loads `data/historical_aapl.csv`, runs PPO/custom/delta over 500 sampled episodes, and writes `results_backtest/*_results_backtest.csv`; also produces error histogram and single-episode behavior plots.

## What to run
1) Train PPO: `python src/agents/train.py` (or randomized variant). 2) Train custom A2C: `python src/agents/train_custom_ac.py`. 3) Evaluate on synthetic: `python evaluation_script.py` or `python run_evaluation_suite.py`. 4) Backtest on historical AAPL: `python evaluation_script.py` (backtest section).
