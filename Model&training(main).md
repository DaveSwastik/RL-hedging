# RL Hedging – Training & Evaluation Guide (Main Branch)

This README describes how to train and evaluate the hedging model in the `main` branch. Update any placeholders (`<...>`) with your project’s specifics.

## Project Overview
- **Goal:** Describe the hedging objective and target instruments/markets.
- **Approach:** Briefly summarize the RL formulation (state, action, reward), model class (e.g., policy gradient, actor-critic, DQN), and data source.

## Repository Structure (main)
- `src/` – Core training & evaluation code (list key modules/classes)
- `configs/` – Training/evaluation configs (YAML/JSON)
- `data/` – Data location or download instructions (avoid committing proprietary data)
- `scripts/` – Helper scripts (e.g., preprocessing, training entrypoints)
- `notebooks/` – Exploratory analysis (optional)
- `models/` – Saved checkpoints (if versioned)
- `reports/` – Metrics, plots, and logs

## Setup
1. **Clone & branch**
   ```bash
   git clone https://github.com/DaveSwastik/RL-hedging.git
   cd RL-hedging
   git checkout main
   ```
2. **Environment**
   - Python version: `<e.g., 3.10>`
   - Create environment:
     ```bash
     python -m venv .venv
     source .venv/bin/activate  # Windows: .venv\Scripts\activate
     pip install -r requirements.txt
     ```
   - Optional: install editable
     ```bash
     pip install -e .
     ```

## Data
- **Source:** `<describe data source or path>`
- **Format:** `<CSV/Parquet/etc., schema summary>`
- **Preprocessing:** Run
  ```bash
  python scripts/preprocess.py --config configs/preprocess.yaml
  ```
  (Adjust script/config names to match your repo.)

## Training
- **Entry point:** `<e.g., python src/train.py --config configs/train.yaml>`
- **Key config fields:** batch size, learning rate, discount factor, risk-aversion terms, reward shaping, episode length, seed.
- **Example:**
  ```bash
  python src/train.py \
    --config configs/train.yaml \
    --seed 42 \
    --log_dir runs/main/train-$(date +%Y%m%d-%H%M%S)
  ```
- **Resuming from checkpoint:**
  ```bash
  python src/train.py --config configs/train.yaml --resume checkpoints/<ckpt>.pt
  ```

## Evaluation
- **Entry point:** `<e.g., python src/eval.py --config configs/eval.yaml --checkpoint checkpoints/best.pt>`
- **Metrics:** PnL distribution, drawdown, hedging error, Sharpe/Sortino, VaR/ES, turnover/costs, stability across seeds.
- **Example:**
  ```bash
  python src/eval.py \
    --config configs/eval.yaml \
    --checkpoint checkpoints/best.pt \
    --seed 123 \
    --eval_episodes 200
  ```
- **Out-of-sample / rolling windows:** Describe splits and methodology.

## Logging & Tracking
- **Logging:** `<stdout/JSON/CSV>` and `<tensorboard/wandb/mlflow>`
- **Where to find logs:** `runs/main/...` (or your actual path)
- **Plotting:** `<scripts/notebooks to visualize metrics>`

## Model Details
- **Observation space:** `<features used>`
- **Action space:** `<hedge ratios/position sizes>`
- **Reward:** `<PnL - costs; risk penalty; inventory penalty>`
- **Algorithm:** `<PPO/A2C/DDPG/SAC/DQN/etc.>` with key hyperparameters.
- **Exploration:** `<epsilon/entropy bonus/OU noise>`
- **Regularization:** `<weight decay, gradient clipping, risk penalties>`

## Reproducibility
- **Seeding:** `<seed strategy>`
- **Determinism:** Note any nondeterministic ops (GPU, env).
- **Config versioning:** Track config hashes and checkpoint metadata.

## Deployment / Inference
- **Loading checkpoint:** `<script or function>`
- **Live hedging loop / simulation:** `<entry point>`
- **Risk limits & guards:** `<positions, exposure, stop-loss>`

## Common Issues
- GPU/CPU requirements; memory notes.
- Data alignment/timezone issues.
- Numerical stability (clipping, normalization, scaling).

## Branch Notes
- This document is tailored to `main`.
- I will add per-branch deltas (scripts/configs/commands) once you share the other branch names and their differences.

## License
- `<license info>`