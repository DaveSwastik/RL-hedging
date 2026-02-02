# Training Strategy (ERA-RL v2)

This document describes the training loop implemented in [src/agents/train_era_rl.py](src/agents/train_era_rl.py), including what environment/data the agent sees and what is novel in the optimization scheme.

## High-level idea

Training is **on-policy** over simulated hedging episodes:

- The environment generates a full price/vol path per episode (regime-switching Heston).
- The agent observes a feature vector each step; the trainer feeds the agent an increasing **history window** (`obs_sequence`).
- The critic learns **distributional value** (quantiles) and a separate **tail model**.
- The actor learns a **multimodal policy** and is trained with an advantage signal that is adjusted by a **risk penalty**.

## Environment and simulated “data”

### Environment

Gymnasium environment: `HedgingEnv` in [src/envs/hedging_env.py](src/envs/hedging_env.py).

- Action: hedge position (continuous), clipped to `[-2, 2]`.
- Observation (6D):
  - `t/T`
  - `S_t / S0`
  - `sqrt(v_t)` (vol)
  - current position
  - running average price / `S0`
  - running max price / `S0`

### Market generator (training distribution)

Simulator: `RegimeHestonSimulator` in [src/simulators/regime_heston.py](src/simulators/regime_heston.py).

- Regime-switching parameters are defined by `regimes` + `trans_mat`.
- Each episode draws a path (spot, variance, regimes) of length `steps + 1`.

Default distribution is configured in [src/configs/default.yaml](src/configs/default.yaml). Your trained policy is therefore specific to that simulated regime mixture (unless you change the config).

### Option pricing + reward/P&L

At each step, the env computes an option mark-to-market using Black–Scholes (`bs_price_and_delta` in [src/utils/bs.py](src/utils/bs.py)) to define hedger P&L:

- Hedger is short the option.
- Hedger holds `prev_pos` shares of underlying.
- Transaction costs apply: `abs(trade_amount) * price * trading_cost`.

Reward is scaled by `reward_scale` and at terminal includes liquidation and terminal hedging error.

## What’s novel in the training setup

### 1) Curriculum via phases (ANCHOR → DISCOVERY → MASTERY)

The script trains for `n_episodes = 5000` with two phase boundaries:

- `phase_1_end = 20%` of episodes
- `phase_2_end = 50%` of episodes

Per phase:

- **ANCHOR**
  - `shock_prob = 0.0`
  - `lambda_cvar = 0.0` (no explicit risk penalty)

- **DISCOVERY**
  - `shock_prob = 0.05`
  - `lambda_cvar` ramps from `0.0` to `0.1`

- **MASTERY**
  - `shock_prob` increases from `0.05` up to `0.20`
  - `lambda_cvar = 0.5`

This is an explicit form of curriculum: learn basic hedging first, then learn robustness and tail awareness.

### 2) Adversarial shock injection

During rollout steps (after step 5), the trainer may call `inject_adversarial_shock(env)`:

- `S_t <- S_t * (1 - severity)`
- `V_t <- V_t + 0.20`

This creates “crash-like” events that are not just a natural outcome of the base simulator, forcing the policy to learn under distribution shift.

### 3) Time-scale separation / two optimizers

Two Adam optimizers are used:

- `optimizer_main` (faster):
  - encoder
  - actor
  - critic body (quantiles)
  - critic uncertainty head

- `optimizer_tail` (slower):
  - critic tail net only

This stabilizes learning because the EVT tail fit is only meaningful once there are enough tail samples.

### 4) Distributional critic loss (quantile regression with Huber)

Body critic trains `quantiles` against discounted return targets using `generalized_quantile_huber_loss`.

- `quantiles`: `[T, N]` stacked across the episode
- `returns`: `[T, 1]` discounted Monte Carlo target with `gamma=0.99`

The weighting term uses $|\tau - \mathbb{1}[target < q]|$ which is standard quantile regression.

### 5) Actor objective with risk-adjusted advantage

The actor uses an advantage estimate:

- `advantages = returns - median_quantile_value`
- normalized (to reduce NaNs during shocks)

Risk penalty is derived from the critic’s tail threshold estimates `u` (saved per step):

- `cvar_proxy = episode_us`
- `risk_penalty = lambda_cvar * abs(min(u, 0))`
- `adjusted_advantage = advantages - risk_penalty`

Actor loss:

- `actor_loss = -(log_prob(action) * adjusted_advantage).mean()`

Plus entropy regularization:

- `entropy_loss = -0.01 * entropy.mean()`

### 6) Tail learning with a replay buffer of exceedances

A deque `tail_buffer` stores tuples when a step is classified as a tail event:

- event condition: `return_t < u_t`
- stored: `(latent_t, padded_gru_history_t, return_t)`

Every `tail_update_freq=10` episodes (and not in ANCHOR), if buffer size >= `k_min=32`:

- sample a batch of 32 latents/returns
- compute tail params `(sigma, xi)` from `tail_net`
- recompute `u_batch` from the body (no grad)
- optimize GPD negative log likelihood `gpd_negative_log_likelihood`

This creates a hybrid: on-policy main update + off-policy-ish tail fitting over rare events.

## Logging, outputs, and evaluation hooks

### Per-step training log

During training, the script appends a per-step record into `training_data_log`:

- episode, step, phase
- price, volatility
- action, reward
- regime mode (argmax mixture)
- `xi`, `uncertainty`
- whether a shock occurred

Saved at the end to:

- `training_data_log.xlsx` (preferred)
- fallback `training_data_log.csv` if Excel writer deps are missing

### Final diagnostic dashboard

After training, the script runs one diagnostic episode and saves:

- `training_mastery_report.png`

It also computes and plots a Black–Scholes delta benchmark (via `bs_greeks`) for visual comparison.

## How to run

From repo root:

- `python -m src.agents.train_era_rl`

To change the training distribution or episode length, edit:

- [src/configs/default.yaml](src/configs/default.yaml)

## Reproducibility notes

- The env supports seeding via `env.reset(seed=...)`, but the training loop also uses `np.random.rand()` for shock triggering without a fixed NumPy seed in the script.
- If strict reproducibility matters, add explicit seeds for NumPy and PyTorch at the start of `train_era_rl()`.
