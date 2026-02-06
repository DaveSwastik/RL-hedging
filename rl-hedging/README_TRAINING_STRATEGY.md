# Training Strategy (ERA-RL v2)

This document describes the training loop implemented in [src/agents/train_era_rl.py](src/agents/train_era_rl.py), including what environment/data the agent sees and what is novel in the optimization scheme.

## High-level idea

Training is **on-policy** over simulated hedging episodes:

- The environment generates a full price/vol path per episode (regime-switching Heston).
- The agent observes a feature vector each step; the trainer feeds the agent a fixed-length **history window** (`obs_sequence`) using a deque with padding.
- The critic learns **distributional value** (quantiles) and a separate **tail model**.
- The actor learns a **multimodal policy** and is trained with a normalized advantage signal (risk metrics are logged and fed as actor features, but the current code does not subtract an explicit risk penalty term in the actor loss).

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

### 1) Curriculum via phases (FORCED_ENTRY → STABILIZATION → MASTERY)

The script trains for `n_episodes = 5000` with two phase boundaries:

- `phase_1_end = 15%` of episodes (`p1 = 0.15`)
- `phase_2_end = 75%` of episodes (`p1 + p2` with `p2 = 0.60`)

Per phase:

- **FORCED_ENTRY**
  - `cost_multiplier = 0.0` (transaction costs are effectively “given back” via reward adjustment)
  - `entropy_coef = 0.05`
  - `shock_prob = 0.0`

- **STABILIZATION**
  - `cost_multiplier = 1.0` (full costs applied)
  - `entropy_coef = 0.01`
  - `shock_prob = 0.05`

- **MASTERY**
  - `cost_multiplier = 1.0`
  - `entropy_coef = 0.005`
  - `shock_prob = 0.15`

This is an explicit form of curriculum: learn a stable baseline first, then train under costs and occasional shocks, with lower entropy regularization over time.

### 2) Adversarial shock injection

During rollout steps in **MASTERY**, the trainer directly mutates the environment state once per episode:

- only when `t > 0.2 * env.n_steps` and with probability `shock_prob / 10` per step
- applies:
  - `S_t <- 0.90 * S_t`
  - `V_t <- V_t + 0.15`

This creates “crash-like” events that are not just a natural outcome of the base simulator, forcing the policy to learn under distribution shift.

Note: the code enforces **at most one shock per episode** (`episode_shock_triggered`).

### 3) Cost annealing via reward adjustment

The environment already applies costs internally; the training loop additionally computes an implied transaction cost:

- `actual_trade = abs(action_t - prev_hedge) * env.S_t`
- `implied_cost = actual_trade * trading_cost`

Then adjusts the observed reward:

- `adjusted_reward = raw_reward + implied_cost * (1 - cost_multiplier)`

So when `cost_multiplier = 0.0` (FORCED_ENTRY) the loop adds back the implied cost term, and when `cost_multiplier = 1.0` (STABILIZATION/MASTERY) the reward is unchanged.

### 4) Time-scale separation / two optimizers

Two Adam optimizers are used:

- `optimizer_main` (faster):
  - encoder
  - actor
  - critic body (quantiles)
  - critic uncertainty head

- `optimizer_tail` (slower):
  - critic tail net only

This stabilizes learning because the EVT tail fit is only meaningful once there are enough tail samples.

### 5) Distributional critic loss (quantile regression with Huber)

Body critic trains `quantiles` against discounted return targets using `generalized_quantile_huber_loss`.

- `quantiles`: `[T, N]` stacked across the episode
- `returns`: `[T, 1]` discounted Monte Carlo target with `gamma=0.99`

The weighting term uses $|\tau - \mathbb{1}[target < q]|$ which is standard quantile regression.

### 6) Actor objective (normalized advantage + entropy)

The actor uses an advantage estimate:

- `advantages = returns - median_quantile_value`
- normalized (to reduce NaNs during shocks)

Implementation details (current code in [src/agents/train_era_rl.py](src/agents/train_era_rl.py)):

- baseline value per step is taken as the middle quantile index: `quantiles[..., num_quantiles // 2]`
- advantages are normalized: `(adv - adv.mean()) / (adv.std() + 1e-8)`

The code does compute and log a CVaR-like proxy from the critic (`CVaR_5pct`), but it is not currently subtracted from the advantage inside the actor loss.

Actor loss:

- `actor_loss = -(log_prob(action) * advantages).mean()`

Entropy regularization (phase-dependent `entropy_coef`):

- `ent_loss = -entropy_coef * entropy.mean()`

Total main loss:

- `loss = body_loss + actor_loss + ent_loss`

Stability details:

- gradient clipping: `clip_grad_norm_(agent.parameters(), 0.5)`

### 7) Tail learning with a replay buffer of exceedances

A deque `tail_buffer` stores tuples when a step is classified as a tail event:

- event condition: `return_t < u_t`
- stored (normalized shapes): `(latent_t, gru_history_t, return_t_scalar)` where:
  - `latent_t` is stored as `[H]`
  - `gru_history_t` is stored as `[T, H]`
  - `return_t_scalar` is stored as Python `float`

Every 10 episodes (`episode % 10 == 0`), if `len(tail_buffer) > 64`:

- sample a batch of 64 latents/returns (no replacement)
- compute tail params `(sigma, xi)` from `tail_net`
- recompute `u_current` from the critic on the sampled `(latent, gru_history)` (no grad)
- optimize a stable GPD negative log likelihood with a branch for `xi -> 0`

This creates a hybrid: on-policy main update + off-policy-ish tail fitting over rare events.

## Logging, outputs, and evaluation hooks

### Per-step training log

During training, the script appends a per-step record into `training_log` with these fields:

- `Episode`, `Step`, `Phase`
- `Price`, `Action`, `Reward`, `Cost_Mult`
- `Mode` and per-mode probabilities: `Prob_Mode0`, `Prob_Mode1`, `Prob_Mode2`
- risk diagnostics: `Xi`, `Uncertainty`, `CVaR_5pct`
- `Shock`
- per-episode backfilled summary fields on every row: `EpisodeTotalPnl`, `EpisodeAvgPnlPerStep`, `EpisodeLen`

Saved at the end to:

- `training_log_v2(5k-episodes).csv` (step-level)
- `episode_summaries(5k-episodes).csv` (episode-level)

The model checkpoint is saved to:

- `models/era_rl_v2(5k-episodes).pth`

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
