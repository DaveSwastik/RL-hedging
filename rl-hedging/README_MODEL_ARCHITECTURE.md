# Model Architecture (ERA-RL v2)

This document describes the implemented architecture in [src/agents/models.py](src/agents/models.py).

## What the model is trying to learn

The agent learns a **hedge ratio** (position in the underlying) for a short option position, in a simulated market. The policy is intentionally designed to:

- Represent **multiple behavioral regimes** (e.g., “normal”, “defensive”, “panic”).
- Learn **distributional value / risk** (not just a single expected value).
- Explicitly model **left-tail risk** with an EVT-inspired tail model.
- Limit destabilizing feedback loops by **blocking gradients** from actor → critic/encoder signals.

## End-to-end module graph

Implemented agent: `ERARL_Agent_V2`

```
obs_sequence  ──► DualStreamEncoder ──► latent, gru_history
                                  │
                                  ├──► HybridDistributionalCritic(latent, gru_history) ──► risk_out
                                  │         ├─ quantiles (monotone)
                                  │         ├─ u (tail threshold; detached)
                                  │         ├─ sigma, xi (GPD tail params)
                                  │         └─ uncertainty
                                  │
                                  └──► MixtureActor(latent, risk_out) ──► MixtureTanhNormal distribution
                                                     │
                                                     ├─ sample() -> action
                                                     └─ log_prob(action), entropy(), mode()
```

## Inputs and outputs

### Observation sequence

The model is called with an **observation sequence** `obs_sequence` of shape `[B, T, D]`:

- `B`: batch size (training uses `B=1` for on-policy rollout; tail updates batch over stored latents)
- `T`: history length (grows over an episode)
- `D`: observation dimension (`D=6` in the current env)

The env observation is produced in [src/envs/hedging_env.py](src/envs/hedging_env.py) and is:

`[time_norm, spot/S0, vol, position, running_avg/S0, running_max/S0]`

### Action

The policy outputs an action in `[-1, 1]` due to `tanh` squashing.

- Environment action space is `Box(low=-2.0, high=2.0, shape=(1,))` and clips any out-of-range action.
- In practice, the policy operates within `[-1, 1]` unless you change the squashing or rescale actions.

## Environment + “data” the model is trained on

This agent is trained on **synthetic trajectories** generated on-the-fly by a simulator (no historical price dataset is required for ERA-RL training).

### Market simulator: regime-switching Heston

Simulator: `RegimeHestonSimulator` in [src/simulators/regime_heston.py](src/simulators/regime_heston.py).

- Stochastic variance (Heston-style) with **full truncation** to keep variance nonnegative.
- **Regime switching** via a Markov transition matrix; each regime has its own `(theta, kappa, sigma_v, mu)`.
- Spot step uses exponential Euler.

Default regime config lives in [src/configs/default.yaml](src/configs/default.yaml):

- `S0=100`, `v0=0.04`, `dt=0.004`, `steps=20`, `rho=-0.7`
- 3 regimes: Low Vol / High Vol / Crisis
- transition matrix `trans_mat` defines regime persistence and switching

### Option pricing proxy inside the environment

The environment uses Black–Scholes pricing and delta as a **mark-to-market proxy** at each step:

- Wrapper: `bs_price_and_delta` in [src/utils/bs.py](src/utils/bs.py)
- Used by the env step logic in [src/envs/hedging_env.py](src/envs/hedging_env.py)

Note: this does *not* mean the policy observes BS delta; by default it does **not** (it’s only used for reward/P&L computation).

## Core architectural components

### 1) DualStreamEncoder (sequence + current features)

Class: `DualStreamEncoder`

- **Path stream**: GRU over the full `obs_sequence` (shape `[B, T, H]`).
- **Current stream**: MLP over the last observation `obs_sequence[:, -1, :]` (shape `[B, D]`).
- **Fusion**: concatenate `(last_gru_state, current_embed)` → MLP + LayerNorm + `tanh`.

Outputs:

- `latent`: `[B, H]` (used by actor and critic)
- `gru_history`: `[B, T, H]` (passed into the critic for potential trajectory awareness)

### 2) HybridDistributionalCritic (body + tail + uncertainty)

Class: `HybridDistributionalCritic`

This critic produces a *risk representation* rather than a single scalar value.

#### Body: monotonic quantiles (distributional value)

- Body network predicts `num_quantiles` (default 32).
- Uses `MonotonicQuantileLayer` to prevent **quantile crossing**.

Mechanism:

- Predict raw `[q0, d1, d2, ..., d_{N-1}]`.
- Force `d_i > 0` via `softplus`.
- Recover quantiles via cumulative sum.

Output: `quantiles` with shape `[B, N]` and guaranteed nondecreasing across quantile index.

#### Tail: GPD parameters (EVT-inspired)

The tail head predicts:

- `sigma = softplus(raw_sigma) + 1e-4`
- `xi = 0.5 * tanh(raw_xi)` (bounded)

These are used during training with a GPD negative log likelihood over sampled tail events.

#### Uncertainty head

Outputs `uncertainty = sigmoid(linear(latent))` in `[0, 1]`.

This is used by the actor as a confidence gate for risk features.

#### Tail threshold `u`

The critic defines a tail threshold:

- `u = quantiles[:, 0]` (first quantile)
- Detached before being returned to avoid backprop through threshold selection.

### 3) MixtureActor (multimodal policy)

Class: `MixtureActor`

- Learns a **mixture-of-Gaussians** policy with `K` modes (default 3).
- Uses a custom `MixtureTanhNormal` wrapper that:
  - samples a mode via a `Categorical`
  - samples a Gaussian in that mode
  - applies `tanh` squashing

#### Risk-conditioned policy inputs

The actor receives:

- `latent` (detached)
- risk metrics `u, sigma, xi, uncertainty` (detached)

It constructs a risk embedding:

- `risk_confidence = 1 - uncertainty`
- `risk_embedding = concat(u, sigma, xi) * risk_confidence`

Then concatenates:

- `x = concat(latent, risk_embedding)`

and passes through an MLP torso to produce:

- mixture logits (mode probabilities)
- per-mode means
- per-mode stds (`softplus + 1e-4`)

#### Gradient blocking (stability)

The actor explicitly detaches `latent` and the critic-derived risk signals.

Practical implication:

- The critic can influence the policy *via features*.
- But the policy loss cannot “fight” the critic/encoder through those features, reducing actor–critic coupling instability.

## Diagnostics / interpretability hooks

`ERARL_Agent_V2.get_diagnostics(...)` returns (no gradients):

- `u, sigma, xi, uncertainty`
- selected mode index (argmax of mixture logits)
- deterministic action (`tanh(best_mean)`) used for plotting

This is used by the training script for the final dashboard visualization.
