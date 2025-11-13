# src/agents/cvar_ctitic.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# Lightweight linear util (keeps parity with models.py)
def linear_layer(in_dim, out_dim, bias=True, gain=1.0):
    ly = nn.Linear(in_dim, out_dim, bias=bias)
    nn.init.orthogonal_(ly.weight, gain=gain)
    if ly.bias is not None:
        nn.init.zeros_(ly.bias)
    return ly

# -------------------------
# ExDrlCritic (Hybrid quantile + GPD tail)
# -------------------------
class ExDrlCritic(nn.Module):
    """
    Critic that predicts:
      - n_body_quantiles (raw scalars) for the distribution body (we sort them)
      - GPD tail parameters (raw -> scale, shape)
    Output:
      u: [B,1]  - threshold chosen as the smallest quantile
      body_quantiles: [B, n_body_quantiles-1] - ascending
      scale: [B,1] - positive scale parameter for GPD
      shape: [B,1] - shape param in (0,0.99)
    """
    def __init__(self, state_dim, hidden_dim=128, n_body_quantiles=32, use_layernorm=False):
        super().__init__()
        self.n_quantiles = n_body_quantiles
        n_outputs = self.n_quantiles + 2  # n quantiles + 2 gpd params
        self.net = nn.Sequential(
            linear_layer(state_dim, hidden_dim // 2),
            nn.ReLU(),
            linear_layer(hidden_dim // 2, n_outputs)
        )
        self._eps = 1e-6

    def forward(self, state_ctx: torch.Tensor):
        """
        state_ctx: [B, state_dim]
        returns: (u, body_q, scale, shape)
        """
        out = self.net(state_ctx)  # [B, nq + 2]
        quantiles_raw = out[:, :self.n_quantiles]   # [B, nq]
        gpd_raw = out[:, self.n_quantiles:]         # [B, 2]

        # sort quantiles ascending
        sorted_q, _ = torch.sort(quantiles_raw, dim=1)  # [B, nq]
        u = sorted_q[:, :1]                              # [B,1] threshold (lowest)
        body_q = sorted_q[:, 1:]                         # [B, nq-1]

        # parse gpd params
        log_scale = gpd_raw[:, 0:1]
        raw_shape = gpd_raw[:, 1:2]

        # positive scale via softplus; add epsilon
        scale = F.softplus(log_scale) + self._eps

        # shape in (0, 0.99)
        shape = torch.sigmoid(raw_shape) * 0.99 + 1e-6

        return u, body_q, scale, shape


# -------------------------
# Pinball loss (supports optional mask) — kept for compatibility
# -------------------------
def pinball_loss(quantiles: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor = None):
    """
    quantiles: [B, N]
    targets: [B] or [B,1]
    mask: optional boolean [B] where True includes sample in loss.
    returns scalar mean loss.
    """
    if targets.dim() == 1:
        targets = targets.unsqueeze(-1)  # [B,1]
    B, N = quantiles.shape
    taus = (torch.arange(N, device=quantiles.device, dtype=quantiles.dtype) + 0.5) / float(N)  # [N]
    diff = targets.unsqueeze(1) - quantiles  # [B, N]
    loss = torch.max(taus.view(1, N) * diff, (taus.view(1, N) - 1.0) * diff)  # [B, N]
    if mask is not None:
        mask = mask.view(-1, 1).to(dtype=loss.dtype, device=loss.device)
        loss = loss * mask
        denom = mask.sum() * N
        if denom.item() == 0:
            return torch.tensor(0.0, device=quantiles.device, dtype=quantiles.dtype)
        return loss.sum() / denom
    else:
        return loss.mean()


# -------------------------
# Vectorized per-sample GPD negative log-likelihood
# -------------------------
def gpd_log_likelihood_loss_vectorized(shape: torch.Tensor, scale: torch.Tensor, tail_y: torch.Tensor):
    """
    Vectorized per-sample GPD negative log-likelihood.

    Args:
      shape: [K,1] or [K]  - k per-tail-sample
      scale: [K,1] or [K]  - sigma per-tail-sample
      tail_y: [K]         - exceedances y = u - r (positive)
    Returns:
      scalar: mean NLL across tail samples
    """
    device = shape.device
    dtype = shape.dtype

    if tail_y is None or tail_y.numel() == 0:
        return torch.tensor(0.0, device=device, dtype=dtype)

    y = tail_y.view(-1, 1).to(device=device, dtype=dtype)      # [K,1]
    k = shape.view(-1, 1).to(device=device, dtype=dtype)       # [K,1]
    sigma = scale.view(-1, 1).to(device=device, dtype=dtype)   # [K,1]

    sigma = torch.clamp(sigma, min=1e-6)
    k = torch.clamp(k, min=1e-6, max=0.99)

    inner = 1.0 + (k * y) / sigma
    inner = torch.clamp(inner, min=1e-12)

    # nll per sample
    nll = torch.log(sigma) + (1.0 + 1.0 / k) * torch.log(inner)
    return nll.mean()


# -------------------------
# (Legacy) batch-averaged GPD NLL (kept for reference)
# -------------------------
def gpd_log_likelihood_loss(shape: torch.Tensor, scale: torch.Tensor, tail_returns: torch.Tensor):
    """
    Older, batch-averaged version kept as reference. Prefer vectorized one above.
    """
    device = shape.device
    dtype = shape.dtype

    if tail_returns is None or tail_returns.numel() == 0:
        return torch.tensor(0.0, device=device, dtype=dtype)

    shape_mean = shape.view(-1).mean()
    scale_mean = scale.view(-1).mean()

    y = tail_returns
    inner = 1.0 + (shape_mean * y) / scale_mean
    inner = torch.clamp(inner, min=1e-12)

    if torch.abs(shape_mean) < 1e-6:
        nll = torch.log(scale_mean) + (y / scale_mean)
    else:
        nll = torch.log(scale_mean) + (1.0 + 1.0 / shape_mean) * torch.log(inner)
    return nll.mean()


# -------------------------
# CVaR estimator from GPD params
# -------------------------
def estimate_cvar_from_gpd(u: torch.Tensor, scale: torch.Tensor, shape: torch.Tensor, alpha: float = 0.05):
    """
    Returns CVaR_alpha estimate (vector) with shape [B].
    For our left-tail setup (r < u), we model y = u - r > 0 as GPD.
    E[y | y > 0] = scale / (1 - shape) (for shape < 1).
    CVaR = u - E[y | y > 0] = u - scale / (1 - shape).
    """
    shape_c = torch.clamp(shape, min=-0.49, max=0.99)
    denom = (1.0 - shape_c)
    denom = torch.clamp(denom, min=1e-6)
    expected_tail = scale / denom
    cvar = u - expected_tail
    return cvar.view(-1)


# -------------------------
# Generalized Quantile Huber Loss (from 2401.02325v2)
# -------------------------
def generalized_quantile_huber_loss(quantiles: torch.Tensor,
                                    targets: torch.Tensor,
                                    b: torch.Tensor,
                                    mask: torch.Tensor = None):
    """
    Implements the Generalized Quantile Huber Loss (GQHL) based on the 1-Wasserstein distance
    between Gaussian-noised quantiles as per paper 2401.02325v2.
    quantiles: [B, N]
    targets: [B] or [B,1]
    b: scalar tensor (>0) estimated noise disparity |sigma_1 - sigma_2|
    mask: optional boolean [B] mask indicating body samples
    """
    if targets.dim() == 1:
        targets = targets.unsqueeze(-1)  # [B,1]
    B, N = quantiles.shape
    device = quantiles.device
    dtype = quantiles.dtype

    # taus: [1, N]
    taus = (torch.arange(N, device=device, dtype=dtype) + 0.5) / float(N)
    taus = taus.view(1, N)

    # error u = targets - quantiles  => [B, N]
    u = targets.unsqueeze(1) - quantiles

    # ensure positive b
    b_safe = torch.clamp(b, min=1e-6).to(device=device, dtype=dtype)

    # compute |u|/b
    abs_u = torch.abs(u)
    abs_u_over_b = abs_u / b_safe

    # cdf term: 0.5*(1 + erf(-|u|/(b*sqrt(2))))
    cdf_term = 0.5 * (1.0 + torch.erf(-abs_u_over_b / math.sqrt(2.0)))

    # exp_term: b * sqrt(2/pi) * exp(-0.5 * (|u|/b)^2)
    exp_term = b_safe * math.sqrt(2.0 / math.pi) * torch.exp(-0.5 * abs_u_over_b.pow(2))

    # C_GL^b(u) = |u| * (1 - 2*CDF) + exp_term - b*sqrt(2/pi)
    cost = abs_u * (1.0 - 2.0 * cdf_term) + exp_term - (b_safe * math.sqrt(2.0 / math.pi))

    # Asymmetric quantile weighting
    asymmetric_cost = torch.max(taus * cost, (taus - 1.0) * cost)

    if mask is not None:
        mask = mask.view(-1, 1).to(dtype=asymmetric_cost.dtype, device=device)
        asymmetric_cost = asymmetric_cost * mask
        denom = mask.sum() * N
        if denom.item() == 0:
            return torch.tensor(0.0, device=device, dtype=dtype)
        return asymmetric_cost.sum() / denom
    else:
        return asymmetric_cost.mean()


def estimate_noise_disparity(target_returns: torch.Tensor, predicted_values: torch.Tensor):
    """
    Estimates b = |sigma_1 - sigma_2| as per the paper.
    Here:
      - sigma_1 is empirical std of target_returns
      - sigma_2 is empirical std of predicted_values
    Returns scalar tensor detached from graph.
    """
    if target_returns.numel() == 0 or predicted_values.numel() == 0:
        return torch.tensor(1e-3, device=target_returns.device if target_returns.numel() > 0 else predicted_values.device)

    sigma_1_sq = torch.var(target_returns, unbiased=False)
    sigma_2_sq = torch.var(predicted_values, unbiased=False)

    sigma_1 = torch.sqrt(sigma_1_sq + 1e-8)
    sigma_2 = torch.sqrt(sigma_2_sq + 1e-8)

    b = torch.abs(sigma_1 - sigma_2)
    return b.detach()
