# src/agents/models.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------------
# Utilities
# -------------------------
def orthogonal_init(m, gain=1.0):
    if isinstance(m, (nn.Linear, nn.Conv1d, nn.Conv2d)):
        nn.init.orthogonal_(m.weight, gain=gain)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

def linear_layer(in_dim, out_dim, bias=True, gain=1.0):
    ly = nn.Linear(in_dim, out_dim, bias=bias)
    orthogonal_init(ly, gain=gain)
    return ly

# -------------------------
# Tanh-squashed Normal distribution wrapper
# -------------------------
class TanhNormal:
    """
    Minimal Tanh-squashed normal wrapper with sample(), rsample(), log_prob(), entropy().
    - mean/std shaped [..., action_dim]
    - actions are in (-1,1) after tanh
    """
    def __init__(self, mean: torch.Tensor, std: torch.Tensor):
        self.base = torch.distributions.Normal(mean, std.clamp(min=1e-6))
        self.mean = mean
        self.std = std

    def rsample(self):
        z = self.base.rsample()
        return torch.tanh(z)

    def sample(self):
        z = self.base.sample()
        return torch.tanh(z)

    def deterministic(self):
        return torch.tanh(self.mean)

    @staticmethod
    def _atanh(x: torch.Tensor):
        # numerical atanh: 0.5 * ln((1+x)/(1-x))
        eps = 1e-6
        a = x.clamp(-1 + eps, 1 - eps)
        return 0.5 * (torch.log1p(a) - torch.log1p(-a))

    def log_prob(self, action: torch.Tensor):
        # action in (-1,1)
        eps = 1e-6
        a = action.clamp(-1 + eps, 1 - eps)
        z = TanhNormal._atanh(a)
        log_prob_z = self.base.log_prob(z)
        # correction term: log|d(tanh)/dz| = log(1 - tanh(z)^2) = log(1 - a^2)
        correction = torch.log(torch.clamp(1 - a.pow(2), min=1e-12))
        return (log_prob_z - correction).sum(dim=-1)

    def entropy(self):
        # approximate entropy by base entropy (not exact after tanh but fine for regularization)
        return self.base.entropy().sum(dim=-1)

# -------------------------
# MLP Actor (compatible with A2CTrainer)
# -------------------------
class MlpActor(nn.Module):
    """
    A simple MLP actor compatible with the trainer's flattened batch API.
    - forward(obs: [B, obs_dim]) -> (dist: TanhNormal, ctx: [B, hid])
    - step(obs: [B, obs_dim] or [B,1,obs_dim], h_prev=None) -> (dist, None)
    """
    def __init__(self, obs_dim: int, hidden_dim: int = 128, action_dim: int = 1, state_dependent_std: bool = False):
        super().__init__()
        self.obs_dim = obs_dim
        self.hidden_dim = hidden_dim
        # Simple backbone
        self.backbone = nn.Sequential(
            linear_layer(obs_dim, hidden_dim),
            nn.Tanh(),
            linear_layer(hidden_dim, hidden_dim // 2),
            nn.Tanh()
        )
        # mean head
        gain = nn.init.calculate_gain('tanh')
        self.action_mean = linear_layer(hidden_dim // 2, action_dim, gain=gain)
        # log-std: either global parameter or state-dependent head
        self.state_dependent_std = state_dependent_std
        if not state_dependent_std:
            self.action_log_std = nn.Parameter(torch.tensor([[-3.0]], dtype=torch.float32))  # small std initially
        else:
            self.logstd_head = nn.Sequential(
                linear_layer(hidden_dim // 2, hidden_dim // 4),
                nn.ReLU(),
                linear_layer(hidden_dim // 4, action_dim)
            )

    def _make_dist(self, mu: torch.Tensor):
        if not self.state_dependent_std:
            std = torch.exp(self.action_log_std) + 1e-6
            std = std.expand_as(mu)
        else:
            std = torch.exp(self.logstd_head(mu.detach())) + 1e-6
            std = std.expand_as(mu)
        return TanhNormal(mu, std)

    def forward(self, obs: torch.Tensor):
        """
        obs: [B, obs_dim] (trainer uses flattened obs)
        returns: (dist, ctx)
        """
        if obs.dim() == 3 and obs.size(1) == 1:
            # accept [B,1,obs_dim] by squeezing
            obs = obs.squeeze(1)
        ctx = self.backbone(obs)
        mu = self.action_mean(ctx)
        dist = self._make_dist(mu)
        return dist, ctx

    def step(self, obs: torch.Tensor, h_prev=None):
        """
        Fast single-step API used by the collector.
        obs: [1, obs_dim] or [B, obs_dim] or [B,1,obs_dim]
        returns (dist, None)
        """
        # ensure 2D input
        if obs.dim() == 3 and obs.size(1) == 1:
            obs_in = obs.squeeze(1)
        else:
            obs_in = obs
        with torch.no_grad():
            dist, ctx = self.forward(obs_in)
        return dist, None

# -------------------------
# MLP Critic
# -------------------------
class MlpCritic(nn.Module):
    """
    Simple MLP value function.
    - forward(obs: [B, obs_dim]) -> v [B]
    - step(obs, h_prev) -> (v, None)
    """
    def __init__(self, obs_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            linear_layer(obs_dim, hidden_dim),
            nn.ReLU(),
            linear_layer(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            linear_layer(hidden_dim // 2, 1)
        )

    def forward(self, obs: torch.Tensor):
        if obs.dim() == 3 and obs.size(1) == 1:
            obs = obs.squeeze(1)
        v = self.net(obs).squeeze(-1)
        return v

    def step(self, obs: torch.Tensor, h_prev=None):
        with torch.no_grad():
            v = self.forward(obs)
        return v, None
# -------------------------