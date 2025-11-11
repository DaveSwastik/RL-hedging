# src/agents/models.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Squashed Normal (tanh) ---
class TanhNormal:
    def __init__(self, mean, std):
        self.base = torch.distributions.Normal(mean, std)

    def sample(self):
        z = self.base.rsample()
        return torch.tanh(z)

    def rsample(self):
        z = self.base.rsample()
        return torch.tanh(z)

    def log_prob(self, a):
        a = torch.clamp(a, -0.999999, 0.999999)
        z = torch.atanh(a)
        log_prob_z = self.base.log_prob(z)           # [..., action_dim]
        corr = torch.log1p(-a.pow(2) + 1e-12)        # log(1 - a^2)
        return (log_prob_z - corr).sum(dim=-1)       # reduce over action dim

    def entropy(self):
        # proxy: base entropy
        return self.base.entropy().sum(dim=-1)

class InterpretableHedger(nn.Module):
    """
    Actor with GRU memory. Provides:
      - step(obs_t, h): O(1) per step (fast mode)
    """
    def __init__(self, obs_dim: int, hidden_dim: int = 64):  # 64 is CPU-friendly
        super().__init__()
        self.gru = nn.GRU(obs_dim, hidden_dim, batch_first=True)

        self.action_mean = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )
        self.action_log_std_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )

    # FAST path: single time step with hidden state
    def step(self, obs_t: torch.Tensor, h: torch.Tensor | None = None):
        """
        obs_t: [B, 1, obs_dim]
        h:     [1, B, hidden_dim] or None
        returns: (dist, h_next)
        """
        g, h_next = self.gru(obs_t, h)          # g: [B,1,H]
        ctx = g[:, -1, :]                       # [B,H]
        mu = self.action_mean(ctx)              # [B,1]
        log_std = self.action_log_std_head(ctx).clamp(-5, 2)
        std = torch.exp(log_std)
        dist = TanhNormal(mu, std)
        return dist, h_next

    # Full sequence path (for interpretability, but slower)
    def forward(self, obs_sequence: torch.Tensor):
        g, _ = self.gru(obs_sequence)           # [B,T,H]
        ctx = g[:, -1, :]                       # last hidden
        mu = self.action_mean(ctx)
        log_std = self.action_log_std_head(ctx).clamp(-5, 2)
        std = torch.exp(log_std)
        dist = TanhNormal(mu, std)
        # attention placeholder
        B, T, _ = g.shape
        attn = torch.full((B, T), 1.0 / T, device=g.device, dtype=g.dtype)
        return dist, attn

class Critic(nn.Module):
    """
    Value function sharing the GRU idea; exposes step() for O(1).
    """
    def __init__(self, obs_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.gru = nn.GRU(obs_dim, hidden_dim, batch_first=True)
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def step(self, obs_t: torch.Tensor, h: torch.Tensor | None = None):
        g, h_next = self.gru(obs_t, h)                # [B,1,H]
        v = self.value_head(g[:, -1, :])              # [B,1]
        return v.squeeze(-1), h_next                  # return scalar per batch

    def forward(self, obs_sequence: torch.Tensor):
        g, _ = self.gru(obs_sequence)
        v = self.value_head(g[:, -1, :])
        return v