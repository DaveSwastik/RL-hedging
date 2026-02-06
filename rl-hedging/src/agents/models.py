# model_v2.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Categorical, MixtureSameFamily

# -------------------------------------------------------------------
# 1. UTILITIES & ROBUST LAYERS
# -------------------------------------------------------------------

def robust_softplus(x, beta=1.0, threshold=20.0):
    return F.softplus(x, beta, threshold)

class MonotonicQuantileLayer(nn.Module):
    """
    Fix Gap 5: Quantile Crossing.
    Instead of predicting values directly, we predict the first quantile
    and positive increments (deltas) for the rest.
    q_i = q_{i-1} + softplus(delta_i)
    """
    def __init__(self, input_dim, num_quantiles):
        super().__init__()
        self.num_quantiles = num_quantiles
        # Predict base (q0) and deltas (d1...dN)
        self.net = nn.Linear(input_dim, num_quantiles)
    
    def forward(self, x):
        raw_out = self.net(x)
        # q0 is raw, q1..qN are strictly positive deltas
        q0 = raw_out[:, 0:1]
        deltas = robust_softplus(raw_out[:, 1:]) 
        
        # Pad q0 and cumsum deltas
        # Shape: [Batch, Num_Quantiles]
        quantiles = torch.cat([q0, deltas], dim=1)
        quantiles = torch.cumsum(quantiles, dim=1)
        return quantiles

class MixtureTanhNormal:
    """
    Fix Gap 1: Regime Switching.
    Represents a Mixture of Gaussians squashed by Tanh.
    Can represent multimodal policies (e.g. "Stay" vs "Panic Sell").
    """
    def __init__(self, categorical_logits, means, stds):
        # categorical_logits: [B, K]
        # means: [B, K, Action_Dim]
        # stds:  [B, K, Action_Dim]
        
        self.cat_dist = Categorical(logits=categorical_logits)
        self.means = means
        self.stds = stds
        
    def sample(self):
        # 1. Pick a mode (Regime)
        # indices: [B]
        mode_indices = self.cat_dist.sample()
        # Expose sampled mode for logging/debugging
        self.last_mode_indices = mode_indices
        
        # Sample one z from each component, then pick based on mode
        z = torch.normal(self.means, self.stds)  # [B, K] (scalar action) or [B, K, D]
        # Select z based on mode indices
        chosen_z = z[torch.arange(z.size(0)), mode_indices]
        return torch.tanh(chosen_z)

    def sample_with_mode(self):
        """Samples an action and also returns the sampled regime index."""
        mode_indices = self.cat_dist.sample()
        self.last_mode_indices = mode_indices
        z = torch.normal(self.means, self.stds)
        chosen_z = z[torch.arange(z.size(0)), mode_indices]
        return torch.tanh(chosen_z), mode_indices

    def log_prob(self, action):
        # Log prob of Mixture is log(sum(exp(log_probs)))
        # For tanh correction: log_prob(a) = log(sum(pi_k * N(atanh(a)|mu_k, sigma_k))) - log(1-a^2)
        
        # Numerical stability clamp
        action_unsquashed = torch.atanh(torch.clamp(action, -0.999999, 0.999999))
        
        # Ensure shape compatibility
        if action_unsquashed.dim() == 0:
            action_unsquashed = action_unsquashed.unsqueeze(0)
        if action_unsquashed.dim() == 1:
            action_unsquashed = action_unsquashed.unsqueeze(-1)  # [B, 1]

        # Component log_probs: [B, K]
        action_expanded = action_unsquashed.unsqueeze(1)  # [B, 1, 1]
        comp_lp = Normal(self.means.unsqueeze(-1), self.stds.unsqueeze(-1)).log_prob(action_expanded)
        comp_log_probs = comp_lp.sum(dim=-1)  # sum over action dims
        
        # Mix with categorical logits
        # mixture_log_prob = logsumexp(logits + comp_log_probs) - logsumexp(logits)
        mixed_log_prob = torch.logsumexp(self.cat_dist.logits + comp_log_probs, dim=1) - \
                         torch.logsumexp(self.cat_dist.logits, dim=1)
        
        # Jacobian correction
        # Jacobian correction: sum over action dims
        if action.dim() == 0:
            action = action.unsqueeze(0)
        if action.dim() == 1:
            action = action.unsqueeze(-1)
        jacobian = torch.log1p(-action.pow(2) + 1e-12).sum(dim=-1)
        
        return mixed_log_prob - jacobian

    def entropy(self):
        """Approximate entropy (ignores tanh squashing correction)."""
        # Categorical entropy + expected Normal entropy (per component)
        cat_ent = self.cat_dist.entropy()  # [B]
        comp_ent = Normal(self.means, self.stds).entropy()  # [B, K]
        weights = torch.softmax(self.cat_dist.logits, dim=-1)  # [B, K]
        mix_ent = (weights * comp_ent).sum(dim=-1)  # [B]
        return cat_ent + mix_ent

    def mode(self):
        # Returns the deterministic "most likely" action (from the most likely mode)
        best_mode = torch.argmax(self.cat_dist.logits, dim=1)
        best_mean = self.means[torch.arange(self.means.size(0)), best_mode]
        return torch.tanh(best_mean)

# -------------------------------------------------------------------
# 2. ENCODER WITH GRADIENT BLOCKING (Fix Gap 6)
# -------------------------------------------------------------------

class DualStreamEncoder(nn.Module):
    def __init__(self, obs_dim, hidden_dim=128):
        super().__init__()
        # Path Stream (GRU)
        self.gru = nn.GRU(obs_dim, hidden_dim, batch_first=True)
        # Trajectory Attention for Critic (Fix Gap 4)
        self.traj_attn = nn.Linear(hidden_dim, 1)
        
        # Current Features Stream
        self.current_net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim // 2),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 2)
        )
        
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim + (hidden_dim // 2), hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh()
        )

    def forward(self, obs_sequence, prev_hedge):
        # 1. Path Processing
        # Condition the GRU on the *current* previous hedge via the initial hidden state.
        # This makes the recurrent features explicitly aware of the current position context
        # (in addition to any position history already present in obs_sequence).
        B = obs_sequence.size(0)
        device = obs_sequence.device
        dtype = obs_sequence.dtype

        if prev_hedge is None:
            prev_hedge_feat = torch.zeros(B, 1, device=device, dtype=dtype)
        else:
            prev_hedge_feat = prev_hedge.to(device=device, dtype=dtype)
            # Flatten to [B] if necessary
            if prev_hedge_feat.dim() > 1:
                prev_hedge_feat = prev_hedge_feat.reshape(B, -1)[:, 0]  # take first column
            if prev_hedge_feat.dim() == 1:
                prev_hedge_feat = prev_hedge_feat.unsqueeze(-1)         #  [B,1]

            if prev_hedge_feat.size(0) != B:
                raise ValueError(f"prev_hedge batch size {prev_hedge_feat.size(0)}  obs batch {B}")

        # Final hidden init
        h0 = torch.tanh(prev_hedge_feat) \
               .expand(-1, self.gru.hidden_size) \
               .unsqueeze(0)                                 # [1, B, hidden_dim]
        gru_out, _ = self.gru(obs_sequence, h0)  # [B, T, H]
        
        # 2. Trajectory Risk Awareness (Fix Gap 4)
        # Instead of just last state, we keep the full sequence for the critic
        # to perform "Trajectory Attention"
        
        # 3. Last State Context (Standard)
        last_context = gru_out[:, -1, :]
        
        # 4. Current Features
        current_obs = obs_sequence[:, -1, :]
        curr_embed = self.current_net(current_obs)
        
        # 5. Fusion
        latent = self.fusion(torch.cat([last_context, curr_embed], dim=-1))
        
        return latent, gru_out # Return full history for critic

# -------------------------------------------------------------------
# 3. HYBRID CRITIC (Corrected & Patched)
# -------------------------------------------------------------------

class HybridDistributionalCritic(nn.Module):
    def __init__(self, latent_dim, num_quantiles=32):
        super().__init__()
        self.num_quantiles = num_quantiles
        
        # 1. Trajectory Attention Head (The Fix)
        # Calculates relevance of each history step to the current state
        self.attention_score = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.Tanh(),
            nn.Linear(latent_dim, 1)
        )
        
        # 2. Body: Monotonic Quantiles
        # Input is now the weighted context vector
        self.body_net = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LeakyReLU(),
            # We use the class output directly here for cleanliness
        )
        self.quantile_layer = MonotonicQuantileLayer(256, num_quantiles)
        
        # 3. Tail: GPD Parameters
        self.tail_net = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.SiLU(),
            nn.Linear(64, 2) # Sigma, Xi
        )
        
        # 4. Uncertainty Head
        self.uncertainty_net = nn.Linear(latent_dim, 1)

    def forward(self, latent, gru_history):
        # --- ATTENTION MECHANISM ---
        # Expand latent to broadcast: [B, H] -> [B, 1, H]
        latent_expanded = latent.unsqueeze(1)
        
        # Combine history [B, T, H] with current latent [B, 1, H]
        # This determines "how similar was the past to right now?"
        combined = torch.tanh(gru_history + latent_expanded)
        
        # Calculate scores: [B, T, 1]
        scores = self.attention_score(combined)
        weights = F.softmax(scores, dim=1)
        
        # Context Vector: Weighted sum of history
        context_vector = torch.sum(gru_history * weights, dim=1) # [B, H]
        
        # --- PREDICTIONS ---
        # 1. Quantiles (using context)
        body_feat = self.body_net(context_vector)
        quantiles = self.quantile_layer(body_feat)
        
        # 2. Consistency Threshold (First quantile)
        u_threshold = quantiles[:, 0].unsqueeze(1).detach()
        
        # 3. Tail (using current latent is usually fine, or use context)
        raw_tail = self.tail_net(latent) 
        sigma = robust_softplus(raw_tail[:, 0:1]) + 1e-4
        xi = torch.tanh(raw_tail[:, 1:2]) * 0.5
        
        # 4. Uncertainty
        uncertainty = torch.sigmoid(self.uncertainty_net(latent))
        
        return {
            'quantiles': quantiles,
            'u': u_threshold,
            'sigma': sigma,
            'xi': xi,
            'uncertainty': uncertainty
        }

# -------------------------------------------------------------------
# 4. MIXTURE ACTOR (Fix Gap 1, 3, 6)
# -------------------------------------------------------------------

class MixtureActor(nn.Module):
    def __init__(self, latent_dim, num_modes=3):
        super().__init__()
        self.num_modes = num_modes
        
        # Input: Latent + Risk Embeddings
        input_dim = latent_dim + 3 
        
        # Shared torso
        self.torso = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 128),
            nn.Tanh()
        )
        
        # Heads for Mixture Parameters
        self.logits_head = nn.Linear(128, num_modes) # P(Mode)
        self.means_head = nn.Linear(128, num_modes)  # Mu per mode
        self.stds_head = nn.Linear(128, num_modes)   # Sigma per mode

    def forward(self, latent, risk_metrics):
        # Fix Gap 6: Gradient Blocking
        # Actor cannot update encoder or critic
        latent = latent.detach()
        u = risk_metrics['u'].detach()
        sigma = risk_metrics['sigma'].detach()
        xi = risk_metrics['xi'].detach()
        uncertainty = risk_metrics['uncertainty'].detach()
        
        # Fix Gap 3: Structural Uncertainty Usage
        # Gating the risk signal: If uncertain, reduce risk feature magnitude
        # Make gating sharper: if uncertainty > 0.5, kill the risk signal
        risk_confidence = torch.sigmoid((0.5 - uncertainty) * 10.0)
        risk_embedding = torch.cat([u, sigma, xi], dim=-1) * risk_confidence
        
        # Input construction
        x = torch.cat([latent, risk_embedding], dim=-1)
        x = self.torso(x)
        
        # Outputs
        logits = self.logits_head(x)
        means = self.means_head(x)
        stds = F.softplus(self.stds_head(x)) + 1e-4
        
        # Return distribution object for loss calculation
        # (Custom wrapper needed for PyTorch MixtureSameFamily with Tanh)
        return MixtureTanhNormal(logits, means, stds)

# -------------------------------------------------------------------
# 5. ERA-RL V2 AGENT
# -------------------------------------------------------------------

class ERARL_Agent_V2(nn.Module):
    def __init__(self, obs_dim, hidden_dim=128, num_quantiles=32):
        super().__init__()
        self.encoder = DualStreamEncoder(obs_dim, hidden_dim)
        self.critic = HybridDistributionalCritic(hidden_dim, num_quantiles)
        self.actor = MixtureActor(hidden_dim, num_modes=3) # Normal, Defensive, Panic
        
    def forward(self, obs_sequence, prev_hedge):
        # 1. Encode
        latent, gru_history = self.encoder(obs_sequence, prev_hedge)
        
        # 2. Critic (Full Gradients allowed)
        risk_out = self.critic(latent, gru_history)
        
        # 3. Actor (Gradients blocked from flowing back to encoder/critic inside Actor)
        dist = self.actor(latent, risk_out)
        
        return dist, risk_out

    # Fix Gap 7: Modular Risk Functional
    def compute_risk_penalty(self, risk_out, alpha=0.05):
        """
        Computes the penalty term. Can be easily swapped for other metrics.
        """
        u = risk_out['u']
        sigma = risk_out['sigma']
        xi = risk_out['xi']
        
        # CVaR Formula (Exceedance)
        cvar = u + (sigma / (1 - xi))
        return cvar

    def get_diagnostics(self, obs_sequence, prev_hedge):
        """Introspection method for live visualization.

        Returns the internal risk view and mode selection without affecting gradients.
        """
        with torch.no_grad():
            # 1. Generate representation and risk metrics
            latent, gru_history = self.encoder(obs_sequence, prev_hedge)
            risk_out = self.critic(latent, gru_history)
            
            # 2. Get the distribution from the actor
            dist = self.actor(latent, risk_out)

            # 3. Extract diagnostics
            # 0: Normal, 1: Defensive, 2: Panic (usually)
            current_mode = torch.argmax(dist.cat_dist.logits, dim=-1).item()
            
            # Deterministic peak of the most likely mode
            expected_action = dist.mode().item()
            
            # Stochastic sample to reveal multimodal behavior in dashboards
            sampled_action = dist.sample().item() 

            return {
                'u': risk_out['u'].item(),
                'sigma': risk_out['sigma'].item(),
                'xi': risk_out['xi'].item(),
                'uncertainty': risk_out['uncertainty'].item(),
                'mode': current_mode,
                'action_mean': expected_action,
                'sampled_action': sampled_action, # Added this line
            }