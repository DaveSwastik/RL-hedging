# # src/agents/models.py    ##Mult-Layer Perceptron model
# import torch
# import torch.nn as nn

# class CustomMLPPolicy(nn.Module):
#     """
#     A simple custom policy network.
#     It takes an observation and outputs a continuous action.
#     """
#     def __init__(self, obs_dim: int, action_dim: int):
#         super(CustomMLPPolicy, self).__init__()
        
#         # Define the network layers
#         self.network = nn.Sequential(
#             nn.Linear(obs_dim, 256),
#             nn.Tanh(),
#             nn.Linear(256, 256),
#             nn.Tanh(),
#             nn.Linear(256, action_dim) # Output layer
#         )

#     def forward(self, obs: torch.Tensor) -> torch.Tensor:
#         """
#         Defines the forward pass of the model.
#         """
#         return self.network(obs)




##GRU 
# src/agents/models.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class InterpretableHedger(nn.Module):
    """
    A GRU-based recurrent policy with an Attention mechanism.
    
    This model addresses the "interpretability" research gap by exposing
    attention weights, which show which past time steps the agent
    focused on when making a decision.
    """
    def __init__(self, obs_dim: int, hidden_dim: int = 128):
        super(InterpretableHedger, self).__init__()
        self.hidden_dim = hidden_dim
        
        # 1. GRU Layer to process sequence and create memory
        self.gru = nn.GRU(obs_dim, hidden_dim, batch_first=True)
        
        # 2. Attention Mechanism Layers
        self.attention_net = nn.Linear(hidden_dim, 1)
        
        # 3. Final output layer
        self.output_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1) # Outputs a single continuous action
        )

    def forward(self, obs_sequence: torch.Tensor):
        """
        Defines the forward pass of the model.
        
        Args:
            obs_sequence (torch.Tensor): A tensor of shape 
                                         (batch_size, sequence_length, obs_dim)

        Returns:
            torch.Tensor: The hedging action.
            torch.Tensor: The attention weights for interpretability.
        """
        # Pass sequence through GRU
        # gru_outputs shape: (batch_size, seq_len, hidden_dim)
        gru_outputs, _ = self.gru(obs_sequence)
        
        # --- Attention Mechanism ---
        # Compute attention scores for each time step
        # energy shape: (batch_size, seq_len, 1)
        energy = self.attention_net(gru_outputs)
        
        # Convert scores to probabilities (weights)
        # attention_weights shape: (batch_size, seq_len, 1)
        attention_weights = F.softmax(energy, dim=1)
        
        # Create context vector by taking a weighted average of GRU outputs
        # context_vector shape: (batch_size, 1, hidden_dim)
        context_vector = torch.bmm(attention_weights.transpose(1, 2), gru_outputs)
        context_vector = context_vector.squeeze(1) # Shape: (batch_size, hidden_dim)
        
        # --- Final Action ---
        # Pass the context vector through the output network
        action = self.output_net(context_vector)
        
        return action, attention_weights.squeeze(-1)