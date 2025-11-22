import torch
import torch.nn as nn

class LatentPredictor(nn.Module):
    """
    LSTM-based predictor that learns latent dynamics.
    
    Takes:
        - z_t: latent state at time t (from canvas encoder) [B, latent_dim]
        - a_t: action at time t [B, action_dim]
    
    Predicts:
        - z_{t+1}: latent state at time t+1 [B, latent_dim]
    """
    
    def __init__(self, latent_dim=8192, action_dim=4, hidden_dim=512, num_layers=2):
        """
        Args:
            latent_dim: Dimension of latent state z (from canvas encoder)
            action_dim: Dimension of action (dx, dy, eos, eod = 4)
            hidden_dim: Hidden dimension of LSTM
            num_layers: Number of LSTM layers
        """
        super().__init__()
        
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Input: concatenate z_t and a_t
        input_dim = latent_dim + action_dim
        
        # LSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        
        # Output projection to predict z_{t+1}
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )
        
    def forward(self, z_t, a_t, hidden_state=None):
        """
        Forward pass for single timestep or sequence.
        
        Args:
            z_t: Latent state [B, latent_dim] or [B, T, latent_dim]
            a_t: Action [B, action_dim] or [B, T, action_dim]
            hidden_state: Optional LSTM hidden state (h, c) for sequential prediction
        
        Returns:
            z_next: Predicted next latent state [B, latent_dim] or [B, T, latent_dim]
            hidden_state: Updated LSTM hidden state (h, c)
        """
        # Handle single timestep vs sequence
        if z_t.dim() == 2:
            # Single timestep: [B, latent_dim] -> [B, 1, latent_dim]
            z_t = z_t.unsqueeze(1)
            a_t = a_t.unsqueeze(1)
            squeeze_output = True
        else:
            squeeze_output = False
        
        # Concatenate z_t and a_t
        x = torch.cat([z_t, a_t], dim=-1)  # [B, T, latent_dim + action_dim]
        
        # LSTM forward
        if hidden_state is None:
            lstm_out, hidden_state = self.lstm(x)
        else:
            lstm_out, hidden_state = self.lstm(x, hidden_state)
        
        # Project to latent space
        z_next = self.output_proj(lstm_out)  # [B, T, latent_dim]
        
        # Squeeze if single timestep
        if squeeze_output:
            z_next = z_next.squeeze(1)  # [B, latent_dim]
        
        return z_next, hidden_state
    
    def predict_sequence(self, z_0, actions, return_all=True):
        """
        Predict a sequence of latent states given initial state and actions.
        
        Args:
            z_0: Initial latent state [B, latent_dim]
            actions: Sequence of actions [B, T, action_dim]
            return_all: If True, return all intermediate states, else only final
        
        Returns:
            z_sequence: Predicted latent states [B, T, latent_dim] or [B, latent_dim]
        """
        batch_size, seq_len, _ = actions.shape
        device = z_0.device
        
        z_sequence = []
        z_t = z_0
        hidden_state = None
        
        for t in range(seq_len):
            a_t = actions[:, t, :]  # [B, action_dim]
            z_next, hidden_state = self.forward(z_t, a_t, hidden_state)
            z_sequence.append(z_next)
            z_t = z_next
        
        z_sequence = torch.stack(z_sequence, dim=1)  # [B, T, latent_dim]
        
        if return_all:
            return z_sequence
        else:
            return z_sequence[:, -1, :]  # Return only final state


class LatentPredictorGRU(nn.Module):
    """
    GRU-based variant of LatentPredictor (alternative to LSTM).
    """
    
    def __init__(self, latent_dim=8192, action_dim=4, hidden_dim=512, num_layers=2):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        input_dim = latent_dim + action_dim
        
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )
    
    def forward(self, z_t, a_t, hidden_state=None):
        if z_t.dim() == 2:
            z_t = z_t.unsqueeze(1)
            a_t = a_t.unsqueeze(1)
            squeeze_output = True
        else:
            squeeze_output = False
        
        x = torch.cat([z_t, a_t], dim=-1)
        
        if hidden_state is None:
            gru_out, hidden_state = self.gru(x)
        else:
            gru_out, hidden_state = self.gru(x, hidden_state)
        
        z_next = self.output_proj(gru_out)
        
        if squeeze_output:
            z_next = z_next.squeeze(1)
        
        return z_next, hidden_state
    
    def predict_sequence(self, z_0, actions, return_all=True):
        batch_size, seq_len, _ = actions.shape
        
        z_sequence = []
        z_t = z_0
        hidden_state = None
        
        for t in range(seq_len):
            a_t = actions[:, t, :]
            z_next, hidden_state = self.forward(z_t, a_t, hidden_state)
            z_sequence.append(z_next)
            z_t = z_next
        
        z_sequence = torch.stack(z_sequence, dim=1)
        
        if return_all:
            return z_sequence
        else:
            return z_sequence[:, -1, :]
