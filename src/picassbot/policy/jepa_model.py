import torch
import torch.nn as nn
import copy
from picassbot.policy.joint_model import Encoder
from picassbot.policy.predictor import LatentPredictor

class JEPAWorldModel(nn.Module):
    """
    JEPA (Joint Embedding Predictive Architecture) World Model.
    
    Components:
    1. Online Encoder: Encodes current state s_t -> z_t. Updated via Gradient Descent.
    2. Target Encoder: Encodes next state s_{t+1} -> z_{t+1}. Updated via EMA (Exponential Moving Average).
    3. Predictor: Predicts z_{t+1} from z_t and action a_t.
    
    The Target Encoder provides a stable learning target and prevents latent collapse
    without requiring negative samples or explicit variance regularization (though var reg helps).
    """
    def __init__(self, action_dim=4, hidden_dim=256, ema_momentum=0.996):
        super().__init__()
        
        # 1. Online Encoder
        self.online_encoder = Encoder()
        
        # 2. Target Encoder (Copy of Online, frozen gradients)
        self.target_encoder = copy.deepcopy(self.online_encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad = False
            
        # 3. Predictor
        self.predictor = LatentPredictor(
            latent_dim=self.online_encoder.flatten_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim
        )
        
        self.ema_momentum = ema_momentum
        
    def forward(self, canvas, next_canvas, action):
        """
        Forward pass for training.
        
        Args:
            canvas: Current observation (t)
            next_canvas: Next observation (t+1)
            action: Action taken (t -> t+1)
            
        Returns:
            z_next_pred: Predicted latent state for t+1 (from Online branch)
            z_next_target: Actual latent state for t+1 (from Target branch)
        """
        # Online Branch: Encode t -> Predict t+1
        z_curr = self.online_encoder(canvas)
        z_next_pred, _ = self.predictor(z_curr, action)
        
        # Target Branch: Encode t+1 (No Grad)
        with torch.no_grad():
            z_next_target = self.target_encoder(next_canvas)
            
        return z_next_pred, z_next_target
    
    @torch.no_grad()
    def update_target_encoder(self):
        """
        Update Target Encoder weights using EMA.
        theta_target = m * theta_target + (1 - m) * theta_online
        """
        for param_online, param_target in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
            param_target.data = self.ema_momentum * param_target.data + (1 - self.ema_momentum) * param_online.data
