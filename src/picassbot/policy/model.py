import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        return self.pool(self.relu(self.bn(self.conv(x))))

class PolicyNetwork(nn.Module):
    def __init__(self, input_channels=1, hidden_dim=256):
        super().__init__()
        
        # Canvas Encoder (128x128 -> 4x4 feature map)
        self.canvas_encoder = nn.Sequential(
            ConvBlock(input_channels, 32),  # 64x64
            ConvBlock(32, 64),              # 32x32
            ConvBlock(64, 128),             # 16x16
            ConvBlock(128, 256),            # 8x8
            ConvBlock(256, 512)             # 4x4
        )
        
        # Target Encoder (Same architecture)
        self.target_encoder = self.canvas_encoder
        
        # Flatten size: 512 * 4 * 4 = 8192
        self.flatten_dim = 512 * 4 * 4
        
        # Fusion Layer
        self.fusion = nn.Sequential(
            nn.Linear(self.flatten_dim * 2, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU()
        )
        
        # Action Heads
        # 1. Continuous Head: dx, dy (Mean and LogStd) -> 4 outputs
        self.fc_action_mean = nn.Linear(hidden_dim, 2)
        self.fc_action_logstd = nn.Linear(hidden_dim, 2)
        
        # 2. Discrete Head: EOS (Pen Lift) -> 1 logit
        self.fc_eos = nn.Linear(hidden_dim, 1)
        
        # 3. Discrete Head: EOD (End of Drawing) -> 1 logit
        self.fc_eod = nn.Linear(hidden_dim, 1)

    def forward(self, canvas, target):
        """
        Args:
            canvas: (B, 1, 128, 128)
            target: (B, 1, 128, 128)
        Returns:
            action_mean: (B, 2)
            action_logstd: (B, 2)
            eos_logit: (B, 1)
            eod_logit: (B, 1)
        """
        # Encode
        c_feat = self.canvas_encoder(canvas)
        t_feat = self.target_encoder(target)
        
        # Flatten
        c_flat = c_feat.view(c_feat.size(0), -1)
        t_flat = t_feat.view(t_feat.size(0), -1)
        
        # Concatenate
        combined = torch.cat([c_flat, t_flat], dim=1)
        
        # Fuse
        latent = self.fusion(combined)
        
        # Heads
        action_mean = torch.tanh(self.fc_action_mean(latent)) # Bound means to [-1, 1] roughly, or let it be free?
        # Usually for relative actions in [-0.3, 0.3], tanh is good if we scale it, or just linear.
        # Let's use Linear for now to allow full range, but maybe clamp later.
        # Actually, our actions are small. Let's stick to Linear.
        action_mean = self.fc_action_mean(latent)
        
        action_logstd = self.fc_action_logstd(latent)
        action_logstd = torch.clamp(action_logstd, min=-5, max=2) # Stability
        
        eos_logit = self.fc_eos(latent)
        eod_logit = self.fc_eod(latent)
        
        return action_mean, action_logstd, eos_logit, eod_logit

    def get_action(self, canvas, target, deterministic=False):
        """Sample an action from the policy."""
        mean, logstd, eos_logit, eod_logit = self(canvas, target)
        
        std = torch.exp(logstd)
        
        if deterministic:
            dx_dy = mean
        else:
            dist = torch.distributions.Normal(mean, std)
            dx_dy = dist.sample()
            
        eos_prob = torch.sigmoid(eos_logit)
        eod_prob = torch.sigmoid(eod_logit)
        
        if deterministic:
            eos = (eos_prob > 0.5).float()
            eod = (eod_prob > 0.5).float()
        else:
            eos = torch.bernoulli(eos_prob)
            eod = torch.bernoulli(eod_prob)
            
        return torch.cat([dx_dy, eos, eod], dim=1)
