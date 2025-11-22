import torch
import torch.nn as nn
from policy.model import ConvBlock  # Ton code existant
from policy.predictor import LatentPredictor # Ton code existant

class Encoder(nn.Module):
    """L'encodeur partagé (CNN)"""
    def __init__(self, input_channels=1):
        super().__init__()
        self.net = nn.Sequential(
            ConvBlock(input_channels, 32),
            ConvBlock(32, 64),
            ConvBlock(64, 128),
            ConvBlock(128, 256),
            ConvBlock(256, 512)
        )
        self.flatten_dim = 512 * 4 * 4

    def forward(self, x):
        return self.net(x).view(x.size(0), -1)

class PolicyHead(nn.Module):
    """La tête de décision (MLP) qui prend les latents"""
    def __init__(self, latent_dim, hidden_dim=512):
        super().__init__()
        # Entrée: Latent Actuel + Latent Target
        self.net = nn.Sequential(
            nn.Linear(latent_dim * 2, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, hidden_dim),
            nn.ReLU()
        )
        self.fc_mean = nn.Linear(hidden_dim, 2)
        self.fc_logstd = nn.Linear(hidden_dim, 2)
        self.fc_eos = nn.Linear(hidden_dim, 1)
        self.fc_eod = nn.Linear(hidden_dim, 1)

    def forward(self, z_current, z_target):
        feat = self.net(torch.cat([z_current, z_target], dim=1))
        mean = self.fc_mean(feat)
        logstd = torch.clamp(self.fc_logstd(feat), min=-20, max=2)  # Prevent numerical overflow
        return mean, logstd, self.fc_eos(feat), self.fc_eod(feat)

class FullAgent(nn.Module):
    def __init__(self, action_dim=2, hidden_dim=512):
        super().__init__()
        self.encoder = Encoder()
        
        # 1. Branche Predictor (World Model)
        self.predictor = LatentPredictor(
            latent_dim=self.encoder.flatten_dim,
            action_dim=action_dim,
            hidden_dim=256
        )
        
        # 2. Branche Policy (Actor)
        self.policy = PolicyHead(
            latent_dim=self.encoder.flatten_dim,
            hidden_dim=hidden_dim
        )

    def forward(self, canvas, target, next_canvas, action_taken):
        """
        Retourne tout ce qu'il faut pour calculer les deux pertes.
        Args:
            canvas: Image actuelle (t)
            target: Image finale objectif
            next_canvas: Image suivante (t+1) -> Pour le Predictor
            action_taken: L'action réelle entre t et t+1
        """
        # A. Encodage (Partagé)
        z_curr = self.encoder(canvas)
        z_targ = self.encoder(target)
        z_next_true = self.encoder(next_canvas) # Pour la loss du predictor

        # B. Prediction (JEPA Loss)
        # On essaie de deviner z_next à partir de z_curr + action
        z_next_pred, _ = self.predictor(z_curr, action_taken)

        # C. Policy (Actor Loss)
        # On prédit l'action à partir de z_curr + z_targ
        mean, logstd, eos, eod = self.policy(z_curr, z_targ)

        return {
            "z_next_pred": z_next_pred,
            "z_next_true": z_next_true,  # Target pour le predictor
            "action_pred": (mean, logstd, eos, eod) # Prediction pour la policy
        }

    def load_pretrained_encoder(self, checkpoint_path):
        """Load encoder weights from a pre-trained policy checkpoint."""
        print(f"Loading pretrained encoder from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'encoder_state_dict' in checkpoint:
            state_dict = checkpoint['encoder_state_dict']
        else:
            state_dict = checkpoint
            
        # Filter for encoder keys
        encoder_dict = {}
        for k, v in state_dict.items():
            if k.startswith('canvas_encoder.'):
                # Remove 'canvas_encoder.' prefix to match self.encoder.net
                new_key = k.replace('canvas_encoder.', 'net.')
                encoder_dict[new_key] = v
            elif k.startswith('encoder.'):
                 # If loading from another joint model
                new_key = k.replace('encoder.', '')
                encoder_dict[new_key] = v
                
        # Load weights
        missing, unexpected = self.encoder.load_state_dict(encoder_dict, strict=False)
        print(f"Encoder loaded. Missing: {len(missing)}, Unexpected: {len(unexpected)}")
        if len(missing) > 0:
            print(f"Missing keys (example): {missing[:5]}")