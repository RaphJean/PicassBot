import numpy as np
import copy
import torch
from picassbot.engine import DrawingWorld
from picassbot.policy.model import PolicyNetwork

class SearchStrategy:
    def __init__(self, world_config, target_image, step_penalty=0.00001, action_scale=0.3, allow_early_stopping=True, policy_model_path=None):
        self.world_config = world_config
        self.target_image = target_image # Numpy array (H, W)
        self.height, self.width = target_image.shape
        self.step_penalty = step_penalty
        self.action_scale = action_scale
        self.allow_early_stopping = allow_early_stopping
        
        self.policy_model = None
        self.device = None
        if policy_model_path:
            try:
                # Determine device first
                self.device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
                # Load checkpoint (supports raw state_dict or full checkpoint dict)
                checkpoint = torch.load(policy_model_path, map_location=self.device)
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                else:
                    state_dict = checkpoint
                # Infer hidden_dim from fusion layer shape
                fusion_weight = state_dict.get('fusion.0.weight')
                if fusion_weight is not None:
                    hidden_dim = fusion_weight.shape[0] // 2  # because first linear outputs hidden_dim*2
                else:
                    hidden_dim = 256  # fallback to default
                # Instantiate model with inferred hidden_dim
                self.policy_model = PolicyNetwork(hidden_dim=hidden_dim).to(self.device)
                self.policy_model.load_state_dict(state_dict)
                self.policy_model.eval()
                print(f"Loaded policy guidance from {policy_model_path} (hidden_dim={hidden_dim})")
            except Exception as e:
                print(f"Failed to load policy: {e}")
                self.policy_model = None

    def sample_action(self, state, step_idx, min_steps=5):
        """Sample an action from Policy if available, else Random."""
        if self.policy_model:
            with torch.no_grad():
                # Prepare inputs
                current_tensor = torch.from_numpy(state).float().unsqueeze(0).unsqueeze(0) / 255.0
                target_tensor = torch.from_numpy(self.target_image).float().unsqueeze(0).unsqueeze(0) / 255.0
                
                current_tensor = current_tensor.to(self.device)
                target_tensor = target_tensor.to(self.device)
                
                # Get action distribution
                action_tensor = self.policy_model.get_action(current_tensor, target_tensor, deterministic=False)
                action = action_tensor.cpu().numpy()[0] # (dx, dy, eos, eod)
                
                dx, dy, eos, eod = action
                
                # Enforce constraints
                if not self.allow_early_stopping or step_idx < min_steps:
                    eod = 0.0
                else:
                    # Policy outputs 0 or 1 for EOD, we keep it
                    pass
                    
                return (dx, dy, eos, eod)
        else:
            # Fallback to random
            dx = np.random.uniform(-self.action_scale, self.action_scale)
            dy = np.random.uniform(-self.action_scale, self.action_scale)
            eos = 1 if np.random.random() < 0.2 else 0
            
            if self.allow_early_stopping and step_idx >= min_steps:
                eod = 1 if np.random.random() < 0.05 else 0
            else:
                eod = 0
            
            return (dx, dy, eos, eod)

    def loss(self, image):
        """Calculate Mean Squared Error between blurred image and target."""
        from scipy.ndimage import gaussian_filter
        
        # Normalize to [0, 1] and cast to float
        img_norm = image.astype(np.float32) / 255.0
        target_norm = self.target_image.astype(np.float32) / 255.0
        
        # Apply Gaussian Blur to smooth the loss landscape
        sigma = 2.0
        img_blur = gaussian_filter(img_norm, sigma=sigma)
        target_blur = gaussian_filter(target_norm, sigma=sigma)
        
        return np.mean((img_blur - target_blur) ** 2)

    def run(self, max_steps=50):
        raise NotImplementedError
