import numpy as np
import copy
import torch
from picassbot.engine import DrawingWorld
from policy.model import PolicyNetwork

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
                self.device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
                self.policy_model = PolicyNetwork().to(self.device)
                checkpoint = torch.load(policy_model_path, map_location=self.device)
                if 'model_state_dict' in checkpoint:
                    self.policy_model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    self.policy_model.load_state_dict(checkpoint)
                self.policy_model.eval()
                print(f"Loaded policy guidance from {policy_model_path}")
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
    
    def loss(self, state):
        from scipy.ndimage import gaussian_filter
        target_smooth = gaussian_filter(self.target_image.astype(float), sigma=1.0)
        state_smooth = gaussian_filter(state.astype(float), sigma=1.0)
        diff = (target_smooth - state_smooth) / 255.0
        return np.mean(diff ** 2)

# ... (rest of the existing strategies remain the same)
