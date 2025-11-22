# Joint dataset for encoder + predictor + policy training
"""Dataset that returns canvas, target, next_canvas, and action for FullAgent training.
"""

import torch
from torch.utils.data import Dataset
from policy.data import PolicyDataset

class JointLatentDynamicsDataset(Dataset):
    """Returns (canvas, target, next_canvas, action) for FullAgent training.
    - canvas: current state at time t (1, 128, 128) normalized to [0,1]
    - target: the final target image (1, 128, 128) normalized to [0,1]
    - next_canvas: state at time t+1 (1, 128, 128) normalized to [0,1]
    - action: action taken between t and t+1 (action_dim,)
    """
    def __init__(self, data_dir, categories, max_samples=1000, action_dim=4):
        self.policy_dataset = PolicyDataset(data_dir, categories, max_samples)
        self.action_dim = action_dim

    def __len__(self):
        # Need at least two consecutive frames
        return max(0, len(self.policy_dataset) - 1)

    def __getitem__(self, idx):
        # Get current state (t), target, and action
        canvas_t, target, action_t = self.policy_dataset[idx]
        # Get next state (t+1)
        canvas_t1, _, _ = self.policy_dataset[idx + 1]

        # Normalize to [0,1]
        if not isinstance(canvas_t, torch.Tensor):
            canvas_t = torch.from_numpy(canvas_t)
            target = torch.from_numpy(target)
            canvas_t1 = torch.from_numpy(canvas_t1)
        
        canvas_t = canvas_t.float() / 255.0
        target = target.float() / 255.0
        canvas_t1 = canvas_t1.float() / 255.0

        # Add channel dimension if needed
        if canvas_t.dim() == 2:
            canvas_t = canvas_t.unsqueeze(0)
        if target.dim() == 2:
            target = target.unsqueeze(0)
        if canvas_t1.dim() == 2:
            canvas_t1 = canvas_t1.unsqueeze(0)

        # Convert action to tensor
        if isinstance(action_t, (list, tuple)):
            action_t = torch.tensor(action_t[:self.action_dim], dtype=torch.float32)
        elif isinstance(action_t, torch.Tensor):
            action_t = action_t[:self.action_dim].float()
        else:
            # numpy array
            action_t = torch.from_numpy(action_t)[:self.action_dim].float()

        return canvas_t, target, canvas_t1, action_t
