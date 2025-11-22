import torch
from torch.utils.data import Dataset
import numpy as np
from policy.data import PolicyDataset
from policy.model import PolicyNetwork

class LatentDynamicsDataset(Dataset):
    """
    Dataset for training the latent predictor.
    
    Generates sequences of (z_t, a_t, z_{t+1}) where:
    - z_t is the latent encoding of canvas at time t
    - a_t is the action taken at time t
    - z_{t+1} is the latent encoding of canvas at time t+1
    """
    
    def __init__(self, data_dir, categories, encoder, device, max_samples=1000, seq_length=10):
        """
        Args:
            data_dir: Path to QuickDraw data
            categories: List of categories to use
            encoder: Pre-trained canvas encoder (from PolicyNetwork)
            device: Device to run encoder on
            max_samples: Max drawings per category
            seq_length: Length of sequences to generate
        """
        self.policy_dataset = PolicyDataset(data_dir, categories, max_samples)
        self.encoder = encoder
        self.device = device
        self.seq_length = seq_length
        
        # Pre-compute all sequences
        print("Pre-computing latent sequences...")
        self.sequences = []
        self._precompute_sequences()
        
    def _precompute_sequences(self):
        """Pre-compute latent sequences for efficiency."""
        self.encoder.eval()
        
        # Group samples by drawing
        from collections import defaultdict
        drawings = defaultdict(list)
        
        for idx in range(len(self.policy_dataset)):
            canvas, target, action = self.policy_dataset[idx]
            # Extract drawing ID (we'll use a simple counter per category)
            # For now, we'll process sequentially
            
            # Get latent encoding
            with torch.no_grad():
                canvas_tensor = canvas.unsqueeze(0).to(self.device)
                z_t = self.encoder(canvas_tensor)  # [1, latent_dim]
                z_t = z_t.squeeze(0).cpu()  # [latent_dim]
            
            # Store (z_t, action)
            # We need to group these by drawing to create sequences
            # For simplicity, we'll create sequences on-the-fly in __getitem__
            # This is a placeholder - in practice, we'd need drawing IDs
            
        # For now, we'll just use the policy dataset directly
        # and create sequences by sampling consecutive steps
        print(f"Dataset ready with {len(self.policy_dataset)} samples")
    
    def __len__(self):
        # Return number of possible sequences
        return max(0, len(self.policy_dataset) - self.seq_length)
    
    def __getitem__(self, idx):
        """
        Returns a sequence of (z_t, a_t, z_{t+1}) pairs.
        
        Returns:
            z_sequence: [seq_length, latent_dim]
            actions: [seq_length, action_dim]
            z_next_sequence: [seq_length, latent_dim]
        """
        # Get consecutive samples
        z_sequence = []
        actions = []
        
        self.encoder.eval()
        with torch.no_grad():
            for t in range(self.seq_length + 1):
                canvas, target, action = self.policy_dataset[idx + t]
                
                # Encode canvas
                canvas_tensor = canvas.unsqueeze(0).to(self.device)
                z_t = self.encoder(canvas_tensor).squeeze(0).cpu()
                
                z_sequence.append(z_t)
                if t < self.seq_length:
                    actions.append(torch.tensor(action, dtype=torch.float32))
        
        z_sequence = torch.stack(z_sequence)  # [seq_length+1, latent_dim]
        actions = torch.stack(actions)  # [seq_length, action_dim]
        
        # Split into input and target
        z_input = z_sequence[:-1]  # [seq_length, latent_dim]
        z_target = z_sequence[1:]  # [seq_length, latent_dim]
        
        return z_input, actions, z_target


class SimpleLatentDynamicsDataset(Dataset):
    """
    Simplified version that generates single (z_t, a_t, z_{t+1}) transitions.
    More memory efficient than sequence-based approach.
    """
    
    def __init__(self, data_dir, categories, encoder, device, max_samples=1000):
        self.policy_dataset = PolicyDataset(data_dir, categories, max_samples)
        self.encoder = encoder
        self.device = device
        
    def __len__(self):
        return max(0, len(self.policy_dataset) - 1)
    
    def __getitem__(self, idx):
        """
        Returns a single transition (z_t, a_t, z_{t+1}).
        """
        self.encoder.eval()
        with torch.no_grad():
            # Get current state
            canvas_t, _, action_t = self.policy_dataset[idx]
            canvas_t = canvas_t.unsqueeze(0).to(self.device)
            z_t = self.encoder(canvas_t)  # [1, 512, 4, 4]
            z_t = z_t.view(z_t.size(0), -1).squeeze(0).cpu()  # Flatten to [8192]
            
            # Get next state
            canvas_t1, _, _ = self.policy_dataset[idx + 1]
            canvas_t1 = canvas_t1.unsqueeze(0).to(self.device)
            z_t1 = self.encoder(canvas_t1)  # [1, 512, 4, 4]
            z_t1 = z_t1.view(z_t1.size(0), -1).squeeze(0).cpu()  # Flatten to [8192]
            
            # Convert action to tensor (fix warning)
            action_t = torch.from_numpy(action_t).float() if isinstance(action_t, np.ndarray) else torch.tensor(action_t, dtype=torch.float32)
        
        return z_t, action_t, z_t1
