"""
Latent MPC Strategy - Model Predictive Control in Latent Space

This strategy performs planning in the learned latent space using:
1. Encoder: Maps state s_t to latent z_t
2. Predictor: Predicts z_{t+1} = f(z_t, a_t) (learned dynamics)
3. Policy heads: Generate actions from (z_t, z_target)
"""

import torch
import torch.nn as nn
import numpy as np
from picassbot.engine import DrawingWorld

class LatentMPC:
    """
    Model Predictive Control in latent space.
    
    Uses:
    - encoder: Maps canvas to latent representation
    - predictor: Predicts next latent state given current state and action
    - policy_heads: Generate actions from latent state and target
    - target_encoder: Encode target image to latent
    """
    
    def __init__(self, world_config, target_image, encoder, predictor, policy_heads, 
                 target_encoder, device, step_penalty=0.00001):
        self.world_config = world_config
        self.target_image = target_image
        self.encoder = encoder
        self.predictor = predictor
        self.policy_heads = policy_heads  # Dict with 'fusion', 'mean', 'logstd', 'eos', 'eod'
        self.target_encoder = target_encoder
        self.device = device
        self.step_penalty = step_penalty
        
        # Encode target once
        with torch.no_grad():
            target_tensor = torch.from_numpy(target_image).float().unsqueeze(0).unsqueeze(0) / 255.0
            target_tensor = target_tensor.to(device)
            z_target_feat = self.target_encoder(target_tensor)  # [1, 512, 4, 4]
            self.z_target = z_target_feat.view(z_target_feat.size(0), -1)  # [1, 8192]
        
        self.encoder.eval()
        self.predictor.eval()
        self.target_encoder.eval()
        
    def encode_state(self, state):
        """Encode canvas state to latent representation."""
        with torch.no_grad():
            state_tensor = torch.from_numpy(state).float().unsqueeze(0).unsqueeze(0) / 255.0
            state_tensor = state_tensor.to(self.device)
            z_feat = self.encoder(state_tensor)  # [1, 512, 4, 4]
            z = z_feat.view(z_feat.size(0), -1)  # [1, 8192]
        return z
    
    def generate_action_from_latent(self, z_t, deterministic=False):
        """
        Generate action from latent state and target.
        Uses the policy network's fusion + action heads.
        """
        with torch.no_grad():
            # Concatenate z_t and z_target
            combined = torch.cat([z_t, self.z_target], dim=1)  # [1, 16384]
            
            # Fusion
            latent = self.policy_heads['fusion'](combined)  # [1, hidden_dim]
            
            # Action heads
            mean = self.policy_heads['mean'](latent)
            logstd = self.policy_heads['logstd'](latent)
            logstd = torch.clamp(logstd, min=-5, max=2)
            
            eos_logit = self.policy_heads['eos'](latent)
            eod_logit = self.policy_heads['eod'](latent)
            
            # Sample action
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
            
            action = torch.cat([dx_dy, eos, eod], dim=1).squeeze(0).cpu().numpy()
        
        return action
    
    def latent_loss(self, z_t):
        """
        Compute loss in latent space.
        This is a simple L2 distance to target latent.
        """
        loss = torch.nn.functional.mse_loss(z_t, self.z_target)
        return loss.item()
    
    def run(self, max_steps=50, horizon=5, num_sequences=20):
        """
        Run Latent MPC.
        
        For each step:
        1. Encode current state to z_t
        2. For each candidate sequence:
            a. Simulate in latent space using predictor
            b. Compute latent loss
        3. Apply first action of best sequence in real world
        """
        world = DrawingWorld(**self.world_config)
        history = []
        
        for step in range(max_steps):
            # Encode current state
            z_t = self.encode_state(world.get_state())
            
            best_sequence = None
            best_cost = float('inf')
            
            # Generate and evaluate sequences in latent space
            for _ in range(num_sequences):
                z_sim = z_t.clone()
                sequence = []
                seq_cost = float('inf')
                
                for t in range(horizon):
                    # Generate action from current latent state
                    action = self.generate_action_from_latent(z_sim, deterministic=False)
                    dx, dy, eos, eod = action
                    sequence.append(action)
                    
                    if eod > 0.5:
                        # End of drawing
                        loss = self.latent_loss(z_sim)
                        seq_cost = loss + self.step_penalty * t
                        break
                    
                    # Predict next latent state
                    action_tensor = torch.from_numpy(action).float().unsqueeze(0).to(self.device)
                    z_next, _ = self.predictor(z_sim, action_tensor)
                    z_sim = z_next
                
                if seq_cost == float('inf'):
                    # Did not stop within horizon
                    loss = self.latent_loss(z_sim)
                    seq_cost = loss + self.step_penalty * horizon
                
                if seq_cost < best_cost:
                    best_cost = seq_cost
                    best_sequence = sequence
            
            # Apply first action in real world
            if best_sequence:
                first_action = best_sequence[0]
                dx, dy, eos, eod = first_action
                
                if eod > 0.5:
                    print(f"Step {step}: Stopping (EOD selected). Latent Cost {best_cost:.4f}")
                    break
                
                world.step(first_action)
                history.append(world.get_state())
                print(f"Step {step}: Latent Forecast Cost {best_cost:.4f}")
            else:
                break
        
        return history


def load_latent_mpc_components(policy_checkpoint_path, predictor_checkpoint_path, device):
    """
    Helper function to load all components needed for LatentMPC.
    
    Returns:
        encoder, predictor, policy_heads, target_encoder
    """
    from policy.model import PolicyNetwork
    from policy.predictor import LatentPredictor
    
    # Load policy network
    policy_net = PolicyNetwork()
    policy_checkpoint = torch.load(policy_checkpoint_path, map_location='cpu')
    if 'model_state_dict' in policy_checkpoint:
        policy_net.load_state_dict(policy_checkpoint['model_state_dict'])
    else:
        policy_net.load_state_dict(policy_checkpoint)
    
    policy_net.to(device)
    policy_net.eval()
    
    # Extract components
    encoder = policy_net.canvas_encoder
    target_encoder = policy_net.target_encoder
    
    policy_heads = {
        'fusion': policy_net.fusion,
        'mean': policy_net.fc_action_mean,
        'logstd': policy_net.fc_action_logstd,
        'eos': policy_net.fc_eos,
        'eod': policy_net.fc_eod
    }
    
    # Load predictor
    predictor_checkpoint = torch.load(predictor_checkpoint_path, map_location='cpu')
    latent_dim = predictor_checkpoint.get('latent_dim', 8192)
    
    predictor = LatentPredictor(latent_dim=latent_dim, action_dim=4)
    predictor.load_state_dict(predictor_checkpoint['model_state_dict'])
    predictor.to(device)
    predictor.eval()
    
    return encoder, predictor, policy_heads, target_encoder
