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
from picassbot.policy.joint_model import FullAgent
from picassbot.planning.base import SearchStrategy

class LatentMPC(SearchStrategy):
    """
    Model Predictive Control in latent space using FullAgent.
    
    Uses FullAgent's learned encoder, predictor, and policy for planning.
    """
    
    def __init__(self, world_config, target_image, joint_model_path, device=None, step_penalty=0.00001, **kwargs):
        # Initialize parent
        super().__init__(world_config, target_image, step_penalty=step_penalty, **kwargs)
        
        if device is None:
            self.device = torch.device(
                "mps" if torch.backends.mps.is_available() 
                else "cuda" if torch.cuda.is_available() 
                else "cpu"
            )
        else:
            self.device = device
            
        # Load FullAgent
        try:
            checkpoint = torch.load(joint_model_path, map_location=self.device)
            
            # Get config from checkpoint or use defaults
            if 'config' in checkpoint:
                cfg = checkpoint['config']
                # Check if config has model section, otherwise use defaults
                if 'model' in cfg:
                    action_dim = cfg['model'].get('action_dim', 4)
                    hidden_dim = cfg['model'].get('hidden_dim', 512)
                else:
                    action_dim = 4
                    hidden_dim = 512
            else:
                action_dim = 4
                hidden_dim = 512
            
            self.full_agent = FullAgent(
                action_dim=action_dim,
                hidden_dim=hidden_dim
            ).to(self.device)
            
            # Load state dicts
            # Check if it's a full checkpoint or just state dict
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
            
            # Infer hidden_dim from predictor weights if possible
            # predictor.lstm.weight_ih_l0 shape is [4*hidden_dim, input_dim]
            # or predictor.output_proj.0.weight is [hidden_dim, hidden_dim]
            inferred_hidden_dim = None
            if 'predictor.lstm.weight_ih_l0' in state_dict:
                inferred_hidden_dim = state_dict['predictor.lstm.weight_ih_l0'].shape[0] // 4
            elif 'predictor.output_proj.0.weight' in state_dict:
                inferred_hidden_dim = state_dict['predictor.output_proj.0.weight'].shape[0]
            
            if inferred_hidden_dim and inferred_hidden_dim != 256: # 256 is default
                print(f"Inferred predictor hidden_dim={inferred_hidden_dim} from checkpoint. Re-initializing FullAgent.")
                self.full_agent = FullAgent(
                    action_dim=action_dim,
                    hidden_dim=hidden_dim,
                    predictor_hidden_dim=inferred_hidden_dim
                ).to(self.device)
            
            # Handle JEPAWorldModel checkpoints (online_encoder -> encoder)
            new_state_dict = {}
            has_policy = False
            for k, v in state_dict.items():
                if k.startswith('online_encoder.'):
                    new_key = k.replace('online_encoder.', 'encoder.')
                    new_state_dict[new_key] = v
                elif k.startswith('predictor.'):
                    new_state_dict[k] = v
                elif k.startswith('policy.'):
                    new_state_dict[k] = v
                    has_policy = True
                elif k.startswith('encoder.'): # Already in FullAgent format
                    new_state_dict[k] = v
                # Ignore target_encoder for inference
            
            # Load what we can
            missing, unexpected = self.full_agent.load_state_dict(new_state_dict, strict=False)
            print(f"Loaded components. Missing: {missing}")
            
            if not has_policy:
                print("WARNING: No policy found in checkpoint. LatentMPC will use random actions for sampling.")
                self.full_agent.policy = None
            
            self.full_agent.eval()
            print(f"Loaded FullAgent from {joint_model_path}")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load FullAgent: {e}")
        
        # Encode target once
        with torch.no_grad():
            target_tensor = torch.from_numpy(target_image).float().unsqueeze(0).unsqueeze(0) / 255.0
            target_tensor = target_tensor.to(self.device)
            self.z_target = self.full_agent.encoder(target_tensor)
        
    def encode_state(self, state):
        """Encode canvas state to latent representation."""
        with torch.no_grad():
            state_tensor = torch.from_numpy(state).float().unsqueeze(0).unsqueeze(0) / 255.0
            state_tensor = state_tensor.to(self.device)
            z = self.full_agent.encoder(state_tensor)
        return z
    
    def generate_action_from_latent(self, z_t, deterministic=False):
        """
        Generate action from latent state and target using FullAgent policy.
        If policy is missing, sample random action.
        """
        if self.full_agent.policy is None:
            # Random action
            dx = np.random.uniform(-0.3, 0.3)
            dy = np.random.uniform(-0.3, 0.3)
            eos = 1.0 if np.random.random() < 0.1 else 0.0
            eod = 1.0 if (np.random.random() < 0.05 and self.allow_early_stopping) else 0.0
            return np.array([dx, dy, eos, eod])
            
        with torch.no_grad():
            # FullAgent policy takes (z_curr, z_targ)
            mean, logstd, eos_logit, eod_logit = self.full_agent.policy(z_t, self.z_target)
            
            logstd = torch.clamp(logstd, min=-5, max=2)
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
                eod = (eod_prob > 0.5).float() if self.allow_early_stopping else torch.zeros_like(eod_prob)
            else:
                eos = torch.bernoulli(eos_prob)
                eod = torch.bernoulli(eod_prob) if self.allow_early_stopping else torch.zeros_like(eod_prob)
            
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
                    z_next, _ = self.full_agent.predictor(z_sim, action_tensor)
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
                
                if eod > 0.5 and self.allow_early_stopping:
                    print(f"Step {step}: Stopping (EOD selected). Latent Cost {best_cost:.4f}")
                    break
                
                world.step(first_action)
                history.append(world.get_state())
                print(f"Step {step}: Latent Forecast Cost {best_cost:.4f}")
            else:
                break
        
        return history
