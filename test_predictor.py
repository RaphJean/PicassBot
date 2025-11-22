"""
Test script for the latent predictor.
Verifies that the model can be instantiated and performs a forward pass.
"""

import torch
from policy.predictor import LatentPredictor, LatentPredictorGRU

def test_predictor():
    print("Testing LatentPredictor...")
    
    # Parameters
    batch_size = 4
    latent_dim = 8192  # From canvas encoder (512 * 4 * 4)
    action_dim = 4
    hidden_dim = 256
    num_layers = 2
    
    # Create model
    predictor = LatentPredictor(
        latent_dim=latent_dim,
        action_dim=action_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers
    )
    
    print(f"Model created: {sum(p.numel() for p in predictor.parameters())} parameters")
    
    # Test single timestep
    print("\n1. Testing single timestep prediction...")
    z_t = torch.randn(batch_size, latent_dim)
    a_t = torch.randn(batch_size, action_dim)
    
    z_next, hidden = predictor(z_t, a_t)
    print(f"  Input z_t: {z_t.shape}")
    print(f"  Input a_t: {a_t.shape}")
    print(f"  Output z_next: {z_next.shape}")
    print(f"  Hidden state: h={hidden[0].shape}, c={hidden[1].shape}")
    
    assert z_next.shape == (batch_size, latent_dim), "Output shape mismatch!"
    print("  ✓ Single timestep test passed")
    
    # Test sequence prediction
    print("\n2. Testing sequence prediction...")
    seq_len = 10
    z_0 = torch.randn(batch_size, latent_dim)
    actions = torch.randn(batch_size, seq_len, action_dim)
    
    z_sequence = predictor.predict_sequence(z_0, actions, return_all=True)
    print(f"  Input z_0: {z_0.shape}")
    print(f"  Input actions: {actions.shape}")
    print(f"  Output sequence: {z_sequence.shape}")
    
    assert z_sequence.shape == (batch_size, seq_len, latent_dim), "Sequence shape mismatch!"
    print("  ✓ Sequence prediction test passed")
    
    # Test GRU variant
    print("\n3. Testing GRU variant...")
    predictor_gru = LatentPredictorGRU(
        latent_dim=latent_dim,
        action_dim=action_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers
    )
    
    z_next_gru, _ = predictor_gru(z_t, a_t)
    print(f"  GRU output: {z_next_gru.shape}")
    assert z_next_gru.shape == (batch_size, latent_dim), "GRU output shape mismatch!"
    print("  ✓ GRU test passed")
    
    print("\n✅ All tests passed!")

if __name__ == "__main__":
    test_predictor()
