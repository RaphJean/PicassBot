"""
Test script for Latent MPC strategy.
Demonstrates planning in latent space using learned dynamics.
"""

import torch
import numpy as np
from PIL import Image
import os
import argparse

from research.latent_mpc import LatentMPC, load_latent_mpc_components

def create_simple_target(width=128, height=128):
    """Create a simple square target."""
    img = Image.new("L", (width, height), color=255)
    from PIL import ImageDraw
    draw = ImageDraw.Draw(img)
    draw.rectangle([30, 30, 98, 98], outline=0, width=2)
    return np.array(img)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy_path", type=str, required=True, help="Path to policy checkpoint")
    parser.add_argument("--predictor_path", type=str, required=True, help="Path to predictor checkpoint")
    parser.add_argument("--max_steps", type=int, default=30)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--num_sequences", type=int, default=20)
    args = parser.parse_args()
    
    # Device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    
    # Load components
    print("Loading models...")
    encoder, predictor, policy_heads, target_encoder = load_latent_mpc_components(
        args.policy_path,
        args.predictor_path,
        device
    )
    
    # Create target
    target = create_simple_target()
    
    # Setup world config
    world_config = {"width": 128, "height": 128, "line_width": 2}
    
    # Create Latent MPC
    latent_mpc = LatentMPC(
        world_config=world_config,
        target_image=target,
        encoder=encoder,
        predictor=predictor,
        policy_heads=policy_heads,
        target_encoder=target_encoder,
        device=device,
        step_penalty=0.00001
    )
    
    print(f"Running Latent MPC (horizon={args.horizon}, sequences={args.num_sequences})...")
    history = latent_mpc.run(
        max_steps=args.max_steps,
        horizon=args.horizon,
        num_sequences=args.num_sequences
    )
    
    # Save results
    output_dir = "research_output/latent_mpc"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save target
    Image.fromarray(target.astype(np.uint8)).save(os.path.join(output_dir, "target.png"))
    
    # Save frames and create GIF
    images = []
    for i, state in enumerate(history):
        img = Image.fromarray(state.astype(np.uint8))
        img.save(os.path.join(output_dir, f"step_{i:03d}.png"))
        images.append(img)
    
    if images:
        gif_path = os.path.join(output_dir, "latent_mpc.gif")
        images[0].save(
            gif_path,
            save_all=True,
            append_images=images[1:],
            duration=200,
            loop=0
        )
        print(f"\nSaved GIF to {gif_path}")
        print(f"Total steps: {len(history)}")
    else:
        print("No steps generated!")

if __name__ == "__main__":
    main()
