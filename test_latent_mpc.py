"""
Test script for Latent MPC strategy.
Demonstrates planning in latent space using learned dynamics.
"""

import torch
import numpy as np
from PIL import Image
import os
import argparse

from picassbot.planning.latent_mpc import LatentMPC

def create_simple_target(width=128, height=128):
    """Create a simple square target."""
    img = Image.new("L", (width, height), color=255)
    from PIL import ImageDraw
    draw = ImageDraw.Draw(img)
    draw.rectangle([30, 30, 98, 98], outline=0, width=2)
    return np.array(img)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--joint_model_path", type=str, required=True, help="Path to joint FullAgent checkpoint")
    parser.add_argument("--max_steps", type=int, default=30)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--num_sequences", type=int, default=20)
    args = parser.parse_args()
    
    # Create target
    target = create_simple_target()
    
    # Setup world config
    world_config = {"width": 128, "height": 128, "line_width": 2}
    
    # Create Latent MPC
    print(f"Loading Latent MPC from {args.joint_model_path}...")
    latent_mpc = LatentMPC(
        world_config=world_config,
        target_image=target,
        joint_model_path=args.joint_model_path,
        step_penalty=0.00001
    )
    
    print(f"Running Latent MPC (horizon={args.horizon}, sequences={args.num_sequences})...")
    history = latent_mpc.run(
        max_steps=args.max_steps,
        horizon=args.horizon,
        num_sequences=args.num_sequences
    )
    
    # Save results
    output_dir = "planning_output/latent_mpc"
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
