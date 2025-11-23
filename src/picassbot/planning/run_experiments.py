import os
import numpy as np
from PIL import Image, ImageDraw
from picassbot.planning.strategies import GreedySearch, RandomShootingMPC, GeneticSearch, CEMSearch, MCTSSearch, PolicyStrategy, LatentMPC
from picassbot.engine import DrawingWorld

def create_target_image(shape_type="triangle", width=128, height=128):
    """Create a simple target image."""
    img = Image.new("L", (width, height), color=255)
    draw = ImageDraw.Draw(img)
    
    if shape_type == "triangle":
        points = [(64, 20), (20, 100), (108, 100), (64, 20)]
        draw.line(points, fill=0, width=2)
    elif shape_type == "square":
        draw.rectangle([30, 30, 98, 98], outline=0, width=2)
    elif shape_type == "circle":
        draw.ellipse([30, 30, 98, 98], outline=0, width=2)
    else:
        raise ValueError(f"Unknown shape type: {shape_type}")
        
    return np.array(img)

def save_sequence(history, output_dir, prefix, target_image=None, save_gif=True):
    os.makedirs(output_dir, exist_ok=True)
    images = []
    
    # Prepare target overlay if provided
    overlay_img = None
    if target_image is not None:
        # Target is (H, W) grayscale. 0=black (line), 255=white.
        # Invert so lines are 255
        inv_target = 255 - target_image
        # Create Red image
        red = np.zeros((target_image.shape[0], target_image.shape[1], 3), dtype=np.uint8)
        red[:, :, 0] = 255 # Red channel
        
        # Create Alpha channel: 0.2 * 255 = 51 where target is line
        # We scale inv_target (0-255) to (0-51)
        alpha = (inv_target.astype(float) / 255.0) * 51
        
        # Combine to RGBA
        overlay_rgba = np.dstack((red, alpha.astype(np.uint8)))
        overlay_img = Image.fromarray(overlay_rgba, 'RGBA')

    for i, state in enumerate(history):
        # State is grayscale
        img = Image.fromarray(state.astype(np.uint8)).convert("RGBA")
        
        if overlay_img:
            img = Image.alpha_composite(img, overlay_img)
            
        img.save(os.path.join(output_dir, f"{prefix}_step_{i:03d}.png"))
        if save_gif:
            images.append(img)
    
    if save_gif and images:
        gif_path = os.path.join(output_dir, f"{prefix}.gif")
        images[0].save(
            gif_path,
            save_all=True,
            append_images=images[1:],
            duration=200,
            loop=0
        )
        print(f"Saved GIF to {gif_path}")

import argparse

from picassbot.quickdraw import QuickDrawDataset

def get_dataset_target(category, data_dir="data/raw", index=None, width=128, height=128):
    """Load a target image from QuickDraw dataset."""
    qd = QuickDrawDataset(data_dir)
    # Load a few drawings to find one
    drawings = qd.load_drawings(category, max_drawings=1000)
    
    if not drawings:
        raise ValueError(f"No drawings found for category {category}")
        
    if index is None:
        # Pick a random one from the end (simulating unseen if trained on beginning)
        index = np.random.randint(len(drawings) // 2, len(drawings))
    
    if index >= len(drawings):
        index = index % len(drawings)
        
    target_img = qd.vector_to_image(drawings[index], image_size=width, line_width=2)
    return np.array(target_img)

def main():
    parser = argparse.ArgumentParser(description="Run drawing search experiments.")
    parser.add_argument("--strategy", type=str, default="greedy", choices=["greedy", "mpc", "genetic", "cem", "mcts", "policy", "latent_mpc", "all"], help="Strategy to use for drawing")
    parser.add_argument("--target_type", type=str, choices=["triangle", "square", "circle", "dataset"], default="triangle", help="Type of target to draw")
    parser.add_argument("--target_category", type=str, help="Category for dataset target (e.g. 'cat')")
    parser.add_argument("--target_index", type=int, help="Index of drawing to use as target")
    parser.add_argument("--data_dir", type=str, default="data/raw", help="Path to QuickDraw data")
    
    parser.add_argument("--step_penalty", type=float, default=0.00001, help="Penalty for each step taken")
    parser.add_argument("--step_variance", type=float, default=0.5, help="Initial variance (std) for CEM search")
    parser.add_argument("--action_scale", type=float, default=0.3, help="Scale of random actions (dx, dy range)")
    
    parser.add_argument("--mcts_horizon", type=int, default=5, help="Horizon length for MCTS rollouts")
    parser.add_argument("--mcts_simulations", type=int, default=100, help="Number of simulations per step for MCTS")
    parser.add_argument("--no_early_stopping", action="store_true", help="Disable early stopping (EOD action)")
    parser.add_argument("--model_path", type=str, default="policy_checkpoints/policy_epoch_2.pth", help="Path to trained policy model checkpoint")
    parser.add_argument("--joint_model_path", type=str, help="Path to joint FullAgent checkpoint (for LatentMPC)")
    parser.add_argument("--predictor_path", type=str, help="[DEPRECATED] Use --joint_model_path instead")
    parser.add_argument("--horizon", type=int, default=5, help="Planning horizon for Latent MPC")
    parser.add_argument("--num_sequences", type=int, default=20, help="Number of sequences for Latent MPC")
    args = parser.parse_args()

    width, height = 128, 128
    
    # Create target
    if args.target_type == "dataset":
        if not args.target_category:
            raise ValueError("--target_category is required when target_type is 'dataset'")
        print(f"Loading target from dataset: {args.target_category}")
        target = get_dataset_target(args.target_category, args.data_dir, args.target_index, width, height)
    else:
        print(f"Creating geometric target: {args.target_type}")
        target = create_target_image(args.target_type, width, height)
        
    # Determine output subdirectory
    if args.target_type == "dataset":
        sub_dir = f"{args.target_category}_{args.target_index}" if args.target_index is not None else args.target_category
    else:
        sub_dir = args.target_type
        
    base_output_dir = os.path.join("planning_output", sub_dir)
    os.makedirs(base_output_dir, exist_ok=True)
    Image.fromarray(target.astype(np.uint8)).save(os.path.join(base_output_dir, "target.png"))
    
    world_config = {"width": width, "height": height, "line_width": 2}
    
    if args.strategy in ["greedy", "all"]:
        print("Running Greedy Search...")
        greedy = GreedySearch(world_config, target, step_penalty=args.step_penalty, action_scale=args.action_scale, allow_early_stopping=not args.no_early_stopping, policy_model_path=args.model_path)
        history_greedy = greedy.run(max_steps=30, samples_per_step=50)
        save_sequence(history_greedy, os.path.join(base_output_dir, "greedy"), "greedy", target_image=target)
    
    if args.strategy in ["mpc", "all"]:
        print("\nRunning Random Shooting MPC...")
        # Increased samples and horizon for better performance
        # User requested horizon=2 previously
        mpc = RandomShootingMPC(world_config, target, step_penalty=args.step_penalty, action_scale=args.action_scale, allow_early_stopping=not args.no_early_stopping, policy_model_path=args.model_path)
        history_mpc = mpc.run(max_steps=300, horizon=2, num_sequences=500)
        save_sequence(history_mpc, os.path.join(base_output_dir, "mpc"), "mpc", target_image=target)
    
    if args.strategy in ["genetic", "all"]:
        print("\nRunning Genetic Search...")
        genetic = GeneticSearch(world_config, target, step_penalty=args.step_penalty, action_scale=args.action_scale, allow_early_stopping=not args.no_early_stopping, policy_model_path=args.model_path)
        history_genetic = genetic.run(max_steps=30, population_size=30, generations=5)
        save_sequence(history_genetic, os.path.join(base_output_dir, "genetic"), "genetic", target_image=target)

    if args.strategy in ["cem", "all"]:
        print("\nRunning CEM Search...")
        cem = CEMSearch(world_config, target, step_penalty=args.step_penalty, action_scale=args.action_scale, allow_early_stopping=not args.no_early_stopping, policy_model_path=args.model_path)
        history_cem = cem.run(max_steps=30, horizon=10, num_sequences=50, num_elites=10, num_iterations=5, initial_std=args.step_variance)
        save_sequence(history_cem, os.path.join(base_output_dir, "cem"), "cem", target_image=target)

    if args.strategy in ["mcts", "all"]:
        print("\nRunning MCTS Search...")
        mcts = MCTSSearch(world_config, target, step_penalty=args.step_penalty, action_scale=args.action_scale, allow_early_stopping=not args.no_early_stopping, policy_model_path=args.model_path)
        history_mcts = mcts.run(max_steps=30, horizon=args.mcts_horizon, num_simulations=args.mcts_simulations)
        save_sequence(history_mcts, os.path.join(base_output_dir, "mcts"), "mcts", target_image=target)

    if args.strategy in ["policy", "all"]:
        print("\nRunning Policy Inference...")
        policy = PolicyStrategy(world_config, target, model_path=args.model_path, step_penalty=args.step_penalty, action_scale=args.action_scale, allow_early_stopping=not args.no_early_stopping)
        history_policy = policy.run(max_steps=30, deterministic=True) # Use deterministic for demo
        save_sequence(history_policy, os.path.join(base_output_dir, "policy"), "policy", target_image=target)

    if args.strategy in ["latent_mpc", "all"]:
        if not args.joint_model_path:
            print("\nSkipping Latent MPC (requires --joint_model_path)")
        else:
            print("\nRunning Latent MPC...")
            latent_mpc = LatentMPC(world_config, target, step_penalty=args.step_penalty, joint_model_path=args.joint_model_path, allow_early_stopping=not args.no_early_stopping)
            history_latent_mpc = latent_mpc.run(max_steps=30, horizon=args.horizon, num_sequences=args.num_sequences)
            save_sequence(history_latent_mpc, os.path.join(base_output_dir, "latent_mpc"), "latent_mpc", target_image=target)
if __name__ == "__main__":
    main()
