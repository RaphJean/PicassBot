import torch
import numpy as np
from PIL import Image
import argparse
import os

from policy.model import PolicyNetwork
from picassbot.engine import DrawingWorld
from picassbot.quickdraw import QuickDrawDataset

def run_inference(args):
    # Setup device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        
    print(f"Using device: {device}")
    
    # Load Model
    model = PolicyNetwork().to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()
    print(f"Loaded model from {args.model_path}")
    
    # Get Target
    # For demo, let's load a real drawing from QuickDraw to use as target
    # Or create a synthetic one.
    width, height = 128, 128
    
    if args.target_category:
        qd = QuickDrawDataset(args.data_dir)
        drawings = qd.load_drawings(args.target_category, max_drawings=1)
        if not drawings:
            print(f"No drawings found for {args.target_category}")
            return
        
        # Render target
        target_img = qd.vector_to_image(drawings[0], image_size=width, line_width=2)
        target_arr = np.array(target_img)
    else:
        # Simple square target
        target_img = Image.new("L", (width, height), color=255)
        from PIL import ImageDraw
        draw = ImageDraw.Draw(target_img)
        draw.rectangle([30, 30, 90, 90], outline=0, width=2)
        target_arr = np.array(target_img)
        
    # Save target
    os.makedirs(args.output_dir, exist_ok=True)
    Image.fromarray(target_arr).save(os.path.join(args.output_dir, "target.png"))
    
    # Run Policy
    world = DrawingWorld(width=width, height=height)
    frames = []
    
    with torch.no_grad():
        for step in range(args.max_steps):
            current_state = world.get_state()
            frames.append(Image.fromarray(current_state))
            
            # Prepare inputs
            current_tensor = torch.from_numpy(current_state).float().unsqueeze(0).unsqueeze(0) / 255.0
            target_tensor = torch.from_numpy(target_arr).float().unsqueeze(0).unsqueeze(0) / 255.0
            
            current_tensor = current_tensor.to(device)
            target_tensor = target_tensor.to(device)
            
            # Get action
            action_tensor = model.get_action(current_tensor, target_tensor, deterministic=not args.stochastic)
            action = action_tensor.cpu().numpy()[0]
            
            dx, dy, eos, eod = action
            print(f"Step {step}: dx={dx:.2f}, dy={dy:.2f}, eos={eos:.2f}, eod={eod:.2f}")
            
            if eod > 0.5:
                print("Policy chose to stop (EOD).")
                break
                
            world.step((dx, dy, eos, 0.0))
            
    # Save result
    frames.append(Image.fromarray(world.get_state()))
    frames[0].save(
        os.path.join(args.output_dir, "inference.gif"),
        save_all=True,
        append_images=frames[1:],
        duration=100,
        loop=0
    )
    print(f"Saved inference GIF to {args.output_dir}/inference.gif")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_dir", type=str, default="data/raw")
    parser.add_argument("--target_category", type=str, help="Category to draw (e.g. 'cat')")
    parser.add_argument("--output_dir", type=str, default="inference_output")
    parser.add_argument("--max_steps", type=int, default=50)
    parser.add_argument("--stochastic", action="store_true", help="Sample from distribution instead of mean")
    args = parser.parse_args()
    
    run_inference(args)
