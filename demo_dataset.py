import os
import torch
import numpy as np
from PIL import Image
from picassbot.dataset import QuickDrawRLDataset

def main():
    data_dir = "data/raw"
    output_dir = "output_dataset_demo"
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize dataset with a few categories
    categories = ['apple', 'face', 'car']
    print(f"Initializing dataset with categories: {categories}")
    
    dataset = QuickDrawRLDataset(
        data_dir=data_dir,
        categories=categories,
        max_drawings_per_category=10
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    # Test getting a few items
    for i in range(5):
        print(f"\n--- Sample {i} ---")
        sample = dataset[i]
        
        state = sample['state']
        action = sample['action']
        next_state = sample['next_state']
        pen_pos = sample['pen_pos']
        next_pen_pos = sample['next_pen_pos']
        
        print(f"State shape: {state.shape}")
        print(f"Action: {action} (dx, dy, eos, eod)")
        print(f"Pen Pos: {pen_pos} -> {next_pen_pos}")
        
        # Save images
        s_img = Image.fromarray((state.squeeze().numpy() * 255).astype(np.uint8))
        ns_img = Image.fromarray((next_state.squeeze().numpy() * 255).astype(np.uint8))
        
        s_img.save(os.path.join(output_dir, f"sample_{i}_state.png"))
        ns_img.save(os.path.join(output_dir, f"sample_{i}_next_state.png"))
        print(f"Saved sample_{i}_state.png and sample_{i}_next_state.png")

if __name__ == "__main__":
    main()
