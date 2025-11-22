import os
import numpy as np
from picassbot.engine import DrawingWorld

def main():
    output_dir = "output_engine_demo"
    os.makedirs(output_dir, exist_ok=True)
    
    print("Initializing Drawing World...")
    world = DrawingWorld(width=128, height=128)
    
    # Reset
    state = world.reset()
    world.render().save(os.path.join(output_dir, "step_0_blank.png"))
    
    # Define some actions (normalized coordinates)
    # Drawing a shape with curves
    actions = [
        (0.2, 0.5, 0.8, 0.5, 0.2),  # Curve up
        (0.8, 0.5, 0.2, 0.5, 0.2),  # Curve down (relative to direction, so this might overlap or flip depending on normal)
        (0.2, 0.2, 0.8, 0.2, -0.5), # Deep curve up
        (0.5, 0.5, 0.5, 0.9, 0.0),  # Straight line
    ]
    
    print(f"Executing {len(actions)} actions...")
    for i, action in enumerate(actions):
        state = world.step(action)
        filename = f"step_{i+1}.png"
        world.render().save(os.path.join(output_dir, filename))
        print(f"Action {i+1}: {action} -> Saved {filename}")
        
    print("Demo completed.")

if __name__ == "__main__":
    main()
