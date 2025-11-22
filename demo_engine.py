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
    
    # Define some actions (dx, dy, eos, eod)
    # Drawing a square: Start at (0.2, 0.2)
    # 1. Move to start (pen up)
    # 2. Draw right
    # 3. Draw down
    # 4. Draw left
    # 5. Draw up
    actions = [
        (0.2, 0.2, 1, 0),   # Move to (0.2, 0.2) - Pen Up
        (0.6, 0.0, 0, 0),   # Draw to (0.8, 0.2) - Pen Down
        (0.0, 0.6, 0, 0),   # Draw to (0.8, 0.8) - Pen Down
        (-0.6, 0.0, 0, 0),  # Draw to (0.2, 0.8) - Pen Down
        (0.0, -0.6, 0, 0),  # Draw to (0.2, 0.2) - Pen Down
        (0.0, 0.0, 1, 1),   # End drawing
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
