import os
import json
import torch
import numpy as np
from torch.utils.data import Dataset
from picassbot.engine import DrawingWorld

class QuickDrawRLDataset(Dataset):
    def __init__(self, data_dir, categories=None, max_drawings_per_category=1000, width=64, height=64):
        """
        PyTorch Dataset for Reinforcement Learning / World Model training.
        
        Args:
            data_dir (str): Path to the directory containing .ndjson files.
            categories (list): List of categories to load. If None, load all found.
            max_drawings_per_category (int): Limit number of drawings per category to save memory/time.
            width (int): Canvas width.
            height (int): Canvas height.
        """
        self.data_dir = data_dir
        self.width = width
        self.height = height
        self.drawings = [] # List of (category, raw_strokes)
        
        # Discover categories
        if categories is None:
            files = [f for f in os.listdir(data_dir) if f.endswith('.ndjson')]
            self.categories = [f.replace('.ndjson', '') for f in files]
        else:
            self.categories = categories
            
        print(f"Loading dataset with {len(self.categories)} categories...")
        
        for cat in self.categories:
            filepath = os.path.join(data_dir, f"{cat}.ndjson")
            if not os.path.exists(filepath):
                continue
                
            with open(filepath, 'r') as f:
                count = 0
                for line in f:
                    try:
                        data = json.loads(line)
                        if data['recognized']: # Only use recognized drawings
                            self.drawings.append(data['drawing'])
                            count += 1
                    except Exception:
                        continue
                        
                    if count >= max_drawings_per_category:
                        break
        
        print(f"Loaded {len(self.drawings)} drawings total.")

    def __len__(self):
        return len(self.drawings)

    def __getitem__(self, idx):
        """
        Returns a full episode or a random transition?
        For World Model training, we usually want a sequence or a transition.
        Here, let's return a random transition (s_t, a_t, s_{t+1}) from the drawing.
        """
        raw_strokes = self.drawings[idx]
        
        # Convert raw strokes to a sequence of relative actions (dx, dy, eos, eod)
        actions = self._strokes_to_actions(raw_strokes)
        
        if len(actions) == 0:
            # Handle empty drawing case (should be rare)
            return self.__getitem__((idx + 1) % len(self))

        # Pick a random step t
        t = np.random.randint(0, len(actions))
        action = actions[t]
        
        # Reconstruct state s_t by replaying actions 0 to t-1
        world = DrawingWorld(width=self.width, height=self.height)
        # Optimization: We could cache states, but for now we replay.
        # Replaying is slow but memory efficient.
        
        for i in range(t):
            world.step(actions[i])
            
        state_t = world.get_state().astype(np.float32) / 255.0 # Normalize [0, 1]
        pen_t = world.pen_pos.copy()
        
        # Apply action to get s_{t+1}
        world.step(action)
        state_next = world.get_state().astype(np.float32) / 255.0
        pen_next = world.pen_pos.copy()
        
        # Convert to tensors
        # State: (1, H, W)
        s_t = torch.from_numpy(state_t).unsqueeze(0)
        s_next = torch.from_numpy(state_next).unsqueeze(0)
        
        # Action: (4,) -> (dx, dy, eos, eod)
        a_t = torch.tensor(action, dtype=torch.float32)
        
        # Pen state: (2,) -> (x, y)
        p_t = torch.from_numpy(pen_t).float()
        p_next = torch.from_numpy(pen_next).float()
        
        return {
            "state": s_t,
            "action": a_t,
            "next_state": s_next,
            "pen_pos": p_t,
            "next_pen_pos": p_next
        }

    def _strokes_to_actions(self, strokes):
        """
        Convert QuickDraw strokes format [[x_pts], [y_pts], [t_pts]] 
        to relative actions (dx, dy, eos, eod).
        """
        actions = []
        # QuickDraw coordinates are 0-255. We need to normalize to 0-1.
        scale = 255.0
        
        current_pen = np.array([0.0, 0.0])
        
        for stroke_idx, stroke in enumerate(strokes):
            x_pts, y_pts = stroke[0], stroke[1]
            
            for i in range(len(x_pts)):
                # Target point normalized
                target = np.array([x_pts[i], y_pts[i]]) / scale
                
                # Calculate delta
                delta = target - current_pen
                dx, dy = delta[0], delta[1]
                
                # Determine EOS/EOD
                # If it's the first point of a stroke, we moved there with pen UP (unless it's the very first point of drawing at 0,0)
                # Actually, QuickDraw data assumes pen down for the stroke points.
                # But we need to move TO the start of the stroke first.
                
                if i == 0:
                    # Move to start of stroke (Pen Up)
                    # Check if we are already there (e.g. continued line) - unlikely in raw data
                    if np.linalg.norm(delta) > 1e-4:
                        # Action: Move pen up
                        actions.append((dx, dy, 1.0, 0.0)) # EOS=1 (Lift)
                        current_pen = target
                        # Now we are at the start, we need to draw to the next point?
                        # No, the first point is just a position. We don't draw "to" the first point, we move "to" it.
                        # The drawing happens between points.
                        continue
                    else:
                        # We are already at start
                        pass
                else:
                    # Draw to next point (Pen Down)
                    actions.append((dx, dy, 0.0, 0.0)) # EOS=0 (Draw)
                    current_pen = target
            
            # End of stroke
            # The last point was reached. The next action will handle the move to the next stroke.
            
        # End of Drawing
        actions.append((0.0, 0.0, 1.0, 1.0)) # EOD=1
        
        return actions
