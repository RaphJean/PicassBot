import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image, ImageDraw
from picassbot.quickdraw import QuickDrawDataset
from picassbot.engine import DrawingWorld

class PolicyDataset(Dataset):
    def __init__(self, data_dir, categories, max_samples=1000, img_size=128):
        self.qd = QuickDrawDataset(data_dir)
        self.drawings = []
        
        # Load data for all categories
        for cat in categories:
            try:
                cat_drawings = self.qd.load_drawings(cat, max_drawings=max_samples)
                # Extract the 'drawing' field which contains the strokes
                self.drawings.extend([d['drawing'] for d in cat_drawings])
            except FileNotFoundError:
                print(f"Warning: Category {cat} not found.")
                
        self.img_size = img_size
        
        # Pre-process data into triplets: (drawing_idx, step_idx)
        self.samples = []
        for i, drawing in enumerate(self.drawings):
            # drawing is a list of strokes: [ [[x...], [y...]], [[x...], [y...]] ]
            
            # First, convert drawing to a sequence of actions
            actions = self._drawing_to_actions(drawing)
            
            # Add each step as a sample
            for t in range(len(actions)):
                self.samples.append((i, t))
                
    def _drawing_to_actions(self, drawing):
        """Convert raw QuickDraw strokes to (dx, dy, eos, eod) actions."""
        actions = []
        current_pos = np.array([0.0, 0.0]) 
        
        # Scale factor: 255 -> 1.0
        scale = 1.0 / 255.0
        
        for stroke in drawing:
            # QuickDraw format: stroke = [[x1, x2, ...], [y1, y2, ...]]
            x_points = stroke[0]
            y_points = stroke[1]
            num_points = len(x_points)
            
            for i in range(num_points):
                x, y = x_points[i], y_points[i]
                target_pos = np.array([x, y]) * scale
                
                delta = target_pos - current_pos
                dx, dy = delta[0], delta[1]
                
                # EOS: 1 if end of stroke (last point in stroke), else 0
                eos = 1.0 if i == num_points - 1 else 0.0
                eod = 0.0 
                
                actions.append((dx, dy, eos, eod))
                current_pos = target_pos
                
        # Add EOD action
        actions.append((0.0, 0.0, 0.0, 1.0))
        
        return actions

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        drawing_idx, step_idx = self.samples[idx]
        drawing = self.drawings[drawing_idx]
        
        # 1. Render Target Image
        target_img = self._render_full_drawing(drawing)
        
        # 2. Render Current State (up to step_idx)
        actions = self._drawing_to_actions(drawing)
        
        # Initialize world
        world = DrawingWorld(width=self.img_size, height=self.img_size)
        
        # Replay actions 0 to step_idx-1
        for t in range(step_idx):
            world.step(actions[t])
            
        current_state = world.get_state() # (H, W) numpy array
        
        # 3. Get Target Action (at step_idx)
        target_action = actions[step_idx]
        
        # Convert to tensors
        current_state_tensor = torch.from_numpy(current_state).float().unsqueeze(0) / 255.0
        target_img_tensor = torch.from_numpy(target_img).float().unsqueeze(0) / 255.0
        
        action_tensor = torch.tensor(target_action, dtype=torch.float32)
        
        return current_state_tensor, target_img_tensor, action_tensor

    def _render_full_drawing(self, drawing):
        """Render the complete drawing to an image."""
        img = Image.new("L", (self.img_size, self.img_size), color=255)
        draw = ImageDraw.Draw(img)
        
        scale = self.img_size / 255.0
        
        for stroke in drawing:
            x_points = stroke[0]
            y_points = stroke[1]
            points = list(zip(x_points, y_points))
            
            if len(points) > 1:
                scaled_points = [(x * scale, y * scale) for x, y in points]
                draw.line(scaled_points, fill=0, width=2)
            elif len(points) == 1:
                x, y = points[0]
                x, y = x * scale, y * scale
                draw.point((x, y), fill=0)
                
        return np.array(img)
