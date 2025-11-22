import numpy as np
from PIL import Image, ImageDraw

class DrawingWorld:
    def __init__(self, width=128, height=128, line_width=2):
        """
        Initialize the Drawing World.
        
        Args:
            width (int): Width of the canvas.
            height (int): Height of the canvas.
            line_width (int): Width of the drawn strokes.
        """
        self.width = width
        self.height = height
        self.line_width = line_width
        self.canvas = Image.new("L", (width, height), color=255) # White canvas
        self.draw = ImageDraw.Draw(self.canvas)
        self.pen_pos = np.array([0.0, 0.0]) # Normalized pen position (0-1)

    def reset(self):
        """
        Reset the canvas to blank (white) and pen to (0,0).
        
        Returns:
            np.ndarray: The initial state (blank canvas image).
        """
        self.canvas = Image.new("L", (self.width, self.height), color=255)
        self.draw = ImageDraw.Draw(self.canvas)
        self.pen_pos = np.array([0.0, 0.0])
        return self.get_state()

    def step(self, action):
        """
        Apply an action to the environment.
        
        Args:
            action (tuple/list): A tuple (dx, dy, eos, eod).
                                 dx, dy: Relative movement (normalized).
                                 eos: End of stroke (1 if pen up, 0 if pen down).
                                 eod: End of drawing (1 if finished).
        
        Returns:
            np.ndarray: The new state (canvas image).
        """
        dx, dy, eos, eod = action
        
        # Calculate new position
        new_pos = self.pen_pos + np.array([dx, dy])
        # Clip to canvas boundaries
        new_pos = np.clip(new_pos, 0.0, 1.0)
        
        # If pen is down (eos < 0.5), draw line
        if eos < 0.5:
            # Denormalize coordinates
            p1 = (int(self.pen_pos[0] * self.width), int(self.pen_pos[1] * self.height))
            p2 = (int(new_pos[0] * self.width), int(new_pos[1] * self.height))
            
            # Draw the line (black on white)
            self.draw.line([p1, p2], fill=0, width=self.line_width)
        
        # Update pen position
        self.pen_pos = new_pos
        
        return self.get_state()

    def get_state(self):
        """
        Get the current state as a numpy array.
        
        Returns:
            np.ndarray: The current canvas as a numpy array (H, W), values 0-255.
        """
        return np.array(self.canvas)

    def render(self):
        """
        Return the current PIL Image for visualization.
        """
        return self.canvas

    def copy(self):
        """
        Create a deep copy of the DrawingWorld.
        """
        new_world = DrawingWorld(width=self.width, height=self.height, line_width=self.line_width)
        new_world.canvas = self.canvas.copy()
        new_world.draw = ImageDraw.Draw(new_world.canvas)
        new_world.pen_pos = self.pen_pos.copy()
        return new_world
