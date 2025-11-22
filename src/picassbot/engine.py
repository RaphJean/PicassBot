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

    def reset(self):
        """
        Reset the canvas to blank (white).
        
        Returns:
            np.ndarray: The initial state (blank canvas image).
        """
        self.canvas = Image.new("L", (self.width, self.height), color=255)
        self.draw = ImageDraw.Draw(self.canvas)
        return self.get_state()

    def step(self, action):
        """
        Apply an action to the environment.
        
        Args:
            action (tuple/list): A tuple (x_start, y_start, x_end, y_end, [curvature]).
                                 Coordinates should be normalized [0, 1].
                                 Curvature is optional (default 0).
                                 - 0: Straight line
                                 - >0: Curve to one side
                                 - <0: Curve to the other side
        
        Returns:
            np.ndarray: The new state (canvas image).
        """
        if len(action) == 4:
            x1, y1, x2, y2 = action
            curvature = 0.0
        elif len(action) == 5:
            x1, y1, x2, y2, curvature = action
        else:
            raise ValueError("Action must be a tuple of length 4 or 5")
        
        # Denormalize coordinates
        p1 = np.array([x1 * self.width, y1 * self.height])
        p2 = np.array([x2 * self.width, y2 * self.height])
        
        if abs(curvature) < 1e-3:
            # Draw straight line
            self.draw.line([tuple(p1), tuple(p2)], fill=0, width=self.line_width)
        else:
            # Draw Bezier curve
            self._draw_bezier(p1, p2, curvature)
        
        return self.get_state()

    def _draw_bezier(self, p1, p2, curvature, num_points=20):
        """
        Draw a quadratic Bezier curve.
        Control point is calculated based on curvature offset from midpoint.
        """
        # Midpoint
        mid = (p1 + p2) / 2
        
        # Vector from p1 to p2
        vec = p2 - p1
        dist = np.linalg.norm(vec)
        
        if dist < 1e-6:
            return # Points are too close
            
        # Normal vector (perpendicular to vec)
        normal = np.array([-vec[1], vec[0]])
        normal = normal / np.linalg.norm(normal)
        
        # Control point: midpoint + curvature * distance * normal
        # Curvature represents the "bulge" relative to the length of the segment
        control = mid + normal * curvature * dist
        
        # Generate points along the curve
        t = np.linspace(0, 1, num_points)
        # Quadratic Bezier formula: B(t) = (1-t)^2 * P0 + 2(1-t)t * P1 + t^2 * P2
        # We need to reshape for broadcasting
        p1_ = p1.reshape(1, 2)
        p2_ = p2.reshape(1, 2)
        c_ = control.reshape(1, 2)
        t_ = t.reshape(-1, 1)
        
        points = (1-t_)**2 * p1_ + 2*(1-t_)*t_ * c_ + t_**2 * p2_
        
        # Convert to list of tuples for PIL
        points_list = [tuple(p) for p in points]
        self.draw.line(points_list, fill=0, width=self.line_width)

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
