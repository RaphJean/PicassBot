import json
import os
import numpy as np
from PIL import Image, ImageDraw

class QuickDrawDataset:
    def __init__(self, data_dir):
        """
        Initialize the dataset loader.
        
        Args:
            data_dir (str): Path to the directory containing .ndjson files.
        """
        self.data_dir = data_dir
        self.categories = [f.split('.')[0] for f in os.listdir(data_dir) if f.endswith('.ndjson')]
        print(f"Found {len(self.categories)} categories: {self.categories}")

    def load_drawings(self, category, max_drawings=1000):
        """
        Load drawings for a specific category.
        
        Args:
            category (str): The category name (e.g., 'apple').
            max_drawings (int): Maximum number of drawings to load.
            
        Returns:
            list: A list of dictionaries, each representing a drawing.
        """
        file_path = os.path.join(self.data_dir, f"{category}.ndjson")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Category '{category}' not found in {self.data_dir}")
        
        drawings = []
        with open(file_path, 'r') as f:
            for i, line in enumerate(f):
                if i >= max_drawings:
                    break
                try:
                    drawings.append(json.loads(line))
                except json.JSONDecodeError:
                    print(f"Error decoding line {i} in {file_path}")
                    continue
        return drawings

    @staticmethod
    def vector_to_image(drawing, image_size=256, line_width=2):
        """
        Convert a vector drawing to a PIL Image.
        
        Args:
            drawing (dict): The drawing dictionary (must contain 'drawing' key).
            image_size (int): Size of the output square image.
            line_width (int): Width of the strokes.
            
        Returns:
            PIL.Image: A grayscale image of the drawing.
        """
        image = Image.new("L", (image_size, image_size), color=255)
        draw = ImageDraw.Draw(image)
        
        strokes = drawing['drawing']
        
        for stroke in strokes:
            x_points = stroke[0]
            y_points = stroke[1]
            
            # Normalize coordinates to image size (Quick Draw is usually 255x255)
            scale = image_size / 256.0
            x_points = [x * scale for x in stroke[0]]
            y_points = [y * scale for y in stroke[1]]
            
            points = list(zip(x_points, y_points))
            if len(points) > 1:
                draw.line(points, fill=0, width=line_width)
            elif len(points) == 1:
                 draw.point(points, fill=0)
                 
        return image

    @staticmethod
    def vector_to_raster_sequence(drawing, image_size=256, line_width=2):
        """
        Convert a vector drawing to a sequence of images (one per stroke).
        Useful for visualizing the drawing process.
        """
        frames = []
        image = Image.new("L", (image_size, image_size), color=255)
        draw = ImageDraw.Draw(image)
        
        strokes = drawing['drawing']
        for stroke in strokes:
            x_points = stroke[0]
            y_points = stroke[1]
            scale = image_size / 256.0
            x_points = [x * scale for x in stroke[0]]
            y_points = [y * scale for y in stroke[1]]
            points = list(zip(x_points, y_points))
            
            if len(points) > 1:
                draw.line(points, fill=0, width=line_width)
            elif len(points) == 1:
                 draw.point(points, fill=0)
            
            frames.append(image.copy())
            
        return frames
