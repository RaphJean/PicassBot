import os
from picassbot.quickdraw import QuickDrawDataset

def main():
    # Setup paths
    data_dir = "data/raw"
    output_dir = "output_demo"
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize dataset
    print(f"Initializing dataset from {data_dir}...")
    dataset = QuickDrawDataset(data_dir)
    
    # Check if we have any categories
    if not dataset.categories:
        print("No categories found yet. Waiting for downloads to finish...")
        return

    # Pick a category (e.g., the first one found)
    category = dataset.categories[0]
    print(f"Loading drawings for category: {category}")
    
    try:
        drawings = dataset.load_drawings(category, max_drawings=5)
        print(f"Loaded {len(drawings)} drawings.")
        
        for i, drawing in enumerate(drawings):
            # Convert to image
            img = dataset.vector_to_image(drawing)
            
            # Save image
            filename = f"{category}_{i}.png"
            filepath = os.path.join(output_dir, filename)
            img.save(filepath)
            print(f"Saved {filepath}")
            
            # Also save sequence for the first one
            if i == 0:
                frames = dataset.vector_to_raster_sequence(drawing)
                for j, frame in enumerate(frames):
                    frame.save(os.path.join(output_dir, f"{category}_{i}_step_{j}.png"))
                print(f"Saved {len(frames)} step frames for the first drawing.")
                
    except Exception as e:
        print(f"Error processing category {category}: {e}")

if __name__ == "__main__":
    main()
