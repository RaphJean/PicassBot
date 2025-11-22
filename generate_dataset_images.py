import os
from picassbot.quickdraw import QuickDrawDataset

def main():
    categories = [
        "cat", "dog", "face", "apple", "tree", 
        "car", "house", "flower", "fish", "bird"
    ]
    
    data_dir = "data/raw"
    output_dir = "image_mosaique/dataset"
    os.makedirs(output_dir, exist_ok=True)
    
    qd = QuickDrawDataset(data_dir)
    
    for category in categories:
        print(f"Processing {category}...")
        drawings = qd.load_drawings(category, max_drawings=1000)
        
        for idx in range(3):
            if idx < len(drawings):
                img = qd.vector_to_image(drawings[idx], image_size=128, line_width=2)
                output_path = os.path.join(output_dir, f"{category}_{idx}.png")
                img.save(output_path)
                print(f"  Saved {output_path}")
    
    print(f"\nDone! Saved {len(categories) * 3} images to {output_dir}/")

if __name__ == "__main__":
    main()
