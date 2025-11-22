import os
import glob
from PIL import Image

def make_gif(input_folder, output_file, duration=200):
    images = []
    # Sort files to ensure correct order
    files = sorted(glob.glob(os.path.join(input_folder, "*.png")))
    
    if not files:
        print(f"No images found in {input_folder}")
        return

    print(f"Found {len(files)} images in {input_folder}")
    for filename in files:
        images.append(Image.open(filename))

    if images:
        # Save as GIF
        images[0].save(
            output_file,
            save_all=True,
            append_images=images[1:],
            duration=duration,
            loop=0
        )
        print(f"Saved GIF to {output_file}")

def main():
    base_dir = "research_output"
    strategies = ["greedy", "mpc", "genetic", "cem"]
    
    for strategy in strategies:
        input_folder = os.path.join(base_dir, strategy)
        output_file = os.path.join(base_dir, f"{strategy}.gif")
        make_gif(input_folder, output_file)

if __name__ == "__main__":
    main()
