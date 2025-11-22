import subprocess
import os

def main():
    categories = [
        "cat", "dog", "face", "apple", "tree", 
        "car", "house", "flower", "fish", "bird"
    ]
    
    indices = [0, 1, 2]
    strategy = "policy" # Fast and uses the trained model
    model_path = "policy_checkpoints/policy_epoch_2.pth"
    
    print(f"Generating samples for {len(categories)} categories, {len(indices)} samples each.")
    print(f"Strategy: {strategy}")
    
    for category in categories:
        for idx in indices:
            print(f"\n--- Processing {category} (Index {idx}) ---")
            cmd = [
                "uv", "run", "python", "-m", "research.run_experiments",
                "--strategy", strategy,
                "--model_path", model_path,
                "--target_type", "dataset",
                "--target_category", category,
                "--target_index", str(idx),
                "--no_early_stopping" # Let it draw fully for demo
            ]
            
            try:
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError as e:
                print(f"Error processing {category} index {idx}: {e}")

if __name__ == "__main__":
    main()
