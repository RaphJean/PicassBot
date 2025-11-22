import os
import requests
from tqdm import tqdm
import concurrent.futures

# Base URL for the simplified dataset
BASE_URL = "https://storage.googleapis.com/quickdraw_dataset/full/simplified/"

# List of categories (we can fetch this dynamically or hardcode/read from a file)
# For robustness, we'll fetch the list from the official repo's categories.txt if possible,
# but since that might be complex, we can use a known list or just try to download common ones.
# Actually, the best way is to read the categories from a source. 
# Let's try to fetch the list of categories from the GitHub repo first.
CATEGORIES_URL = "https://raw.githubusercontent.com/googlecreativelab/quickdraw-dataset/master/categories.txt"

DATA_DIR = "data/raw"

def get_categories():
    print("Fetching category list...")
    response = requests.get(CATEGORIES_URL)
    if response.status_code == 200:
        categories = response.text.strip().split('\n')
        print(f"Found {len(categories)} categories.")
        return categories
    else:
        print("Failed to fetch categories list.")
        return []

def download_category(category):
    filename = f"{category}.ndjson"
    url = f"{BASE_URL}{filename}"
    filepath = os.path.join(DATA_DIR, filename)
    
    if os.path.exists(filepath):
        # Check if file is not empty (basic check)
        if os.path.getsize(filepath) > 0:
            return f"Skipped {category} (already exists)"
    
    try:
        # Stream download to handle large files
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            total_size = int(r.headers.get('content-length', 0))
            
            with open(filepath, 'wb') as f, tqdm(
                desc=category,
                total=total_size,
                unit='iB',
                unit_scale=True,
                unit_divisor=1024,
                leave=False
            ) as bar:
                for chunk in r.iter_content(chunk_size=8192):
                    size = f.write(chunk)
                    bar.update(size)
        return f"Downloaded {category}"
    except Exception as e:
        return f"Error downloading {category}: {e}"

def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    categories = get_categories()
    
    if not categories:
        return

    print(f"Starting download of {len(categories)} categories to {DATA_DIR}...")
    
    # Use ThreadPoolExecutor for parallel downloads
    # Be careful not to use too many workers to avoid rate limiting or bandwidth saturation
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        # Submit all download tasks
        future_to_category = {executor.submit(download_category, cat): cat for cat in categories}
        
        # Process results as they complete
        for future in tqdm(concurrent.futures.as_completed(future_to_category), total=len(categories), desc="Overall Progress"):
            cat = future_to_category[future]
            try:
                result = future.result()
                # print(result) # Optional: print result of each download
            except Exception as exc:
                print(f'{cat} generated an exception: {exc}')

    print("Download complete.")

if __name__ == "__main__":
    main()
