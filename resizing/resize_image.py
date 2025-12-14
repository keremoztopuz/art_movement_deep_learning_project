import os
from PIL import Image
from tqdm import tqdm

INPUT_DIR = "/Users/keremoztopuz/Desktop/art_movement_deep_learning_project/cleaned_wikiart"
OUTPUT_DIR = "/Users/keremoztopuz/Desktop/art_movement_deep_learning_project/resized_wikiart"

os.makedirs(OUTPUT_DIR, exist_ok=True)

size = (380, 380)

subfolders = os.listdir(INPUT_DIR)

for subfolder in tqdm(subfolders):
    subfolder_path = os.path.join(INPUT_DIR, subfolder)

    if os.path.isdir(subfolder_path):
        files = os.listdir(subfolder_path)
        os.makedirs(os.path.join(OUTPUT_DIR, subfolder), exist_ok=True)

        for file in files:
            if file.endswith(".jpg"):
                img = Image.open(os.path.join(subfolder_path, file))
                img = img.resize(size, Image.NEAREST)
                img.save(os.path.join(OUTPUT_DIR, subfolder, file))
            
        
    print("Done!")
