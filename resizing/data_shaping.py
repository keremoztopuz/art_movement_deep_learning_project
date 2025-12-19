import os 
import random
import shutil

DATA_DIR = "/Users/keremoztopuz/Desktop/art_movement_deep_learning_project/Images/resized_wikiart"
OUTPUT_DIR = "/Users/keremoztopuz/Desktop/art_movement_deep_learning_project/Images/balanced_images"

# Kaggle'da seçtiğimiz 15 sınıf (doğru isimlerle)
SELECTED_CLASSES = [
    "Symbolism", "Impressionism", "Baroque", "Expressionism",
    "Romanticism", "Post_Impressionism", "Art_Nouveau_Modern",
    "Realism", "Abstract_Expressionism", "Northern_Renaissance",
    "Naive_Art_Primitivism", "Cubism", "Rococo",
    "Color_Field_Painting", "Pop_Art"
]

SAMPLES_PER_CLASS = 650

os.makedirs(OUTPUT_DIR, exist_ok=True)

for class_name in SELECTED_CLASSES:
    src_folder = os.path.join(DATA_DIR, class_name)
    dst_folder = os.path.join(OUTPUT_DIR, class_name)
    os.makedirs(dst_folder, exist_ok=True)
    
    files = [f for f in os.listdir(src_folder) if f.endswith(".jpg")]
    random.seed(42)
    random.shuffle(files)

    selected_files = files[:SAMPLES_PER_CLASS]
    
    for f in selected_files:
        shutil.copy(os.path.join(src_folder, f), os.path.join(dst_folder, f))

    print(f"{class_name}: {len(selected_files)} image copied")

print(f"\nTotal: {len(SELECTED_CLASSES) * SAMPLES_PER_CLASS} images")
print(f"Destination: {OUTPUT_DIR}")

logger 