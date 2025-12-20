import os
import random
import shutil
from config import SELECTED_CLASSES


def create_balanced_dataset(source_dir, output_dir, samples_per_class=650, seed=42):
    os.makedirs(output_dir, exist_ok=True)
    
    results = {}
    
    for class_name in SELECTED_CLASSES:
        src = os.path.join(source_dir, class_name)
        dst = os.path.join(output_dir, class_name)
        
        if not os.path.exists(src):
            print(f"warning: {class_name} not found in source directory")
            continue
        
        os.makedirs(dst, exist_ok=True)
        
        files = [f for f in os.listdir(src) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        random.seed(seed)
        random.shuffle(files)
        
        selected_files = files[:samples_per_class]
        
        for f in selected_files:
            shutil.copy(os.path.join(src, f), os.path.join(dst, f))
        
        results[class_name] = len(selected_files)
        print(f"{class_name}: {len(selected_files)} images")
    
    total = sum(results.values())
    print(f"\ntotal: {len(results)} classes, {total} images")
    
    return results


def get_class_counts(data_dir):
    counts = {}
    
    for folder in os.listdir(data_dir):
        folder_path = os.path.join(data_dir, folder)
        if os.path.isdir(folder_path):
            files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            counts[folder] = len(files)
    
    return counts


def print_class_distribution(data_dir):
    counts = get_class_counts(data_dir)
    
    print("\nclass distribution:")
    for cls, count in sorted(counts.items(), key=lambda x: x[1], reverse=True):
        bar = "█" * (count // 50)
        print(f"{cls:30s} {count:4d} {bar}")
    print(f"\ntotal: {sum(counts.values())} images in {len(counts)} classes")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--balance", action="store_true")
    parser.add_argument("--source", type=str)
    parser.add_argument("--output", type=str)
    parser.add_argument("--samples", type=int, default=650)
    parser.add_argument("--info", type=str)
    args = parser.parse_args()
    
    if args.balance:
        create_balanced_dataset(args.source, args.output, args.samples)
    elif args.info:
        print_class_distribution(args.info)