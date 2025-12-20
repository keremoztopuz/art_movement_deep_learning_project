import os
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns

from config import DEVICE, NUM_CLASSES, IMAGE_SIZE, MEAN, STD, CHECKPOINT_DIR, MODEL_SAVE_PATH
from model import create_model
from dataset import test_loader, test_dataset, classes


def get_tta_transforms():
    return [
        transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ]),
        transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ]),
        transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomRotation(degrees=(5, 5)),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ]),
        transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomRotation(degrees=(-5, -5)),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ]),
    ]


def evaluate_model(model_path=None, use_tta=False):
    model_path = model_path or MODEL_SAVE_PATH
    
    model = create_model()
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    
    all_preds = []
    all_labels = []
    
    if use_tta:
        tta_transforms = get_tta_transforms()
        
        with torch.no_grad():
            for i in tqdm(range(len(test_dataset)), desc="tta evaluation"):
                img_path = test_dataset.image_paths[i]
                label = test_dataset.labels[i]
                
                probs_sum = None
                for tta in tta_transforms:
                    img = Image.open(img_path).convert("RGB")
                    img_tensor = tta(img).unsqueeze(0).to(DEVICE)
                    probs = F.softmax(model(img_tensor), dim=1)
                    probs_sum = probs if probs_sum is None else probs_sum + probs
                
                pred = probs_sum.argmax(dim=1).item()
                all_preds.append(pred)
                all_labels.append(label)
    else:
        with torch.no_grad():
            for images, labels in tqdm(test_loader, desc="evaluation"):
                images = images.to(DEVICE)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.numpy())
    
    metrics = {
        "accuracy": accuracy_score(all_labels, all_preds),
        "precision": precision_score(all_labels, all_preds, average="macro", zero_division=0),
        "recall": recall_score(all_labels, all_preds, average="macro", zero_division=0),
        "f1": f1_score(all_labels, all_preds, average="macro", zero_division=0),
    }
    
    return metrics, all_preds, all_labels


def evaluate_ensemble(model_paths=None, use_tta=True):
    if model_paths is None:
        model_paths = [
            ("convnext_tiny", os.path.join(CHECKPOINT_DIR, "model_convnext.pth")),
            ("efficientnet_b4", os.path.join(CHECKPOINT_DIR, "model_effnet.pth")),
            ("mobilenetv3_large_100", os.path.join(CHECKPOINT_DIR, "model_mobile.pth")),
        ]
    
    models = []
    for model_name, path in model_paths:
        if os.path.exists(path):
            model = create_model(model_name=model_name, pretrained=False)
            model.load_state_dict(torch.load(path, map_location=DEVICE))
            model.to(DEVICE)
            model.eval()
            models.append(model)
            print(f"loaded: {model_name}")
        else:
            print(f"warning: {path} not found")
    
    if len(models) == 0:
        raise ValueError("no models loaded for ensemble")
    
    tta_transforms = get_tta_transforms() if use_tta else [get_tta_transforms()[0]]
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for i in tqdm(range(len(test_dataset)), desc="ensemble + tta"):
            img_path = test_dataset.image_paths[i]
            label = test_dataset.labels[i]
            
            ensemble_probs = None
            
            for model in models:
                for tta in tta_transforms:
                    img = Image.open(img_path).convert("RGB")
                    img_tensor = tta(img).unsqueeze(0).to(DEVICE)
                    probs = F.softmax(model(img_tensor), dim=1)
                    ensemble_probs = probs if ensemble_probs is None else ensemble_probs + probs
            
            pred = ensemble_probs.argmax(dim=1).item()
            all_preds.append(pred)
            all_labels.append(label)
    
    metrics = {
        "accuracy": accuracy_score(all_labels, all_preds),
        "precision": precision_score(all_labels, all_preds, average="macro", zero_division=0),
        "recall": recall_score(all_labels, all_preds, average="macro", zero_division=0),
        "f1": f1_score(all_labels, all_preds, average="macro", zero_division=0),
    }
    
    return metrics, all_preds, all_labels


def print_results(metrics, all_preds, all_labels, save_confusion_matrix=True):
    print(f"\naccuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"precision: {metrics['precision']:.4f}")
    print(f"recall:    {metrics['recall']:.4f}")
    print(f"f1 score:  {metrics['f1']:.4f}")
    
    print("\nclassification report:")
    print(classification_report(all_labels, all_preds, target_names=classes))
    
    if save_confusion_matrix:
        cm = confusion_matrix(all_labels, all_preds)
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                    xticklabels=classes, yticklabels=classes)
        plt.xlabel("predicted")
        plt.ylabel("true")
        plt.title("confusion matrix")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig("confusion_matrix.png", dpi=150)
        print("\nconfusion matrix saved to: confusion_matrix.png")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--ensemble", action="store_true")
    parser.add_argument("--tta", action="store_true")
    parser.add_argument("--model", type=str, default=None)
    args = parser.parse_args()
    
    if args.ensemble:
        metrics, preds, labels = evaluate_ensemble(use_tta=args.tta)
    else:
        metrics, preds, labels = evaluate_model(model_path=args.model, use_tta=args.tta)
    
    print_results(metrics, preds, labels)