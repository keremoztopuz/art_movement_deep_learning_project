import os
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns

from config import DEVICE, NUM_CLASSES, IMAGE_SIZE, MEAN, STD, MODEL_SAVE_PATH
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
    ]


def evaluate_model(model_path=None, use_tta=False):
    model_path = model_path or MODEL_SAVE_PATH
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"model not found: {model_path}\nplease provide correct path or train model first.")
    
    model = create_model(pretrained=False)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    
    all_preds = []
    all_labels = []
    
    if use_tta:
        tta_transforms = get_tta_transforms()
        
        with torch.no_grad():
            for i in tqdm(range(len(test_dataset)), desc="evaluation with tta"):
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


def print_results(metrics, all_preds, all_labels, save_plots=True):
    print(f"\n{'='*50}")
    print(f"accuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"precision: {metrics['precision']:.4f}")
    print(f"recall:    {metrics['recall']:.4f}")
    print(f"f1 score:  {metrics['f1']:.4f}")
    print(f"{'='*50}")
    
    print("\nclassification report:")
    print(classification_report(all_labels, all_preds, target_names=classes))
    
    if save_plots:
        cm = confusion_matrix(all_labels, all_preds)
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                    xticklabels=classes, yticklabels=classes)
        plt.xlabel("predicted")
        plt.ylabel("true")
        plt.title(f"confusion matrix - accuracy: {metrics['accuracy']*100:.2f}%")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig("confusion_matrix.png", dpi=150)
        print("\nconfusion matrix saved to: confusion_matrix.png")


if __name__ == "__main__":
    # notebook uyumlu - argparse yerine doğrudan çağrı
    metrics, preds, labels = evaluate_model(use_tta=False)
    print_results(metrics, preds, labels)