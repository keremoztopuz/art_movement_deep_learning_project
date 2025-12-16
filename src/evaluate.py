#test and metrics

import torch
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.metrics import (accuracy_score, precision_score, 
recall_score, f1_score, confusion_matrix, classification_report)
from tqdm import tqdm
from config import DEVICE, NUM_CLASSES
from model import create_model
from dataset import test_loader, classes

model = create_model()
model.load_state_dict(torch.load("best_model.pth"))
model.eval()
model.to(DEVICE)

all_predictions = []
all_labels = []

with torch.no_grad():
    for images, labels in tqdm(test_loader):
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)
        outputs = model(images)
        max_value = torch.argmax(outputs, dim=1)

        all_predictions.extend(max_value.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

final_acc = accuracy_score(all_labels, all_predictions)
final_prec = precision_score(all_labels, all_predictions, average="macro", zero_division=0)
final_rec = recall_score(all_labels, all_predictions, average="macro", zero_division=0)
final_f1 = f1_score(all_labels, all_predictions, average="macro", zero_division=0)

print("Final Accuracy: ", final_acc)
print("Final Precision: ", final_prec)
print("Final Recall: ", final_rec)
print("Final F1 Score: ", final_f1)

cm = confusion_matrix(all_labels, all_predictions)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.savefig("confusion_matrix.png")

print(classification_report(all_labels, all_predictions))