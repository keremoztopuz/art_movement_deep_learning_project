#training loop

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import numpy as np 
from config import DEVICE, LEARNING_RATE, EPOCHS
from model import create_model
from dataset import train_loader, val_loader, classes, y_train
from sklearn.metrics import accuracy_score

# Class weights for imbalanced data
class_counts = np.bincount(y_train)
class_weights = 1.0 / class_counts
class_weights = class_weights / class_weights.sum() * len(class_counts)
class_weights = torch.FloatTensor(class_weights).to(DEVICE)

model = create_model()
model.to(DEVICE)
criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
best_val_loss = float('inf')
scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

patience = 10
counter = 0

def mixup_data(x, y, alpha=0.2):
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        # MixUp uygula
        images, targets_a, targets_b, lam = mixup_data(images, labels, alpha=0.2)

        optimizer.zero_grad()
        outputs = model(images)
        loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    
    model.eval()
    val_loss = 0.0
    val_acc = 0.0
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Val"):
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            val_acc += accuracy_score(labels.cpu().numpy(), predicted.cpu().numpy())

    epoch_loss = running_loss / len(train_loader)
    val_epoch_loss = val_loss / len(val_loader)
    val_epoch_acc = val_acc / len(val_loader)

    scheduler.step()

    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {epoch_loss:.4f}, Val Loss: {val_epoch_loss:.4f}, Val Acc: {val_epoch_acc:.4f}")

    if val_epoch_loss < best_val_loss:
        best_val_loss = val_epoch_loss
        counter = 0
        torch.save(model.state_dict(), 'best_model.pth')
    else:
        counter += 1
        if counter >= patience:
            print("Early stopping")
            break

print("Training complete!")