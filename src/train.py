#training loop with Focal Loss + CutMix

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import numpy as np
from config import DEVICE, LEARNING_RATE, EPOCHS
from model import create_model
from dataset import train_loader, val_loader, classes
from sklearn.metrics import accuracy_score

# Focal Loss 
class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None, label_smoothing=0.1):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.label_smoothing = label_smoothing
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', label_smoothing=self.label_smoothing)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()

# CutMix 
def cutmix_data(x, y, alpha=1.0):
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)
    
    # Random box koordinatları
    W, H = x.size(2), x.size(3)
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    
    x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
    
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
    
    return x, y, y[index], lam

def cutmix_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# Model ve optimizer
model = create_model()
model.to(DEVICE)
criterion = FocalLoss(gamma=2, label_smoothing=0.1)
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
best_val_loss = float('inf')
scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

patience = 15
counter = 0
cutmix_prob = 0.5  

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        if np.random.random() < cutmix_prob:
            images, targets_a, targets_b, lam = cutmix_data(images, labels, alpha=1.0)
            optimizer.zero_grad()
            outputs = model(images)
            loss = cutmix_criterion(criterion, outputs, targets_a, targets_b, lam)
        else:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
        
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