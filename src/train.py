import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score

from config import (
    DEVICE, LEARNING_RATE, EPOCHS, PATIENCE,
    FOCAL_GAMMA, LABEL_SMOOTHING, CUTMIX_PROB, CUTMIX_ALPHA,
    CHECKPOINT_DIR, MODEL_SAVE_PATH
)
from model import create_model
from dataset import train_loader, val_loader


class FocalLoss(nn.Module):
    def __init__(self, gamma=FOCAL_GAMMA, label_smoothing=LABEL_SMOOTHING):
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(
            inputs, targets, reduction='none', 
            label_smoothing=self.label_smoothing
        )
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()


def cutmix_data(x, y, alpha=CUTMIX_ALPHA):
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)
    
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


def train_model(model_name=None, save_path=None, epochs=None):
    epochs = epochs or EPOCHS
    save_path = save_path or MODEL_SAVE_PATH
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    model = create_model(model_name=model_name)
    model.to(DEVICE)
    
    criterion = FocalLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    
    best_val_acc = 0.0
    patience_counter = 0
    
    print(f"\ntraining: {model_name or 'convnext_tiny'}")
    print(f"device: {DEVICE}, epochs: {epochs}\n")
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        for images, labels in tqdm(train_loader, desc=f"epoch {epoch+1}/{epochs}"):
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            
            if np.random.random() < CUTMIX_PROB:
                images, targets_a, targets_b, lam = cutmix_data(images, labels)
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
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(DEVICE)
                labels = labels.to(DEVICE)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        epoch_loss = running_loss / len(train_loader)
        val_acc = val_correct / val_total
        
        scheduler.step()
        
        print(f"epoch {epoch+1}/{epochs} | train loss: {epoch_loss:.4f} | val acc: {val_acc:.4f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
            print(f"  model saved: {save_path}")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"\nearly stopping at epoch {epoch+1}")
                break
    
    print(f"\ntraining completed! best val accuracy: {best_val_acc:.4f}\n")
    return best_val_acc


if __name__ == "__main__":
    train_model()