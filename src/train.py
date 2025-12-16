#training loop

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
from config import DEVICE, LEARNING_RATE, EPOCHS
from model import create_model
from dataset import train_loader, val_loader, classes
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

model = create_model()
model.to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
best_val_loss = float('inf')
scheduler = StepLR(optimizer, step_size=7, gamma=0.1)

patient = 5
counter = 0

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    for images, labels in tqdm(train_loader):
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    
    model.eval()
    with torch.no_grad():
        val_running_loss = 0.0
        val_accuracy = 0.0
        val_precision = 0.0
        val_recall = 0.0
        val_f1 = 0.0

        for images, labels in tqdm(val_loader):
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, labels)
            val_running_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            val_accuracy += accuracy_score(labels.cpu().numpy(), predicted.cpu().numpy())
            val_precision += precision_score(labels.cpu().numpy(), predicted.cpu().numpy(), average='macro', zero_division=0)
            val_recall += recall_score(labels.cpu().numpy(), predicted.cpu().numpy(), average='macro', zero_division=0)
            val_f1 += f1_score(labels.cpu().numpy(), predicted.cpu().numpy(), average='macro', zero_division=0)

    epoch_loss = running_loss / len(train_loader)
    val_epoch_loss = val_running_loss / len(val_loader)
    val_epoch_accuracy = val_accuracy / len(val_loader)
    val_epoch_precision = val_precision / len(val_loader)
    val_epoch_recall = val_recall / len(val_loader)
    val_epoch_f1 = val_f1 / len(val_loader)

    scheduler.step()

    if val_epoch_loss < best_val_loss:
        best_val_loss = val_epoch_loss
        counter = 0
        torch.save(model.state_dict(), 'best_model.pth')
    else:
        counter += 1
        if counter >= patient:
            print("Early stopping")
            break

    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {epoch_loss:.4f}, Val Loss: {val_epoch_loss:.4f}, Val Accuracy: {val_epoch_accuracy:.4f}, Val Precision: {val_epoch_precision:.4f}, Val Recall: {val_epoch_recall:.4f}, Val F1: {val_epoch_f1:.4f}")

    if val_epoch_loss < best_val_loss:
        best_val_loss = val_epoch_loss
        torch.save(model.state_dict(), 'best_model.pth')
    