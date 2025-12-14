#training loop

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from config import DEVICE, LEARNING_RATE, EPOCHS
from model import create_model
from dataset import train_loader, val_loader, classes

