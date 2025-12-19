#paths and hyperparameters

import os
import torch

DATA_DIR = os.path.join("..", "Images", "resized_wikiart")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "convnext_tiny"

NUM_CLASSES = 15
LEARNING_RATE = 0.00001
BATCH_SIZE = 24
EPOCHS = 50

IMAGE_SIZE = 224
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

