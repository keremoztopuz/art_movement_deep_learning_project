#paths and hyperparameters

import os
import torch

DATA_DIR = os.path.join("..", "Images", "resized_wikiart")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "efficientnet_b3"

NUM_CLASSES = 27
LEARNING_RATE = 0.0001
BATCH_SIZE = 24
EPOCHS = 50

IMAGE_SIZE = 300
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

