import os
import torch

# paths
DATA_DIR = os.path.join("..", "Images", "balanced_images")

# model
MODEL_NAME = "convnext_tiny"
NUM_CLASSES = 10
DROP_RATE = 0.3

# training
LEARNING_RATE = 0.0001
BATCH_SIZE = 32
EPOCHS = 50
PATIENCE = 15

# augmentation
IMAGE_SIZE = 224
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

# device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# selected 10 classes
SELECTED_CLASSES = [
    "Baroque", "Cubism", "Impressionism", "Abstract_Expressionism",
    "Pop_Art", "Rococo", "Northern_Renaissance", "Expressionism",
    "Art_Nouveau_Modern", "Color_Field_Painting"
]

# loss settings
FOCAL_GAMMA = 2
LABEL_SMOOTHING = 0.1
CUTMIX_PROB = 0.5
CUTMIX_ALPHA = 1.0

# paths for saving
CHECKPOINT_DIR = "."
MODEL_SAVE_PATH = "best_convnext_acc.pth"
