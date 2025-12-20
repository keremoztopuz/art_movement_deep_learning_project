import os
import torch

# paths
DATA_DIR_LOCAL = os.path.join("..", "Images", "balanced_images")
DATA_DIR_KAGGLE = "/kaggle/working/balanced_10class"
DATA_DIR = DATA_DIR_KAGGLE if os.path.exists("/kaggle") else DATA_DIR_LOCAL

# model
MODEL_NAME = "convnext_tiny"
NUM_CLASSES = 10
DROP_RATE = 0.3

# training
LEARNING_RATE = 0.0001
BATCH_SIZE = 32
EPOCHS = 30
PATIENCE = 10

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
CHECKPOINT_DIR = os.path.join("..", "checkpoints")
MODEL_SAVE_PATH = os.path.join(CHECKPOINT_DIR, "best_model.pth")
