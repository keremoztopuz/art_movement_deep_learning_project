import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from config import DATA_DIR, IMAGE_SIZE, MEAN, STD, BATCH_SIZE

# train augmentation
train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.2),
    transforms.RandomRotation(30),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.15),
    transforms.RandomAffine(degrees=15, translate=(0.15, 0.15), scale=(0.85, 1.15)),
    transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
    transforms.RandomErasing(p=0.2),
])

# validation transform
val_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])


class ArtDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


def load_data():
    images = []
    labels = []
    
    if not os.path.exists(DATA_DIR):
        raise FileNotFoundError(f"data directory not found: {DATA_DIR}")
    
    for folder in os.listdir(DATA_DIR):
        folder_path = os.path.join(DATA_DIR, folder)
        if os.path.isdir(folder_path):
            for file in os.listdir(folder_path):
                if file.lower().endswith((".jpg", ".jpeg", ".png")):
                    images.append(os.path.join(folder_path, file))
                    labels.append(folder)
    
    if len(images) == 0:
        raise ValueError(f"no images found in {DATA_DIR}")
    
    classes = sorted(list(set(labels)))
    class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
    labels = [class_to_idx[label] for label in labels]
    
    X_train, X_temp, y_train, y_temp = train_test_split(
        images, labels, stratify=labels, test_size=0.3, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, stratify=y_temp, test_size=0.5, random_state=42
    )
    
    return (X_train, y_train), (X_val, y_val), (X_test, y_test), classes


def create_dataloaders(batch_size=None):
    batch_size = batch_size or BATCH_SIZE
    
    (X_train, y_train), (X_val, y_val), (X_test, y_test), classes = load_data()
    
    train_dataset = ArtDataset(X_train, y_train, transform=train_transform)
    val_dataset = ArtDataset(X_val, y_val, transform=val_transform)
    test_dataset = ArtDataset(X_test, y_test, transform=val_transform)
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        num_workers=2, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, 
        num_workers=2, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, 
        num_workers=2, pin_memory=True
    )
    
    return train_loader, val_loader, test_loader, classes, test_dataset


try:
    train_loader, val_loader, test_loader, classes, test_dataset = create_dataloaders()
except FileNotFoundError:
    train_loader = val_loader = test_loader = classes = test_dataset = None
    print("warning: data not loaded. set DATA_DIR correctly.")
