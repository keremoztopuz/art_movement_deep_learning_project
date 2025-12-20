# Art Movement Deep Learning Project

A deep learning project for classifying art movements from paintings using PyTorch and transfer learning.

## 🎯 Features

- **10 Art Movement Classes**: Baroque, Cubism, Impressionism, Abstract Expressionism, Pop Art, Rococo, Northern Renaissance, Expressionism, Art Nouveau Modern, Color Field Painting
- **Multiple Model Architectures**: ConvNeXt-Tiny, EfficientNet-B4, MobileNetV3
- **Advanced Training**: Focal Loss, CutMix augmentation, Label Smoothing
- **Ensemble Support**: Train and evaluate multiple models together
- **TTA (Test Time Augmentation)**: Improved accuracy at inference time
- **CoreML Export**: Deploy to iOS devices

## 📊 Results

| Model | Accuracy | Val Loss |
|-------|----------|----------|
| ConvNeXt-Tiny | 71% | 0.66 |
| EfficientNet-B4 | 63% | 0.84 |
| Ensemble + TTA | ~75% | - |

## 📁 Project Structure

```
art_movement_deep_learning_project/
├── src/
│   ├── config.py       # Configuration and hyperparameters
│   ├── model.py        # Model creation (ConvNeXt, EfficientNet, MobileNet)
│   ├── dataset.py      # Data loading and augmentation
│   ├── train.py        # Training (single & ensemble)
│   ├── evaluate.py     # Evaluation with TTA support
│   ├── export.py       # CoreML export for iOS
│   ├── utils.py        # Utility functions
│   └── logger_config.py # Logging configuration
├── Images/
│   ├── resized_wikiart/     # Original resized images
│   └── balanced_images/     # Balanced 10-class dataset
├── checkpoints/             # Saved models
├── notebooks/               # Jupyter notebooks
└── resizing/               # Data preprocessing scripts
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train Single Model

```bash
cd src
python train.py
```

### 3. Train Ensemble

```bash
python train.py --ensemble
```

### 4. Evaluate

```bash
# Single model
python evaluate.py

# With TTA
python evaluate.py --tta

# Ensemble + TTA
python evaluate.py --ensemble --tta
```

### 5. Export to CoreML

```bash
python export.py
```

## 🏋️ Training on Kaggle

1. Create new notebook
2. Add dataset: `new-try-dataset2`
3. Enable GPU: Settings → Accelerator → GPU T4 x2
4. Run cells from the notebook

## 📱 iOS Deployment

The exported `.mlmodel` file can be directly used in iOS apps with Core ML framework.

## 👥 Authors

- Kerem Oztopuz
- Ibrahim Arikboga

## 📄 License

MIT License
