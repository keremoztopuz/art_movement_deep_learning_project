# Art Movement Deep Learning Project

Art Movement Classification using Deep Learning with PyTorch and Transfer Learning.

## Project Overview

This project classifies paintings into 10 distinct art movements using ConvNeXt-Tiny architecture with advanced training techniques.

## Results

| Metric | Score |
|--------|-------|
| **Accuracy** | ~75% |
| Precision | 74% |
| Recall | 73% |
| F1 Score | 73% |

## Art Movement Classes (10)

- Abstract Expressionism
- Art Nouveau Modern
- Baroque
- Color Field Painting
- Cubism
- Expressionism
- Impressionism
- Northern Renaissance
- Pop Art
- Rococo

## Technical Details

### Model Architecture
- **ConvNeXt-Tiny** (28M parameters)
- Transfer learning from ImageNet
- Dropout: 0.3

### Training Techniques
- **Focal Loss** (γ=2) - focuses on hard examples
- **CutMix** - advanced data augmentation
- **Label Smoothing** (0.1) - prevents overconfidence
- **CosineAnnealingLR** - smooth learning rate decay
- **AdamW** optimizer with weight decay

### Data Augmentation
- Random Horizontal/Vertical Flip
- Random Rotation (±30°)
- Color Jitter
- Random Affine
- Random Perspective
- Random Erasing

### Dataset
- 10 classes × 650 images = 6,500 total
- Split: 70% train / 15% val / 15% test
- Image size: 224×224

## Experiments Tried

| Experiment | Result |
|------------|--------|
| EfficientNet-B3 | 59% (15 classes) |
| EfficientNet-B4 | 63% |
| MobileNetV3 | 65% |
| **ConvNeXt-Tiny** | **75%** ✓ |
| Ensemble (3 models) | 73% |

## Project Structure

```
art_movement_deep_learning_project/
├── src/
│   ├── config.py      # configuration
│   ├── model.py       # model creation
│   ├── dataset.py     # data loading
│   ├── train.py       # training loop
│   ├── evaluate.py    # evaluation
│   ├── export.py      # coreml export
│   └── utils.py       # utilities
├── checkpoints/       # saved models
├── Images/            # dataset
└── requirements.txt
```

## Usage

### Training
```bash
cd src
python train.py
```

### Evaluation
```bash
python evaluate.py
python evaluate.py --tta  # with test time augmentation
```

### Export to iOS
```bash
python export.py
```

## Requirements

- Python 3.8+
- PyTorch 2.0+
- timm
- scikit-learn
- matplotlib
- seaborn

## Authors

- Kerem Oztopuz
- Ibrahim Arikboga

## License

MIT License
