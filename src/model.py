#model presentation

import timm
from config import MODEL_NAME, NUM_CLASSES

def create_model():
    model = timm.create_model(
        MODEL_NAME,
        num_classes=NUM_CLASSES,
        pretrained=True
    )
     
    return model
