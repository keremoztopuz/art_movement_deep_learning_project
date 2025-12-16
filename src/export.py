#coreml export

import torch 
import coremltools as ct
from model import create_model
from config import MODEL_NAME, NUM_CLASSES

model = create_model()
model.load_state_dict(torch.load("best_model.pth", map_location="cpu"))
model.eval()

example_input = torch.randn(1, 3, 380, 380)
traced_model = torch.jit.trace(model, example_input)

mlmodel = ct.convert(
    traced_model,
    inputs=[ct.TensorType(shape=example_input.shape)]
)

mlmodel.author = "Kerem Oztopuz, Ibrahim Arıkboga"
mlmodel.license = "MIT"
mlmodel.short_description = "Art Movement Classification"

mlmodel.save("art_movement_classifier.mlmodel")