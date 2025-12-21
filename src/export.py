import os
import torch
import coremltools as ct
from model import create_model
from config import IMAGE_SIZE, CHECKPOINT_DIR, MODEL_SAVE_PATH


def export_to_coreml(model_path=None, output_path=None):
    model_path = model_path or MODEL_SAVE_PATH
    output_path = output_path or os.path.join(CHECKPOINT_DIR, "art_classifier.mlpackage")
    
    model = create_model(pretrained=False)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    
    example_input = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)
    traced_model = torch.jit.trace(model, example_input)
    
    mlmodel = ct.convert(
        traced_model,
        inputs=[ct.TensorType(shape=example_input.shape, name="image")],
        outputs=[ct.TensorType(name="classLabelProbs")],
        minimum_deployment_target=ct.target.iOS15
    )
    
    mlmodel.author = "Kerem Oztopuz, Ibrahim Arikboga"
    mlmodel.license = "MIT"
    mlmodel.short_description = "art movement classification - 10 classes"
    mlmodel.version = "1.0"
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    mlmodel.save(output_path)
    
    print(f"coreml model saved to: {output_path}")
    return output_path


if __name__ == "__main__":
    export_to_coreml()