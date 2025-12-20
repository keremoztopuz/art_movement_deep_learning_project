import os
import torch
import coremltools as ct
from model import create_model
from config import MODEL_NAME, NUM_CLASSES, IMAGE_SIZE, CHECKPOINT_DIR, MODEL_SAVE_PATH


def export_to_coreml(model_path=None, output_path=None, model_name=None):
    model_path = model_path or MODEL_SAVE_PATH
    output_path = output_path or os.path.join(CHECKPOINT_DIR, "art_classifier.mlmodel")
    
    model = create_model(model_name=model_name, pretrained=False)
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


def export_best_model_for_mobile():
    models_to_try = [
        ("mobilenetv3_large_100", os.path.join(CHECKPOINT_DIR, "model_mobile.pth")),
        ("convnext_tiny", os.path.join(CHECKPOINT_DIR, "model_convnext.pth")),
        (MODEL_NAME, MODEL_SAVE_PATH),
    ]
    
    for model_name, model_path in models_to_try:
        if os.path.exists(model_path):
            print(f"exporting: {model_name}")
            return export_to_coreml(model_path=model_path, model_name=model_name)
    
    raise FileNotFoundError("no model found to export")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()
    
    if args.model:
        export_to_coreml(model_path=args.model, output_path=args.output)
    else:
        export_best_model_for_mobile()