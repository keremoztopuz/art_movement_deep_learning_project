import timm
from config import MODEL_NAME, NUM_CLASSES, DROP_RATE


def create_model(model_name=None, num_classes=None, pretrained=True):
    model_name = model_name or MODEL_NAME
    num_classes = num_classes or NUM_CLASSES
    
    model = timm.create_model(
        model_name,
        pretrained=pretrained,
        num_classes=num_classes,
        drop_rate=DROP_RATE
    )
    
    return model


def get_model_info(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "total_params_m": f"{total_params / 1e6:.2f}M",
        "trainable_params_m": f"{trainable_params / 1e6:.2f}M"
    }
