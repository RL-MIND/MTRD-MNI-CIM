"""Registry for the three architectures used in the paper workflows."""

from .dncnn import DnCNN
from .unet import UNet
from .vgg import VGG16


MODEL_REGISTRY = {
    "vgg16": VGG16,
    "dncnn": DnCNN,
    "unet": UNet,
}


def get_model(name, **kwargs):
    """Construct a supported model by its public name."""
    try:
        constructor = MODEL_REGISTRY[name]
    except KeyError as error:
        available = ", ".join(sorted(MODEL_REGISTRY))
        raise ValueError(f"unknown model {name!r}; available models: {available}") from error
    return constructor(**kwargs)


def list_models():
    """Return supported model names in stable order."""
    return sorted(MODEL_REGISTRY)
