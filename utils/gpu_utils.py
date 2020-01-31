import torch
from torch import nn


def get_gpu_if_available(gpu: int = None):
    # Device agnostic setting.
    return torch.device(f'cuda:{gpu}') if torch.cuda.is_available() and (gpu is not None) else torch.device('cpu')


def get_single_model_device(model: nn.Module) -> torch.device:
    """Function for retrieving device of a model, assuming that it is on a single device.

    Args:
        model: The model, assumed to be on a single device.

    Returns:
        The device that the model is in.
    """
    return next(model.parameters()).device
