"""
Loss functions for DAS autoencoder training.

Configured via config.yaml:
    training.loss_function: "l1" | "huber" | "mse"
    training.huber_delta:   float (only used for huber)
"""

import torch.nn as nn


def get_loss_function(config):
    """
    Factory that returns the loss criterion specified in config.

    Args:
        config: Parsed config dict.

    Returns:
        nn.Module: Loss function ready for training.

    Raises:
        ValueError: If an unknown loss function name is specified.
    """
    name = config["training"].get("loss_function", "l1").lower()

    if name == "l1" or name == "mae":
        return nn.L1Loss()

    elif name == "huber" or name == "smooth_l1":
        delta = config["training"].get("huber_delta", 1.0)
        return nn.HuberLoss(delta=delta)

    elif name == "mse":
        return nn.MSELoss()

    else:
        raise ValueError(
            f"Unknown loss function '{name}'. Choose from: l1, huber, mse"
        )
