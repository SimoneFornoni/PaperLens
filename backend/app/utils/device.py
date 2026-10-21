"""Resolve the device for computations"""
import torch

SUPPORTED_DEVICES = ("cpu", "cuda", "mps")

def resolve_device(device: str | None = None) -> str:
    """Return the device to run on, auto-detecting when ``device`` is None.

    Args:
        device: Explicit device name, or None to auto-detect.

    Returns:
        One of the supported devices.
    """
    if device is not None:
        resolved = device.lower()
        if resolved not in SUPPORTED_DEVICES:
            raise ValueError(f"Unsupported device '{device}'; expected one of {SUPPORTED_DEVICES}")
        return resolved

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"