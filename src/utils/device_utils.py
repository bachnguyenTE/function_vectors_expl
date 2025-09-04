"""
Device utility functions for cross-platform compatibility.
Provides MPS support for macOS while maintaining CUDA compatibility for other platforms.
"""

import torch
import platform


def get_optimal_device():
    """
    Get the optimal device for the current platform.
    
    Returns:
        str: Device string ('mps', 'cuda', or 'cpu')
    """
    if torch.cuda.is_available():
        return "cuda"
    elif platform.system() == "Darwin" and torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def to_device(tensor, device=None):
    """
    Move tensor to the specified device or optimal device.
    
    Args:
        tensor: PyTorch tensor to move
        device: Target device (if None, uses optimal device)
        
    Returns:
        torch.Tensor: Tensor moved to the specified device
    """
    if device is None:
        device = get_optimal_device()
    return tensor.to(device)


def manual_seed_all(seed):
    """
    Set random seeds for all available backends.
    
    Args:
        seed (int): Random seed value
    """
    import random
    import numpy as np
    import os
    
    # Random seed
    random.seed(seed)
    
    # Numpy seed
    np.random.seed(seed)
    
    # Torch seed
    torch.manual_seed(seed)
    
    # Platform-specific seeds
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True
    
    if platform.system() == "Darwin" and torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    
    # OS seed
    os.environ['PYTHONHASHSEED'] = str(seed)


def zeros_like_on_device(tensor, device=None):
    """
    Create a zeros tensor with the same shape as input tensor on specified device.
    
    Args:
        tensor: Reference tensor for shape
        device: Target device (if None, uses optimal device)
        
    Returns:
        torch.Tensor: Zero tensor on the specified device
    """
    if device is None:
        device = get_optimal_device()
    return torch.zeros_like(tensor).to(device)


def randn_like_on_device(tensor, device=None):
    """
    Create a random normal tensor with the same shape as input tensor on specified device.
    
    Args:
        tensor: Reference tensor for shape
        device: Target device (if None, uses optimal device)
        
    Returns:
        torch.Tensor: Random normal tensor on the specified device
    """
    if device is None:
        device = get_optimal_device()
    return torch.randn_like(tensor).to(device)
