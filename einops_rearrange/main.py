from typing import Union
import numpy as np
from .transform import extract_information, apply_transform, ExtractedInfo

def rearrange(tensor: np.ndarray, pattern: str, **axes_lengths: int) -> np.ndarray:
    """
    Rearranges an input tensor according to the specified pattern.

    """
    if not isinstance(tensor,np.ndarray) : raise TypeError("Expected numpy array as input")
    
    for v in axes_lengths.values():
        if not isinstance(v, int):
            raise TypeError("All provided axis lengths must be integers.")
        if v <= 0:
            raise ValueError("All provided axis lengths must be positive.")

    ndim = tensor.ndim

    # Extract transformation details
    extracted_info: ExtractedInfo = extract_information(pattern, axes_lengths, ndim)
    
    # Apply transformation using extracted information
    return apply_transform(tensor, extracted_info)