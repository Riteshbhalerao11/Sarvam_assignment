from typing import Union
import numpy as np
from .transform import extract_information, apply_transform, ExtractedInfo

def rearrange(tensor: Union[np.ndarray, list], pattern: str, **axes_lengths: int) -> np.ndarray:
    """
    Rearranges an input tensor according to the specified pattern.

    Args:
        tensor (Union[np.ndarray, list]): Input tensor (NumPy array or list of NumPy arrays).
        pattern (str): String specifying the rearrangement pattern.
        axes_lengths (dict): Dictionary mapping axis names to their lengths.

    Returns:
        np.ndarray: Rearranged NumPy array.

    Raises:
        ValueError: If the input tensor is not a valid NumPy array or cannot be transformed.
        EinopsError: If there are inconsistencies in the pattern transformation.
    """
    if isinstance(tensor, list):
        tensor = np.array(tensor)

    ndim = tensor.ndim

    # Extract transformation details
    extracted_info: ExtractedInfo = extract_information(pattern, axes_lengths, ndim)
    
    # Apply transformation using extracted information
    return apply_transform(tensor, extracted_info)