from typing import Union, List, Dict
import numpy as np
from .transform import extract_information, apply_transform

def rearrange(tensor: Union[np.ndarray, List[np.ndarray]], pattern: str, **axes_lengths: Dict[str, int]) -> np.ndarray:
    """
    Rearranges an input tensor according to the specified pattern.
    
    :param tensor: Input tensor (numpy array or list of numpy arrays)
    :param pattern: String specifying the rearrangement pattern
    :param axes_lengths: Dictionary mapping axis names to their lengths
    :return: Rearranged numpy array
    """
    if isinstance(tensor, list):
        tensor = np.array(tensor)
    
    ndim = tensor.ndim
    
    # Extract pattern information
    extracted_info = extract_information(pattern, axes_lengths, ndim)
    (
        in_known_unk,
        axis_len_map,
        permutation_order,
        repeat_axes_pos,
        axis_pos_map,
        out_pattern_structure,
        added_axes
    ) = extracted_info

    # print(extracted_info)
    
    # Apply transformation
    rearranged_tensor = apply_transform(tensor,
        in_known_unk, axis_len_map, tensor.shape, permutation_order,
        axis_pos_map, out_pattern_structure, added_axes
    )
    
    return rearranged_tensor