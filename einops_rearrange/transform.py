import itertools
import numpy as np
import math
from collections import OrderedDict
from typing import Dict, List, Tuple, Union, NamedTuple, Optional

from .errors import EinopsError
from .parser import AnonymousAxis, Parser


class ExtractedInfo(NamedTuple):
    """
    Extracted information from pattern necessary for further analysis

    """
    known_unk: List[Tuple[List[int], List[int]]] 
    axis_lengths: List[int]
    permutation_order: List[int]
    axis_position_map: Dict[Union[str, AnonymousAxis], int]
    output_structure: List[List[Union[str, AnonymousAxis]]]
    added_axes: Dict[int, int]

    def __str__(self):
        return "\n".join([f"{k} : {v}" for k,v in self._asdict().items()]) 
    

def extract_information(pattern: str, known_axes: Dict[str, int], ndim: int) -> ExtractedInfo:
    """
    Parses the einops pattern and extracts relevant transformation information.

    """
    splits = pattern.split("->")
    if len(splits) != 2:
        raise EinopsError(f"Invalid pattern: {pattern}")

    in_pattern, out_pattern = Parser(splits[0], is_input=True), Parser(splits[1], is_input=False)

    if not in_pattern.identifiers or not out_pattern.identifiers:
        raise EinopsError(f"Missing identifiers in the pattern : {pattern}")

    if in_pattern.has_ellipsis and not out_pattern.has_ellipsis:
        raise EinopsError(f"Ellipsis present in the left side but missing from the right: {pattern}")

    difference = in_pattern.identifiers - out_pattern.identifiers
    if difference:
        raise EinopsError(f"Unexpected identifiers on the left side: {difference}")

    axes_without_size = (
        {ax for ax in out_pattern.identifiers if not isinstance(ax, AnonymousAxis)} - in_pattern.identifiers - known_axes.keys()
    )
    if axes_without_size:
        raise EinopsError(f"Specify sizes for new axes: {axes_without_size}")

    # Update pattern structures to account for ellipsis
    if in_pattern.has_ellipsis:
        static_dims = len(in_pattern.structure) - 1
        if ndim < static_dims:
            raise EinopsError(f"Shape mismatch: pattern requires at least {static_dims} dimensions, but tensor has {ndim}.")

        ellipsis_count = ndim - static_dims
        generated_axes = [f"ELLIPSIS_{i}" for i in range(ellipsis_count)]

        in_pattern_struct = []
        for axis in in_pattern.structure:
            if axis == "ELLIPSIS":
                in_pattern_struct.extend([[ax] for ax in generated_axes])
            else:
                in_pattern_struct.append(axis)

        out_pattern_struct = []
        for axis in out_pattern.structure:

            if isinstance(axis, str) and axis == "ELLIPSIS":
                out_pattern_struct.extend([[ax] for ax in generated_axes])
            
            elif isinstance(axis, list) and "ELLIPSIS" in axis:
                index = axis.index("ELLIPSIS")
                new_axis = axis.copy()  
                new_axis[index:index+1] = generated_axes
                out_pattern_struct.append(new_axis)
            
            else:
                out_pattern_struct.append(axis)

    else:
        expected_dims = len(in_pattern.structure)
        if ndim != expected_dims:
            raise EinopsError(f"Shape mismatch: pattern requires {expected_dims} dimensions, but tensor has {ndim}.")
        in_pattern_struct, out_pattern_struct = in_pattern.structure, out_pattern.structure

    axis_len_map: Dict[Union[str, AnonymousAxis], int] = OrderedDict() # Axis-to-Length mapping

    in_pattern_ordered = list(itertools.chain.from_iterable(in_pattern_struct))

    _unknown = -1
    for ax in in_pattern_ordered:
        axis_len_map[ax] = known_axes[ax] if ax in known_axes else _unknown

    out_pattern_ordered = list(itertools.chain.from_iterable(out_pattern_struct))

    repeat_axes_pos: Dict[Union[str, AnonymousAxis], int] = {}

    # Mapping the axis lengths that are already known  
    for pos, ax in enumerate(out_pattern_ordered):
        if ax not in axis_len_map:
            if isinstance(ax, AnonymousAxis):
                axis_len_map[ax] = ax.value
            else:
                axis_len_map[ax] = known_axes[ax] if ax in known_axes else _unknown

            repeat_axes_pos[ax] = pos

    for ax in known_axes:
        if ax not in axis_len_map:
            raise EinopsError(f"Unspecified axis passed as an argument: {ax}")
        
    # Input known_unknown mapping (Number of known and unknown lengths for every axis) 
    in_known_unk: List[Tuple[List[int], List[int]]] = [] 
    axis_pos_map = {name: pos for pos, name in enumerate(axis_len_map)} # Axis-to-Position mapping

    for composite_axis in in_pattern_struct:
        known = [axis_pos_map[axis] for axis in composite_axis if axis_len_map[axis] != _unknown]
        unknown = [axis_pos_map[axis] for axis in composite_axis if axis_len_map[axis] == _unknown]

        if len(unknown) > 1:
            raise EinopsError(f"Could not infer sizes for axes {[in_pattern_ordered[i] for i in unknown]}.")

        in_known_unk.append((known, unknown))

    # Evaluate permutation order 
    permutation_order = [in_pattern_ordered.index(axis) for axis in out_pattern_ordered if axis in in_pattern_ordered]
    # Additional axes for repetition
    added_axes = {pos: axis_len_map[ax] for ax, pos in repeat_axes_pos.items()}

    return ExtractedInfo(in_known_unk, list(axis_len_map.values()), permutation_order, axis_pos_map, out_pattern_struct, added_axes)


def infer_shape(shape: Tuple[int, ...], extracted_info: ExtractedInfo) -> Tuple[Optional[List[int]], List[int], bool]:
    """
    Infers any required shape changes during transformations.

    """
    init_reshape_dims = None # shape after initial reshaping
    need_init_reshape = False 

    # Calculating remaining axis lengths
    for (known, unk), dim in zip(extracted_info.known_unk, shape):
        product = math.prod(extracted_info.axis_lengths[ax] for ax in known) if known else 1

        if len(unk) == 1:
            extracted_info.axis_lengths[unk[0]] = dim // product
        elif dim != product:
            raise EinopsError(f"Shape mismatch: {dim} != {product}")

        if len(known) + len(unk) != 1:
            need_init_reshape = True

    init_reshape_dims = extracted_info.axis_lengths[:len(extracted_info.permutation_order)] if need_init_reshape else None
    final_shapes = [] 

    # Calculating resultant final shape
    for axis in extracted_info.output_structure:
        prod = math.prod(extracted_info.axis_lengths[extracted_info.axis_position_map[ax]] for ax in axis) if axis else 1
        final_shapes.append(prod)
    
    # If final reshape requried
    need_final_reshape = any(len(axis) != 1 for axis in extracted_info.output_structure)
    
    return init_reshape_dims, final_shapes, need_final_reshape


def apply_transform(tensor: np.ndarray, extracted_info: ExtractedInfo) -> np.ndarray:
    """
    Transforms the input tensor based on provided pattern

    """
    shape = tensor.shape
    init_reshape_dims, final_shapes, need_final_reshape = infer_shape(shape, extracted_info)

    # Initial reshape 
    if init_reshape_dims:
        tensor = tensor.reshape(init_reshape_dims)

    # Rearrangement
    if extracted_info.permutation_order and extracted_info.permutation_order != list(range(len(extracted_info.permutation_order))):
        tensor = tensor.transpose(extracted_info.permutation_order)

    # Repetition
    for axis_position, axis_length in extracted_info.added_axes.items():
        tensor = np.expand_dims(tensor, axis=axis_position)
        tensor = np.broadcast_to(tensor, [
            axis_length if i == axis_position else size for i, size in enumerate(tensor.shape)
        ])

    # Final reshaping
    if final_shapes and need_final_reshape:
        tensor = tensor.reshape(final_shapes)

    return tensor
