import itertools
from typing import Dict, Optional, Union, List, Tuple
from collections import OrderedDict
import numpy as np

from . import EinopsError
from .parser import AnonymousAxis, Parser

def extract_information(pattern: str, known_axes: Dict[str, int], ndim: int):
    splits = pattern.split("->")
    if len(splits) != 2:
        raise EinopsError(f"Invalid pattern: {pattern}")
    
    in_pattern, out_pattern = Parser(splits[0], is_input=True), Parser(splits[1], is_input=False)
    
    if in_pattern.has_ellipsis and not out_pattern.has_ellipsis:
        raise EinopsError(f"Ellipsis present in the left side but missing from the right: {pattern}")
    
    difference = in_pattern.identifiers - out_pattern.identifiers
    if difference:
        raise EinopsError(f"Unexpected identifiers on the left side: {difference}")
    
    axes_without_size = (
        {ax for ax in out_pattern.identifiers if not isinstance(ax, AnonymousAxis)}
        - in_pattern.identifiers - known_axes.keys()
    )
    if axes_without_size:
        raise EinopsError(f"Specify sizes for new axes: {axes_without_size}")
    
    if in_pattern.has_ellipsis:
        static_dims = len(in_pattern.structure) - 1
        if ndim < static_dims:
            raise EinopsError(f"Shape mismatch: pattern requires at least {static_dims} dimensions, but tensor has {ndim}.")
        
        ellipsis_count = ndim - static_dims
        generated_axes = [f"ELLIPSIS_{i}" for i in range(ellipsis_count)]
        in_pattern_struct = []
        
        for axis_group in in_pattern.structure:
            if axis_group == "ELLIPSIS":
                in_pattern_struct.extend([[ax] for ax in generated_axes])
            else:
                in_pattern_struct.append(axis_group)
        
        out_pattern_struct = []
        for axis_group in out_pattern.structure:
            if axis_group == "ELLIPSIS":
                out_pattern_struct.extend([[ax] for ax in generated_axes])
            else:
                refined_group = [ax if ax != "ELLIPSIS" else generated_axes for ax in axis_group]
                out_pattern_struct.append(
                    [elem for sublist in refined_group for elem in (sublist if isinstance(sublist, list) else [sublist])]
                )
        
        in_pattern.identifiers.update(generated_axes)
        in_pattern.identifiers.discard("ELLIPSIS")
        if out_pattern.has_ellipsis:
            out_pattern.identifiers.update(generated_axes)
            out_pattern.identifiers.discard("ELLIPSIS")
    else:
        expected_dims = len(in_pattern.structure)
        if ndim != expected_dims:
            raise EinopsError(f"Shape mismatch: pattern requires {expected_dims} dimensions, but tensor has {ndim}.")
        in_pattern_struct, out_pattern_struct = in_pattern.structure, out_pattern.structure
    
    axis_len_map: Dict[Union[str, AnonymousAxis], int] = OrderedDict()
    in_pattern_ordered = list(itertools.chain(*in_pattern_struct))
    _unknown = -1
    
    for ax in in_pattern_ordered:
        axis_len_map[ax] = known_axes.get(ax, _unknown)
    
    out_pattern_ordered = list(itertools.chain(*out_pattern_struct))
    repeat_axes_pos: Dict[Union[str, AnonymousAxis], int] = {}
    
    for pos, ax in enumerate(out_pattern_ordered):
        if ax not in axis_len_map:
            if isinstance(ax, AnonymousAxis):
                axis_len_map[ax] = ax.value
            else:
                axis_len_map[ax] = known_axes.get(ax, _unknown)
            repeat_axes_pos[ax] = pos
    
    for ax in known_axes:
        if ax not in axis_len_map:
            raise EinopsError(f"Unspecified axis passed as an argument: {ax}")
    
    in_known_unk: List[Tuple[List[int], List[int]]] = []
    axis_pos_map = {name: pos for pos, name in enumerate(axis_len_map)}
    
    for composite_axis in in_pattern_struct:
        known = [axis_pos_map[axis] for axis in composite_axis if axis_len_map[axis] != _unknown]
        unknown = [axis_pos_map[axis] for axis in composite_axis if axis_len_map[axis] == _unknown]
        
        if len(unknown) > 1:
            raise EinopsError(f"Could not infer sizes for {unknown}")
        
        in_known_unk.append((known, unknown))
    
    permutation_order = [in_pattern_ordered.index(axis) for axis in out_pattern_ordered if axis in in_pattern_ordered]
    added_axes = {pos: axis_len_map[ax] for ax, pos in repeat_axes_pos.items()}
    
    # print(in_known_unk)
    # print("+"*30)
    # print(axis_len_map)
    # print("+"*30)
    # print(permutation_order)
    # print("+"*30)
    # print(repeat_axes_pos)
    # print("+"*30)
    # print(axis_pos_map)
    # print("+"*30)
    # print(out_pattern_struct)
    # print("+"*30)
    # print(added_axes)

    return in_known_unk, list(axis_len_map.values()), permutation_order, repeat_axes_pos, axis_pos_map, out_pattern_struct, added_axes

def infer_shape(known_unk, axis_lens, shape, perm_order, axis_pos_map, out_struct):

    init_reshape_dims: Optional[List[int]] = None
    
    for axis, dim in zip(known_unk, shape):
        known, unk = axis
        product = np.prod([axis_lens[ax] for ax in known]) if known else 1
        
        if len(unk) == 1:
            axis_lens[unk[0]] = dim // product
        elif dim != product:
            raise EinopsError(f"Shape mismatch: {dim} != {product}")
        
        if len(known) + len(unk) != 1:
            init_reshape_dims = [axis_lens[i] for i in range(len(perm_order))]
    
    final_shapes = [np.prod([axis_lens[axis_pos_map[ax]] for ax in axis]) if axis else 1 for axis in out_struct]
    print(final_shapes)

    need_final_reshape = any(len(axis) != 1 for axis in out_struct)
    
    return init_reshape_dims, final_shapes, need_final_reshape

def apply_transform(tensor, known_unk, axis_lens, shape, perm_order, axis_pos_map, out_struct, added_axes):
    init_reshape_dims, final_shapes, need_final_reshape = infer_shape(
        known_unk, axis_lens, shape, perm_order, axis_pos_map, out_struct
    )
    
    if init_reshape_dims:
        tensor = np.reshape(tensor, init_reshape_dims)
    if perm_order and perm_order != list(range(len(perm_order))):
        tensor = np.transpose(tensor, perm_order)
    
    for axis_position, axis_length in added_axes.items():
        tensor = np.expand_dims(tensor, axis_position)
        tensor = np.tile(tensor, [axis_length if i == axis_position else 1 for i in range(len(tensor.shape))])
    
    if final_shapes and need_final_reshape:
        tensor = np.reshape(tensor, final_shapes)
    
    return tensor
