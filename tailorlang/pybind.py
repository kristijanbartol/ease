import os
from pathlib import Path
import numpy as np

# Import the C++ module
from .lib.remesh_module import apply_remesh_and_trace


def _dict_to_ordered_array(seamline_pairs_dict):
    ordered_array = []
    for seam_name in seamline_pairs_dict:
        ordered_array.append(list(seamline_pairs_dict[seam_name]))
    return np.array(ordered_array, dtype=np.int32)


def _ordered_array_to_dict(new_seamlines_array, seam_names):
    d = {}
    for seam_idx, seam_name in enumerate(seam_names):
        d[seam_name] = new_seamlines_array[seam_idx]
    return d


def apply_remesh(vertices, faces, seamline_pairs_dict, reduction_factor=0.5):
    """
    Remesh a mesh using the C++ remeshing function.
    """
    # Input validation and conversion
    vertices = np.ascontiguousarray(vertices, dtype=np.float32)
    faces = np.ascontiguousarray(faces, dtype=np.int32)
    seamline_pairs = np.ascontiguousarray(_dict_to_ordered_array(seamline_pairs_dict), dtype=np.int32)
    
    # Call C++ function
    new_verts, new_faces, new_seamline_pairs = apply_remesh_and_trace(vertices, faces, seamline_pairs, reduction_factor)
    
    return new_verts, new_faces, _ordered_array_to_dict(new_seamline_pairs, seamline_pairs_dict.keys())
