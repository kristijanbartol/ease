import os
from pathlib import Path
import numpy as np

# Import the C++ module
from .lib.remesh_module import apply_remesh_and_trace

def apply_remesh(vertices, faces, seamline_pairs, reduction_factor=1.0):
    """
    Remesh a mesh using the C++ remeshing function.
    """
    # Input validation and conversion
    vertices = np.ascontiguousarray(vertices, dtype=np.float32)
    faces = np.ascontiguousarray(faces, dtype=np.int32)
    seamline_pairs = np.ascontiguousarray(seamline_pairs, dtype=np.int32)
    
    # Call C++ function
    new_verts, new_faces, new_seamline_pairs = apply_remesh_and_trace(vertices, faces, seamline_pairs, reduction_factor)
    
    return new_verts, new_faces, new_seamline_pairs
