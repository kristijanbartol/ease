# Misc geometric utils (geometry.py).

import numpy as np
from typing import Tuple, List

def compute_triangle_directions(vertices: np.ndarray, faces: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Compute warp and weft directions and unit distance points for each triangle in a mesh.
    
    Args:
        vertices: (V, 3) array of vertex coordinates
        faces: (F, 3) array of face indices
    
    Returns:
        Tuple containing:
        - List of barycentric coordinates for C_x points
        - List of barycentric coordinates for C_y points
    """
    UNIT_DISTANCE = 0.005
    c_x_coords = []
    c_y_coords = []
    
    def compute_barycentric(p: np.ndarray, v1: np.ndarray, v2: np.ndarray, v3: np.ndarray) -> np.ndarray:
        """
        Compute barycentric coordinates using area method.
        """
        # Compute vectors
        v0 = v2 - v1
        v1_vec = v3 - v1
        v2_vec = p - v1
        
        # Compute dot products
        d00 = np.dot(v0, v0)
        d01 = np.dot(v0, v1_vec)
        d11 = np.dot(v1_vec, v1_vec)
        d20 = np.dot(v2_vec, v0)
        d21 = np.dot(v2_vec, v1_vec)
        
        # Compute barycentric coordinates
        denom = d00 * d11 - d01 * d01
        beta = (d11 * d20 - d01 * d21) / denom
        gamma = (d00 * d21 - d01 * d20) / denom
        alpha = 1.0 - beta - gamma
        
        return np.array([alpha, beta, gamma])
    
    for face in faces:
        # Get triangle vertices
        v1, v2, v3 = vertices[face]
        
        # Compute triangle normal
        edge1 = v2 - v1
        edge2 = v3 - v1
        normal = np.cross(edge1, edge2)
        normal = normal / np.linalg.norm(normal)
        
        # Project global X and Y axes onto triangle plane
        global_x = np.array([1.0, 0.0, 0.0])
        global_y = np.array([0.0, 1.0, 0.0])
        
        weft = global_x - np.dot(global_x, normal) * normal
        warp = global_y - np.dot(global_y, normal) * normal
        
        # Normalize projected directions
        weft = weft / np.linalg.norm(weft)
        warp = warp / np.linalg.norm(warp)
        
        # Compute triangle centroid in world coordinates
        centroid = (v1 + v2 + v3) / 3.0
        
        # Compute unit distance points
        c_x = centroid + weft * UNIT_DISTANCE
        c_y = centroid + warp * UNIT_DISTANCE
        
        # Compute barycentric coordinates
        c_x_coords.append(compute_barycentric(c_x, v1, v2, v3))
        c_y_coords.append(compute_barycentric(c_y, v1, v2, v3))
    
    return c_x_coords, c_y_coords
